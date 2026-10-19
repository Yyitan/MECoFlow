"""Validation-only entropy/support/source-disagreement risk probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def entropy(x: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(x.float(), dim=1)
    return normalize(-(p * p.clamp_min(1e-12).log()).sum(1))


def support_seen(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    ai = {n: i for i, n in enumerate(meta["attrs"])}
    oi = {n: i for i, n in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pa[train_pair], minlength=len(ai)).float()
    oc = torch.bincount(po[train_pair], minlength=len(oi)).float()
    support = normalize(torch.log1p(torch.minimum(ac[pa], oc[po])))
    seen = torch.bincount(train_pair, minlength=len(meta["pairs"])) > 0
    return support, seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_validation":
        raise ValueError("protocol must be preregistered")
    cache_path = Path(protocol["cache"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    sources = []
    for item in protocol["validation_sources"]:
        path = Path(item["path"]).resolve()
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if set(payload) not in ({"val"}, {"val", "test"}):
            raise ValueError("unexpected source keys")
        sources.append(payload["val"].float())
    n = int(protocol["family_size"])
    iso = torch.stack([zscore(x) for x in sources[:n]]).mean(0)
    diag = torch.stack([zscore(x) for x in sources[n:]]).mean(0)
    disagreement = normalize((iso - diag).abs().mean(1))
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    support, pair_seen = support_seen(cache)
    pair = val["pair"].long()
    inverse = (1.0 - support).clamp_min(1e-6)[None] * diag + support[None] * iso
    soft = support.clamp_min(1e-6).pow(0.25)[None] * diag + (1.0 - support.clamp_min(1e-6).pow(0.25))[None] * iso
    scope = pair_seen.float() + (1.0 - pair_seen.float()) * 0.25
    delta_a = inverse - text_logits
    delta_b = soft * scope[pair][:, None] * 1.5
    base = val["logits"].float()
    import dataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator
    evaluator = FastClosedWorldEvaluator(dataset.CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base_stats = evaluator(base, val["attr"], val["obj"], pair)
    u = entropy(val["logits"])
    rows = []
    for beta in protocol["beta_values"]:
        for gamma in protocol["gamma_values"]:
            score = normalize(u + float(beta) * (1.0 - support[pair]) + float(gamma) * disagreement)
            for slope in protocol["slope_values"]:
                for threshold in protocol["threshold_values"]:
                    raw = torch.sigmoid(float(slope) * (float(threshold) - score))
                    alpha = pair_seen[pair].float() + (1.0 - pair_seen[pair].float()) * raw
                    fused = base + (1.0 - alpha[:, None]) * delta_a + alpha[:, None] * delta_b
                    stats = evaluator(fused, val["attr"], val["obj"], pair)
                    rows.append({"beta":float(beta),"gamma":float(gamma),"slope":float(slope),"threshold":float(threshold),"auc_gain":stats["AUC"]-base_stats["AUC"],"hm_gain":stats["best_hm"]-base_stats["best_hm"],"mean_alpha":float(alpha.mean()),"fused_val":stats})
    eligible = [r for r in rows if r["auc_gain"] >= float(protocol["auc_gate"])]
    result = {"status":"passed","mode":"validation_only_entropy_support_disagreement_risk","test_read":False,"protocol":str(protocol_path),"protocol_sha256":sha256(protocol_path),"cache":{"path":str(cache_path),"sha256":sha256(cache_path)},"sources":[{"path":str(Path(x["path"]).resolve()),"sha256":x["sha256"]} for x in protocol["validation_sources"]],"candidate_count":len(rows),"selection":"retain AUC gain >= gate and maximize HM, then AUC","base_val":base_stats,"best":max(eligible,key=lambda r:(r["hm_gain"],r["auc_gain"])) if eligible else None,"rows":rows}
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps({"best":result["best"],"candidate_count":len(rows)},indent=2))


if __name__ == "__main__":
    main()
