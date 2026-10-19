"""Validation-only structural-risk selective transport probe.

The gate combines classifier entropy with unseen pair support deficiency.  It
never reads test logits and uses a preregistered finite candidate family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


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


def entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.float(), dim=1)
    return normalize(-(p * p.clamp_min(1e-12).log()).sum(1))


def support_seen(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    ai = {name: i for i, name in enumerate(meta["attrs"])}
    oi = {name: i for i, name in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pa[train_pair], minlength=len(ai)).float()
    obj_count = torch.bincount(po[train_pair], minlength=len(oi)).float()
    support = normalize(torch.log1p(torch.minimum(attr_count[pa], obj_count[po])))
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
        raise ValueError("protocol must be preregistered before validation")
    cache_path = Path(protocol["cache"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    loaded = []
    for item in protocol["validation_sources"]:
        path = Path(item["path"]).resolve()
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if set(payload) not in ({"val"}, {"val", "test"}):
            raise ValueError("unexpected source keys")
        loaded.append(payload["val"].float())
    family_size = int(protocol["family_size"])
    iso = torch.stack([zscore(x) for x in loaded[:family_size]]).mean(0)
    diag = torch.stack([zscore(x) for x in loaded[family_size:]]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    support, pair_seen = support_seen(cache)
    inverse = (1.0 - support).clamp_min(1e-6)[None] * diag + support[None] * iso
    soft = support.clamp_min(1e-6).pow(0.25)[None] * diag + (1.0 - support.clamp_min(1e-6).pow(0.25))[None] * iso
    pair_scope = pair_seen.float() + (1.0 - pair_seen.float()) * 0.25
    pair = val["pair"].long()
    delta_a = inverse - text_logits
    delta_b = soft * pair_scope[pair][:, None] * 1.5
    u = entropy(val["logits"])
    risk = []
    for beta in protocol["beta_values"]:
        risk.append((float(beta), normalize(u + float(beta) * (1.0 - support[pair]))))
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], pair)
    rows = []
    for beta, score in risk:
        for slope in protocol["slope_values"]:
            for threshold in protocol["threshold_values"]:
                alpha_raw = torch.sigmoid(float(slope) * (float(threshold) - score))
                alpha = pair_seen[pair].float() + (1.0 - pair_seen[pair].float()) * alpha_raw
                fused = val["logits"].float() + (1.0 - alpha[:, None]) * delta_a + alpha[:, None] * delta_b
                stats = evaluator(fused, val["attr"], val["obj"], pair)
                rows.append({"beta": beta, "slope": slope, "threshold": threshold,
                             "mean_alpha": float(alpha.mean()),
                             "auc_gain": stats["AUC"] - base["AUC"],
                             "hm_gain": stats["best_hm"] - base["best_hm"],
                             "fused_val": stats})
    eligible = [row for row in rows if row["auc_gain"] >= float(protocol["auc_gate"])]
    result = {
        "status": "passed",
        "mode": "validation_only_entropy_plus_support_risk",
        "test_read": False,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(Path(x["path"]).resolve()), "sha256": x["sha256"]} for x in protocol["validation_sources"]],
        "candidate_count": len(rows),
        "selection": "retain AUC gain >= gate and maximize HM, then AUC",
        "base_val": base,
        "best": max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None,
        "rows": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best": result["best"], "candidate_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
