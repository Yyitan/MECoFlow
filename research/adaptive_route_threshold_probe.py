"""Validation-only threshold/temperature schedules for frozen route blending.

This probe never loads test logits. It extends the fixed two-route adaptive
blend with a preregistered monotone entropy threshold family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.float(), dim=1)
    return -(p * p.clamp_min(1e-12).log()).sum(1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def pair_support(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    ai = {name: i for i, name in enumerate(meta["attrs"])}
    oi = {name: i for i, name in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pa[train_pair], minlength=len(ai)).float()
    oc = torch.bincount(po[train_pair], minlength=len(oi)).float()
    support = normalize(torch.log1p(torch.minimum(ac[pa], oc[po])))
    seen = torch.bincount(train_pair, minlength=len(meta["pairs"])) > 0
    return support, seen


def schedules(u: torch.Tensor) -> list[dict[str, object]]:
    rows = []
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        rows.append({"name": f"hard_{threshold:.1f}", "alpha": (u >= threshold).float()})
        rows.append({"name": f"reverse_hard_{threshold:.1f}", "alpha": (u <= threshold).float()})
    for slope in (1.0, 2.0, 4.0, 8.0, 16.0):
        for threshold in (0.25, 0.5, 0.75):
            rows.append({
                "name": f"sigmoid_{slope:g}_{threshold:.2f}",
                "alpha": torch.sigmoid(slope * (u - threshold)),
            })
            rows.append({
                "name": f"reverse_sigmoid_{slope:g}_{threshold:.2f}",
                "alpha": torch.sigmoid(slope * (threshold - u)),
            })
    return rows


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
    sources = protocol["validation_sources"]
    loaded = []
    for item in sources:
        path = Path(item["path"]).resolve()
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        # The archived source artifacts also contain test logits for later
        # fixed replay; this probe deliberately accesses only payload["val"].
        if set(payload) not in ({"val"}, {"val", "test"}):
            raise ValueError("unexpected source keys")
        loaded.append(payload["val"].float())
    n = int(protocol["family_size"])
    iso = torch.stack([zscore(x) for x in loaded[:n]]).mean(0)
    diag = torch.stack([zscore(x) for x in loaded[n:]]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    support, pair_seen = pair_support(cache)
    # A is the frozen AUC specialist; B is the frozen balanced specialist.
    inverse = (1.0 - support).clamp_min(1e-6)[None] * diag + support[None] * iso
    soft = support.clamp_min(1e-6).pow(0.25)[None] * diag + (1.0 - support.clamp_min(1e-6).pow(0.25))[None] * iso
    pair_scope = pair_seen.float() + (1.0 - pair_seen.float()) * 0.25
    delta_a = inverse - text_logits
    delta_b = soft * pair_scope[val["pair"].long()][:, None] * 1.5
    u = normalize(entropy(val["logits"]))
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    for item in schedules(u):
        alpha = pair_seen[val["pair"].long()].float() + (1.0 - pair_seen[val["pair"].long()].float()) * item["alpha"]
        fused = evaluator(val["logits"].float() + (1.0 - alpha[:, None]) * delta_a + alpha[:, None] * delta_b, val["attr"], val["obj"], val["pair"])
        rows.append({"name": item["name"], "mean_alpha": float(alpha.mean()), "auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"], "fused_val": fused})
    eligible = [row for row in rows if row["auc_gain"] >= float(protocol["auc_gate"])]
    result = {
        "status": "passed",
        "mode": "validation_only_entropy_threshold_temperature",
        "test_read": False,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(Path(x["path"]).resolve()), "sha256": x["sha256"]} for x in sources],
        "selection": "retain AUC gain >= gate and maximize HM, then AUC",
        "base_val": base,
        "best": max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None,
        "rows": rows,
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best": result["best"], "candidate_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
