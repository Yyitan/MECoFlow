"""Validation-only pair-level routing between isotropic and diagonal sources."""

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

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def supports(cache: dict[str, object]) -> dict[str, torch.Tensor]:
    meta = cache["meta"]
    attrs = {n: i for i, n in enumerate(meta["attrs"])}
    objs = {n: i for i, n in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    oc = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    attr = normalize(torch.log1p(ac[pair_attr]))
    obj = normalize(torch.log1p(oc[pair_obj]))
    pair = normalize(torch.log1p(torch.minimum(ac[pair_attr], oc[pair_obj])))
    return {"pair": pair, "attr": attr, "obj": obj}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--iso-logits", nargs=3, required=True)
    parser.add_argument("--diag-logits", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    iso_paths = [Path(p).resolve() for p in args.iso_logits]
    diag_paths = [Path(p).resolve() for p in args.diag_logits]
    iso = torch.stack([zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in iso_paths])
    diag = torch.stack([zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in diag_paths])
    iso_mean, diag_mean = iso.mean(0), diag.mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    s = supports(cache)
    # Family confidence is unsupervised: route toward the family with lower
    # within-family disagreement for each image/class column.
    iso_disp = (iso - iso_mean).abs().mean(0).mean(0)
    diag_disp = (diag - diag_mean).abs().mean(0).mean(0)
    reliability_diag = (1.0 / (diag_disp + 1e-6)) / (1.0 / (diag_disp + 1e-6) + 1.0 / (iso_disp + 1e-6))
    reliability_diag = normalize(reliability_diag)
    route_rows = []
    for route_name, raw in {
        "constant": torch.full_like(s["pair"], 0.5),
        "pair_support": s["pair"],
        "attr_support": s["attr"],
        "obj_support": s["obj"],
        "inverse_pair_support": 1.0 - s["pair"],
        "disagreement_reliability": reliability_diag,
        "inverse_disagreement_reliability": 1.0 - reliability_diag,
    }.items():
        for power in (0.25, 0.5, 1.0, 2.0):
            if route_name == "constant":
                alpha = raw
            else:
                alpha = raw.clamp_min(1e-6).pow(power)
            # alpha is diagonal weight; retain the mean scale of both experts.
            score = alpha[None, :] * diag_mean + (1.0 - alpha[None, :]) * iso_mean
            for delta_name, delta in (("full", score), ("update", score - text_logits)):
                for strength in (0.5, 0.75, 1.0, 1.25, 1.5):
                    route_rows.append((route_name, power, delta_name, strength, delta))
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    for route_name, power, delta_name, strength, delta in route_rows:
        fused = evaluator(val["logits"].float() + strength * delta, val["attr"], val["obj"], val["pair"])
        rows.append({"route": route_name, "power": power, "delta": delta_name, "strength": strength, "auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"], "fused_val": fused})
    eligible = [r for r in rows if r["auc_gain"] >= 0.002]
    best_balanced = max(eligible, key=lambda r: (r["hm_gain"], r["auc_gain"])) if eligible else None
    best_auc = max(rows, key=lambda r: (r["auc_gain"], r["hm_gain"]))
    result = {"mode":"validation_only_pair_source_router", "test_read":False, "cache":{"path":str(cache_path),"sha256":sha256(cache_path)}, "sources":[{"path":str(p),"sha256":sha256(p)} for p in iso_paths+diag_paths], "base_val":base, "candidate_grid":{"routes":["constant","pair_support","attr_support","obj_support","inverse_pair_support","disagreement_reliability","inverse_disagreement_reliability"],"powers":[0.25,0.5,1.0,2.0],"strengths":[0.5,0.75,1.0,1.25,1.5],"selection":"retain AUC gain >= 0.002, maximize HM"}, "best_balanced":best_balanced, "best_auc":best_auc, "rows":rows}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_balanced":best_balanced, "best_auc":best_auc}, indent=2))


if __name__ == "__main__":
    main()
