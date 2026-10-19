"""Validation-only Pareto blend for the original UT checkpoint."""

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
from research.fast_feature_evaluation import FastClosedWorldEvaluator, real_centroids, zscore  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def support(cache: dict[str, object]) -> torch.Tensor:
    meta = cache["meta"]
    ai = {name: index for index, name in enumerate(meta["attrs"])}
    oi = {name: index for index, name in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    tp = cache["train"]["pair"].long()
    ac = torch.bincount(pa[tp], minlength=len(ai)).float()
    oc = torch.bincount(po[tp], minlength=len(oi)).float()
    raw = torch.log1p(torch.minimum(ac[pa], oc[po]))
    return (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)


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
    iso_paths = [Path(path).resolve() for path in args.iso_logits]
    diag_paths = [Path(path).resolve() for path in args.diag_logits]
    iso = torch.stack([zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float()) for path in iso_paths]).mean(0)
    diag = torch.stack([zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float()) for path in diag_paths]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    s = support(cache)
    global_score = s.clamp_min(1e-6).pow(0.25)[None] * diag + (1.0 - s.clamp_min(1e-6).pow(0.25)[None]) * iso
    selective_score = s.clamp_min(1e-6).pow(0.5)[None] * diag + (1.0 - s.clamp_min(1e-6).pow(0.5)[None]) * iso
    _, counts = real_centroids(cache)
    unseen = (counts == 0).float()
    global_delta = global_score - text_logits
    selective_delta = selective_score * unseen[None] * 1.5
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    for blend_selective in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        delta = (1.0 - blend_selective) * global_delta * 1.25 + blend_selective * selective_delta
        fused = evaluator(val["logits"].float() + delta, val["attr"], val["obj"], val["pair"])
        rows.append({"blend_selective": blend_selective, "auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"], "fused_val": fused})
    eligible = [row for row in rows if row["auc_gain"] >= 0.002]
    result = {"mode":"validation_only_ut_original_pareto_blend","test_read":False,"cache":{"path":str(cache_path),"sha256":sha256(cache_path)},"sources":[{"path":str(path),"sha256":sha256(path)} for path in iso_paths+diag_paths],"candidates":{"A":"pair_support^0.25/all/update x 1.25","B":"pair_support^0.5/unseen-only/full x 1.5"},"blend_grid":[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],"base_val":base,"best_balanced":max(eligible,key=lambda row:(row["hm_gain"],row["auc_gain"])) if eligible else None,"best_auc":max(rows,key=lambda row:(row["auc_gain"],row["hm_gain"])),"rows":rows}
    output_path = Path(args.output); output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps({"best_balanced":result["best_balanced"],"best_auc":result["best_auc"]}, indent=2))


if __name__ == "__main__":
    main()
