"""Validation-only cross-seed check for a frozen C-GQA topology gate.

The candidate is frozen from the earlier topology diagnostic. This script never
accesses cache['test'] or logits['test']; it reports only validation metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    real_centroids,
    zscore,
)


def harmonic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 2.0 * left * right / (left + right).clamp_min(1e-6)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--strength", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device(args.device)
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Deliberately bind only the validation branch. The full cache may contain
    # a test branch for legacy compatibility, but this audit never touches it.
    val = {key: value.to(device) for key, value in cache["val"].items()}
    meta = cache["meta"]
    attr_index = {value: index for index, value in enumerate(meta["attrs"])}
    obj_index = {value: index for index, value in enumerate(meta["objs"])}
    pair_attr = torch.tensor(
        [attr_index[a] for a, _ in meta["pairs"]], device=device, dtype=torch.long
    )
    pair_obj = torch.tensor(
        [obj_index[o] for _, o in meta["pairs"]], device=device, dtype=torch.long
    )
    _, pair_counts = real_centroids(cache)
    seen = (pair_counts > 0).to(device)
    # The original diagnostic uses train-image support. Reconstruct it from
    # train pair labels without accessing validation/test image features.
    train_pair = cache["train"]["pair"].long()
    train_pair_attr = pair_attr[train_pair]
    train_pair_obj = pair_obj[train_pair]
    attr_image_support = torch.bincount(
        train_pair_attr.cpu(), minlength=len(meta["attrs"])
    ).float().to(device)
    obj_image_support = torch.bincount(
        train_pair_obj.cpu(), minlength=len(meta["objs"])
    ).float().to(device)
    support = torch.minimum(
        attr_image_support[pair_attr], obj_image_support[pair_obj]
    )
    support_log = torch.log1p(support)
    support_score = (support_log - support_log.min()) / (
        support_log.max() - support_log.min()
    ).clamp_min(1e-6)
    # Unseen-only mask is not used: the frozen candidate gates all pair logits
    # with a continuous reliability weight, matching the prior diagnostic.
    mask = support_score.to(device)

    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_text = zscore(val["com"].float() @ text.T)
    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), device
    )
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    runs = []
    for logits_path in args.classifier_logits:
        path = Path(logits_path).resolve()
        item = torch.load(path, map_location="cpu", weights_only=True)
        if "val" not in item:
            raise ValueError(f"validation logits missing from {path}")
        val_full = zscore(item["val"].float().to(device))
        delta = val_full
        fused = evaluator(
            val["logits"].float() + args.strength * delta * mask[None],
            val["attr"], val["obj"], val["pair"],
        )
        runs.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "base_val": base,
                "fused_val": fused,
                "auc_gain": float(fused["AUC"] - base["AUC"]),
                "hm_gain": float(fused["best_hm"] - base["best_hm"]),
            }
        )
    output = {
        "protocol": {
            "mode": "validation_only",
            "cache": str(cache_path),
            "cache_sha256": file_sha256(cache_path),
            "candidate": "min_image_support_continuous_full",
            "strength": args.strength,
            "test_read": False,
        },
        "base_val": base,
        "runs": runs,
        "mean": {
            "auc_gain": float(np.mean([item["auc_gain"] for item in runs])),
            "hm_gain": float(np.mean([item["hm_gain"] for item in runs])),
            "positive_auc_runs": int(sum(item["auc_gain"] > 0 for item in runs)),
            "positive_hm_runs": int(sum(item["hm_gain"] > 0 for item in runs)),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["mean"], indent=2))


if __name__ == "__main__":
    main()
