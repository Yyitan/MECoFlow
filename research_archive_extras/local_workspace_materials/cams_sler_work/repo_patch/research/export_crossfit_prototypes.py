"""Export pair-held-out flow prototypes for leakage-free metric learning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.fast_feature_evaluation import additive_completion, real_centroids  # noqa: E402
from research.pair_holdout_flow_eval import edge_preserving_holdout, gaussianize  # noqa: E402
from research.piflow_pilot import PrimitiveConditionalFlow  # noqa: E402
from research.warmstart_flow_classifier import generate_unseen, residual_bank  # noqa: E402
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--heldout-fold", type=int, required=True)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-distribution", default="gaussian_isotropic")
    parser.add_argument("--transport-objective", default="flow_matching")
    parser.add_argument("--pair-cond-scale", type=float, default=1.0)
    parser.add_argument("--pair-guidance-scale", type=float, default=0.5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed + 1000)
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    fold_cache, heldout = edge_preserving_holdout(
        cache, args.folds, args.heldout_fold, args.seed
    )
    centroids, counts = real_centroids(fold_cache)
    additive = additive_completion(fold_cache, centroids, counts, 30)
    residuals = gaussianize(
        residual_bank(fold_cache, additive), args.residual_distribution, args.seed
    )
    active = torch.zeros(len(counts), dtype=torch.bool)
    active[heldout] = True
    generation_cache = dict(fold_cache)
    generation_cache["meta"] = dict(fold_cache["meta"])
    generation_cache["meta"]["active"] = active.tolist()
    flow = PrimitiveConditionalFlow(
        fold_cache["train"]["com"].shape[1], args.hidden_dim, args.cond_dim
    ).to(device)
    flow.load_state_dict(torch.load(args.flow_path, map_location="cpu", weights_only=True))
    flow.eval()
    generated, labels = generate_unseen(
        flow, generation_cache, additive, residuals, counts,
        args.residual_scale, args, device, local_tables=None,
    )
    prototypes, variance = [], []
    for pair_id in heldout.tolist():
        samples = generated[labels == pair_id].float()
        mean = F.normalize(samples.mean(0), dim=-1)
        prototypes.append(mean)
        variance.append((samples - mean).square().sum(-1).mean())
    prototypes_tensor = torch.stack(prototypes)
    variance_tensor = torch.stack(variance)

    full_centroids, _ = real_centroids(cache)
    alignment = F.cosine_similarity(prototypes_tensor, full_centroids[heldout], dim=-1)
    additive_alignment = F.cosine_similarity(
        additive[heldout], full_centroids[heldout], dim=-1
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pair_ids": heldout,
            "prototypes": prototypes_tensor.half(),
            "additive": additive[heldout].half(),
            "variance": variance_tensor,
            "fold": args.heldout_fold,
            "seed": args.seed,
        },
        output,
    )
    summary = {
        "heldout_pairs": len(heldout),
        "mean_real_alignment": alignment.mean().item(),
        "median_real_alignment": alignment.median().item(),
        "mean_additive_alignment": additive_alignment.mean().item(),
        "mean_flow_minus_additive_alignment": (
            alignment - additive_alignment
        ).mean().item(),
        "mean_variance": variance_tensor.mean().item(),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
