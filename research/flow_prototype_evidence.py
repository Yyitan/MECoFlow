"""Validation-only prototype evidence from a frozen compositional flow.

This diagnostic bypasses the learned global cosine classifier.  Seen classes use
real feature centroids and unseen classes use means of samples from a frozen flow.
It tests whether classifier competition, rather than generated support quality, is
the bottleneck in a large CZSL candidate space.
"""

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

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    FastOpenWorldEvaluator,
    additive_completion,
    real_centroids,
    zscore,
)
from research.piflow_pilot import PrimitiveConditionalFlow  # noqa: E402
from research.warmstart_flow_classifier import (  # noqa: E402
    generate_unseen,
    residual_bank,
)
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--residual-distribution",
        choices=("empirical", "gaussian_diag", "gaussian_isotropic"),
        default="gaussian_isotropic",
    )
    parser.add_argument("--transport-objective", choices=("flow_matching", "direct_endpoint"), default="flow_matching")
    parser.add_argument("--pair-cond-scale", type=float, default=1.0)
    parser.add_argument("--pair-guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def gaussianize(residuals: torch.Tensor, distribution: str, seed: int) -> torch.Tensor:
    if distribution == "empirical":
        return residuals
    generator = torch.Generator().manual_seed(seed + 7919)
    mean = residuals.mean(0)
    std = residuals.std(0, unbiased=False).clamp_min(1e-6)
    if distribution == "gaussian_isotropic":
        std = std.square().mean().sqrt().expand_as(std)
    return mean + torch.randn(residuals.shape, generator=generator) * std


def main() -> None:
    args = parse_args()
    set_seed(args.seed + 1000)
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    additive = additive_completion(cache, centroids, counts, 30)
    residuals = gaussianize(
        residual_bank(cache, additive), args.residual_distribution, args.seed
    )
    flow = PrimitiveConditionalFlow(
        cache["train"]["com"].shape[1], args.hidden_dim, args.cond_dim
    ).to(device)
    flow.load_state_dict(torch.load(args.flow_path, map_location="cpu", weights_only=True))
    flow.eval()
    generated, generated_labels = generate_unseen(
        flow, cache, additive, residuals, counts, args.residual_scale,
        args, device, local_tables=None,
    )

    prototypes = centroids.clone().float()
    for pair_id in torch.unique(generated_labels).tolist():
        prototypes[pair_id] = F.normalize(
            generated[generated_labels == pair_id].float().mean(0), dim=-1
        )
    prototypes = F.normalize(prototypes, dim=-1)
    torch.save(prototypes.half(), work_dir / "flow_prototypes.pt")

    val = {key: value.to(device) for key, value in cache["val"].items()}
    active_mask = torch.tensor(
        cache["meta"].get("active", [True] * len(prototypes)),
        dtype=torch.float32,
        device=device,
    )
    prototype_logits = val["com"].float() @ prototypes.to(device).T
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    text_logits = val["com"].float() @ text.T
    if cache["meta"].get("open_world", False):
        evaluator = FastOpenWorldEvaluator(cache, device)
    else:
        evaluator = FastClosedWorldEvaluator(
            CompositionDataset(args.dataset_path, phase="val"), device
        )
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    prototype_stats = evaluator(
        prototype_logits, val["attr"], val["obj"], val["pair"]
    )
    proto_z = zscore(prototype_logits)
    text_z = zscore(text_logits)
    trials: dict[str, object] = {}
    best = None
    for delta_name, delta in {
        "full": proto_z * active_mask[None],
        "update": (proto_z - text_z) * active_mask[None],
    }.items():
        for strength in (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            stats = evaluator(
                val["logits"].float() + strength * delta,
                val["attr"], val["obj"], val["pair"],
            )
            trials[f"{delta_name}_{strength:.2f}"] = stats
            candidate = (stats["AUC"], stats["best_hm"], delta_name, strength)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    gain = best[0] - base["AUC"]
    result = {
        "method": {
            "evidence": "flow_prototype",
            "residual_distribution": args.residual_distribution,
            "transport_objective": args.transport_objective,
            "pair_cond_scale": args.pair_cond_scale,
            "pair_guidance_scale": args.pair_guidance_scale,
            "samples_per_class": args.samples_per_class,
            "seed": args.seed,
        },
        "base_val": base,
        "prototype_val": prototype_stats,
        "selection": {
            "delta": best[2],
            "strength": best[3] if gain >= args.min_val_auc_gain else 0.0,
            "raw_strength": best[3],
            "val_auc": best[0],
            "val_hm": best[1],
            "val_auc_gain": gain,
            "min_val_auc_gain": args.min_val_auc_gain,
            "enabled": gain >= args.min_val_auc_gain,
        },
        "trials": trials,
    }
    (work_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selection": result["selection"], "prototype_val": prototype_stats}, indent=2), flush=True)


if __name__ == "__main__":
    main()
