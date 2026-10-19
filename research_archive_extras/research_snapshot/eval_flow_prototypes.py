"""Evaluate flow-generated class distributions without a learned classifier."""

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
    additive_completion,
    real_centroids,
    zscore,
)
from research.piflow_pilot import PrimitiveConditionalFlow  # noqa: E402
from research.warmstart_flow_classifier import (  # noqa: E402
    generate_unseen,
    local_residual_tables,
    residual_bank,
)
from utils import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--pair-cond-scale", type=float, default=1.0)
    parser.add_argument("--residual-bank-mode", choices=("global", "local"), default="global")
    parser.add_argument("--semantic-neighbors", type=int, default=16)
    parser.add_argument("--primitive-bonus", type=float, default=0.25)
    parser.add_argument("--residual-scales", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    additive = additive_completion(cache, centroids, counts, 30)
    residuals = residual_bank(cache, additive)
    local_tables = local_residual_tables(cache, cache["train"]["pair"].long(), args, device)
    flow = PrimitiveConditionalFlow(
        cache["train"]["com"].shape[1], args.hidden_dim, args.cond_dim
    ).to(device)
    flow.load_state_dict(torch.load(args.flow_path, map_location="cpu", weights_only=True))
    flow.eval()

    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.to(device).T)
    test_text = zscore(test["com"].float() @ text.to(device).T)
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    output: dict[str, object] = {"base_val": base_val, "base_test": base_test, "variants": {}}
    best = None
    strengths = (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
    for residual_scale in args.residual_scales:
        generated, labels = generate_unseen(
            flow, cache, additive, residuals, counts, residual_scale, args, device, local_tables
        )
        generated_sum = torch.zeros_like(centroids)
        generated_sum.index_add_(0, labels, generated)
        generated_count = torch.bincount(labels, minlength=len(centroids)).clamp_min(1)
        generated_mean = F.normalize(generated_sum / generated_count[:, None], dim=-1)
        unseen = counts == 0
        concentration_sum = torch.zeros(len(centroids))
        concentration_sum.index_add_(
            0, labels, (generated * generated_mean[labels]).sum(-1)
        )
        reliability = {
            "concentration": concentration_sum / generated_count,
            "text_alignment": (generated_mean * text).sum(-1),
            "prior_agreement": (generated_mean * additive).sum(-1),
        }
        standardized = []
        for score in reliability.values():
            values = score[unseen]
            standardized.append(
                (score - values.mean()) / values.std().clamp_min(1e-6)
            )
        reliability["combined"] = torch.stack(standardized).mean(0)
        masks = {
            "all": torch.ones(len(centroids), dtype=torch.float32, device=device),
            "unseen": unseen.float().to(device),
        }
        for reliability_name, score in reliability.items():
            for keep_fraction in (0.25, 0.5, 0.75):
                threshold = torch.quantile(score[unseen], 1.0 - keep_fraction)
                masks[f"{reliability_name}_top{keep_fraction:g}"] = (
                    unseen & (score >= threshold)
                ).float().to(device)
        prototypes = {
            "text_seen": text.clone(),
            "visual_seen": centroids.clone(),
        }
        for prototype in prototypes.values():
            prototype[unseen] = generated_mean[unseen]
        for prototype_name, prototype in prototypes.items():
            val_proto = zscore(val["com"].float() @ prototype.to(device).T)
            test_proto = zscore(test["com"].float() @ prototype.to(device).T)
            for delta_name, val_delta, test_delta in (
                ("full", val_proto, test_proto),
                ("update", val_proto - val_text, test_proto - test_text),
            ):
                for mask_name, mask in masks.items():
                    masked_val_delta = val_delta * mask[None]
                    masked_test_delta = test_delta * mask[None]
                    for strength in strengths:
                        stats = val_eval(
                            val["logits"].float() + strength * masked_val_delta,
                            val["attr"], val["obj"], val["pair"],
                        )
                        name = (
                            f"r{residual_scale:g}_{prototype_name}_{delta_name}_"
                            f"{mask_name}_{strength:g}"
                        )
                        output["variants"][name] = {"val": stats}
                        candidate = (
                            stats["AUC"], stats["best_hm"], name, strength,
                            masked_test_delta,
                        )
                        if best is None or candidate[:2] > best[:2]:
                            best = candidate
    assert best is not None
    enabled = best[0] - base_val["AUC"] >= args.min_val_auc_gain
    strength = best[3] if enabled else 0.0
    fused_test = test_eval(
        test["logits"].float() + strength * best[4],
        test["attr"], test["obj"], test["pair"],
    )
    output["selection"] = {
        "variant": best[2], "strength": strength, "raw_strength": best[3],
        "val_auc": best[0], "val_hm": best[1],
        "val_auc_gain": best[0] - base_val["AUC"],
        "min_val_auc_gain": args.min_val_auc_gain, "enabled": enabled,
    }
    output["fused_test"] = fused_test
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"selection": output["selection"], "fused_test": fused_test}, indent=2))


if __name__ == "__main__":
    main()
