"""Pair-held-out evaluation for compositional feature generators.

This protocol never reads the benchmark validation or test split.  It removes a
deterministic, edge-preserving fold of *training* compositions, trains a generator
on the remaining composition graph, synthesizes support for the removed edges,
and measures generalized seen/held-out recognition on their real training images.
The purpose is to select transport mechanisms for unseen-pair transfer without
using downstream validation fusion as a proxy for generator quality.
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

from research.fast_feature_evaluation import additive_completion, real_centroids, zscore  # noqa: E402
from research.generative_classifier_pilot import classifier_logits, train_classifier  # noqa: E402
from research.piflow_pilot import PrimitiveConditionalFlow  # noqa: E402
from research.warmstart_flow_classifier import (  # noqa: E402
    balanced_seen_samples,
    generate_unseen,
    pair_maps,
    residual_bank,
    train_flow,
)
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--flow-path")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--heldout-fold", type=int, default=0)
    parser.add_argument("--flow-epochs", type=int, default=20)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--classifier-batch-size", type=int, default=2048)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument(
        "--transport-objective",
        choices=("flow_matching", "direct_endpoint"),
        default="flow_matching",
    )
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--classifier-lr", type=float, default=3e-3)
    parser.add_argument("--aux-weight", type=float, default=0.05)
    parser.add_argument(
        "--pair-aux-target",
        choices=("real_centroid", "additive_barycenter"),
        default="real_centroid",
    )
    parser.add_argument("--primitive-aux-weight", type=float, default=0.0)
    parser.add_argument("--primitive-transport-weight", type=float, default=0.0)
    parser.add_argument("--pair-cond-scale", type=float, default=1.0)
    parser.add_argument("--pair-guidance-scale", type=float, default=1.0)
    parser.add_argument("--recenter-generated", action="store_true")
    parser.add_argument("--variation-scale", type=float, default=1.0)
    parser.add_argument("--cond-drop", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--residual-distribution",
        choices=("empirical", "gaussian_diag", "gaussian_isotropic"),
        default="gaussian_isotropic",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def edge_preserving_holdout(
    cache: dict[str, object], folds: int, heldout_fold: int, seed: int
) -> tuple[dict[str, object], torch.Tensor]:
    """Remove a pair fold while keeping every held-out primitive in training."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0 <= heldout_fold < folds:
        raise ValueError(f"heldout-fold must be in [0, {folds})")
    labels = cache["train"]["pair"].long()
    seen_ids = torch.unique(labels).sort().values
    generator = torch.Generator().manual_seed(seed + 1729)
    permutation = seen_ids[torch.randperm(len(seen_ids), generator=generator)]
    fold = torch.full((len(cache["meta"]["pairs"]),), -1, dtype=torch.long)
    fold[permutation] = torch.arange(len(permutation)) % folds
    proposed = seen_ids[fold[seen_ids] == heldout_fold]
    pair_attr, pair_obj = pair_maps(cache)

    keep_pair = torch.zeros(len(fold), dtype=torch.bool)
    keep_pair[seen_ids] = True
    keep_pair[proposed] = False
    attr_degree = torch.bincount(pair_attr[keep_pair], minlength=len(cache["meta"]["attrs"]))
    obj_degree = torch.bincount(pair_obj[keep_pair], minlength=len(cache["meta"]["objs"]))
    supported = (attr_degree[pair_attr[proposed]] > 0) & (obj_degree[pair_obj[proposed]] > 0)
    heldout = proposed[supported]
    keep_pair[proposed[~supported]] = True
    keep_instance = keep_pair[labels]

    reduced = dict(cache)
    reduced_train: dict[str, object] = {}
    for key, value in cache["train"].items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == len(labels):
            reduced_train[key] = value[keep_instance]
        else:
            reduced_train[key] = value
    reduced["train"] = reduced_train
    return reduced, heldout


def gaussianize(
    residuals: torch.Tensor, distribution: str, seed: int
) -> torch.Tensor:
    if distribution == "empirical":
        return residuals
    generator = torch.Generator().manual_seed(seed + 7919)
    mean = residuals.mean(0)
    std = residuals.std(0, unbiased=False).clamp_min(1e-6)
    if distribution == "gaussian_isotropic":
        std = std.square().mean().sqrt().expand_as(std)
    return mean + torch.randn(residuals.shape, generator=generator) * std


def generalized_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    seen_ids: torch.Tensor,
    heldout_ids: torch.Tensor,
) -> dict[str, float]:
    logits = zscore(logits.float())
    seen_score, seen_index = logits[:, seen_ids].max(-1)
    heldout_score, heldout_index = logits[:, heldout_ids].max(-1)
    seen_prediction = seen_ids[seen_index]
    heldout_prediction = heldout_ids[heldout_index]
    class_count = logits.shape[1]
    label_count = torch.bincount(labels, minlength=class_count).float().clamp_min(1)
    best_hm = best_seen = best_heldout = 0.0
    curve = []
    for bias in torch.linspace(-4.0, 4.0, 161):
        choose_seen = seen_score - bias >= heldout_score
        prediction = torch.where(choose_seen, seen_prediction, heldout_prediction)
        correct_count = torch.bincount(
            labels,
            weights=(prediction == labels).float(),
            minlength=class_count,
        )
        per_class = correct_count / label_count
        seen_acc = per_class[seen_ids].mean().item()
        heldout_acc = per_class[heldout_ids].mean().item()
        hm = 2.0 * seen_acc * heldout_acc / max(seen_acc + heldout_acc, 1e-12)
        curve.append((seen_acc, heldout_acc))
        if hm > best_hm:
            best_hm, best_seen, best_heldout = hm, seen_acc, heldout_acc
    # Integrate the monotone envelope in seen-accuracy order.  The scalar is a
    # model-selection diagnostic, not a replacement for benchmark AUC.
    curve.sort()
    envelope, points = 0.0, []
    for seen_acc, heldout_acc in reversed(curve):
        envelope = max(envelope, heldout_acc)
        points.append((seen_acc, envelope))
    points.reverse()
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        auc += max(x1 - x0, 0.0) * (y0 + y1) * 0.5
    return {
        "best_seen": best_seen,
        "best_heldout": best_heldout,
        "best_hm": best_hm,
        "curve_auc": auc,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    flow_cache, heldout = edge_preserving_holdout(
        cache, args.folds, args.heldout_fold, args.seed
    )
    centroids, counts = real_centroids(flow_cache)
    additive = additive_completion(flow_cache, centroids, counts, 30)
    residuals = gaussianize(
        residual_bank(flow_cache, additive), args.residual_distribution, args.seed
    )

    # train_flow expects these shared warm-flow options.
    args.residual_bank_mode = "global"
    args.semantic_neighbors = 16
    args.primitive_bonus = 0.25
    if args.flow_path:
        flow = PrimitiveConditionalFlow(
            flow_cache["train"]["com"].shape[1], args.hidden_dim, args.cond_dim
        ).to(device)
        flow.load_state_dict(
            torch.load(args.flow_path, map_location="cpu", weights_only=True)
        )
        flow.eval()
    else:
        flow = train_flow(
            flow_cache, additive, centroids, residuals, args, device, local_tables=None
        )
        torch.save(flow.state_dict(), work_dir / "flow.pt")

    active = torch.zeros(len(counts), dtype=torch.bool)
    active[heldout] = True
    generation_cache = dict(flow_cache)
    generation_cache["meta"] = dict(flow_cache["meta"])
    generation_cache["meta"]["active"] = active.tolist()
    generated, generated_labels = generate_unseen(
        flow,
        generation_cache,
        additive,
        residuals,
        counts,
        args.residual_scale,
        args,
        device,
        local_tables=None,
    )
    seen_features, seen_labels_global = balanced_seen_samples(
        flow_cache, counts, args.samples_per_class, args.seed
    )

    seen_global = torch.where(counts > 0)[0]
    active_global = torch.cat([seen_global, heldout]).sort().values
    global_to_local = torch.full((len(counts),), -1, dtype=torch.long)
    global_to_local[active_global] = torch.arange(len(active_global))
    training_features = torch.cat([seen_features, generated])
    training_labels = global_to_local[torch.cat([seen_labels_global, generated_labels])]
    classifier_args = argparse.Namespace(
        batch_size=args.classifier_batch_size,
        epochs=args.classifier_epochs,
        lr=args.classifier_lr,
        fixed_weight_mask=None,
    )
    text = cache["prompts"]["pair"].float()[active_global]
    classifier = train_classifier(
        training_features, training_labels, text, classifier_args, device
    )

    real_features = cache["train"]["com"].float()
    real_labels = global_to_local[cache["train"]["pair"].long()]
    valid = real_labels >= 0
    real_features, real_labels = real_features[valid], real_labels[valid]
    seen_local = global_to_local[seen_global]
    heldout_local = global_to_local[heldout]
    classifier_score = classifier_logits(classifier, real_features, device)
    text_score = 20.0 * F.normalize(real_features, dim=-1) @ F.normalize(text, dim=-1).T

    prototypes = torch.zeros_like(text)
    for local_id, global_id in enumerate(active_global.tolist()):
        if counts[global_id] > 0:
            members = flow_cache["train"]["com"][flow_cache["train"]["pair"] == global_id]
        else:
            members = generated[generated_labels == global_id]
        prototypes[local_id] = F.normalize(members.float().mean(0), dim=-1)
    prototype_score = 20.0 * F.normalize(real_features, dim=-1) @ prototypes.T

    result = {
        "protocol": {
            "folds": args.folds,
            "heldout_fold": args.heldout_fold,
            "training_pairs": int(len(seen_global)),
            "heldout_pairs": int(len(heldout)),
            "heldout_instances": int(
                torch.isin(cache["train"]["pair"].long(), heldout).sum()
            ),
            "residual_distribution": args.residual_distribution,
            "pair_cond_scale": args.pair_cond_scale,
            "pair_guidance_scale": args.pair_guidance_scale,
            "primitive_transport_weight": args.primitive_transport_weight,
            "pair_aux_target": args.pair_aux_target,
            "recenter_generated": args.recenter_generated,
            "variation_scale": args.variation_scale,
            "residual_scale": args.residual_scale,
            "seed": args.seed,
        },
        "text": generalized_curve(text_score, real_labels, seen_local, heldout_local),
        "flow_prototype": generalized_curve(
            prototype_score, real_labels, seen_local, heldout_local
        ),
        "flow_classifier": generalized_curve(
            classifier_score, real_labels, seen_local, heldout_local
        ),
    }
    (work_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
