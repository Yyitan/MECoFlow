"""Reverse FlowComposer primitive fields into unseen composition prototypes.

This is a validation-only research diagnostic.  FlowComposer learns visual-to-text
velocity fields.  We integrate the frozen fields from t=1 back to t=0, starting
from composition text embeddings perturbed by isotropic maximum-entropy noise, and
measure whether the resulting prototypes align with real validation features.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--attribute-flow", required=True)
    parser.add_argument("--object-flow", required=True)
    parser.add_argument("--gate-checkpoint")
    parser.add_argument("--feature-cache")
    parser.add_argument("--official-logit-cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("val",), default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--reverse-steps", type=int, default=8)
    parser.add_argument(
        "--reverse-end-times",
        type=float,
        nargs="+",
        default=[0.0],
        help="stop reverse integration at these times; 0 is full inversion",
    )
    parser.add_argument("--skip-reverse", action="store_true")
    parser.add_argument(
        "--tangent-radii",
        type=float,
        nargs="+",
        default=[],
        help="text-boundary angular trust radii for normalized tangent correction",
    )
    parser.add_argument(
        "--adaptive-tangent-scales",
        type=float,
        nargs="+",
        default=[],
        help="velocity-norm scales for composition-adaptive tangent correction",
    )
    parser.add_argument(
        "--adaptive-tangent-caps",
        type=float,
        nargs="+",
        default=[],
        help="maximum angular trust radii paired with every adaptive tangent scale",
    )
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_state(path: str) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def pair_index_tensor(dataset: object, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [(dataset.attr2idx[a], dataset.obj2idx[o]) for a, o in dataset.pairs],
        dtype=torch.long,
        device=device,
    )


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataset: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if dataset.phase == "train":
        local_to_global = torch.tensor(
            [dataset.pair2idx[pair] for pair in dataset.train_pairs], dtype=torch.long
        )
    else:
        local_to_global = None
    features, labels = [], []
    model.eval()
    for batch in loader:
        images, _, _, pair = batch[:4]
        images = images.to(device, non_blocking=True).type(model.clip.dtype)
        composition, _ = model.encode_image(images)
        features.append(F.normalize(composition.float(), dim=-1).cpu())
        pair = pair.long()
        if local_to_global is not None:
            pair = local_to_global[pair]
        labels.append(pair)
    return torch.cat(features), torch.cat(labels)


def centroids(
    features: torch.Tensor, labels: torch.Tensor, num_pairs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.zeros(num_pairs, features.shape[1])
    counts = torch.bincount(labels, minlength=num_pairs)
    result.index_add_(0, labels, features)
    valid = counts > 0
    result[valid] = F.normalize(result[valid] / counts[valid, None], dim=-1)
    return result, counts


@torch.no_grad()
def reverse_prototypes(
    pair_text: torch.Tensor,
    pair_ids: torch.Tensor,
    attr_flow: torch.nn.Module,
    obj_flow: torch.nn.Module,
    gate: torch.nn.Module | None,
    noise_scale: float,
    steps: int,
    end_time: float,
    samples_per_class: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if not 0.0 <= end_time < 1.0:
        raise ValueError("reverse end time must be in [0, 1)")
    generator = torch.Generator(device=device).manual_seed(seed)
    prototypes = torch.zeros(len(pair_text), pair_text.shape[1])
    for chunk in pair_ids.split(128):
        labels = chunk.to(device).repeat_interleave(samples_per_class)
        center = pair_text[labels]
        noise = torch.randn(center.shape, generator=generator, device=device)
        # Keep the perturbation tangential so scale controls angular coverage.
        noise = noise - (noise * center).sum(-1, keepdim=True) * center
        noise = F.normalize(noise, dim=-1)
        state = F.normalize(center + noise_scale * noise, dim=-1)
        step_size = (1.0 - end_time) / steps
        for index in range(steps):
            time_value = 1.0 - index * step_size
            time = torch.full((len(state), 1), time_value, device=device)
            velocity_attr = attr_flow(state, time)
            velocity_obj = obj_flow(state, time)
            if gate is None:
                velocity = velocity_attr + velocity_obj
            else:
                _, _, _, velocity = gate(velocity_attr, velocity_obj, tau=1.0)
            state = state - step_size * velocity
        state = F.normalize(state, dim=-1)
        state = state.view(len(chunk), samples_per_class, -1).mean(1)
        prototypes[chunk] = F.normalize(state, dim=-1).cpu()
    return prototypes


@torch.no_grad()
def tangent_prototypes(
    pair_text: torch.Tensor,
    pair_ids: torch.Tensor,
    attr_flow: torch.nn.Module,
    obj_flow: torch.nn.Module,
    gate: torch.nn.Module | None,
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    """Take one magnitude-invariant reverse step in the text tangent space."""
    if radius < 0.0:
        raise ValueError("tangent radius must be nonnegative")
    prototypes = torch.zeros(len(pair_text), pair_text.shape[1])
    for chunk in pair_ids.split(128):
        center = pair_text[chunk.to(device)]
        time = torch.ones((len(center), 1), device=device)
        velocity_attr = attr_flow(center, time)
        velocity_obj = obj_flow(center, time)
        if gate is None:
            velocity = velocity_attr + velocity_obj
        else:
            _, _, _, velocity = gate(velocity_attr, velocity_obj, tau=1.0)
        tangent = velocity - (velocity * center).sum(-1, keepdim=True) * center
        direction = F.normalize(tangent, dim=-1)
        state = F.normalize(center - radius * direction, dim=-1)
        prototypes[chunk] = state.cpu()
    return prototypes


@torch.no_grad()
def adaptive_tangent_prototypes(
    pair_text: torch.Tensor,
    pair_ids: torch.Tensor,
    attr_flow: torch.nn.Module,
    obj_flow: torch.nn.Module,
    gate: torch.nn.Module | None,
    scale: float,
    cap: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use primitive-field magnitude as a class-specific step inside a trust region."""
    if scale < 0.0 or cap <= 0.0:
        raise ValueError("adaptive tangent scale must be nonnegative and cap positive")
    prototypes = torch.zeros(len(pair_text), pair_text.shape[1])
    displacements = torch.zeros(len(pair_ids))
    offset = 0
    for chunk in pair_ids.split(128):
        center = pair_text[chunk.to(device)]
        time = torch.ones((len(center), 1), device=device)
        velocity_attr = attr_flow(center, time)
        velocity_obj = obj_flow(center, time)
        if gate is None:
            velocity = velocity_attr + velocity_obj
        else:
            _, _, _, velocity = gate(velocity_attr, velocity_obj, tau=1.0)
        tangent = velocity - (velocity * center).sum(-1, keepdim=True) * center
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        direction = tangent / tangent_norm.clamp_min(1e-12)
        displacement = (scale * tangent_norm).clamp_max(cap)
        state = F.normalize(center - displacement * direction, dim=-1)
        prototypes[chunk] = state.cpu()
        displacements[offset : offset + len(chunk)] = displacement.squeeze(-1).cpu()
        offset += len(chunk)
    return prototypes, displacements


def curve_metrics(
    scores: torch.Tensor,
    truth: torch.Tensor,
    seen_classes: torch.Tensor,
) -> dict[str, float]:
    unseen_classes = ~seen_classes
    neg = torch.finfo(scores.dtype).min
    seen_scores = scores.masked_fill(~seen_classes[None], neg)
    unseen_scores = scores.masked_fill(~unseen_classes[None], neg)
    max_seen, pred_seen = seen_scores.max(1)
    max_unseen, pred_unseen = unseen_scores.max(1)
    sample_seen = seen_classes[truth]
    sample_unseen = ~sample_seen
    correct = scores[torch.arange(len(scores)), truth]
    valid_unseen = sample_unseen & (pred_unseen == truth)
    thresholds = (max_seen[valid_unseen] - correct[valid_unseen] - 1e-4).sort().values
    skip = max(len(thresholds) // 20, 1)
    biases = thresholds[::skip]
    seen_curve, unseen_curve = [], []
    for bias in biases:
        prediction = torch.where(max_unseen + bias > max_seen, pred_unseen, pred_seen)
        match = prediction == truth
        seen_curve.append(match[sample_seen].float().mean().item())
        unseen_curve.append(match[sample_unseen].float().mean().item())
    seen_curve.append(0.0)
    unseen_curve.append((pred_unseen[sample_unseen] == truth[sample_unseen]).float().mean().item())
    seen_array = np.asarray(seen_curve)
    unseen_array = np.asarray(unseen_curve)
    denominator = seen_array + unseen_array
    hm = np.divide(
        2 * seen_array * unseen_array,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return {
        "best_seen": float(seen_array.max()),
        "best_unseen": float(unseen_array.max()),
        "best_hm": float(hm.max()),
        "AUC": float(np.trapezoid(seen_array, unseen_array)),
    }


def prototype_scores(
    prototypes: torch.Tensor,
    train_centroids: torch.Tensor,
    train_counts: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    train_ids: torch.Tensor,
    unseen_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_ids = torch.cat([train_ids, unseen_ids]).unique().sort().values
    global_to_local = torch.full((len(prototypes),), -1, dtype=torch.long)
    global_to_local[active_ids] = torch.arange(len(active_ids))
    active_prototypes = prototypes[active_ids].clone()
    seen_mask = train_counts[active_ids] > 0
    active_prototypes[seen_mask] = train_centroids[active_ids[seen_mask]]
    valid = global_to_local[eval_labels] >= 0
    truth = global_to_local[eval_labels[valid]]
    scores = 20.0 * eval_features[valid] @ F.normalize(active_prototypes, dim=-1).T
    return scores, truth, seen_mask


def evaluate_prototypes(
    prototypes: torch.Tensor,
    train_centroids: torch.Tensor,
    train_counts: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    train_ids: torch.Tensor,
    unseen_ids: torch.Tensor,
) -> dict[str, float]:
    return curve_metrics(
        *prototype_scores(
            prototypes,
            train_centroids,
            train_counts,
            eval_features,
            eval_labels,
            train_ids,
            unseen_ids,
        )
    )


def zscore(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True).clamp_min(1e-6)


def fusion_grid(
    base_scores: torch.Tensor,
    evidence_scores: torch.Tensor,
    truth: torch.Tensor,
    seen_mask: torch.Tensor,
) -> dict[str, object]:
    base = zscore(base_scores)
    evidence = zscore(evidence_scores)
    rows = []
    for weight in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
        metrics = curve_metrics(base + weight * evidence, truth, seen_mask)
        rows.append({"weight": weight, "metrics": metrics})
    best_auc = max(rows, key=lambda row: (row["metrics"]["AUC"], row["metrics"]["best_hm"]))
    best_hm = max(rows, key=lambda row: (row["metrics"]["best_hm"], row["metrics"]["AUC"]))
    return {"best_auc": best_auc, "best_hm": best_hm, "grid": rows}


def resolved(path: str | None) -> str | None:
    return str(Path(path).resolve()) if path else None


def require_cache_provenance(cache: dict[str, object], expected: dict[str, object]) -> None:
    actual = cache.get("provenance")
    if actual != expected:
        raise ValueError(
            "cache provenance mismatch; remove the stale cache or use a new cache path: "
            f"expected={expected}, actual={actual}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    official_root = str(Path(args.official_root).resolve())
    sys.path.insert(0, official_root)

    from config import DefaultConfig  # noqa: E402
    from dataset import CompositionDataset_u  # noqa: E402
    from fm import DeepFlowMatchingNet  # noqa: E402
    from model.model_factory import get_model  # noqa: E402
    from s2_gate import VelocityGater  # noqa: E402
    import test as official_test  # noqa: E402

    config = Namespace(**json.loads(Path(args.config_json).read_text()))
    train_dataset = CompositionDataset_u(
        args.dataset_path, phase="train", split="compositional-split-natural"
    )
    eval_dataset = CompositionDataset_u(
        args.dataset_path, phase=args.phase, split="compositional-split-natural"
    )
    # Seen-class prototypes are evaluation references, not augmented training
    # examples.  Reuse the deterministic evaluation transform so rebuilding a
    # feature cache cannot change the baseline through random image augmentation.
    train_dataset.transform = eval_dataset.transform
    attributes = [value.replace(".", " ").lower() for value in train_dataset.attrs]
    classes = [value.replace(".", " ").lower() for value in train_dataset.objs]
    model = get_model(config, attributes, classes, len(attributes)).to(device)
    model.load_state_dict(load_state(args.base_checkpoint))
    model.eval()

    flow_config = DefaultConfig()
    attr_flow = DeepFlowMatchingNet(768, 768, 768, flow_config.blocks).to(device)
    obj_flow = DeepFlowMatchingNet(768, 768, 768, flow_config.blocks).to(device)
    attr_flow.load_state_dict(load_state(args.attribute_flow))
    obj_flow.load_state_dict(load_state(args.object_flow))
    attr_flow.eval()
    obj_flow.eval()
    gate = None
    if args.gate_checkpoint:
        gate = VelocityGater(768, hidden=256, sum_to_one=config.sum_to_one, with_scale=True).to(device)
        gate.load_state_dict(load_state(args.gate_checkpoint))
        gate.eval()

    pairs = pair_index_tensor(eval_dataset, device)
    with torch.no_grad():
        pair_text = model.encode_text_for_open(pairs)[0].float()
        pair_text = F.normalize(pair_text, dim=-1)

    feature_cache = Path(args.feature_cache) if args.feature_cache else None
    feature_provenance = {
        "base_checkpoint": resolved(args.base_checkpoint),
        "dataset_path": resolved(args.dataset_path),
        "phase": args.phase,
        "train_feature_transform": "eval",
    }
    if feature_cache is not None and feature_cache.exists():
        cached = torch.load(feature_cache, map_location="cpu", weights_only=False)
        require_cache_provenance(cached, feature_provenance)
        train_features, train_labels = cached["train_features"], cached["train_labels"]
        eval_features, eval_labels = cached["eval_features"], cached["eval_labels"]
        print(f"loaded feature cache: {feature_cache}", flush=True)
    else:
        print("extracting train composition features", flush=True)
        train_features, train_labels = extract_features(
            model, train_dataset, args.batch_size, args.num_workers, device
        )
        print(f"extracting {args.phase} composition features", flush=True)
        eval_features, eval_labels = extract_features(
            model, eval_dataset, args.batch_size, args.num_workers, device
        )
        if feature_cache is not None:
            feature_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "provenance": feature_provenance,
                    "train_features": train_features,
                    "train_labels": train_labels,
                    "eval_features": eval_features,
                    "eval_labels": eval_labels,
                },
                feature_cache,
            )
            print(f"saved feature cache: {feature_cache}", flush=True)
    train_centroids, train_counts = centroids(train_features, train_labels, len(eval_dataset.pairs))
    eval_centroids, eval_counts = centroids(eval_features, eval_labels, len(eval_dataset.pairs))
    train_ids = torch.tensor(
        [eval_dataset.pair2idx[pair] for pair in eval_dataset.train_pairs], dtype=torch.long
    )
    unseen_ids = torch.tensor(
        [eval_dataset.pair2idx[pair] for pair in eval_dataset.val_pairs], dtype=torch.long
    )
    unseen_ids = unseen_ids[eval_counts[unseen_ids] > 0]

    base_bundle: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    official_logit_cache = Path(args.official_logit_cache) if args.official_logit_cache else None
    official_provenance = {
        "base_checkpoint": resolved(args.base_checkpoint),
        "attribute_flow": resolved(args.attribute_flow),
        "object_flow": resolved(args.object_flow),
        "gate_checkpoint": resolved(args.gate_checkpoint),
        "dataset_path": resolved(args.dataset_path),
        "phase": args.phase,
        "reverse_steps": args.reverse_steps,
    }
    if official_logit_cache is not None:
        if official_logit_cache.exists():
            cached_logits = torch.load(
                official_logit_cache, map_location="cpu", weights_only=False
            )
            require_cache_provenance(cached_logits, official_provenance)
            composed_logits = cached_logits["composed_logits"]
            official_truth = cached_logits["pair_truth"]
            print(f"loaded official logit cache: {official_logit_cache}", flush=True)
        else:
            flow_models = [None, attr_flow, obj_flow]
            if gate is None:
                _, _, _, official_truth, composed_logits = (
                    official_test.predict_logits_fm_joint_troika_v2_3path(
                        model,
                        flow_models,
                        eval_dataset,
                        args.reverse_steps,
                        path_2=True,
                        compose=True,
                    )
                )
            else:
                _, _, _, official_truth, composed_logits = (
                    official_test.predict_logits_fm_joint_troika_v2_3path_wGate_v2(
                        model,
                        flow_models,
                        gate,
                        eval_dataset,
                        args.reverse_steps,
                        path_2=True,
                        compose=True,
                    )
                )
            official_logit_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "provenance": official_provenance,
                    "composed_logits": composed_logits,
                    "pair_truth": official_truth,
                },
                official_logit_cache,
            )
            print(f"saved official logit cache: {official_logit_cache}", flush=True)
        active_ids = torch.cat([train_ids, unseen_ids]).unique().sort().values
        global_to_local = torch.full((len(eval_dataset.pairs),), -1, dtype=torch.long)
        global_to_local[active_ids] = torch.arange(len(active_ids))
        valid = global_to_local[official_truth] >= 0
        base_scores = composed_logits[valid][:, active_ids]
        base_truth = global_to_local[official_truth[valid]]
        base_seen_mask = train_counts[active_ids] > 0
        base_bundle = base_scores, base_truth, base_seen_mask

    pair_text_cpu = pair_text.cpu()
    text_scores, text_truth, text_seen_mask = prototype_scores(
        pair_text_cpu,
        train_centroids,
        train_counts,
        eval_features,
        eval_labels,
        train_ids,
        unseen_ids,
    )
    text_result: dict[str, object] = {
        "alignment_mean": float(
            (pair_text_cpu[unseen_ids] * eval_centroids[unseen_ids]).sum(-1).mean()
        ),
        "alignment_median": float(
            (pair_text_cpu[unseen_ids] * eval_centroids[unseen_ids]).sum(-1).median()
        ),
        "prototype": curve_metrics(text_scores, text_truth, text_seen_mask),
    }
    if base_bundle is not None:
        base_scores, base_truth, base_seen_mask = base_bundle
        if not torch.equal(text_truth, base_truth) or not torch.equal(text_seen_mask, base_seen_mask):
            raise ValueError("official and prototype evaluation order differs")
        text_result["fusion"] = fusion_grid(
            base_scores, text_scores, base_truth, base_seen_mask
        )
    result: dict[str, object] = {
        "protocol": vars(args),
        "text_prototype": text_result,
    }
    if base_bundle is not None:
        result["official_flowcomposer"] = curve_metrics(*base_bundle)
    if not args.skip_reverse:
        for end_time in args.reverse_end_times:
            for scale in args.noise_scales:
                generated = reverse_prototypes(
                    pair_text,
                    unseen_ids,
                    attr_flow,
                    obj_flow,
                    gate,
                    scale,
                    args.reverse_steps,
                    end_time,
                    args.samples_per_class,
                    args.seed,
                    device,
                )
                alignment = (generated[unseen_ids] * eval_centroids[unseen_ids]).sum(-1)
                text_cosine = (generated[unseen_ids] * pair_text_cpu[unseen_ids]).sum(-1)
                key = f"reverse_end_{end_time:.3f}_noise_{scale:.3f}"
                generated_scores, generated_truth, generated_seen_mask = prototype_scores(
                    generated,
                    train_centroids,
                    train_counts,
                    eval_features,
                    eval_labels,
                    train_ids,
                    unseen_ids,
                )
                result[key] = {
                    "alignment_mean": float(alignment.mean()),
                    "alignment_median": float(alignment.median()),
                    "text_cosine_mean": float(text_cosine.mean()),
                    "prototype": curve_metrics(
                        generated_scores, generated_truth, generated_seen_mask
                    ),
                }
                if base_bundle is not None:
                    base_scores, base_truth, base_seen_mask = base_bundle
                    result[key]["fusion"] = fusion_grid(
                        base_scores, generated_scores, base_truth, base_seen_mask
                    )
                print(key, json.dumps(result[key]), flush=True)

    for radius in args.tangent_radii:
        generated = tangent_prototypes(
            pair_text,
            unseen_ids,
            attr_flow,
            obj_flow,
            gate,
            radius,
            device,
        )
        alignment = (generated[unseen_ids] * eval_centroids[unseen_ids]).sum(-1)
        text_cosine = (generated[unseen_ids] * pair_text_cpu[unseen_ids]).sum(-1)
        key = f"tangent_radius_{radius:.5f}"
        generated_scores, generated_truth, generated_seen_mask = prototype_scores(
            generated,
            train_centroids,
            train_counts,
            eval_features,
            eval_labels,
            train_ids,
            unseen_ids,
        )
        result[key] = {
            "alignment_mean": float(alignment.mean()),
            "alignment_median": float(alignment.median()),
            "text_cosine_mean": float(text_cosine.mean()),
            "prototype": curve_metrics(
                generated_scores, generated_truth, generated_seen_mask
            ),
        }
        if base_bundle is not None:
            base_scores, base_truth, base_seen_mask = base_bundle
            result[key]["fusion"] = fusion_grid(
                base_scores, generated_scores, base_truth, base_seen_mask
            )
        print(key, json.dumps(result[key]), flush=True)

    for scale in args.adaptive_tangent_scales:
        for cap in args.adaptive_tangent_caps:
            generated, displacements = adaptive_tangent_prototypes(
                pair_text,
                unseen_ids,
                attr_flow,
                obj_flow,
                gate,
                scale,
                cap,
                device,
            )
            alignment = (generated[unseen_ids] * eval_centroids[unseen_ids]).sum(-1)
            text_cosine = (generated[unseen_ids] * pair_text_cpu[unseen_ids]).sum(-1)
            key = f"adaptive_tangent_scale_{scale:.5f}_cap_{cap:.5f}"
            generated_scores, generated_truth, generated_seen_mask = prototype_scores(
                generated,
                train_centroids,
                train_counts,
                eval_features,
                eval_labels,
                train_ids,
                unseen_ids,
            )
            result[key] = {
                "alignment_mean": float(alignment.mean()),
                "alignment_median": float(alignment.median()),
                "text_cosine_mean": float(text_cosine.mean()),
                "displacement_mean": float(displacements.mean()),
                "displacement_max": float(displacements.max()),
                "cap_fraction": float((displacements >= cap - 1e-8).float().mean()),
                "prototype": curve_metrics(
                    generated_scores, generated_truth, generated_seen_mask
                ),
            }
            if base_bundle is not None:
                base_scores, base_truth, base_seen_mask = base_bundle
                result[key]["fusion"] = fusion_grid(
                    base_scores, generated_scores, base_truth, base_seen_mask
                )
            print(key, json.dumps(result[key]), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
