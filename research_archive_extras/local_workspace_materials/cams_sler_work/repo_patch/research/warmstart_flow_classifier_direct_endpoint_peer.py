"""Warm-started visual residual flow for generative CZSL data completion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


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
from research.generative_classifier_pilot import (  # noqa: E402
    classifier_logits,
    train_classifier,
)
from research.piflow_pilot import PrimitiveConditionalFlow  # noqa: E402
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--additive-path")
    parser.add_argument("--flow-path")
    parser.add_argument(
        "--flow-pair-folds",
        type=int,
        default=1,
        help="train the flow on all but one deterministic composition fold",
    )
    parser.add_argument("--flow-heldout-fold", type=int, default=0)
    parser.add_argument("--flow-epochs", type=int, default=20)
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--classifier-batch-size", type=int, default=2048)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument(
        "--transport-objective",
        choices=("flow_matching", "direct_endpoint"),
        default="flow_matching",
        help=(
            "train the time-conditioned rectified-flow objective (main method) or "
            "an exactly parameter-matched direct endpoint regressor"
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--classifier-lr", type=float, default=3e-3)
    parser.add_argument("--aux-weight", type=float, default=0.05)
    parser.add_argument("--primitive-aux-weight", type=float, default=0.0)
    parser.add_argument("--pair-cond-scale", type=float, default=1.0)
    parser.add_argument("--residual-bank-mode", choices=("global", "local"), default="global")
    parser.add_argument(
        "--residual-distribution",
        choices=("empirical", "gaussian_diag", "gaussian_isotropic"),
        default="gaussian_isotropic",
        help="empirical bootstrap or a moment-matched diagonal Gaussian source",
    )
    parser.add_argument(
        "--prior-mode",
        choices=("anova", "leave_pair_out"),
        default="anova",
        help=(
            "leave_pair_out builds every training start distribution without using "
            "statistics from its target composition, matching unseen-pair inference"
        ),
    )
    parser.add_argument(
        "--source-center",
        choices=("additive", "zero"),
        default="additive",
        help=(
            "center the flow source at the compositional additive prior (main method) "
            "or at zero for a controlled source-location ablation"
        ),
    )
    parser.add_argument("--semantic-neighbors", type=int, default=16)
    parser.add_argument("--primitive-bonus", type=float, default=0.25)
    parser.add_argument("--feature-key", choices=("com", "glb"), default="com")
    parser.add_argument("--freeze-seen-classifier", action="store_true")
    parser.add_argument("--cond-drop", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--residual-scales", type=float, nargs="+", default=(0.0, 0.5, 1.0))
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pair_maps(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    pair_attr = torch.tensor([meta["attrs"].index(a) for a, _ in meta["pairs"]], dtype=torch.long)
    pair_obj = torch.tensor([meta["objs"].index(o) for _, o in meta["pairs"]], dtype=torch.long)
    return pair_attr, pair_obj


def residual_bank(cache: dict[str, object], additive: torch.Tensor) -> torch.Tensor:
    train = cache["train"]
    return train["com"].float() - additive[train["pair"].long()]


def marginal_completion(
    cache: dict[str, object],
    centroids: torch.Tensor,
    counts: torch.Tensor,
    leave_seen_pair_out: bool,
) -> torch.Tensor:
    """Primitive marginal prior, optionally excluding the target seen pair.

    The exclusion makes a seen training pair look exactly like an unseen pair:
    its attribute/object/global statistics are estimated only from other
    compositions.  At inference, unseen pairs have count zero, so the same
    formula naturally uses all available seen compositions.
    """
    pair_attr, pair_obj = pair_maps(cache)
    num_attrs = len(cache["meta"]["attrs"])
    num_objs = len(cache["meta"]["objs"])
    weighted = centroids * counts[:, None].float()
    attr_sum = torch.zeros(num_attrs, centroids.shape[1])
    obj_sum = torch.zeros(num_objs, centroids.shape[1])
    attr_count = torch.zeros(num_attrs)
    obj_count = torch.zeros(num_objs)
    attr_sum.index_add_(0, pair_attr, weighted)
    obj_sum.index_add_(0, pair_obj, weighted)
    attr_count.index_add_(0, pair_attr, counts.float())
    obj_count.index_add_(0, pair_obj, counts.float())
    global_sum = weighted.sum(0)
    global_count = counts.sum().float()

    remove = counts.float() if leave_seen_pair_out else torch.zeros_like(counts, dtype=torch.float32)
    removed_weighted = weighted if leave_seen_pair_out else torch.zeros_like(weighted)
    attr_denom = attr_count[pair_attr] - remove
    obj_denom = obj_count[pair_obj] - remove
    global_denom = global_count - remove
    # Every benchmark primitive occurs in multiple train compositions, but keep
    # a safe full-marginal fallback for unusually sparse custom splits.
    full_attr = attr_sum[pair_attr] / attr_count[pair_attr].clamp_min(1)[:, None]
    full_obj = obj_sum[pair_obj] / obj_count[pair_obj].clamp_min(1)[:, None]
    full_global = global_sum / global_count.clamp_min(1)
    attr_mean = (attr_sum[pair_attr] - removed_weighted) / attr_denom.clamp_min(1)[:, None]
    obj_mean = (obj_sum[pair_obj] - removed_weighted) / obj_denom.clamp_min(1)[:, None]
    global_mean = (global_sum[None] - removed_weighted) / global_denom.clamp_min(1)[:, None]
    attr_mean = torch.where((attr_denom > 0)[:, None], attr_mean, full_attr)
    obj_mean = torch.where((obj_denom > 0)[:, None], obj_mean, full_obj)
    global_mean = torch.where((global_denom > 0)[:, None], global_mean, full_global[None])
    return F.normalize(attr_mean + obj_mean - global_mean, dim=-1)


@torch.no_grad()
def local_residual_tables(
    cache: dict[str, object], labels: torch.Tensor, args: argparse.Namespace, device: torch.device
) -> dict[str, torch.Tensor] | None:
    """Build a leakage-free semantic/primitive neighbourhood over seen pairs."""
    if args.residual_bank_mode == "global":
        return None
    pair_attr, pair_obj = pair_maps(cache)
    seen_ids = torch.unique(labels).sort().values.to(device)
    pair_semantic = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    pair_attr_device, pair_obj_device = pair_attr.to(device), pair_obj.to(device)
    similarity = pair_semantic @ pair_semantic[seen_ids].T
    similarity.add_(
        args.primitive_bonus
        * (
            (pair_attr_device[:, None] == pair_attr_device[seen_ids][None]).float()
            + (pair_obj_device[:, None] == pair_obj_device[seen_ids][None]).float()
        )
    )
    # Excluding a seen class itself makes training imitate unseen-pair transfer.
    seen_lookup = torch.full((len(pair_semantic),), -1, dtype=torch.long, device=device)
    seen_lookup[seen_ids] = torch.arange(len(seen_ids), device=device)
    seen_rows = seen_lookup >= 0
    similarity[seen_rows, seen_lookup[seen_rows]] = -torch.inf
    neighbour_count = min(args.semantic_neighbors, max(1, len(seen_ids) - 1))
    neighbours = seen_ids[similarity.topk(neighbour_count, dim=-1).indices]

    counts = torch.bincount(labels, minlength=len(pair_semantic))
    max_count = int(counts.max())
    instance_table = torch.zeros(len(pair_semantic), max_count, dtype=torch.long)
    for pair_id in seen_ids.cpu().tolist():
        indices = torch.where(labels == pair_id)[0]
        instance_table[pair_id, : len(indices)] = indices
    return {
        "neighbours": neighbours,
        "instance_table": instance_table.to(device),
        "counts": counts.to(device),
    }


def sample_residual_indices(
    labels: torch.Tensor,
    residual_count: int,
    tables: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> torch.Tensor:
    if tables is None:
        return torch.randint(residual_count, (len(labels),), device=device)
    neighbours = tables["neighbours"][labels]
    rank = torch.randint(neighbours.shape[1], (len(labels),), device=device)
    source_pair = neighbours[torch.arange(len(labels), device=device), rank]
    offset = (torch.rand(len(labels), device=device) * tables["counts"][source_pair]).long()
    return tables["instance_table"][source_pair, offset]


def sample_residual_vectors(
    labels: torch.Tensor,
    residuals: torch.Tensor,
    tables: dict[str, torch.Tensor] | None,
    distribution: str,
    device: torch.device,
) -> torch.Tensor:
    residual_id = sample_residual_indices(labels, len(residuals), tables, device)
    return residuals[residual_id]


def train_flow(
    cache: dict[str, object],
    train_additive: torch.Tensor,
    centroids: torch.Tensor,
    residuals: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    local_tables: dict[str, torch.Tensor] | None,
) -> PrimitiveConditionalFlow:
    train = cache["train"]
    labels = train["pair"].long()
    features = train["com"].float()
    pair_attr, pair_obj = pair_maps(cache)
    counts = torch.bincount(labels, minlength=len(train_additive)).clamp_min(1)
    sampler = WeightedRandomSampler(counts[labels].float().reciprocal(), len(labels), replacement=True)
    loader = DataLoader(torch.arange(len(labels)), batch_size=args.batch_size, sampler=sampler, num_workers=0)
    model = PrimitiveConditionalFlow(features.shape[1], args.hidden_dim, args.cond_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.flow_epochs)
    pair_attr, pair_obj = pair_attr.to(device), pair_obj.to(device)
    prompts = cache["prompts"]
    attr_prompt = prompts["attr"].float().to(device)
    obj_prompt = prompts["obj"].float().to(device)
    pair_prompt = prompts["pair"].float().to(device)
    labels_device = labels.to(device)
    features_device = features.to(device)
    additive_device = train_additive.to(device)
    residuals_device = residuals.to(device)
    seen_ids = torch.unique(labels_device).sort().values
    seen_lookup = torch.full((len(train_additive),), -1, dtype=torch.long, device=device)
    seen_lookup[seen_ids] = torch.arange(len(seen_ids), device=device)
    seen_centroids = centroids[seen_ids.cpu()].to(device)
    train_attr = pair_attr[labels_device]
    train_obj = pair_obj[labels_device]
    attr_centroids = torch.zeros(len(attr_prompt), features.shape[1], device=device)
    obj_centroids = torch.zeros(len(obj_prompt), features.shape[1], device=device)
    attr_centroids.index_add_(0, train_attr, features_device)
    obj_centroids.index_add_(0, train_obj, features_device)
    attr_counts = torch.bincount(train_attr, minlength=len(attr_prompt)).clamp_min(1)
    obj_counts = torch.bincount(train_obj, minlength=len(obj_prompt)).clamp_min(1)
    attr_centroids = F.normalize(attr_centroids / attr_counts[:, None], dim=-1)
    obj_centroids = F.normalize(obj_centroids / obj_counts[:, None], dim=-1)
    prior_scales = torch.tensor([0.0, 0.5, 1.0], device=device)

    model.train()
    for epoch in range(args.flow_epochs):
        losses, flow_losses, aux_losses = [], [], []
        for indices in loader:
            indices = indices.to(device)
            label = labels_device[indices]
            target = features_device[indices]
            sampled_residual = sample_residual_vectors(
                label,
                residuals_device,
                local_tables,
                args.residual_distribution,
                device,
            )
            scale = prior_scales[torch.randint(len(prior_scales), (len(indices),), device=device)]
            start = F.normalize(
                additive_device[label] + scale[:, None] * sampled_residual, dim=-1
            )
            if args.transport_objective == "direct_endpoint":
                time = torch.zeros(len(indices), device=device)
                noisy = start
            else:
                time = torch.rand(len(indices), device=device)
                noisy = (1.0 - time[:, None]) * start + time[:, None] * target
            velocity_target = target - start
            velocity = model(
                noisy,
                time,
                attr_prompt[pair_attr[label]],
                obj_prompt[pair_obj[label]],
                args.pair_cond_scale * pair_prompt[label],
                cond_drop=args.cond_drop,
            )
            flow_loss = (velocity - velocity_target).square().sum(-1).mean()
            endpoint = F.normalize(noisy + (1.0 - time[:, None]) * velocity, dim=-1)
            centroid_logits = 20.0 * endpoint @ seen_centroids.T
            aux_loss = F.cross_entropy(centroid_logits, seen_lookup[label])
            primitive_aux_loss = F.cross_entropy(
                20.0 * endpoint @ attr_centroids.T, pair_attr[label]
            ) + F.cross_entropy(20.0 * endpoint @ obj_centroids.T, pair_obj[label])
            loss = (
                flow_loss
                + args.aux_weight * aux_loss
                + args.primitive_aux_weight * primitive_aux_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
            flow_losses.append(flow_loss.detach())
            aux_losses.append(aux_loss.detach())
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"epoch={epoch + 1:03d} loss={torch.stack(losses).mean().item():.5f} "
                f"flow={torch.stack(flow_losses).mean().item():.5f} "
                f"aux={torch.stack(aux_losses).mean().item():.5f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def generate_unseen(
    model: PrimitiveConditionalFlow,
    cache: dict[str, object],
    additive: torch.Tensor,
    residuals: torch.Tensor,
    counts: torch.Tensor,
    residual_scale: float,
    args: argparse.Namespace,
    device: torch.device,
    local_tables: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_attr, pair_obj = pair_maps(cache)
    pair_attr, pair_obj = pair_attr.to(device), pair_obj.to(device)
    prompts = cache["prompts"]
    attr_prompt = prompts["attr"].float().to(device)
    obj_prompt = prompts["obj"].float().to(device)
    pair_prompt = prompts["pair"].float().to(device)
    active = torch.tensor(
        cache["meta"].get("active", [True] * len(counts)), dtype=torch.bool
    )
    unseen_ids = torch.where((counts == 0) & active)[0]
    residuals_device = residuals.to(device)
    all_samples, all_labels = [], []
    for chunk in unseen_ids.split(128):
        label = chunk.to(device).repeat_interleave(args.samples_per_class)
        sampled_residual = sample_residual_vectors(
            label,
            residuals_device,
            local_tables,
            args.residual_distribution,
            device,
        )
        start = F.normalize(
            additive[label.cpu()].to(device)
            + residual_scale * sampled_residual,
            dim=-1,
        )
        current = start
        if args.transport_objective == "direct_endpoint":
            time = torch.zeros(len(label), device=device)
            endpoint_delta = model(
                current,
                time,
                attr_prompt[pair_attr[label]],
                obj_prompt[pair_obj[label]],
                args.pair_cond_scale * pair_prompt[label],
            )
            current = current + endpoint_delta
        else:
            dt = 1.0 / args.sample_steps
            for step in range(args.sample_steps):
                time = torch.full((len(label),), step * dt, device=device)
                velocity_cond = model(
                    current,
                    time,
                    attr_prompt[pair_attr[label]],
                    obj_prompt[pair_obj[label]],
                    args.pair_cond_scale * pair_prompt[label],
                )
                if args.guidance_scale == 1.0:
                    velocity = velocity_cond
                else:
                    velocity_uncond = model(
                        current,
                        time,
                        torch.zeros_like(attr_prompt[pair_attr[label]]),
                        torch.zeros_like(obj_prompt[pair_obj[label]]),
                        torch.zeros_like(pair_prompt[label]),
                    )
                    velocity = velocity_uncond + args.guidance_scale * (
                        velocity_cond - velocity_uncond
                    )
                current = current + dt * velocity
        all_samples.append(F.normalize(current, dim=-1).cpu())
        all_labels.append(label.cpu())
    return torch.cat(all_samples), torch.cat(all_labels)


def balanced_seen_samples(
    cache: dict[str, object], counts: torch.Tensor, samples_per_class: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train = cache["train"]
    features, labels = train["com"].float(), train["pair"].long()
    samples, targets = [], []
    for class_id in torch.where(counts > 0)[0].tolist():
        available = torch.where(labels == class_id)[0]
        chosen = available[torch.randint(len(available), (samples_per_class,), generator=generator)]
        samples.append(features[chosen])
        targets.append(torch.full((samples_per_class,), class_id, dtype=torch.long))
    return torch.cat(samples), torch.cat(targets)


def composition_crossfit_cache(
    cache: dict[str, object], folds: int, heldout_fold: int, seed: int
) -> tuple[dict[str, object], torch.Tensor]:
    """Remove a balanced fold of entire seen compositions from flow training."""
    if folds <= 1:
        return cache, torch.empty(0, dtype=torch.long)
    if not 0 <= heldout_fold < folds:
        raise ValueError(f"flow-heldout-fold must be in [0, {folds})")
    labels = cache["train"]["pair"].long()
    seen_ids = torch.unique(labels).sort().values
    generator = torch.Generator().manual_seed(seed + 1729)
    permutation = seen_ids[torch.randperm(len(seen_ids), generator=generator)]
    fold_of_pair = torch.full(
        (len(cache["meta"]["pairs"]),), -1, dtype=torch.long
    )
    fold_of_pair[permutation] = torch.arange(len(permutation)) % folds
    heldout_ids = seen_ids[fold_of_pair[seen_ids] == heldout_fold]
    keep = fold_of_pair[labels] != heldout_fold
    crossfit = dict(cache)
    crossfit_train = {}
    for key, value in cache["train"].items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and len(value) == len(labels)
        ):
            crossfit_train[key] = value[keep]
        else:
            crossfit_train[key] = value
    crossfit["train"] = crossfit_train
    return crossfit, heldout_ids


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    set_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    feature_cache = cache
    if args.feature_key != "com":
        feature_cache = dict(cache)
        for split in ("train", "val", "test"):
            feature_cache[split] = dict(cache[split])
            feature_cache[split]["com"] = cache[split][args.feature_key]
    centroids, counts = real_centroids(feature_cache)
    flow_cache, heldout_flow_pairs = composition_crossfit_cache(
        feature_cache, args.flow_pair_folds, args.flow_heldout_fold, args.seed
    )
    flow_centroids, flow_counts = real_centroids(flow_cache)
    if args.additive_path:
        additive = torch.load(args.additive_path, map_location="cpu", weights_only=True).float()
        train_additive = additive
    elif args.prior_mode == "leave_pair_out":
        additive = marginal_completion(
            flow_cache, flow_centroids, flow_counts, leave_seen_pair_out=False
        )
        train_additive = marginal_completion(
            flow_cache, flow_centroids, flow_counts, leave_seen_pair_out=True
        )
    else:
        additive = additive_completion(flow_cache, flow_centroids, flow_counts, 30)
        train_additive = additive
    residuals = residual_bank(flow_cache, train_additive)
    if args.residual_distribution in {"gaussian_diag", "gaussian_isotropic"}:
        # Materialize a fixed bank from the moment-matched Gaussian.  Sampling
        # this bank is efficient and keeps every other training choice exactly
        # identical to the empirical-bootstrap ablation.
        gaussian_generator = torch.Generator().manual_seed(args.seed + 7919)
        residual_mean = residuals.mean(0)
        residual_std = residuals.std(0, unbiased=False).clamp_min(1e-6)
        if args.residual_distribution == "gaussian_isotropic":
            residual_std = residual_std.square().mean().sqrt().expand_as(residual_std)
        residuals = residual_mean + torch.randn(
            residuals.shape, generator=gaussian_generator
        ) * residual_std
    source_centers = additive
    train_source_centers = train_additive
    if args.source_center == "zero":
        # Keep the exact same residual statistics and conditional flow.  Only
        # remove the compositional location, isolating whether ANOVA anchoring
        # is necessary for short-range transport.
        source_centers = torch.zeros_like(additive)
        train_source_centers = torch.zeros_like(train_additive)
    local_tables = local_residual_tables(
        flow_cache, flow_cache["train"]["pair"].long(), args, device
    )
    torch.save(additive.half(), work_dir / "additive_means.pt")
    if args.prior_mode == "leave_pair_out":
        torch.save(train_additive.half(), work_dir / "train_additive_means.pt")
    flow_started = time.perf_counter()
    if args.flow_path:
        flow = PrimitiveConditionalFlow(
            feature_cache["train"]["com"].shape[1], args.hidden_dim, args.cond_dim
        ).to(device)
        flow.load_state_dict(torch.load(args.flow_path, map_location="cpu", weights_only=True))
        flow.eval()
        print(f"loaded flow: {args.flow_path}", flush=True)
    else:
        flow = train_flow(
            flow_cache,
            train_source_centers,
            flow_centroids,
            residuals,
            args,
            device,
            local_tables,
        )
    torch.cuda.synchronize(device)
    flow_seconds = time.perf_counter() - flow_started
    torch.save(flow.state_dict(), work_dir / "flow.pt")

    # Decouple sampling/classifier randomness from whether the flow was trained in
    # this process or loaded from disk, making cached-flow evaluations reproducible.
    set_seed(args.seed + 1000)
    seen_features, seen_labels = balanced_seen_samples(
        feature_cache, counts, args.samples_per_class, args.seed
    )
    if cache["meta"].get("open_world", False):
        val_eval = FastOpenWorldEvaluator(cache, device)
        test_eval = FastOpenWorldEvaluator(cache, device)
    else:
        val_dataset = CompositionDataset(args.dataset_path, phase="val")
        test_dataset = CompositionDataset(args.dataset_path, phase="test")
        val_eval = FastClosedWorldEvaluator(val_dataset, device)
        test_eval = FastClosedWorldEvaluator(test_dataset, device)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    text = cache["prompts"]["pair"].float()
    text_device = F.normalize(text, dim=-1).to(device)
    active_mask = torch.tensor(
        cache["meta"].get("active", [True] * len(text)), dtype=torch.float32, device=device
    )
    val_score_features = feature_cache["val"]["com"]
    test_score_features = feature_cache["test"]["com"]
    val_text_z = zscore(val_score_features.float().to(device) @ text_device.T)
    test_text_z = zscore(test_score_features.float().to(device) @ text_device.T)
    result: dict[str, object] = {
        "method": {
            "residual_bank_mode": args.residual_bank_mode,
            "residual_distribution": args.residual_distribution,
            "transport_objective": args.transport_objective,
            "prior_mode": args.prior_mode,
            "source_center": args.source_center,
            "semantic_neighbors": args.semantic_neighbors,
            "primitive_bonus": args.primitive_bonus,
            "pair_cond_scale": args.pair_cond_scale,
            "aux_weight": args.aux_weight,
            "primitive_aux_weight": args.primitive_aux_weight,
            "feature_key": args.feature_key,
            "freeze_seen_classifier": args.freeze_seen_classifier,
            "cond_drop": args.cond_drop,
            "guidance_scale": args.guidance_scale,
            "flow_pair_folds": args.flow_pair_folds,
            "flow_heldout_fold": args.flow_heldout_fold,
            "flow_heldout_pair_count": len(heldout_flow_pairs),
            "flow_training_pair_count": int((flow_counts > 0).sum()),
        },
        "base_val": val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"]),
        "base_test": test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"]),
    }
    best = None
    classifier_parameters = 0
    generation_and_classifier_started = time.perf_counter()
    for residual_scale in args.residual_scales:
        unseen_features, unseen_labels = generate_unseen(
            flow,
            feature_cache,
            source_centers,
            residuals,
            counts,
            residual_scale,
            args,
            device,
            local_tables,
        )
        classifier_args = argparse.Namespace(
            batch_size=args.classifier_batch_size,
            epochs=args.classifier_epochs,
            lr=args.classifier_lr,
            fixed_weight_mask=(counts > 0) if args.freeze_seen_classifier else None,
        )
        classifier = train_classifier(
            torch.cat([seen_features, unseen_features]),
            torch.cat([seen_labels, unseen_labels]),
            text,
            classifier_args,
            device,
        )
        classifier_parameters = sum(parameter.numel() for parameter in classifier.parameters())
        val_clf = classifier_logits(classifier, val_score_features, device).to(device)
        test_clf = classifier_logits(classifier, test_score_features, device).to(device)
        name = f"residual_{residual_scale:.2f}"
        torch.save(
            {"val": val_clf.half().cpu(), "test": test_clf.half().cpu()},
            work_dir / f"classifier_logits_{name}.pt",
        )
        trials = {}
        val_clf_z = zscore(val_clf)
        for delta_name, val_delta in {
            "full": val_clf_z * active_mask[None],
            "update": (val_clf_z - val_text_z) * active_mask[None],
        }.items():
            for strength in (-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
                logits = val["logits"].float() + strength * val_delta
                stats = val_eval(logits, val["attr"], val["obj"], val["pair"])
                trials[f"{delta_name}_{strength:.2f}"] = stats
                candidate = (
                    stats["AUC"],
                    stats["best_hm"],
                    name,
                    delta_name,
                    strength,
                    test_clf,
                )
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        result[name] = {
            "classifier_val": val_eval(val_clf, val["attr"], val["obj"], val["pair"]),
            "classifier_test": test_eval(test_clf, test["attr"], test["obj"], test["pair"]),
            "trials": trials,
        }
        print(
            f"{name} clf_auc={result[name]['classifier_val']['AUC']:.6f} "
            f"fused_auc={max(x['AUC'] for x in trials.values()):.6f}",
            flush=True,
        )
        del classifier, unseen_features, unseen_labels, val_clf

    assert best is not None
    val_auc, val_hm, name, delta_name, strength, test_clf = best
    val_gain = val_auc - result["base_val"]["AUC"]
    enabled = val_gain >= args.min_val_auc_gain
    if not enabled:
        strength = 0.0
    test_clf_z = zscore(test_clf.to(device))
    test_delta = test_clf_z if delta_name == "full" else test_clf_z - test_text_z
    test_delta = test_delta * active_mask[None]
    fused_test_logits = test["logits"].float() + strength * test_delta
    result["selection"] = {
        "variant": name,
        "delta": delta_name,
        "strength": strength,
        "val_auc": val_auc,
        "val_hm": val_hm,
        "val_auc_gain": val_gain,
        "min_val_auc_gain": args.min_val_auc_gain,
        "enabled": enabled,
    }
    result["fused_test"] = test_eval(fused_test_logits, test["attr"], test["obj"], test["pair"])
    torch.cuda.synchronize(device)
    result["efficiency"] = {
        "flow_parameters": sum(parameter.numel() for parameter in flow.parameters()),
        "classifier_parameters": classifier_parameters,
        "flow_stage_seconds": flow_seconds,
        "flow_loaded": bool(args.flow_path),
        "generation_classifier_eval_seconds": (
            time.perf_counter() - generation_and_classifier_started
        ),
        "total_seconds": time.perf_counter() - run_started,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "generated_samples_per_unseen_class": args.samples_per_class,
        "flow_sampling_steps": (
            args.sample_steps if args.transport_objective == "flow_matching" else 1
        ),
    }
    (work_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selection": result["selection"], "fused_test": result["fused_test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
