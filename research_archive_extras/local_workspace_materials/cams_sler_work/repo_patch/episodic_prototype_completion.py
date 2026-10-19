"""Leave-one-composition-out visual prototype completion for CZSL.

For every seen training composition, its own visual samples are removed from the
attribute, object and global context used to reconstruct its prototype. A shared
completion network is optimized with centroid reconstruction and primitive-sharing
hard-negative contrastive losses. The same network then completes truly unseen
composition prototypes from their seen primitive neighborhoods.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from eval_reliability_gated_fusion import row_zscore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("select", "apply"), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--anchor-batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def pair_indices(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    attrs = {value: index for index, value in enumerate(meta["attrs"])}
    objs = {value: index for index, value in enumerate(meta["objs"])}
    return (
        torch.tensor([attrs[a] for a, _ in meta["pairs"]]),
        torch.tensor([objs[o] for _, o in meta["pairs"]]),
    )


def real_centroids(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    train = cache["train"]
    num_pairs = len(cache["meta"]["pairs"])
    sums = torch.zeros(num_pairs, train["com"].shape[1])
    counts = torch.bincount(train["pair"].long(), minlength=num_pairs)
    sums.index_add_(0, train["pair"].long(), train["com"].float())
    centroids = torch.zeros_like(sums)
    centroids[counts > 0] = F.normalize(
        sums[counts > 0] / counts[counts > 0, None], dim=-1
    )
    return centroids, counts


def leave_one_composition_sources(
    cache: dict[str, object], centroids: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Build [attr context, object context, additive context, text] per pair."""
    pair_attr, pair_obj = pair_indices(cache)
    num_attrs, num_objs = len(cache["meta"]["attrs"]), len(cache["meta"]["objs"])
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

    own_sum = weighted
    own_count = counts.float()
    seen = counts > 0
    attr_denominator = attr_count[pair_attr] - own_count
    obj_denominator = obj_count[pair_obj] - own_count
    global_denominator = global_count - own_count
    attr_context = (attr_sum[pair_attr] - own_sum) / attr_denominator.clamp_min(1)[:, None]
    obj_context = (obj_sum[pair_obj] - own_sum) / obj_denominator.clamp_min(1)[:, None]
    global_context = (global_sum[None] - own_sum) / global_denominator.clamp_min(1)[:, None]

    # For truly unseen pairs own_count is zero, so the expressions above naturally
    # use every available seen neighbor. Degree-one seen primitives fall back to the
    # leave-one-out global context rather than leaking the target centroid.
    attr_context[attr_denominator <= 0] = global_context[attr_denominator <= 0]
    obj_context[obj_denominator <= 0] = global_context[obj_denominator <= 0]
    additive = F.normalize(attr_context + obj_context - global_context, dim=-1)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    sources = torch.stack(
        [F.normalize(attr_context, dim=-1), F.normalize(obj_context, dim=-1), additive, text],
        dim=1,
    )
    assert torch.isfinite(sources).all()
    assert not torch.any(sources[seen, :3].isnan())
    return sources


class PrototypeCompleter(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(feature_dim, hidden_dim) for _ in range(4)]
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4)
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.cat(
            [projection(sources[:, index]) for index, projection in enumerate(self.projections)],
            dim=-1,
        )
        weights = torch.softmax(self.gate(hidden), dim=-1)
        base = (weights[:, :, None] * sources).sum(1)
        residual_scale = 0.5 * torch.sigmoid(self.residual_logit)
        output = F.normalize(base + residual_scale * torch.tanh(self.residual(hidden)), dim=-1)
        return output, weights


def structured_candidate_batch(
    anchors: torch.Tensor,
    seen_ids: torch.Tensor,
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    candidates = [anchors]
    for primitive_index in (pair_attr, pair_obj):
        selected = []
        for anchor in anchors.cpu().tolist():
            neighbors = seen_ids[
                primitive_index[seen_ids] == primitive_index[anchor]
            ]
            neighbors = neighbors[neighbors != anchor]
            if len(neighbors) == 0:
                neighbors = seen_ids[seen_ids != anchor]
            selected.append(neighbors[torch.randint(len(neighbors), (1,), generator=generator)])
        candidates.append(torch.cat(selected))
    return torch.unique(torch.cat(candidates))


def train_completer(
    model: PrototypeCompleter,
    sources: torch.Tensor,
    centroids: torch.Tensor,
    counts: torch.Tensor,
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    seen_ids = torch.where(counts > 0)[0]
    generator = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        permutation = seen_ids[torch.randperm(len(seen_ids), generator=generator)]
        epoch_losses = []
        for anchors in permutation.split(args.anchor_batch):
            candidates = structured_candidate_batch(
                anchors, seen_ids, pair_attr, pair_obj, generator
            )
            candidate_sources = sources[candidates].to(device)
            predicted, weights = model(candidate_sources)
            targets = centroids[candidates].to(device)
            position = {int(pair): index for index, pair in enumerate(candidates.tolist())}
            anchor_positions = torch.tensor(
                [position[int(pair)] for pair in anchors.tolist()], device=device
            )
            anchor_targets = centroids[anchors].to(device)
            scale = 20.0
            forward_logits = scale * anchor_targets @ predicted.T
            reverse_logits = scale * predicted[anchor_positions] @ targets.T
            contrastive = 0.5 * (
                F.cross_entropy(forward_logits, anchor_positions)
                + F.cross_entropy(reverse_logits, anchor_positions)
            )
            reconstruction = (1.0 - (predicted * targets).sum(1)).mean()
            gate_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1).mean()
            loss = contrastive + 0.5 * reconstruction - 0.01 * gate_entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append((loss.detach(), contrastive.detach(), reconstruction.detach()))
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            values = torch.stack([torch.stack(item) for item in epoch_losses]).mean(0)
            record = {
                "epoch": epoch + 1,
                "loss": float(values[0]),
                "contrastive": float(values[1]),
                "reconstruction": float(values[2]),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    return history


def prototype_variants(
    model: PrototypeCompleter,
    sources: torch.Tensor,
    centroids: torch.Tensor,
    counts: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    with torch.no_grad():
        predicted_parts, weight_parts = [], []
        for chunk in sources.split(1024):
            predicted, weights = model(chunk.to(device))
            predicted_parts.append(predicted.cpu())
            weight_parts.append(weights.cpu())
        predicted = torch.cat(predicted_parts)
        weights = torch.cat(weight_parts)
    seen = counts > 0
    completed_hybrid = predicted.clone()
    completed_hybrid[seen] = centroids[seen]
    additive_hybrid = sources[:, 2].clone()
    additive_hybrid[seen] = centroids[seen]
    diagnostics = {
        "mean_gate_weights": weights.mean(0).tolist(),
        "seen_reconstruction_cosine": float(
            (predicted[seen] * centroids[seen]).sum(1).mean()
        ),
        "learned_residual_scale": float(
            (0.5 * torch.sigmoid(model.residual_logit)).detach().cpu()
        ),
    }
    return {
        "completed_hybrid": completed_hybrid,
        "completed_all": predicted,
        "additive_hybrid": additive_hybrid,
    }, diagnostics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator

    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    sources = leave_one_composition_sources(cache, centroids, counts)
    pair_attr, pair_obj = pair_indices(cache)
    model = PrototypeCompleter(centroids.shape[1], args.hidden_dim).to(device)

    checkpoint_path = Path(args.checkpoint)
    if args.mode == "select":
        history = train_completer(
            model, sources, centroids, counts, pair_attr, pair_obj, args, device
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "hidden_dim": args.hidden_dim},
            checkpoint_path,
        )
        split_name, phase = "val", "val"
    else:
        saved_model = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved_model["hidden_dim"] != args.hidden_dim:
            raise ValueError("--hidden-dim does not match checkpoint")
        model.load_state_dict(saved_model["state_dict"])
        history = []
        split_name, phase = "test", "test"
    model.eval()
    variants, diagnostics = prototype_variants(model, sources, centroids, counts, device)

    split = {key: value.to(device) for key, value in cache[split_name].items()}
    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase=phase), device
    )
    base = split["logits"].float()
    base_stats = evaluator(base, split["attr"], split["obj"], split["pair"])
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    text_z = row_zscore(split["com"].float() @ text.T)

    if args.mode == "select":
        candidates, best = [], None
        for variant_name, prototypes in variants.items():
            prototype_z = row_zscore(split["com"].float() @ prototypes.to(device).T)
            for delta_name, delta in (
                ("full", prototype_z),
                ("update", prototype_z - text_z),
            ):
                for strength in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
                    stats = evaluator(
                        base + strength * delta,
                        split["attr"], split["obj"], split["pair"],
                    )
                    auc_gain = stats["AUC"] - base_stats["AUC"]
                    hm_gain = stats["best_hm"] - base_stats["best_hm"]
                    score = (
                        auc_gain / max(base_stats["AUC"], 1e-6)
                        + hm_gain / max(base_stats["best_hm"], 1e-6)
                    )
                    record = {
                        "variant": variant_name,
                        "delta": delta_name,
                        "strength": strength,
                        "AUC": stats["AUC"],
                        "best_hm": stats["best_hm"],
                        "auc_gain": auc_gain,
                        "hm_gain": hm_gain,
                        "normalized_score": score,
                        "stats": stats,
                    }
                    candidates.append(record)
                    key = (score, stats["AUC"], stats["best_hm"], record)
                    if auc_gain >= 0 and hm_gain >= 0 and (
                        best is None or key[:3] > best[:3]
                    ):
                        best = key
        if best is None:
            raise RuntimeError("No prototype candidate improved both validation metrics")
        candidates.sort(key=lambda item: item["normalized_score"], reverse=True)
        output = {
            "protocol": "train_only_completion_validation_only_selection",
            "base_val": base_stats,
            "selection": best[3],
            "prototype_diagnostics": diagnostics,
            "training_history": history,
            "top_candidates": candidates[:30],
        }
    else:
        if not args.selection:
            raise ValueError("--selection is required in apply mode")
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["selection"]
        prototypes = variants[selection["variant"]].to(device)
        prototype_z = row_zscore(split["com"].float() @ prototypes.T)
        delta = prototype_z if selection["delta"] == "full" else prototype_z - text_z
        fused = evaluator(
            base + selection["strength"] * delta,
            split["attr"], split["obj"], split["pair"],
        )
        output = {
            "protocol": "test_application_of_frozen_train_and_validation_selection",
            "selection": selection,
            "base_test": base_stats,
            "fused_test": fused,
            "gain": {
                "AUC": fused["AUC"] - base_stats["AUC"],
                "best_hm": fused["best_hm"] - base_stats["best_hm"],
            },
            "prototype_diagnostics": diagnostics,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in output if key not in ("top_candidates", "training_history")}, indent=2))


if __name__ == "__main__":
    main()
