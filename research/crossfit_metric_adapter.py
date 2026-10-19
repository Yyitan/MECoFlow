"""Learn a generated-to-real metric from pair-cross-fitted prototypes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import set_seed  # noqa: E402


class CrossfitMetric(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.image_down = nn.Linear(dim, rank, bias=False)
        self.image_up = nn.Linear(rank, dim, bias=False)
        self.proto_down = nn.Linear(dim, rank, bias=False)
        self.proto_up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.image_up.weight)
        nn.init.zeros_(self.proto_up.weight)
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def image(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value + self.image_up(F.gelu(self.image_down(value))), dim=-1)

    def prototype(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value + self.proto_up(F.gelu(self.proto_down(value))), dim=-1)

    def logits(self, image: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100.0) * self.image(image) @ self.prototype(prototype).T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-prototypes", nargs="+", required=True)
    parser.add_argument("--eval-prototypes", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--class-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_crossfit(paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    pair_ids, prototypes = [], []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        pair_ids.append(payload["pair_ids"].long())
        prototypes.append(payload["prototypes"].float())
    ids = torch.cat(pair_ids)
    values = torch.cat(prototypes)
    if len(torch.unique(ids)) != len(ids):
        raise ValueError("crossfit prototype files contain duplicate pair ids")
    return ids, values


def instance_table(
    labels: torch.Tensor, pair_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, counts = [], []
    max_count = max(int((labels == pair_id).sum()) for pair_id in pair_ids.tolist())
    for pair_id in pair_ids.tolist():
        indices = torch.where(labels == pair_id)[0]
        row = torch.empty(max_count, dtype=torch.long)
        row[: len(indices)] = indices
        if len(indices) < max_count:
            row[len(indices) :] = indices[0]
        rows.append(row)
        counts.append(len(indices))
    return torch.stack(rows), torch.tensor(counts)


@torch.no_grad()
def evaluate(
    model: CrossfitMetric,
    features: torch.Tensor,
    labels: torch.Tensor,
    pair_ids: torch.Tensor,
    prototypes: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    lookup = torch.full((int(labels.max()) + 1,), -1, dtype=torch.long)
    lookup[pair_ids] = torch.arange(len(pair_ids))
    mask = lookup[labels] >= 0
    target = lookup[labels[mask]]
    image = features[mask].float().to(device)
    proto = prototypes.float().to(device)
    raw_logits = F.normalize(image, dim=-1) @ F.normalize(proto, dim=-1).T
    adapted_logits = model.logits(image, proto)

    def macro(logits: torch.Tensor) -> float:
        prediction = logits.argmax(-1).cpu()
        correct = torch.bincount(
            target,
            weights=(prediction == target).float(),
            minlength=len(pair_ids),
        )
        count = torch.bincount(target, minlength=len(pair_ids)).clamp_min(1)
        return (correct / count).mean().item()

    real_centroids = []
    for pair_id in pair_ids.tolist():
        real_centroids.append(F.normalize(features[labels == pair_id].float().mean(0), dim=-1))
    real_centroids = torch.stack(real_centroids).to(device)
    raw_alignment = F.cosine_similarity(real_centroids, F.normalize(proto, dim=-1), dim=-1)
    adapted_alignment = F.cosine_similarity(
        model.image(real_centroids), model.prototype(proto), dim=-1
    )
    return {
        "raw_macro_top1": macro(raw_logits),
        "adapted_macro_top1": macro(adapted_logits),
        "raw_mean_alignment": raw_alignment.mean().item(),
        "adapted_mean_alignment": adapted_alignment.mean().item(),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    features = cache["train"]["com"].float()
    labels = cache["train"]["pair"].long()
    train_ids, train_prototypes = load_crossfit(args.train_prototypes)
    eval_ids, eval_prototypes = load_crossfit([args.eval_prototypes])
    table, counts = instance_table(labels, train_ids)

    model = CrossfitMetric(features.shape[1], args.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed + 4000)
    train_prototypes_device = train_prototypes.to(device)
    model.train()
    for epoch in range(args.epochs):
        losses = []
        order = torch.randperm(len(train_ids), generator=generator)
        for local_ids in order.split(args.class_batch_size):
            offsets = (
                torch.rand(len(local_ids), generator=generator) * counts[local_ids]
            ).long()
            image_ids = table[local_ids, offsets]
            image = features[image_ids].to(device)
            proto = train_prototypes_device[local_ids.to(device)]
            logits = model.logits(image, proto)
            target = torch.arange(len(local_ids), device=device)
            loss = 0.5 * (
                F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)
            )
            image_shift = model.image(image) - F.normalize(image, dim=-1)
            proto_shift = model.prototype(proto) - F.normalize(proto, dim=-1)
            loss = loss + args.anchor_weight * (
                image_shift.square().mean() + proto_shift.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1:03d} loss={torch.stack(losses).mean().item():.5f}", flush=True)
    model.eval()
    result = {
        "protocol": {
            "train_pairs": len(train_ids),
            "eval_pairs": len(eval_ids),
            "rank": args.rank,
            "anchor_weight": args.anchor_weight,
            "seed": args.seed,
        },
        "eval": evaluate(
            model, features, labels, eval_ids, eval_prototypes, device
        ),
    }
    torch.save(model.state_dict(), work_dir / "metric.pt")
    (work_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
