"""Primitive-conditioned diffusion energy on complete CAMS latent-query sets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.diffusion_energy_pilot import (  # noqa: E402
    hard_negative_table,
    pair_maps,
    sample_hard_negatives,
)
from research.eval_saved_energy import adjustment, fuse, rank_diagnostics, seen_pair_mask  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator  # noqa: E402
from research.piflow_pilot import timestep_embedding  # noqa: E402
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--rank-weight", type=float, default=10.0)
    parser.add_argument("--geometry-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def normalize_query_set(queries: torch.Tensor) -> torch.Tensor:
    return F.normalize(queries.float(), dim=-1)


def wait_for_file(path: Path) -> None:
    while not path.exists():
        print(f"waiting for query cache: {path}", flush=True)
        time.sleep(10)


class ConditionalSetDenoiser(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.attr = nn.Linear(feature_dim, hidden_dim)
        self.obj = nn.Linear(feature_dim, hidden_dim)
        self.pair = nn.Linear(feature_dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(128, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cond_type = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=6,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, feature_dim))

    def forward(
        self,
        noisy: torch.Tensor,
        signal: torch.Tensor,
        attr_cond: torch.Tensor,
        obj_cond: torch.Tensor,
        pair_cond: torch.Tensor,
    ) -> torch.Tensor:
        conditions = torch.stack(
            [self.attr(attr_cond), self.obj(obj_cond), self.pair(pair_cond)], dim=1
        ) + self.cond_type[None]
        scale, shift = self.film(conditions.mean(1)).chunk(2, dim=-1)
        queries = self.query_norm(self.input(noisy)) * (1.0 + scale[:, None]) + shift[:, None]
        queries = queries + self.time(timestep_embedding(signal, 128))[:, None]
        hidden = self.transformer(torch.cat([conditions, queries], dim=1))[:, 3:]
        return normalize_query_set(self.output(hidden))


def set_similarity(prediction: torch.Tensor, target: torch.Tensor, geometry_weight: float) -> torch.Tensor:
    token_similarity = F.cosine_similarity(prediction, target, dim=-1).mean(1)
    pred_gram = prediction @ prediction.transpose(1, 2)
    target_gram = target @ target.transpose(1, 2)
    geometry_error = (pred_gram - target_gram).square().mean((1, 2))
    return token_similarity - geometry_weight * geometry_error


def train_model(
    queries: torch.Tensor,
    cache: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> ConditionalSetDenoiser:
    train = cache["train"]
    labels = train["pair"].long()
    pair_attr, pair_obj = pair_maps(cache)
    negatives = hard_negative_table(cache, pair_attr, pair_obj)
    counts = torch.bincount(labels, minlength=len(pair_attr)).clamp_min(1)
    sampler = WeightedRandomSampler(counts[labels].float().reciprocal(), len(labels), replacement=True)
    loader = DataLoader(torch.arange(len(labels)), batch_size=args.batch_size, sampler=sampler, num_workers=0)
    pair_attr, pair_obj = pair_attr.to(device), pair_obj.to(device)
    labels_device = labels.to(device)
    attr_prompt = cache["prompts"]["attr"].float().to(device)
    obj_prompt = cache["prompts"]["obj"].float().to(device)
    pair_prompt = cache["prompts"]["pair"].float().to(device)
    model = ConditionalSetDenoiser(queries.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    signal_choices = torch.tensor([0.03, 0.07, 0.15, 0.30], device=device)

    model.train()
    for epoch in range(args.epochs):
        losses, recs, ranks = [], [], []
        for indices in loader:
            clean = normalize_query_set(queries[indices].to(device, non_blocking=True))
            label = labels_device[indices.to(device)]
            negative = sample_hard_negatives(label, negatives, device)
            noise = normalize_query_set(torch.randn_like(clean))
            signal = signal_choices[torch.randint(len(signal_choices), (len(clean),), device=device)]
            noisy = normalize_query_set(
                signal.sqrt()[:, None, None] * clean
                + (1.0 - signal).sqrt()[:, None, None] * noise
            )
            positive = model(
                noisy, signal, attr_prompt[pair_attr[label]], obj_prompt[pair_obj[label]], pair_prompt[label]
            )
            negative_prediction = model(
                noisy,
                signal,
                attr_prompt[pair_attr[negative]],
                obj_prompt[pair_obj[negative]],
                pair_prompt[negative],
            )
            positive_score = set_similarity(positive, clean, args.geometry_weight)
            negative_score = set_similarity(negative_prediction, clean, args.geometry_weight)
            reconstruction = (1.0 - positive_score).mean()
            ranking = F.relu(args.margin - positive_score + negative_score).mean()
            loss = reconstruction + args.rank_weight * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
            recs.append(reconstruction.detach())
            ranks.append(ranking.detach())
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"epoch={epoch + 1:03d} loss={torch.stack(losses).mean().item():.5f} "
                f"rec={torch.stack(recs).mean().item():.5f} rank={torch.stack(ranks).mean().item():.5f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def score_split(
    model: ConditionalSetDenoiser,
    queries: torch.Tensor,
    split: dict[str, torch.Tensor],
    cache: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_attr, pair_obj = pair_maps(cache)
    pair_attr, pair_obj = pair_attr.to(device), pair_obj.to(device)
    attr_prompt = cache["prompts"]["attr"].float().to(device)
    obj_prompt = cache["prompts"]["obj"].float().to(device)
    pair_prompt = cache["prompts"]["pair"].float().to(device)
    topk = min(args.topk, split["logits"].shape[1])
    top_indices = split["logits"].topk(topk, dim=1).indices
    energies = torch.empty(len(queries), topk)
    generator = torch.Generator().manual_seed(seed)
    noises = torch.randn(queries.shape, generator=generator, dtype=torch.float16)
    signal_levels = (0.05, 0.20)

    for start in range(0, len(queries), args.eval_batch_size):
        stop = min(start + args.eval_batch_size, len(queries))
        clean = normalize_query_set(queries[start:stop].to(device))
        noise = normalize_query_set(noises[start:stop].to(device))
        candidates = top_indices[start:stop].to(device)
        flat_candidates = candidates.flatten()
        score = torch.zeros(len(clean), topk, device=device)
        for level in signal_levels:
            signal = torch.full((len(clean),), level, device=device)
            noisy = normalize_query_set(
                math.sqrt(level) * clean + math.sqrt(1.0 - level) * noise
            )
            prediction = model(
                noisy.repeat_interleave(topk, 0),
                signal.repeat_interleave(topk),
                attr_prompt[pair_attr[flat_candidates]],
                obj_prompt[pair_obj[flat_candidates]],
                pair_prompt[flat_candidates],
            ).view(len(clean), topk, clean.shape[1], clean.shape[2])
            repeated_clean = clean[:, None].expand_as(prediction).reshape(-1, clean.shape[1], clean.shape[2])
            score += set_similarity(
                prediction.reshape(-1, clean.shape[1], clean.shape[2]),
                repeated_clean,
                args.geometry_weight,
            ).view(len(clean), topk)
        energies[start:stop] = (score / len(signal_levels)).cpu()
        if start == 0 or stop == len(queries) or (start // args.eval_batch_size + 1) % 100 == 0:
            print(f"score {stop}/{len(queries)}", flush=True)
    return top_indices, energies


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    query_dir = Path(args.query_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_queries = torch.load(query_dir / "train_queries.pt", map_location="cpu", weights_only=True)
    model = train_model(train_queries, cache, args, device)
    torch.save(model.state_dict(), work_dir / "set_denoiser.pt")
    del train_queries

    wait_for_file(query_dir / "val_queries.pt")
    val_queries = torch.load(query_dir / "val_queries.pt", map_location="cpu", weights_only=True)
    val_indices, val_energy = score_split(model, val_queries, cache["val"], cache, args, device, 3000 + args.seed)
    del val_queries
    wait_for_file(query_dir / "test_queries.pt")
    test_queries = torch.load(query_dir / "test_queries.pt", map_location="cpu", weights_only=True)
    test_indices, test_energy = score_split(model, test_queries, cache["test"], cache, args, device, 4000 + args.seed)
    del test_queries
    torch.save(
        {"val_indices": val_indices, "val_energy": val_energy, "test_indices": test_indices, "test_energy": test_energy},
        work_dir / "energies.pt",
    )

    seen_mask = seen_pair_mask(cache, device)
    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    test_dataset = CompositionDataset(args.dataset_path, phase="test")
    val_eval = FastClosedWorldEvaluator(val_dataset, device)
    test_eval = FastClosedWorldEvaluator(test_dataset, device)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    result: dict[str, object] = {
        "base_val": val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"]),
        "base_test": test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"]),
        "rank_val": rank_diagnostics(cache["val"], val_indices.to(device), val_energy.to(device), seen_mask),
        "rank_test": rank_diagnostics(cache["test"], test_indices.to(device), test_energy.to(device), seen_mask),
    }
    strengths = (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
    best = None
    for kind in ("all", "group", "unseen_only"):
        val_delta = adjustment(kind, val_indices.to(device), val_energy.to(device), seen_mask)
        trials = {}
        for strength in strengths:
            logits = fuse(val["logits"].float(), val_indices.to(device), val_delta, strength)
            stats = val_eval(logits, val["attr"], val["obj"], val["pair"])
            trials[f"{strength:.2f}"] = stats
            candidate = (stats["AUC"], stats["best_hm"], kind, strength)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        result[kind] = trials
    assert best is not None
    _, _, kind, strength = best
    test_delta = adjustment(kind, test_indices.to(device), test_energy.to(device), seen_mask)
    fused_test_logits = fuse(test["logits"].float(), test_indices.to(device), test_delta, strength)
    result["selection"] = {"kind": kind, "strength": strength, "val_auc": best[0], "val_hm": best[1]}
    result["fused_test"] = test_eval(fused_test_logits, test["attr"], test["obj"], test["pair"])
    (work_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "rank_val": result["rank_val"],
        "rank_test": result["rank_test"],
        "selection": result["selection"],
        "fused_test": result["fused_test"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
