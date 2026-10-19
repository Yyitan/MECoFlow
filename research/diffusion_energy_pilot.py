"""Low-cost diagnostic for primitive-conditioned denoising energy in CAMS space.

Unlike feature synthesis, this model scores how well an attribute-object condition
explains a corrupted *real* CAMS feature. CAMS supplies a top-K shortlist, so the
generative energy is only evaluated for hard candidate compositions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from research.fast_feature_evaluation import FastClosedWorldEvaluator  # noqa: E402
from research.piflow_pilot import ResidualBlock, timestep_embedding  # noqa: E402
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class PrimitiveDenoiser(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.attr_proj = nn.Linear(feature_dim, cond_dim)
        self.obj_proj = nn.Linear(feature_dim, cond_dim)
        self.pair_proj = nn.Linear(feature_dim, cond_dim)
        self.time_proj = nn.Sequential(nn.Linear(128, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, cond_dim) for _ in range(3)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, feature_dim))

    def forward(
        self,
        noisy: torch.Tensor,
        signal: torch.Tensor,
        attr_cond: torch.Tensor,
        obj_cond: torch.Tensor,
        pair_cond: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.attr_proj(attr_cond) + self.obj_proj(obj_cond) + self.pair_proj(pair_cond)
        cond = cond + self.time_proj(timestep_embedding(signal, 128))
        hidden = self.input(noisy)
        for block in self.blocks:
            hidden = block(hidden, cond)
        return F.normalize(self.output(hidden), dim=-1)


def pair_maps(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    pair_attr = torch.tensor([meta["attrs"].index(a) for a, _ in meta["pairs"]], dtype=torch.long)
    pair_obj = torch.tensor([meta["objs"].index(o) for _, o in meta["pairs"]], dtype=torch.long)
    return pair_attr, pair_obj


def hard_negative_table(cache: dict[str, object], pair_attr: torch.Tensor, pair_obj: torch.Tensor) -> list[torch.Tensor]:
    train_ids = torch.unique(cache["train"]["pair"]).tolist()
    table: list[torch.Tensor] = []
    for pair_id in range(len(pair_attr)):
        candidates = [
            other
            for other in train_ids
            if other != pair_id
            and (pair_attr[other] == pair_attr[pair_id] or pair_obj[other] == pair_obj[pair_id])
        ]
        if not candidates:
            candidates = [other for other in train_ids if other != pair_id]
        table.append(torch.tensor(candidates, dtype=torch.long))
    return table


def sample_hard_negatives(labels: torch.Tensor, table: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    negatives = []
    for label in labels.cpu().tolist():
        candidates = table[label]
        negatives.append(candidates[torch.randint(len(candidates), ())])
    return torch.stack(negatives).to(device)


def train_model(
    cache: dict[str, object], args: argparse.Namespace, device: torch.device
) -> PrimitiveDenoiser:
    train = cache["train"]
    prompts = cache["prompts"]
    labels = train["pair"].long()
    pair_attr, pair_obj = pair_maps(cache)
    negative_table = hard_negative_table(cache, pair_attr, pair_obj)
    counts = torch.bincount(labels, minlength=len(pair_attr)).clamp_min(1)
    sampler = WeightedRandomSampler(counts[labels].float().reciprocal(), len(labels), replacement=True)
    loader = DataLoader(torch.arange(len(labels)), batch_size=args.batch_size, sampler=sampler, num_workers=0)

    features = train["com"].float().to(device)
    labels_device = labels.to(device)
    pair_attr = pair_attr.to(device)
    pair_obj = pair_obj.to(device)
    attr_prompt = prompts["attr"].float().to(device)
    obj_prompt = prompts["obj"].float().to(device)
    pair_prompt = prompts["pair"].float().to(device)
    model = PrimitiveDenoiser(features.shape[1], args.hidden_dim, args.cond_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    signal_choices = torch.tensor([0.05, 0.10, 0.20, 0.35], device=device)

    model.train()
    for epoch in range(args.epochs):
        losses, recs, ranks = [], [], []
        for batch_indices in loader:
            batch_indices = batch_indices.to(device)
            label = labels_device[batch_indices]
            negative = sample_hard_negatives(label, negative_table, device)
            clean = features[batch_indices]
            noise = F.normalize(torch.randn_like(clean), dim=-1)
            signal = signal_choices[torch.randint(len(signal_choices), (len(clean),), device=device)]
            noisy = signal.sqrt()[:, None] * clean + (1.0 - signal).sqrt()[:, None] * noise
            noisy = F.normalize(noisy, dim=-1)

            positive = model(
                noisy,
                signal,
                attr_prompt[pair_attr[label]],
                obj_prompt[pair_obj[label]],
                pair_prompt[label],
            )
            negative_prediction = model(
                noisy,
                signal,
                attr_prompt[pair_attr[negative]],
                obj_prompt[pair_obj[negative]],
                pair_prompt[negative],
            )
            positive_score = F.cosine_similarity(positive, clean)
            negative_score = F.cosine_similarity(negative_prediction, clean)
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
def denoising_energy(
    model: PrimitiveDenoiser,
    split: dict[str, torch.Tensor],
    cache: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    pair_attr, pair_obj = pair_maps(cache)
    pair_attr, pair_obj = pair_attr.to(device), pair_obj.to(device)
    attr_prompt = cache["prompts"]["attr"].float().to(device)
    obj_prompt = cache["prompts"]["obj"].float().to(device)
    pair_prompt = cache["prompts"]["pair"].float().to(device)
    features = split["com"].float()
    logits = split["logits"].float()
    topk = min(args.topk, logits.shape[1])
    top_indices = logits.topk(topk, dim=1).indices
    energies = torch.empty(len(features), topk)
    generator = torch.Generator().manual_seed(seed)
    noises = F.normalize(torch.randn(features.shape, generator=generator), dim=-1)
    signal_levels = (0.05, 0.15, 0.35)

    for start in range(0, len(features), args.eval_batch_size):
        stop = min(start + args.eval_batch_size, len(features))
        clean = features[start:stop].to(device)
        noise = noises[start:stop].to(device)
        candidates = top_indices[start:stop].to(device)
        score_sum = torch.zeros(len(clean), topk, device=device)
        flat_candidates = candidates.flatten()
        for level in signal_levels:
            signal = torch.full((len(clean),), level, device=device)
            noisy = F.normalize(math.sqrt(level) * clean + math.sqrt(1.0 - level) * noise, dim=-1)
            repeated_noisy = noisy.repeat_interleave(topk, dim=0)
            repeated_signal = signal.repeat_interleave(topk)
            prediction = model(
                repeated_noisy,
                repeated_signal,
                attr_prompt[pair_attr[flat_candidates]],
                obj_prompt[pair_obj[flat_candidates]],
                pair_prompt[flat_candidates],
            ).view(len(clean), topk, -1)
            score_sum += F.cosine_similarity(prediction, clean[:, None, :], dim=-1)
        energies[start:stop] = (score_sum / len(signal_levels)).cpu()

    truth = split["pair"].long()
    in_topk = (top_indices == truth[:, None]).any(1)
    meta = cache["meta"]
    train_pair_ids = torch.tensor(
        [meta["pairs"].index(pair) for pair in meta["train_pairs"]], dtype=torch.long
    )
    seen_pair_mask = torch.zeros(len(meta["pairs"]), dtype=torch.bool)
    seen_pair_mask[train_pair_ids] = True
    seen_sample = seen_pair_mask[truth]
    recall = {
        "all": float(in_topk.float().mean()),
        "seen": float(in_topk[seen_sample].float().mean()),
        "unseen": float(in_topk[~seen_sample].float().mean()),
    }
    return top_indices, energies, recall


def fuse_logits(base: torch.Tensor, indices: torch.Tensor, energy: torch.Tensor, strength: float) -> torch.Tensor:
    energy_z = (energy - energy.mean(1, keepdim=True)) / energy.std(1, keepdim=True).clamp_min(1e-6)
    fused = base.clone()
    fused.scatter_add_(1, indices, strength * energy_z)
    return fused


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    model = train_model(cache, args, device)
    torch.save(model.state_dict(), work_dir / "denoiser.pt")

    val_indices, val_energy, val_recall = denoising_energy(model, cache["val"], cache, args, device, 1000 + args.seed)
    test_indices, test_energy, test_recall = denoising_energy(model, cache["test"], cache, args, device, 2000 + args.seed)
    torch.save(
        {"val_indices": val_indices, "val_energy": val_energy, "test_indices": test_indices, "test_energy": test_energy},
        work_dir / "energies.pt",
    )

    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    test_dataset = CompositionDataset(args.dataset_path, phase="test")
    val_eval = FastClosedWorldEvaluator(val_dataset, device)
    test_eval = FastClosedWorldEvaluator(test_dataset, device)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    best = None
    trials: dict[str, object] = {}
    for strength in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0):
        logits = fuse_logits(val["logits"].float(), val_indices.to(device), val_energy.to(device), strength)
        stats = val_eval(logits, val["attr"], val["obj"], val["pair"])
        trials[f"{strength:.2f}"] = stats
        candidate = (stats["AUC"], stats["best_hm"], strength)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    strength = best[2]
    fused_test_logits = fuse_logits(
        test["logits"].float(), test_indices.to(device), test_energy.to(device), strength
    )
    fused_test = test_eval(fused_test_logits, test["attr"], test["obj"], test["pair"])
    result = {
        "base_val": base_val,
        "base_test": base_test,
        "topk": args.topk,
        "topk_recall_val": val_recall,
        "topk_recall_test": test_recall,
        "val_trials": trials,
        "selection": {"strength": strength, "val_auc": best[0], "val_hm": best[1]},
        "fused_test": fused_test,
    }
    (work_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
