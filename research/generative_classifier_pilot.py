"""Balanced classifier trained with generated unseen CAMS features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.eval_saved_energy import adjustment, fuse, seen_pair_mask  # noqa: E402
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    additive_completion,
    real_centroids,
)
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class CosineClassifier(nn.Module):
    def __init__(self, initial_weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(initial_weight.clone())
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100.0) * F.normalize(features, dim=-1) @ F.normalize(self.weight, dim=-1).T


def make_training_set(
    cache: dict[str, object],
    means: torch.Tensor,
    counts: torch.Tensor,
    residual_scale: float,
    samples_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train = cache["train"]
    features = train["com"].float()
    labels = train["pair"].long()
    num_classes = len(means)
    additive_for_seen = means[labels]
    residual_bank = features - additive_for_seen
    samples, targets = [], []
    for class_id in range(num_classes):
        if counts[class_id] > 0:
            available = torch.where(labels == class_id)[0]
            chosen = available[torch.randint(len(available), (samples_per_class,), generator=generator)]
            generated = features[chosen]
        else:
            residual_ids = torch.randint(len(residual_bank), (samples_per_class,), generator=generator)
            generated = F.normalize(means[class_id][None] + residual_scale * residual_bank[residual_ids], dim=-1)
        samples.append(generated)
        targets.append(torch.full((samples_per_class,), class_id, dtype=torch.long))
    return torch.cat(samples), torch.cat(targets)


def train_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial_weight: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> CosineClassifier:
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = CosineClassifier(initial_weight.to(device)).to(device)
    fixed_mask = getattr(args, "fixed_weight_mask", None)
    fixed_initial = None
    if fixed_mask is not None:
        fixed_mask = fixed_mask.to(device=device, dtype=torch.bool)
        fixed_initial = model.weight.detach().clone()
        model.weight.register_hook(
            lambda gradient: gradient.masked_fill(fixed_mask[:, None], 0.0)
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.train()
    for _ in range(args.epochs):
        for batch_features, batch_labels in loader:
            logits = model(batch_features.to(device))
            loss = F.cross_entropy(logits, batch_labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if fixed_initial is not None:
                model.weight.data[fixed_mask] = fixed_initial[fixed_mask]
        scheduler.step()
    return model.eval()


@torch.no_grad()
def classifier_logits(model: CosineClassifier, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    outputs = []
    for chunk in features.split(1024):
        outputs.append(model(chunk.float().to(device)).cpu())
    return torch.cat(outputs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    additive = additive_completion(cache, centroids, counts, 30)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    seen = counts > 0

    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    test_dataset = CompositionDataset(args.dataset_path, phase="test")
    val_eval = FastClosedWorldEvaluator(val_dataset, device)
    test_eval = FastClosedWorldEvaluator(test_dataset, device)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    seen_mask = seen_pair_mask(cache, device)
    result: dict[str, object] = {
        "base_val": val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"]),
        "base_test": test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"]),
    }
    best = None
    variants = []
    for alpha in (0.0, 0.5, 1.0):
        means = F.normalize(alpha * additive + (1.0 - alpha) * text, dim=-1)
        means[seen] = centroids[seen]
        for residual_scale in (0.0, 1.0):
            variants.append((alpha, residual_scale, means.clone()))

    for variant_id, (alpha, residual_scale, means) in enumerate(variants):
        synth_features, synth_labels = make_training_set(
            cache, means, counts, residual_scale, args.samples_per_class, args.seed + variant_id
        )
        model = train_classifier(synth_features, synth_labels, text, args, device)
        val_clf = classifier_logits(model, cache["val"]["com"], device).to(device)
        test_clf = classifier_logits(model, cache["test"]["com"], device).to(device)
        name = f"a{alpha:.2f}_r{residual_scale:.2f}"
        variant_result: dict[str, object] = {
            "classifier_val": val_eval(val_clf, val["attr"], val["obj"], val["pair"]),
            "classifier_test": test_eval(test_clf, test["attr"], test["obj"], test["pair"]),
        }
        trials = {}
        # Treat classifier logits as a generated-data energy over every class.
        delta = (val_clf - val_clf.mean(1, keepdim=True)) / val_clf.std(1, keepdim=True).clamp_min(1e-6)
        for strength in (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
            logits = val["logits"].float() + strength * delta
            stats = val_eval(logits, val["attr"], val["obj"], val["pair"])
            trials[f"{strength:.2f}"] = stats
            candidate = (stats["AUC"], stats["best_hm"], name, strength, test_clf)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        variant_result["trials"] = trials
        result[name] = variant_result
        print(
            f"{name} clf_val_auc={variant_result['classifier_val']['AUC']:.6f} "
            f"best_fused_val={max(x['AUC'] for x in trials.values()):.6f}",
            flush=True,
        )
        del model, synth_features, synth_labels, val_clf

    assert best is not None
    _, _, name, strength, test_clf = best
    test_delta = (test_clf - test_clf.mean(1, keepdim=True)) / test_clf.std(1, keepdim=True).clamp_min(1e-6)
    fused_test_logits = test["logits"].float() + strength * test_delta
    result["selection"] = {"variant": name, "strength": strength, "val_auc": best[0], "val_hm": best[1]}
    result["fused_test"] = test_eval(fused_test_logits, test["attr"], test["obj"], test["pair"])
    (work_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selection": result["selection"], "fused_test": result["fused_test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
