"""Benchmark the training-free ReCoFlow score-fusion inference overhead."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def zscore(logits: torch.Tensor) -> torch.Tensor:
    return (logits - logits.mean(1, keepdim=True)) / logits.std(1, keepdim=True).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    features = cache[args.split]["com"].float().to(device)
    base = cache[args.split]["logits"].float().to(device)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    # The learned cosine classifier has the same shape as the pair-text bank.
    # A fixed perturbation prevents the update path from cancelling numerically.
    generator = torch.Generator(device=device).manual_seed(0)
    learned = F.normalize(text + 0.01 * torch.randn(text.shape, generator=generator, device=device), dim=-1)

    def run() -> torch.Tensor:
        generated_score = zscore(features @ learned.T)
        text_score = zscore(features @ text.T)
        return base + 0.2 * (generated_score - text_score)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(args.repeats):
        output = run()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    seconds = elapsed / args.repeats
    result = {
        "split": args.split,
        "images": len(features),
        "classes": text.shape[0],
        "feature_dim": text.shape[1],
        "mean_seconds": seconds,
        "milliseconds_per_image": 1000.0 * seconds / len(features),
        "images_per_second": len(features) / seconds,
        "peak_cuda_memory_mib_including_cached_scores": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        ),
        "extra_classifier_parameters": text.numel() + 1,
        "checksum": output.mean().item(),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
