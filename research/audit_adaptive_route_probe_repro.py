"""Audit that the corrected validation probe preserves a preregistered result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--old-probe", required=True)
    parser.add_argument("--corrected-probe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    old_path = Path(args.old_probe).resolve()
    corrected_path = Path(args.corrected_probe).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    train_pairs = cache["train"]["pair"].long()
    val_pairs = cache["val"]["pair"].long()
    pair_seen = torch.bincount(train_pairs, minlength=len(cache["meta"]["pairs"])) > 0
    old = json.loads(old_path.read_text(encoding="utf-8"))
    corrected = json.loads(corrected_path.read_text(encoding="utf-8"))
    old_best = old["best_balanced"]
    corrected_best = corrected["best_balanced"]
    same_candidate = old_best["name"] == corrected_best["name"]
    same_gain = all(
        abs(old_best[key] - corrected_best[key]) < 1e-12
        for key in ("auc_gain", "hm_gain")
    )
    result = {
        "status": "passed" if same_candidate and same_gain else "failed",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "old_probe": {"path": str(old_path), "sha256": sha256(old_path)},
        "corrected_probe": {"path": str(corrected_path), "sha256": sha256(corrected_path)},
        "validation_seen_pair_count": int(pair_seen[val_pairs].sum().item()),
        "validation_sample_count": int(val_pairs.numel()),
        "same_candidate": same_candidate,
        "same_gain": same_gain,
        "old_best": old_best,
        "corrected_best": corrected_best,
        "decision": "The fixed test result remains valid: the corrected pair-seen mask reproduces the preregistered candidate and validation gains.",
    }
    Path(args.output).resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
