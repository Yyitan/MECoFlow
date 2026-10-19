"""Validate an extracted CAMS feature cache before any generator is launched."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--builder", required=True)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument(
        "--expected-status",
        default="amended_reproducible_pipeline",
    )
    parser.add_argument("--expect-deterministic-train", action="store_true")
    parser.add_argument("--allow-missing-test", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = {
        "cache": file_record(Path(args.cache).resolve()),
        "config": file_record(Path(args.config).resolve()),
        "checkpoint": file_record(Path(args.checkpoint).resolve()),
        "builder": file_record(Path(args.builder).resolve()),
        "amendment": file_record(Path(args.amendment).resolve()),
    }
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    required_keys = {"meta", "prompts", "train", "val"}
    assert required_keys <= set(cache)
    if not args.allow_missing_test:
        assert "test" in cache
    else:
        assert "test" not in cache
    provenance = cache["meta"]["feature_cache_provenance"]
    assert provenance["status"] == args.expected_status
    assert int(provenance["extraction_seed"]) == args.expected_seed
    assert Path(provenance["dataset_path"]).resolve() == Path(args.dataset_path).resolve()
    for name in ("config", "checkpoint", "builder", "amendment"):
        assert provenance[name] == expected[name]
    dependencies = {
        "dataset": file_record(Path(args.builder).resolve().parents[1] / "dataset.py"),
        "piflow_pilot": file_record(
            Path(args.builder).resolve().parent / "piflow_pilot.py"
        ),
    }
    assert provenance["dependencies"] == dependencies
    train_transform = provenance["transforms"]["train"]
    assert train_transform["stochastic"] is (not args.expect_deterministic_train)
    if args.expect_deterministic_train:
        assert train_transform["random_sequence_fixed_by_extraction_seed"] is False
        assert provenance["test_read"] is False
    else:
        assert train_transform["random_sequence_fixed_by_extraction_seed"] is True
        val_test_transform = provenance["transforms"]["val_test"]
        assert val_test_transform["stochastic"] is False

    output = {
        "status": "passed",
        "pipeline": provenance["status"],
        "extraction_seed": int(provenance["extraction_seed"]),
        "artifacts": expected,
        "dependencies": dependencies,
        "split_sizes": {
            split: int(cache[split]["com"].shape[0])
            for split in (("train", "val") if args.allow_missing_test else ("train", "val", "test"))
        },
        "transforms": provenance["transforms"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
