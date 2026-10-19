import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.piflow_pilot import build_cache, load_config
from utils import set_seed


def file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--amendment")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output)
    config = load_config(str(config_path))
    extraction_seed = int(config.seed)
    set_seed(extraction_seed)
    cache = build_cache(config, args, output_path, torch.device("cuda"))

    provenance = {
        "status": "amended_reproducible_pipeline",
        "extraction_seed": extraction_seed,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "config": file_record(config_path),
        "checkpoint": file_record(checkpoint_path),
        "builder": file_record(Path(__file__)),
        "dependencies": {
            "dataset": file_record(ROOT / "dataset.py"),
            "piflow_pilot": file_record(ROOT / "research" / "piflow_pilot.py"),
        },
        "transforms": {
            "train": {
                "definition": (
                    "Resize(224,BICUBIC)->CenterCrop(224)->RandomHorizontalFlip->"
                    "RandomPerspective->RandomRotation(5deg)->RGB->ToTensor->CLIP normalize"
                ),
                "stochastic": True,
                "random_sequence_fixed_by_extraction_seed": True,
            },
            "val_test": {
                "definition": (
                    "Resize(224,BICUBIC)->CenterCrop(224)->RGB->ToTensor->CLIP normalize"
                ),
                "stochastic": False,
            },
        },
    }
    if args.amendment:
        provenance["amendment"] = file_record(Path(args.amendment).resolve())
    cache["meta"]["feature_cache_provenance"] = provenance
    torch.save(cache, output_path)
    print(json.dumps(provenance, indent=2), flush=True)
    print(f"saved cache: {output_path}", flush=True)


if __name__ == "__main__":
    main()
