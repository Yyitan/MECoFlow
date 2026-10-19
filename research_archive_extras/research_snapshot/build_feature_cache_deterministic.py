"""Build a validation-only cache with deterministic train feature extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset as dataset_module  # noqa: E402
from dataset import CompositionDataset  # noqa: E402
from model.configure_model import configure_model  # noqa: E402
from research.piflow_pilot import extract_split, load_config, prompt_features  # noqa: E402
from utils import set_seed  # noqa: E402


def file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def deterministic_transform_image(split: str = "train", imagenet: bool = False):
    if imagenet:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.27577711, 0.27577711)
    return Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(mean, std),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--amendment", required=True)
    args = parser.parse_args()

    # CompositionDataset resolves transform_image from the dataset module at runtime.
    dataset_module.transform_image = deterministic_transform_image
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    amendment_path = Path(args.amendment).resolve()
    config = load_config(str(config_path))
    extraction_seed = int(config.seed)
    set_seed(extraction_seed)
    train_dataset = CompositionDataset(args.dataset_path, phase="train")
    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    model, _ = configure_model(config, train_dataset)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.to("cuda").eval()
    prompts = prompt_features(model, train_dataset)
    cache = {
        "meta": {
            "attrs": train_dataset.attrs,
            "objs": train_dataset.objs,
            "pairs": train_dataset.pairs,
            "train_pairs": train_dataset.train_pairs,
            "val_pairs": train_dataset.val_pairs,
            "test_pairs": train_dataset.test_pairs,
        },
        "prompts": prompts,
        "train": extract_split(model, train_dataset, prompts, config.beta, torch.device("cuda"), compute_logits=False),
        "val": extract_split(model, val_dataset, prompts, config.beta, torch.device("cuda")),
    }
    provenance = {
        "status": "deterministic_validation_only_robustness",
        "extraction_seed": extraction_seed,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "config": file_record(config_path),
        "checkpoint": file_record(checkpoint_path),
        "builder": file_record(Path(__file__)),
        "amendment": file_record(amendment_path),
        "dependencies": {
            "dataset": file_record(ROOT / "dataset.py"),
            "piflow_pilot": file_record(ROOT / "research" / "piflow_pilot.py"),
        },
        "transforms": {
            "train": {
                "definition": "Resize(224,BICUBIC)->CenterCrop(224)->RGB->ToTensor->CLIP normalize",
                "stochastic": False,
                "random_sequence_fixed_by_extraction_seed": False,
            },
            "val": {
                "definition": "Resize(224,BICUBIC)->CenterCrop(224)->RGB->ToTensor->CLIP normalize",
                "stochastic": False,
            },
        },
        "test_read": False,
    }
    cache["meta"]["feature_cache_provenance"] = provenance
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    print(json.dumps(provenance, indent=2), flush=True)
    print(f"saved cache: {output_path}", flush=True)


if __name__ == "__main__":
    main()
