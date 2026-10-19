"""Extract CAMS attribute/object branch features for cross-primitive generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from model.configure_model import configure_model  # noqa: E402
from research.piflow_pilot import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")
    config = load_config(args.config)
    dataset = CompositionDataset(args.dataset_path, phase="train")
    model, _ = configure_model(config, dataset)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    output = {"attr_feature": [], "obj_feature": [], "attr": [], "obj": [], "pair": []}
    with torch.no_grad():
        for step, batch in enumerate(loader):
            attr, obj, _com, _glb = model.image_encoder(batch[0].to(device, non_blocking=True))
            output["attr_feature"].append(F.normalize(attr.float(), dim=-1).half().cpu())
            output["obj_feature"].append(F.normalize(obj.float(), dim=-1).half().cpu())
            output["attr"].append(torch.as_tensor(batch[1]).long())
            output["obj"].append(torch.as_tensor(batch[2]).long())
            output["pair"].append(torch.as_tensor(batch[3]).long())
            if step == 0 or (step + 1) % 100 == 0:
                print(f"batch={step + 1}/{len(loader)}", flush=True)
    packed = {key: torch.cat(value) for key, value in output.items()}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, path)
    print({key: (tuple(value.shape), str(value.dtype)) for key, value in packed.items()}, flush=True)


if __name__ == "__main__":
    main()
