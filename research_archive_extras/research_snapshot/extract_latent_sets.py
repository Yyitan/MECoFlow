"""Extract CAMS composition latent-query sets before mean pooling."""

from __future__ import annotations

import argparse
import json
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
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def extract_split(
    model: torch.nn.Module,
    dataset: CompositionDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
        captured.append(output.detach())

    visual = model.clip_model.visual
    handle = visual.tr_c.register_forward_hook(hook)
    chunks: list[torch.Tensor] = []
    agreements: list[torch.Tensor] = []
    try:
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            _, _, pooled, _ = model.image_encoder(images)
            assert len(captured) == 1
            queries = captured.pop()
            # Project each composition query through CAMS' learned composition
            # projection, preserving the 32-query set before mean pooling.
            projected = visual.ln_norm(queries.float()) @ visual.proj_com.float()
            projected = F.normalize(projected, dim=-1)
            chunks.append(projected.half().cpu())
            query_mean = F.normalize(projected.mean(1), dim=-1)
            agreements.append(F.cosine_similarity(query_mean, F.normalize(pooled.float(), dim=-1)).cpu())
            if step == 0 or (step + 1) % 100 == 0:
                print(f"phase={dataset.phase} batch={step + 1}/{len(loader)}", flush=True)
    finally:
        handle.remove()
    agreement = torch.cat(agreements)
    return torch.cat(chunks), {
        "mean_cosine_to_cams_pooled": float(agreement.mean()),
        "min_cosine_to_cams_pooled": float(agreement.min()),
        "max_cosine_to_cams_pooled": float(agreement.max()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    train_dataset = CompositionDataset(args.dataset_path, phase="train")
    model, _ = configure_model(config, train_dataset)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    diagnostics: dict[str, object] = {}
    for phase in ("train", "val", "test"):
        dataset = train_dataset if phase == "train" else CompositionDataset(args.dataset_path, phase=phase)
        queries, stats = extract_split(
            model, dataset, args.batch_size, args.num_workers, device, args.max_batches
        )
        torch.save(queries, output_dir / f"{phase}_queries.pt")
        diagnostics[phase] = {
            "shape": list(queries.shape),
            "dtype": str(queries.dtype),
            **stats,
        }
        print(f"saved {phase}: shape={tuple(queries.shape)} stats={stats}", flush=True)
        del queries

    (output_dir / "metadata.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
