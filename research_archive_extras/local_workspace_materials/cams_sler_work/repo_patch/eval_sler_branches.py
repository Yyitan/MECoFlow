"""Validation-only ablation of SLER routed residual branches."""

import torch

import train
from dataset import CompositionDataset
from flags import parser
from model.configure_model import configure_model
from utils import load_args, set_seed


MASKS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
]


def main():
    args = parser.parse_args()
    if args.config:
        load_args(args.config, args)
    args.open_world = False
    set_seed(args.seed)

    dataset = CompositionDataset(
        args.dataset_path,
        phase="val",
        split="compositional-split-natural",
        open_world=False,
    )
    model, _ = configure_model(args, dataset)
    model.to(train.device)
    model.load_state_dict(torch.load(args.load_model, weights_only=True))

    for mask in MASKS:
        args.sler_branch_scales = list(mask)
        print(f"=== branch_scales={mask} ===", flush=True)
        with torch.no_grad():
            train.evaluate(model, dataset, args)


if __name__ == "__main__":
    main()
