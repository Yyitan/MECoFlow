"""Evaluate a saved CAMS checkpoint on the closed-world test split.

This intentionally reuses train.evaluate(), matching the closed-world path
used at the end of training and avoiding test.py's open-world feasibility mask.
"""

import torch

import train
from dataset import CompositionDataset
from flags import parser
from model.configure_model import configure_model
from utils import load_args, set_seed


def main():
    args = parser.parse_args()
    if args.config:
        load_args(args.config, args)
    args.open_world = False
    set_seed(args.seed)

    test_dataset = CompositionDataset(
        args.dataset_path,
        phase="test",
        split="compositional-split-natural",
        open_world=False,
    )
    model, _ = configure_model(args, test_dataset)
    model.to(train.device)
    model.load_state_dict(torch.load(args.load_model, weights_only=True))

    with torch.no_grad():
        train.evaluate(model, test_dataset, args)


if __name__ == "__main__":
    main()
