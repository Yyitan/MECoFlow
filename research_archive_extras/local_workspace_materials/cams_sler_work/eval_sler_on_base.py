"""Evaluate SLER routing on a legacy CAMS checkpoint without retraining it."""

import os

import torch

import test
import train
from dataset import CompositionDataset
from flags import parser
from model.configure_model import configure_model
from utils import load_args, set_seed


def load_legacy_checkpoint(model, path):
    incompatible = model.load_state_dict(
        torch.load(path, weights_only=True), strict=False
    )
    expected_missing = {
        "sler_composer.0.weight",
        "sler_composer.0.bias",
        "sler_composer.2.weight",
        "sler_composer.2.bias",
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(f"Unexpected missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    print(f"legacy_load_missing={sorted(incompatible.missing_keys)}", flush=True)


def evaluate_open(model, dataset, args):
    evaluator = test.Evaluator(dataset)
    logits, attr_gt, obj_gt, pair_gt = test.predict_logits(model, dataset, args)
    feasibility = torch.load(
        args.feasibility, map_location="cpu", weights_only=True
    )["feasibility"]
    logits = test.threshold_with_feasibility(
        logits,
        dataset.seen_mask,
        threshold=args.threshold,
        feasiblity=feasibility,
    )
    stats = test.test(dataset, evaluator, logits, attr_gt, obj_gt, pair_gt)
    keys = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    print("| ".join(f"{key} {stats[key]:.4f}" for key in keys) + "|", flush=True)


def main():
    args = parser.parse_args()
    if args.config:
        load_args(args.config, args)
    if not args.use_sler:
        raise ValueError("This evaluator requires use_sler=true")
    set_seed(args.seed)
    mixes = [float(value) for value in os.environ.get(
        "SLER_EVAL_MIXES", "0,0.25,0.5"
    ).split(",")]

    for open_world in (False, True):
        args.open_world = open_world
        dataset = CompositionDataset(
            args.dataset_path,
            phase="test",
            split="compositional-split-natural",
            open_world=open_world,
        )
        model, _ = configure_model(args, dataset)
        model.to(train.device)
        load_legacy_checkpoint(model, args.load_model)

        world = "open" if open_world else "closed"
        for mix in mixes:
            args.sler_mix = mix
            print(f"=== world={world} sler_mix={mix:.4f} ===", flush=True)
            with torch.no_grad():
                if open_world:
                    evaluate_open(model, dataset, args)
                else:
                    train.evaluate(model, dataset, args)

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
