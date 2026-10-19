"""Cross-seed validation of the single SPR setting selected on seed0."""

import torch

import train
from dataset import CompositionDataset
from eval_spr import (
    closed_candidate_mask,
    collect_branch_logits,
    evaluate_scores,
    reconcile,
)
from flags import parser
from model.configure_model import configure_model
from utils import load_args, set_seed


TEMPERATURE = 2.0
ATTR_WEIGHT = 0.25
OBJ_WEIGHT = 0.5


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
    model.eval()

    att, obj, com, glb, attr_gt, obj_gt, pair_gt = collect_branch_logits(
        model, dataset, args
    )
    attr_prob = torch.softmax(att, dim=1)
    obj_prob = torch.softmax(obj, dim=1)
    pair_attr = torch.as_tensor(dataset.pairs2attr_idx, dtype=torch.long)
    pair_obj = torch.as_tensor(dataset.pairs2obj_idx, dtype=torch.long)
    candidate_mask = closed_candidate_mask(dataset)
    pair_base = args.beta * glb + (1.0 - args.beta) * com
    legacy_scores = pair_base + attr_prob[:, pair_attr] * obj_prob[:, pair_obj]
    targets = (attr_gt, obj_gt, pair_gt)

    legacy = evaluate_scores(dataset, legacy_scores, targets)
    fixed_scores = reconcile(
        legacy_scores,
        attr_prob,
        obj_prob,
        pair_attr,
        pair_obj,
        candidate_mask,
        ATTR_WEIGHT,
        OBJ_WEIGHT,
        TEMPERATURE,
    )
    fixed = evaluate_scores(dataset, fixed_scores, targets)
    print(f"LEGACY {legacy}", flush=True)
    print(
        f"FIXED_SPR temperature={TEMPERATURE} attr={ATTR_WEIGHT} "
        f"obj={OBJ_WEIGHT} {fixed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
