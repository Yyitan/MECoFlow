"""Final closed/open test evaluation for validation-selected SPR settings.

Do not use this script until temperature and marginal weights have been frozen
from validation. It reports both the unchanged legacy score and fixed SPR.
"""

import torch

import test
import train
from dataset import CompositionDataset
from eval_spr import collect_branch_logits, evaluate_scores, reconcile
from flags import parser
from model.configure_model import configure_model
from utils import load_args, set_seed


TEMPERATURE = 2.0
ATTR_WEIGHT = 0.25
OBJ_WEIGHT = 0.5


def candidate_mask(dataset, args, open_world):
    if not open_world:
        allowed = set(dataset.train_pairs + dataset.test_pairs)
        return torch.tensor([pair in allowed for pair in dataset.pairs])

    feasibility = torch.load(
        args.feasibility, map_location="cpu", weights_only=True
    )["feasibility"]
    # Keep every seen composition even when the external feasibility prior is
    # below threshold, matching the repository's legacy open-world protocol.
    return (feasibility >= args.threshold) | dataset.seen_mask.bool()


def evaluate_world(args, open_world):
    args.open_world = open_world
    dataset = CompositionDataset(
        args.dataset_path,
        phase="test",
        split="compositional-split-natural",
        open_world=open_world,
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
    mask = candidate_mask(dataset, args, open_world)
    pair_base = args.beta * glb + (1.0 - args.beta) * com
    legacy_scores = pair_base + attr_prob[:, pair_attr] * obj_prob[:, pair_obj]
    targets = (attr_gt, obj_gt, pair_gt)

    if open_world:
        legacy_eval = test.threshold_with_feasibility(
            legacy_scores,
            dataset.seen_mask,
            threshold=args.threshold,
            feasiblity=torch.load(
                args.feasibility, map_location="cpu", weights_only=True
            )["feasibility"],
        )
    else:
        legacy_eval = legacy_scores

    fixed_scores = reconcile(
        legacy_scores,
        attr_prob,
        obj_prob,
        pair_attr,
        pair_obj,
        mask,
        ATTR_WEIGHT,
        OBJ_WEIGHT,
        TEMPERATURE,
    )
    fixed_scores = fixed_scores.masked_fill(~mask.unsqueeze(0), -1e4)

    legacy = evaluate_scores(dataset, legacy_eval, targets)
    fixed = evaluate_scores(dataset, fixed_scores, targets)
    name = "OPEN" if open_world else "CLOSED"
    print(f"{name}_LEGACY {legacy}", flush=True)
    print(
        f"{name}_SPR temperature={TEMPERATURE} attr={ATTR_WEIGHT} "
        f"obj={OBJ_WEIGHT} {fixed}",
        flush=True,
    )


def main():
    args = parser.parse_args()
    if args.config:
        load_args(args.config, args)
    set_seed(args.seed)
    evaluate_world(args, open_world=False)
    evaluate_world(args, open_world=True)


if __name__ == "__main__":
    main()
