"""Validation-only prototype for Structured Posterior Reconciliation (SPR).

SPR treats the attribute-object candidate scores as a joint posterior and
iteratively reconciles its row/column marginals with the primitive branches.
The base CAMS/SLER score remains an explicit anchor. Hyperparameters are ranked
on validation AUC; this script never evaluates the test split.
"""

import itertools
import math

import torch
from torch.utils.data import DataLoader

import test
import train
from dataset import CompositionDataset
from flags import parser
from model.configure_model import configure_model
from utils import load_args, set_seed


WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
PAIR_TEMPERATURES = (0.5, 1.0, 2.0)
NUM_STEPS = 6


@torch.no_grad()
def collect_branch_logits(model, dataset, args):
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    pairs = torch.tensor(
        [(dataset.attr2idx[a], dataset.obj2idx[o]) for a, o in dataset.pairs]
    )

    composition_text = []
    attribute_text = object_text = None
    for pair_chunk in torch.split(pairs, 64):
        token_com, token_att, token_obj = model.csp.construct_token_tensors(pair_chunk)
        com_text, att_text, obj_text = model.get_text_features(
            token_com, token_att, token_obj
        )
        composition_text.append(com_text)
        if attribute_text is None:
            attribute_text, object_text = att_text, obj_text
    composition_text = torch.cat(composition_text, dim=0)

    outputs = [[], [], [], []]
    attr_gt, obj_gt, pair_gt = [], [], []
    for data in loader:
        data = [x.to(train.device) if not isinstance(x, tuple) else x for x in data]
        branch_logits = model.compute_branch_logits(
            data[0], composition_text, attribute_text, object_text,
            return_routes=False,
        )[:4]
        for bucket, value in zip(outputs, branch_logits):
            bucket.append(value.float().cpu())
        attr_gt.append(data[1].cpu())
        obj_gt.append(data[2].cpu())
        pair_gt.append(data[3].cpu())

    return (
        *(torch.cat(values, dim=0) for values in outputs),
        torch.cat(attr_gt),
        torch.cat(obj_gt),
        torch.cat(pair_gt),
    )


def closed_candidate_mask(dataset):
    allowed = set(dataset.train_pairs + dataset.val_pairs)
    return torch.tensor([pair in allowed for pair in dataset.pairs])


def marginalize(joint, indices, count):
    result = joint.new_zeros(joint.shape[0], count)
    return result.scatter_add(1, indices.unsqueeze(0).expand_as(joint), joint)


def reconcile(
    base_logits,
    attr_prob,
    obj_prob,
    pair_attr,
    pair_obj,
    candidate_mask,
    attr_weight,
    obj_weight,
    temperature,
):
    masked_base = base_logits.masked_fill(~candidate_mask.unsqueeze(0), -1e4)
    log_anchor = torch.log_softmax(masked_base / temperature, dim=1)
    joint = log_anchor.exp()
    eps = 1e-12

    for _ in range(NUM_STEPS):
        attr_margin = marginalize(joint, pair_attr, attr_prob.shape[1])
        obj_margin = marginalize(joint, pair_obj, obj_prob.shape[1])
        correction = (
            attr_weight
            * (attr_prob[:, pair_attr].clamp_min(eps).log()
               - attr_margin[:, pair_attr].clamp_min(eps).log())
            + obj_weight
            * (obj_prob[:, pair_obj].clamp_min(eps).log()
               - obj_margin[:, pair_obj].clamp_min(eps).log())
        )
        log_joint = log_anchor + correction
        log_joint = log_joint.masked_fill(~candidate_mask.unsqueeze(0), -1e4)
        joint = torch.softmax(log_joint, dim=1)

    # Express the projected posterior back in the base-logit scale. Additive
    # row constants are immaterial to ranking and generalized-ZSL calibration.
    return temperature * joint.clamp_min(eps).log()


def evaluate_scores(dataset, scores, targets):
    evaluator = test.Evaluator(dataset)
    stats = test.test(dataset, evaluator, scores, *targets)
    return {key: float(stats[key]) for key in (
        "best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"
    )}


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
    pair_attr = torch.tensor(dataset.pairs2attr_idx, dtype=torch.long)
    pair_obj = torch.tensor(dataset.pairs2obj_idx, dtype=torch.long)
    candidate_mask = closed_candidate_mask(dataset)

    pair_base = args.beta * glb + (1.0 - args.beta) * com
    legacy_scores = pair_base + attr_prob[:, pair_attr] * obj_prob[:, pair_obj]
    targets = (attr_gt, obj_gt, pair_gt)
    baseline = evaluate_scores(dataset, legacy_scores, targets)
    print(f"LEGACY {baseline}", flush=True)

    ranked = []
    for temperature, attr_weight, obj_weight in itertools.product(
        PAIR_TEMPERATURES, WEIGHTS, WEIGHTS
    ):
        if attr_weight == 0.0 and obj_weight == 0.0:
            continue
        scores = reconcile(
            legacy_scores,
            attr_prob,
            obj_prob,
            pair_attr,
            pair_obj,
            candidate_mask,
            attr_weight,
            obj_weight,
            temperature,
        )
        stats = evaluate_scores(dataset, scores, targets)
        ranked.append((stats["AUC"], stats["best_hm"], temperature,
                       attr_weight, obj_weight, stats))

    ranked.sort(reverse=True)
    print("TOP_SPR", flush=True)
    for row in ranked[:12]:
        print(row, flush=True)


if __name__ == "__main__":
    main()
