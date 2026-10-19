import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import CompositionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    return parser.parse_args()


def zscore(logits: torch.Tensor) -> torch.Tensor:
    return (logits - logits.mean(1, keepdim=True)) / logits.std(1, keepdim=True).clamp_min(1e-6)


class FastClosedWorldEvaluator:
    """Exact top-1 equivalent of CAMS' closed-world bias-sweep evaluator."""

    def __init__(self, dataset: CompositionDataset, device: torch.device) -> None:
        pair_to_id = dataset.pair2idx
        closed_pairs = set(dataset.train_pairs)
        closed_pairs.update(dataset.val_pairs if dataset.phase == "val" else dataset.test_pairs)
        seen_pairs = set(dataset.train_pairs)
        self.closed = torch.tensor([pair in closed_pairs for pair in dataset.pairs], device=device)
        self.seen = torch.tensor([pair in seen_pairs for pair in dataset.pairs], device=device)
        self.unseen = self.closed & ~self.seen
        self.pair_attr = torch.tensor(
            [dataset.attr2idx[attr] for attr, _ in dataset.pairs], device=device
        )
        self.pair_obj = torch.tensor(
            [dataset.obj2idx[obj] for _, obj in dataset.pairs], device=device
        )
        # Retain this assertion because cached pair ids must use the same global ordering.
        assert all(pair_to_id[pair] == index for index, pair in enumerate(dataset.pairs))

    @torch.no_grad()
    def __call__(
        self,
        logits: torch.Tensor,
        attr_truth: torch.Tensor,
        obj_truth: torch.Tensor,
        pair_truth: torch.Tensor,
    ) -> dict[str, float]:
        neg = torch.finfo(logits.dtype).min
        seen_scores = logits.masked_fill(~self.seen[None], neg)
        unseen_scores = logits.masked_fill(~self.unseen[None], neg)
        max_seen, pred_seen = seen_scores.max(1)
        max_unseen, pred_unseen = unseen_scores.max(1)
        choose_unseen_zero = max_unseen > max_seen
        pred_zero = torch.where(choose_unseen_zero, pred_unseen, pred_seen)

        sample_seen = self.seen[pair_truth]
        sample_unseen = ~sample_seen
        zero_match = pred_zero == pair_truth
        correct_scores = logits[torch.arange(len(logits), device=logits.device), pair_truth]
        # CAMS constructs its initial ``closed`` prediction with bias=+1000.
        # For top-1 this is exactly the best unseen candidate, and it defines
        # which unseen samples contribute calibration thresholds.
        unseen_potential_match = pred_unseen == pair_truth
        valid_unseen = sample_unseen & unseen_potential_match
        thresholds = (max_seen[valid_unseen] - correct_scores[valid_unseen] - 1e-4).sort().values
        skip = max(len(thresholds) // 20, 1)
        biases = thresholds[::skip]

        seen_curve: list[float] = []
        unseen_curve: list[float] = []
        for bias in biases:
            choose_unseen = max_unseen + bias > max_seen
            pred = torch.where(choose_unseen, pred_unseen, pred_seen)
            match = pred == pair_truth
            seen_curve.append(match[sample_seen].float().mean().item())
            unseen_curve.append(match[sample_unseen].float().mean().item())
        # Replicate CAMS' final endpoint at bias=+1000.
        seen_curve.append(0.0)
        unseen_curve.append(unseen_potential_match[sample_unseen].float().mean().item())

        seen_array = np.asarray(seen_curve)
        unseen_array = np.asarray(unseen_curve)
        denom = seen_array + unseen_array
        hm = np.divide(2 * seen_array * unseen_array, denom, out=np.zeros_like(denom), where=denom > 0)
        return {
            "best_seen": float(seen_array.max()),
            "best_unseen": float(unseen_array.max()),
            "best_hm": float(hm.max()),
            "AUC": float(np.trapz(seen_array, unseen_array)),
            "attr_acc": float((self.pair_attr[pred_zero] == attr_truth).float().mean().item()),
            "obj_acc": float((self.pair_obj[pred_zero] == obj_truth).float().mean().item()),
        }


class FastOpenWorldEvaluator(FastClosedWorldEvaluator):
    """Bias-sweep evaluator over a feasibility-filtered open-world candidate set."""

    def __init__(self, cache: dict[str, object], device: torch.device) -> None:
        meta = cache["meta"]
        attrs = {value: index for index, value in enumerate(meta["attrs"])}
        objs = {value: index for index, value in enumerate(meta["objs"])}
        seen_pairs = set(map(tuple, meta["train_pairs"]))
        self.closed = torch.ones(len(meta["pairs"]), dtype=torch.bool, device=device)
        self.seen = torch.tensor(
            [tuple(pair) in seen_pairs for pair in meta["pairs"]], device=device
        )
        self.unseen = ~self.seen
        self.pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]], device=device)
        self.pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]], device=device)


def real_centroids(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    train = cache["train"]
    num_pairs = len(cache["meta"]["pairs"])
    dim = train["com"].shape[1]
    centroids = torch.zeros(num_pairs, dim)
    counts = torch.bincount(train["pair"], minlength=num_pairs)
    centroids.index_add_(0, train["pair"], train["com"].float())
    centroids[counts > 0] = F.normalize(centroids[counts > 0] / counts[counts > 0, None], dim=-1)
    return centroids, counts


def additive_completion(
    cache: dict[str, object], centroids: torch.Tensor, counts: torch.Tensor, iterations: int
) -> torch.Tensor:
    meta = cache["meta"]
    num_attrs, num_objs = len(meta["attrs"]), len(meta["objs"])
    pair_attr = torch.tensor([meta["attrs"].index(a) for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([meta["objs"].index(o) for _, o in meta["pairs"]])
    seen = counts > 0
    weights = counts[seen, None].float()
    global_mean = (centroids[seen] * weights).sum(0) / weights.sum()
    attr_effect = torch.zeros(num_attrs, centroids.shape[1])
    obj_effect = torch.zeros(num_objs, centroids.shape[1])
    seen_ids = torch.where(seen)[0]
    seen_attr = pair_attr[seen_ids]
    seen_obj = pair_obj[seen_ids]
    seen_counts = counts[seen_ids].float()
    for _ in range(iterations):
        attr_numerator = torch.zeros_like(attr_effect)
        attr_denominator = torch.zeros(num_attrs)
        attr_residual = centroids[seen_ids] - global_mean - obj_effect[seen_obj]
        attr_numerator.index_add_(0, seen_attr, attr_residual * seen_counts[:, None])
        attr_denominator.index_add_(0, seen_attr, seen_counts)
        valid_attr = attr_denominator > 0
        attr_effect[valid_attr] = attr_numerator[valid_attr] / attr_denominator[valid_attr, None]
        attr_effect -= attr_effect.mean(0, keepdim=True)
        obj_numerator = torch.zeros_like(obj_effect)
        obj_denominator = torch.zeros(num_objs)
        obj_residual = centroids[seen_ids] - global_mean - attr_effect[seen_attr]
        obj_numerator.index_add_(0, seen_obj, obj_residual * seen_counts[:, None])
        obj_denominator.index_add_(0, seen_obj, seen_counts)
        valid_obj = obj_denominator > 0
        obj_effect[valid_obj] = obj_numerator[valid_obj] / obj_denominator[valid_obj, None]
        obj_effect -= obj_effect.mean(0, keepdim=True)
    return F.normalize(global_mean + attr_effect[pair_attr] + obj_effect[pair_obj], dim=-1)


@torch.no_grad()
def evaluate_variant(
    name: str,
    prototypes: torch.Tensor,
    cache: dict[str, object],
    val_eval: FastClosedWorldEvaluator,
    test_eval: FastClosedWorldEvaluator,
    device: torch.device,
) -> dict[str, object]:
    prototypes = prototypes.to(device)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    val_proto = val["com"].float() @ prototypes.T
    test_proto = test["com"].float() @ prototypes.T
    val_base_z, test_base_z = zscore(val["logits"].float()), zscore(test["logits"].float())
    val_proto_z, test_proto_z = zscore(val_proto), zscore(test_proto)

    only_val = val_eval(val_proto, val["attr"], val["obj"], val["pair"])
    only_test = test_eval(test_proto, test["attr"], test["obj"], test["pair"])
    best = None
    for weight in np.linspace(0.0, 1.0, 21):
        fused = (1.0 - weight) * val_base_z + weight * val_proto_z
        stats = val_eval(fused, val["attr"], val["obj"], val["pair"])
        candidate = (stats["AUC"], stats["best_hm"], float(weight), stats)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    weight = best[2]
    test_fused = (1.0 - weight) * test_base_z + weight * test_proto_z
    fused_test = test_eval(test_fused, test["attr"], test["obj"], test["pair"])
    print(f"{name}: weight={weight:.2f} val_auc={best[0]:.6f} test_auc={fused_test['AUC']:.6f}", flush=True)
    return {
        "only_val": only_val,
        "only_test": only_test,
        "selected_weight": weight,
        "fused_val": best[3],
        "fused_test": fused_test,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    generated = torch.load(args.generated, map_location="cpu", weights_only=False).float()
    centroids, counts = real_centroids(cache)
    additive = additive_completion(cache, centroids, counts, args.iterations)
    generated_hybrid = generated.clone()
    generated_hybrid[counts > 0] = centroids[counts > 0]
    additive_hybrid = additive.clone()
    additive_hybrid[counts > 0] = centroids[counts > 0]
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)

    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    test_dataset = CompositionDataset(args.dataset_path, phase="test")
    val_eval = FastClosedWorldEvaluator(val_dataset, device)
    test_eval = FastClosedWorldEvaluator(test_dataset, device)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    result: dict[str, object] = {
        "base_val": val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"]),
        "base_test": test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"]),
    }
    del val, test
    variants = {
        "generated": generated,
        "generated_hybrid": generated_hybrid,
        "text": text,
        "additive": additive,
        "additive_hybrid": additive_hybrid,
    }
    for name, prototypes in variants.items():
        result[name] = evaluate_variant(name, prototypes, cache, val_eval, test_eval, device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
