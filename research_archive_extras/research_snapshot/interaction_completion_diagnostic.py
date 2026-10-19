"""Diagnose semantic-to-visual interaction completion for unseen compositions.

The visual prototype is decomposed into an additive primitive prior and a learned
interaction residual.  This script tests two non-generative mean estimators (ridge
and semantic neighbours) before they are used as the starting distribution of the
residual flow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    additive_completion,
    evaluate_variant,
    real_centroids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args()


def pair_maps(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    attr_index = {value: index for index, value in enumerate(meta["attrs"])}
    obj_index = {value: index for index, value in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attr_index[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([obj_index[o] for _, o in meta["pairs"]])
    return pair_attr, pair_obj


def semantic_features(cache: dict[str, object]) -> dict[str, torch.Tensor]:
    pair_attr, pair_obj = pair_maps(cache)
    prompt = cache["prompts"]
    pair = F.normalize(prompt["pair"].float(), dim=-1)
    primitive = F.normalize(
        prompt["attr"].float()[pair_attr] + prompt["obj"].float()[pair_obj], dim=-1
    )
    interaction = F.normalize(pair - primitive, dim=-1)
    return {"pair": pair, "interaction": interaction}


def ridge_predict(
    x: torch.Tensor,
    target: torch.Tensor,
    seen: torch.Tensor,
    regularization: float,
    device: torch.device,
) -> torch.Tensor:
    """Primal multi-output ridge regression with an explicit intercept."""
    x_seen = x[seen].to(device)
    y_seen = target[seen].to(device)
    x_mean, y_mean = x_seen.mean(0), y_seen.mean(0)
    x_centered, y_centered = x_seen - x_mean, y_seen - y_mean
    gram = x_centered.T @ x_centered
    gram.diagonal().add_(regularization)
    weight = torch.linalg.solve(gram, x_centered.T @ y_centered)
    return ((x.to(device) - x_mean) @ weight + y_mean).cpu()


@torch.no_grad()
def neighbour_predict(
    x: torch.Tensor,
    target: torch.Tensor,
    seen: torch.Tensor,
    neighbours: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    x_device = F.normalize(x.to(device), dim=-1)
    seen_ids = torch.where(seen)[0].to(device)
    output = torch.zeros_like(target)
    for ids in torch.arange(len(x), device=device).split(512):
        similarity = x_device[ids] @ x_device[seen_ids].T
        values, indices = similarity.topk(min(neighbours, len(seen_ids)), dim=-1)
        weights = torch.softmax(values / temperature, dim=-1)
        candidates = target[seen_ids[indices].cpu()].to(device)
        output[ids.cpu()] = (weights[..., None] * candidates).sum(1).cpu()
    return output


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    additive = additive_completion(cache, centroids, counts, 30)
    seen = counts > 0
    visual_interaction = torch.zeros_like(centroids)
    visual_interaction[seen] = centroids[seen] - additive[seen]
    semantics = semantic_features(cache)

    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    results: dict[str, object] = {}
    best = None

    estimators: list[tuple[str, torch.Tensor]] = []
    for semantic_name, semantic in semantics.items():
        for regularization in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            prediction = ridge_predict(
                semantic, visual_interaction, seen, regularization, device
            )
            estimators.append((f"ridge_{semantic_name}_l{regularization:g}", prediction))
        for neighbours in (4, 8, 16, 32):
            prediction = neighbour_predict(
                semantic, visual_interaction, seen, neighbours, 0.05, device
            )
            estimators.append((f"knn_{semantic_name}_k{neighbours}", prediction))

    for estimator_name, interaction in estimators:
        for scale in (0.25, 0.5, 1.0, 1.5, 2.0):
            prototypes = F.normalize(additive + scale * interaction, dim=-1)
            prototypes[seen] = centroids[seen]
            name = f"{estimator_name}_s{scale:g}"
            stats = evaluate_variant(name, prototypes, cache, val_eval, test_eval, device)
            results[name] = stats
            candidate = (
                stats["fused_val"]["AUC"],
                stats["fused_val"]["best_hm"],
                name,
                prototypes,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate

    assert best is not None
    torch.save(best[3].half(), work_dir / "best_prototypes.pt")
    summary = {
        "selection": best[2],
        "fused_val": results[best[2]]["fused_val"],
        "fused_test": results[best[2]]["fused_test"],
        "all": results,
    }
    (work_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("selection", "fused_val", "fused_test")}, indent=2))


if __name__ == "__main__":
    main()
