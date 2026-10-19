import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.eval_saved_energy import adjustment, fuse, rank_diagnostics, seen_pair_mask  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


@torch.no_grad()
def sparse_scores(
    queries: torch.Tensor,
    split: dict[str, torch.Tensor],
    prompts: torch.Tensor,
    topk: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    indices = split["logits"].topk(topk, dim=1).indices
    outputs = {"max": [], "top2": [], "top4": [], "top8": [], "lse5": [], "lse10": []}
    prompts = prompts.float().to(device)
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        query = queries[start:stop].float().to(device)
        candidate = indices[start:stop].to(device)
        candidate_prompt = prompts[candidate]
        score = torch.einsum("bnd,bkd->bnk", query, candidate_prompt)
        sorted_score = score.sort(dim=1, descending=True).values
        outputs["max"].append(sorted_score[:, 0].cpu())
        outputs["top2"].append(sorted_score[:, :2].mean(1).cpu())
        outputs["top4"].append(sorted_score[:, :4].mean(1).cpu())
        outputs["top8"].append(sorted_score[:, :8].mean(1).cpu())
        outputs["lse5"].append(torch.logsumexp(score * 5.0, dim=1).sub(torch.log(torch.tensor(score.shape[1], device=device))).div(5.0).cpu())
        outputs["lse10"].append(torch.logsumexp(score * 10.0, dim=1).sub(torch.log(torch.tensor(score.shape[1], device=device))).div(10.0).cpu())
    return indices, {name: torch.cat(parts) for name, parts in outputs.items()}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    prompts = cache["prompts"]["pair"]
    val_queries = torch.load(Path(args.query_dir) / "val_queries.pt", map_location="cpu", weights_only=True)
    test_queries = torch.load(Path(args.query_dir) / "test_queries.pt", map_location="cpu", weights_only=True)
    val_indices, val_scores = sparse_scores(val_queries, cache["val"], prompts, args.topk, args.batch_size, device)
    test_indices, test_scores = sparse_scores(test_queries, cache["test"], prompts, args.topk, args.batch_size, device)
    del val_queries, test_queries

    seen_mask = seen_pair_mask(cache, device)
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
    strengths = (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
    best = None
    for score_name in val_scores:
        result[score_name] = {
            "rank_val": rank_diagnostics(cache["val"], val_indices.to(device), val_scores[score_name].to(device), seen_mask),
            "rank_test": rank_diagnostics(cache["test"], test_indices.to(device), test_scores[score_name].to(device), seen_mask),
        }
        for kind in ("all", "group", "unseen_only"):
            delta = adjustment(kind, val_indices.to(device), val_scores[score_name].to(device), seen_mask)
            trials = {}
            for strength in strengths:
                logits = fuse(val["logits"].float(), val_indices.to(device), delta, strength)
                stats = val_eval(logits, val["attr"], val["obj"], val["pair"])
                trials[f"{strength:.2f}"] = stats
                candidate = (stats["AUC"], stats["best_hm"], score_name, kind, strength)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            result[score_name][kind] = trials
    assert best is not None
    _, _, score_name, kind, strength = best
    test_delta = adjustment(kind, test_indices.to(device), test_scores[score_name].to(device), seen_mask)
    test_logits = fuse(test["logits"].float(), test_indices.to(device), test_delta, strength)
    result["selection"] = {
        "score": score_name,
        "kind": kind,
        "strength": strength,
        "val_auc": best[0],
        "val_hm": best[1],
    }
    result["fused_test"] = test_eval(test_logits, test["attr"], test["obj"], test["pair"])
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection": result["selection"],
        "fused_test": result["fused_test"],
        "selected_rank_val": result[score_name]["rank_val"],
        "selected_rank_test": result[score_name]["rank_test"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
