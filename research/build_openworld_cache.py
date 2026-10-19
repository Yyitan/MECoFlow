"""Build an exact CAMS feature cache over feasibility-filtered open-world pairs."""

from __future__ import annotations

import argparse
import itertools
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feasibility", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


@torch.no_grad()
def candidate_prompts(
    model: torch.nn.Module,
    pair_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pair_output = []
    attr_output = obj_output = None
    for chunk in pair_ids.split(64):
        token_pair, token_attr, token_obj = model.csp.construct_token_tensors(chunk.to(device))
        pair, attr_output, obj_output = model.get_text_features(token_pair, token_attr, token_obj)
        pair_output.append(pair.half().cpu())
    assert attr_output is not None and obj_output is not None
    return {
        "pair": torch.cat(pair_output),
        "attr": attr_output.half().cpu(),
        "obj": obj_output.half().cpu(),
    }


@torch.no_grad()
def extract_split(
    model: torch.nn.Module,
    dataset: CompositionDataset,
    prompts: dict[str, torch.Tensor],
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
    active: torch.Tensor,
    pair_lookup: torch.Tensor,
    beta: float,
    batch_size: int,
    device: torch.device,
    compute_logits: bool,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    pair_prompt = prompts["pair"].float().to(device)
    attr_prompt = prompts["attr"].float().to(device)
    obj_prompt = prompts["obj"].float().to(device)
    pair_attr, pair_obj = pair_attr.to(device), pair_obj.to(device)
    active = active.float().to(device)
    output: dict[str, list[torch.Tensor]] = {
        "com": [], "glb": [], "logits": [], "attr": [], "obj": [], "pair": []
    }
    for step, batch in enumerate(loader):
        images = batch[0].to(device, non_blocking=True)
        attr_truth = torch.as_tensor(batch[1]).long()
        obj_truth = torch.as_tensor(batch[2]).long()
        attr, obj, com, glb = model.image_encoder(images)
        attr = F.normalize(attr.float(), dim=-1)
        obj = F.normalize(obj.float(), dim=-1)
        com = F.normalize(com.float(), dim=-1)
        glb = F.normalize(glb.float(), dim=-1)
        if compute_logits:
            scale = model.temp_logit.float()
            attr_prob = torch.softmax(scale * attr @ attr_prompt.T, dim=-1)
            obj_prob = torch.softmax(scale * obj @ obj_prompt.T, dim=-1)
            composition = scale * com @ pair_prompt.T
            global_score = scale * glb @ pair_prompt.T
            score = beta * global_score + (1.0 - beta) * composition
            score = score + attr_prob[:, pair_attr] * obj_prob[:, pair_obj]
            output["logits"].append((score * active[None]).half().cpu())
        output["com"].append(com.half().cpu())
        output["glb"].append(glb.half().cpu())
        output["attr"].append(attr_truth)
        output["obj"].append(obj_truth)
        output["pair"].append(pair_lookup[attr_truth, obj_truth])
        if step == 0 or (step + 1) % 100 == 0:
            print(f"phase={dataset.phase} batch={step + 1}/{len(loader)}", flush=True)
    return {key: torch.cat(value) for key, value in output.items() if value}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    config = load_config(args.config)
    train_dataset = CompositionDataset(args.dataset_path, phase="train")
    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    test_dataset = CompositionDataset(args.dataset_path, phase="test")
    attrs, objs = train_dataset.attrs, train_dataset.objs
    full_pairs = list(itertools.product(attrs, objs))
    feasibility = torch.load(args.feasibility, map_location="cpu", weights_only=True)["feasibility"]
    if len(feasibility) != len(full_pairs):
        raise ValueError("feasibility ordering does not match the Cartesian pair set")
    known = set(train_dataset.train_pairs + train_dataset.val_pairs + train_dataset.test_pairs)
    keep = [bool(feasibility[index] >= args.threshold) or pair in known for index, pair in enumerate(full_pairs)]
    candidate_ids = torch.tensor([index for index, value in enumerate(keep) if value])
    candidate_pairs = [full_pairs[index] for index in candidate_ids.tolist()]
    attr_index = {value: index for index, value in enumerate(attrs)}
    obj_index = {value: index for index, value in enumerate(objs)}
    pair_ids = torch.tensor([(attr_index[a], obj_index[o]) for a, o in candidate_pairs])
    pair_attr, pair_obj = pair_ids[:, 0], pair_ids[:, 1]
    seen = set(train_dataset.train_pairs)
    active = torch.tensor(
        [pair in seen or bool(feasibility[index] >= args.threshold) for pair, index in zip(candidate_pairs, candidate_ids)]
    )
    pair_lookup = torch.full((len(attrs), len(objs)), -1, dtype=torch.long)
    pair_lookup[pair_attr, pair_obj] = torch.arange(len(candidate_pairs))

    model, _ = configure_model(config, train_dataset)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    prompts = candidate_prompts(model, pair_ids, device)
    cache = {
        "meta": {
            "attrs": attrs,
            "objs": objs,
            "pairs": candidate_pairs,
            "train_pairs": train_dataset.train_pairs,
            "val_pairs": val_dataset.val_pairs,
            "test_pairs": test_dataset.test_pairs,
            "active": active.tolist(),
            "open_world": True,
            "full_pair_count": len(full_pairs),
            "threshold": args.threshold,
        },
        "prompts": prompts,
        "train": extract_split(
            model, train_dataset, prompts, pair_attr, pair_obj, active, pair_lookup,
            config.beta, args.batch_size, device, False,
        ),
        "val": extract_split(
            model, val_dataset, prompts, pair_attr, pair_obj, active, pair_lookup,
            config.beta, args.batch_size, device, True,
        ),
        "test": extract_split(
            model, test_dataset, prompts, pair_attr, pair_obj, active, pair_lookup,
            config.beta, args.batch_size, device, True,
        ),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(
        f"saved {path}: candidates={len(candidate_pairs)} active={int(active.sum())} "
        f"full={len(full_pairs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
