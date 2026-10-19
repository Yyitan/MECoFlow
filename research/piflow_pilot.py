"""Pilot: primitive-conditioned rectified-flow generation in CAMS feature space.

This is intentionally an offline, low-cost diagnostic. It freezes a reproduced CAMS
checkpoint, extracts its composition features, learns p(v_com | attribute, object),
and tests whether generated unseen-pair prototypes add signal beyond CAMS scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import test as cams_test  # noqa: E402
from dataset import CompositionDataset  # noqa: E402
from model.configure_model import configure_model  # noqa: E402
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work-dir", default="output/research/piflow_mit")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--samples-per-pair", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as stream:
        groups = yaml.safe_load(stream)
    values: dict[str, object] = {}
    for group in groups.values():
        values.update(group)
    values.setdefault("weight_decay", 1e-5)
    values.setdefault("eval_batch_size", 64)
    values.setdefault("open_world", False)
    values.setdefault("seed", 0)
    return SimpleNamespace(**values)


@torch.no_grad()
def prompt_features(model: nn.Module, dataset: CompositionDataset) -> dict[str, torch.Tensor]:
    model.eval()
    pairs = torch.tensor(
        [(dataset.attr2idx[a], dataset.obj2idx[o]) for a, o in dataset.pairs],
        dtype=torch.long,
    )
    chunks = torch.split(pairs, 64)
    pair_features = []
    attr_features = obj_features = None
    for chunk in chunks:
        token_com, token_att, token_obj = model.csp.construct_token_tensors(chunk)
        com, attr_features, obj_features = model.get_text_features(token_com, token_att, token_obj)
        pair_features.append(com.float().cpu())
    assert attr_features is not None and obj_features is not None
    return {
        "pair": torch.cat(pair_features).float(),
        "attr": attr_features.float().cpu(),
        "obj": obj_features.float().cpu(),
    }


@torch.no_grad()
def extract_split(
    model: nn.Module,
    dataset: CompositionDataset,
    prompts: dict[str, torch.Tensor],
    beta: float,
    device: torch.device,
    compute_logits: bool = True,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    pair_prompt = prompts["pair"].to(device)
    attr_prompt = prompts["attr"].to(device)
    obj_prompt = prompts["obj"].to(device)
    pair_attr = dataset.pairs2attr_idx.to(device)
    pair_obj = dataset.pairs2obj_idx.to(device)
    pair_lookup = torch.full((len(dataset.attrs), len(dataset.objs)), -1, dtype=torch.long)
    for index, (attr, obj) in enumerate(dataset.pairs):
        pair_lookup[dataset.attr2idx[attr], dataset.obj2idx[obj]] = index

    outputs: dict[str, list[torch.Tensor]] = {
        "com": [],
        "glb": [],
        "logits": [],
        "attr": [],
        "obj": [],
        "pair": [],
    }
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        attr_gt = torch.as_tensor(batch[1], dtype=torch.long)
        obj_gt = torch.as_tensor(batch[2], dtype=torch.long)
        attr, obj, com, glb = model.image_encoder(images)
        attr = F.normalize(attr.float(), dim=-1)
        obj = F.normalize(obj.float(), dim=-1)
        com = F.normalize(com.float(), dim=-1)
        glb = F.normalize(glb.float(), dim=-1)

        if compute_logits:
            scale = model.temp_logit.float()
            attr_score = torch.softmax(scale * attr @ attr_prompt.T, dim=-1)[:, pair_attr]
            obj_score = torch.softmax(scale * obj @ obj_prompt.T, dim=-1)[:, pair_obj]
            com_score = scale * com @ pair_prompt.T
            glb_score = scale * glb @ pair_prompt.T
            logits = beta * glb_score + (1.0 - beta) * com_score + attr_score * obj_score
            outputs["logits"].append(logits.cpu())

        outputs["com"].append(com.cpu())
        outputs["glb"].append(glb.cpu())
        outputs["attr"].append(attr_gt.cpu())
        outputs["obj"].append(obj_gt.cpu())
        outputs["pair"].append(pair_lookup[attr_gt, obj_gt])
    return {key: torch.cat(value) for key, value in outputs.items() if value}


def build_cache(
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    cache_path: Path,
    device: torch.device,
) -> dict[str, object]:
    train_dataset = CompositionDataset(args.dataset_path, phase="train")
    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    test_dataset = CompositionDataset(args.dataset_path, phase="test")
    model, _ = configure_model(cfg, train_dataset)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    prompts = prompt_features(model, train_dataset)
    cache = {
        "meta": {
            "attrs": train_dataset.attrs,
            "objs": train_dataset.objs,
            "pairs": train_dataset.pairs,
            "train_pairs": train_dataset.train_pairs,
            "val_pairs": train_dataset.val_pairs,
            "test_pairs": train_dataset.test_pairs,
        },
        "prompts": prompts,
        "train": extract_split(model, train_dataset, prompts, cfg.beta, device, compute_logits=False),
        "val": extract_split(model, val_dataset, prompts, cfg.beta, device),
        "test": extract_split(model, test_dataset, prompts, cfg.beta, device),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    angles = t[:, None] * frequencies[None] * 1000.0
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cond = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        h = self.fc2(F.silu(self.fc1(h)))
        return x + h


class PrimitiveConditionalFlow(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.attr_proj = nn.Linear(feature_dim, cond_dim)
        self.obj_proj = nn.Linear(feature_dim, cond_dim)
        self.pair_proj = nn.Linear(feature_dim, cond_dim)
        self.time_proj = nn.Sequential(nn.Linear(128, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, cond_dim) for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, feature_dim))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attr_cond: torch.Tensor,
        obj_cond: torch.Tensor,
        pair_cond: torch.Tensor,
        cond_drop: float = 0.0,
    ) -> torch.Tensor:
        cond = self.attr_proj(attr_cond) + self.obj_proj(obj_cond) + self.pair_proj(pair_cond)
        if self.training and cond_drop > 0:
            keep = (torch.rand(x.shape[0], 1, device=x.device) >= cond_drop).to(cond.dtype)
            cond = cond * keep
        cond = cond + self.time_proj(timestep_embedding(t, 128))
        h = self.input(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.output(h)


def train_flow(
    cache: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> PrimitiveConditionalFlow:
    train = cache["train"]
    prompts = cache["prompts"]
    pair_indices = train["pair"].long()
    meta = cache["meta"]
    pair_attr = torch.tensor([meta["attrs"].index(a) for a, _ in meta["pairs"]], dtype=torch.long)
    pair_obj = torch.tensor([meta["objs"].index(o) for _, o in meta["pairs"]], dtype=torch.long)
    counts = torch.bincount(pair_indices, minlength=len(meta["pairs"])).clamp_min(1)
    weights = counts[pair_indices].float().reciprocal()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    indices = torch.arange(len(pair_indices))
    loader = DataLoader(indices, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    feature_dim = train["com"].shape[1]
    model = PrimitiveConditionalFlow(feature_dim, args.hidden_dim, args.cond_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    attr_prompts = prompts["attr"].to(device)
    obj_prompts = prompts["obj"].to(device)
    pair_prompts = prompts["pair"].to(device)
    features = train["com"].to(device)
    pair_indices_device = pair_indices.to(device)
    pair_attr = pair_attr.to(device)
    pair_obj = pair_obj.to(device)

    model.train()
    for epoch in range(args.epochs):
        running = []
        for batch_indices in loader:
            batch_indices = batch_indices.to(device)
            labels = pair_indices_device[batch_indices]
            x1 = features[batch_indices]
            x0 = F.normalize(torch.randn_like(x1), dim=-1)
            t = torch.rand(x1.shape[0], device=device)
            xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
            target = x1 - x0
            prediction = model(
                xt,
                t,
                attr_prompts[pair_attr[labels]],
                obj_prompts[pair_obj[labels]],
                pair_prompts[labels],
                cond_drop=0.1,
            )
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(loss.detach())
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            mean_loss = torch.stack(running).mean().item()
            print(f"flow epoch={epoch + 1:03d} loss={mean_loss:.6f}", flush=True)
    return model.eval()


@torch.no_grad()
def generate_prototypes(
    model: PrimitiveConditionalFlow,
    cache: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    prompts = cache["prompts"]
    meta = cache["meta"]
    pair_attr = torch.tensor([meta["attrs"].index(a) for a, _ in meta["pairs"]], device=device)
    pair_obj = torch.tensor([meta["objs"].index(o) for _, o in meta["pairs"]], device=device)
    attr_prompts = prompts["attr"].to(device)
    obj_prompts = prompts["obj"].to(device)
    pair_prompts = prompts["pair"].to(device)
    feature_dim = pair_prompts.shape[1]
    all_means = []
    for start in range(0, len(meta["pairs"]), 128):
        stop = min(start + 128, len(meta["pairs"]))
        pair_ids = torch.arange(start, stop, device=device).repeat_interleave(args.samples_per_pair)
        x = F.normalize(torch.randn(len(pair_ids), feature_dim, device=device), dim=-1)
        dt = 1.0 / args.sample_steps
        for step in range(args.sample_steps):
            t = torch.full((len(pair_ids),), step * dt, device=device)
            velocity = model(
                x,
                t,
                attr_prompts[pair_attr[pair_ids]],
                obj_prompts[pair_obj[pair_ids]],
                pair_prompts[pair_ids],
            )
            x = x + dt * velocity
        x = F.normalize(x, dim=-1).view(stop - start, args.samples_per_pair, feature_dim)
        all_means.append(F.normalize(x.mean(dim=1), dim=-1).cpu())
    return torch.cat(all_means)


def zscore(logits: torch.Tensor) -> torch.Tensor:
    return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp_min(1e-6)


def stats_for(dataset: CompositionDataset, split: dict[str, torch.Tensor], logits: torch.Tensor) -> dict[str, float]:
    evaluator = cams_test.Evaluator(dataset)
    raw = cams_test.test(
        dataset,
        evaluator,
        logits.cpu(),
        split["attr"].cpu(),
        split["obj"].cpu(),
        split["pair"].cpu(),
    )
    return {key: float(value) for key, value in raw.items() if key in {"best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"}}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_path = work_dir / "features.pt"
    cfg = load_config(args.config)

    if cache_path.exists() and not args.rebuild_cache:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        cache = build_cache(cfg, args, cache_path, device)

    flow = train_flow(cache, args, device)
    torch.save(flow.state_dict(), work_dir / "flow.pt")
    generated = generate_prototypes(flow, cache, args, device)
    torch.save(generated, work_dir / "generated_prototypes.pt")

    meta = cache["meta"]
    train = cache["train"]
    real_centroids = torch.zeros_like(generated)
    real_counts = torch.zeros(len(meta["pairs"]), dtype=torch.long)
    for pair_id in torch.unique(train["pair"]):
        mask = train["pair"] == pair_id
        real_centroids[pair_id] = F.normalize(train["com"][mask].mean(dim=0), dim=-1)
        real_counts[pair_id] = mask.sum()
    seen_mask = real_counts > 0
    hybrid = generated.clone()
    hybrid[seen_mask] = real_centroids[seen_mask]

    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    test_dataset = CompositionDataset(args.dataset_path, phase="test")
    result: dict[str, object] = {}
    result["base_val"] = stats_for(val_dataset, cache["val"], cache["val"]["logits"])
    result["base_test"] = stats_for(test_dataset, cache["test"], cache["test"]["logits"])

    variants = {"generated": generated, "hybrid": hybrid}
    best = None
    for variant_name, prototypes in variants.items():
        val_gen = cache["val"]["com"] @ prototypes.T
        test_gen = cache["test"]["com"] @ prototypes.T
        result[f"{variant_name}_only_val"] = stats_for(val_dataset, cache["val"], val_gen)
        for weight in np.linspace(0.0, 1.0, 21):
            val_logits = (1.0 - weight) * zscore(cache["val"]["logits"]) + weight * zscore(val_gen)
            val_stats = stats_for(val_dataset, cache["val"], val_logits)
            candidate = (val_stats["AUC"], val_stats["best_hm"], variant_name, float(weight))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        result[f"{variant_name}_only_test"] = stats_for(test_dataset, cache["test"], test_gen)

    assert best is not None
    _, _, best_variant, best_weight = best
    best_prototypes = variants[best_variant]
    test_gen = cache["test"]["com"] @ best_prototypes.T
    fused_test = (1.0 - best_weight) * zscore(cache["test"]["logits"]) + best_weight * zscore(test_gen)
    result["selection"] = {"variant": best_variant, "weight": best_weight, "val_auc": best[0], "val_hm": best[1]}
    result["fused_test"] = stats_for(test_dataset, cache["test"], fused_test)

    with open(work_dir / "results.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
