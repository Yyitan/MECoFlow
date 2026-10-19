"""Refine CAMS' shared composition context with ReCoFlow visual prototypes."""

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
from model.configure_model import configure_model  # noqa: E402
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    additive_completion,
    real_centroids,
)
from research.piflow_pilot import PrimitiveConditionalFlow, load_config  # noqa: E402
from research.warmstart_flow_classifier import generate_unseen, residual_bank  # noqa: E402
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--pair-cond-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pair_maps(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    attrs = {value: index for index, value in enumerate(meta["attrs"])}
    objs = {value: index for index, value in enumerate(meta["objs"])}
    return (
        torch.tensor([attrs[a] for a, _ in meta["pairs"]], dtype=torch.long),
        torch.tensor([objs[o] for _, o in meta["pairs"]], dtype=torch.long),
    )


def composition_text(
    model: torch.nn.Module,
    pair_ids: torch.Tensor,
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
) -> torch.Tensor:
    """Construct only the composition tokens, preserving gradients to com context."""
    csp = model.csp
    token_ids = csp.token_ids[0]
    class_token_ids = token_ids.repeat(len(pair_ids), 1)
    tokens = model.clip_model.token_embedding(class_token_ids).type(model.clip_model.dtype)
    eos = int(token_ids.argmax())
    soft = csp.soft_att_obj
    tokens[:, eos - 2, :] = soft[pair_attr[pair_ids]].type(model.clip_model.dtype)
    tokens[:, eos - 1, :] = soft[pair_obj[pair_ids] + csp.num_att].type(model.clip_model.dtype)
    tokens[:, 1 : len(csp.com_ctx_vectors) + 1, :] = csp.com_ctx_vectors.type(
        model.clip_model.dtype
    )
    text = model.text_encoder(token_ids, tokens, enable_pos_emb=True)
    return F.normalize(text.float(), dim=-1)


@torch.no_grad()
def all_composition_text(
    model: torch.nn.Module,
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for ids in torch.arange(len(pair_attr), device=device).split(256):
        outputs.append(composition_text(model, ids, pair_attr, pair_obj).cpu())
    return torch.cat(outputs)


def updated_logits(
    split: dict[str, torch.Tensor],
    old_text: torch.Tensor,
    new_text: torch.Tensor,
    scale: float,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    com = split["com"].float().to(device)
    glb = split["glb"].float().to(device)
    old_text_device, new_text_device = old_text.to(device), new_text.to(device)
    old_visual = scale * ((1.0 - beta) * (com @ old_text_device.T) + beta * (glb @ old_text_device.T))
    new_visual = scale * ((1.0 - beta) * (com @ new_text_device.T) + beta * (glb @ new_text_device.T))
    return split["logits"].float().to(device) - old_visual + new_visual


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    additive = additive_completion(cache, centroids, counts, 30)
    residuals = residual_bank(cache, additive)
    flow = PrimitiveConditionalFlow(
        cache["train"]["com"].shape[1], args.hidden_dim, args.cond_dim
    ).to(device)
    flow.load_state_dict(torch.load(args.flow_path, map_location="cpu", weights_only=True))
    flow.eval()
    generation_args = argparse.Namespace(
        samples_per_class=args.samples_per_class,
        sample_steps=args.sample_steps,
        pair_cond_scale=args.pair_cond_scale,
        guidance_scale=args.guidance_scale,
    )
    generated, labels = generate_unseen(
        flow, cache, additive, residuals, counts, args.residual_scale,
        generation_args, device, None,
    )
    del flow
    torch.cuda.empty_cache()
    generated_sum = torch.zeros_like(centroids)
    generated_sum.index_add_(0, labels, generated)
    generated_count = torch.bincount(labels, minlength=len(centroids)).clamp_min(1)
    prototypes = centroids.clone()
    unseen = counts == 0
    prototypes[unseen] = F.normalize(
        generated_sum[unseen] / generated_count[unseen, None], dim=-1
    )

    cfg = load_config(args.config)
    dataset = CompositionDataset(args.dataset_path, phase="train")
    model, _ = configure_model(cfg, dataset)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.csp.com_ctx_vectors.requires_grad = True
    optimizer = torch.optim.AdamW(
        [model.csp.com_ctx_vectors], lr=args.lr, weight_decay=0.0
    )
    anchor = model.csp.com_ctx_vectors.detach().clone()
    pair_attr, pair_obj = pair_maps(cache)
    pair_attr, pair_obj = pair_attr.to(device), pair_obj.to(device)
    prototypes = prototypes.to(device)
    old_text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    scale = float(model.temp_logit.detach())
    history, best = [], None
    generator = torch.Generator().manual_seed(args.seed + 2000)
    for epoch in range(args.epochs):
        order = torch.randperm(len(prototypes), generator=generator)
        losses = []
        for ids_cpu in order.split(args.batch_size):
            ids = ids_cpu.to(device)
            text = composition_text(model, ids, pair_attr, pair_obj)
            logits = 20.0 * prototypes[ids] @ text.T
            target = torch.arange(len(ids), device=device)
            contrastive = 0.5 * (
                F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)
            )
            anchor_loss = (model.csp.com_ctx_vectors - anchor).float().square().mean()
            loss = contrastive + args.anchor_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        new_text = all_composition_text(model, pair_attr, pair_obj, device)
        val_logits = updated_logits(
            cache["val"], old_text, new_text, scale, cfg.beta, device
        )
        stats = val_eval(val_logits, val["attr"], val["obj"], val["pair"])
        record = {"epoch": epoch + 1, "loss": float(torch.stack(losses).mean()), **stats}
        history.append(record)
        candidate = (stats["AUC"], stats["best_hm"], epoch + 1, model.csp.com_ctx_vectors.detach().cpu().clone())
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        print(
            f"epoch={epoch + 1:02d} loss={record['loss']:.5f} "
            f"val_auc={stats['AUC']:.6f} val_hm={stats['best_hm']:.6f}",
            flush=True,
        )
    assert best is not None
    model.csp.com_ctx_vectors.data.copy_(best[3].to(device))
    best_text = all_composition_text(model, pair_attr, pair_obj, device)
    raw_test_logits = updated_logits(
        cache["test"], old_text, best_text, scale, cfg.beta, device
    )
    raw_test = test_eval(raw_test_logits, test["attr"], test["obj"], test["pair"])
    val_gain = best[0] - base_val["AUC"]
    enabled = val_gain >= args.min_val_auc_gain
    selected_test = raw_test if enabled else base_test
    output = {
        "base_val": base_val, "base_test": base_test,
        "selection": {
            "epoch": best[2], "val_auc": best[0], "val_hm": best[1],
            "val_auc_gain": val_gain, "min_val_auc_gain": args.min_val_auc_gain,
            "enabled": enabled,
        },
        "raw_test": raw_test, "selected_test": selected_test, "history": history,
    }
    torch.save({"com_ctx_vectors": best[3]}, work_dir / "prompt_refinement.pt")
    (work_dir / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"selection": output["selection"], "raw_test": raw_test, "selected_test": selected_test}, indent=2))


if __name__ == "__main__":
    main()
