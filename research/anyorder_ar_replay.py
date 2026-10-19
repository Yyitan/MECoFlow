"""Any-order autoregressive residual replay for compositional ZSL.

This is a lightweight feature-space pilot inspired by order-marginalized visual
autoregressive classifiers.  It models additive-centred CAMS residuals as a
random-order sequence of continuous PCA tokens, generates balanced unseen-pair
features offline, and then uses the same classifier/fusion protocol as MECoFlow.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    FastOpenWorldEvaluator,
    additive_completion,
    real_centroids,
    zscore,
)
from research.generative_classifier_pilot import (  # noqa: E402
    classifier_logits,
    train_classifier,
)
from research.warmstart_flow_classifier import (  # noqa: E402
    balanced_seen_samples,
    pair_maps,
)
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pca-max-samples", type=int, default=50000)
    parser.add_argument("--pair-cond-scale", type=float, default=1.0)
    parser.add_argument(
        "--token-order", choices=("random", "raster"), default="random"
    )
    parser.add_argument(
        "--generation-mode", choices=("sample", "mean"), default="sample"
    )
    parser.add_argument("--generation-temperature", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--classifier-batch-size", type=int, default=2048)
    parser.add_argument("--classifier-lr", type=float, default=3e-3)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    parser.add_argument(
        "--shared-selection",
        help="validation-only aggregate JSON fixing delta/strength for test evaluation",
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.latent_dim % args.token_dim:
        parser.error("--latent-dim must be divisible by --token-dim")
    if args.hidden_dim % args.heads:
        parser.error("--hidden-dim must be divisible by --heads")
    if not args.validation_only and not args.shared_selection:
        parser.error("test evaluation requires --shared-selection from validation-only seeds")
    return args


class AnyOrderResidualAR(nn.Module):
    """Causal continuous-token model with explicit target-position instructions."""

    def __init__(
        self,
        prompt_dim: int,
        token_dim: int,
        num_tokens: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        pair_cond_scale: float,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.pair_cond_scale = pair_cond_scale
        self.attr_proj = nn.Linear(prompt_dim, hidden_dim)
        self.obj_proj = nn.Linear(prompt_dim, hidden_dim)
        self.pair_proj = nn.Linear(prompt_dim, hidden_dim)
        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.position = nn.Embedding(num_tokens, hidden_dim)
        self.value_proj = nn.Linear(token_dim, hidden_dim)
        self.instruction_type = nn.Parameter(torch.zeros(hidden_dim))
        self.value_type = nn.Parameter(torch.zeros(hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.output = nn.Linear(hidden_dim, token_dim * 2)
        nn.init.normal_(self.instruction_type, std=0.02)
        nn.init.normal_(self.value_type, std=0.02)

    def condition(
        self,
        pair_ids: torch.Tensor,
        pair_attr: torch.Tensor,
        pair_obj: torch.Tensor,
        attr_prompt: torch.Tensor,
        obj_prompt: torch.Tensor,
        pair_prompt: torch.Tensor,
    ) -> torch.Tensor:
        attr = attr_prompt[pair_attr[pair_ids]]
        obj = obj_prompt[pair_obj[pair_ids]]
        pair = pair_prompt[pair_ids]
        return self.cond_norm(
            self.attr_proj(attr)
            + self.obj_proj(obj)
            + self.pair_cond_scale * self.pair_proj(pair)
        )

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)

    def teacher_forced(
        self, code: torch.Tensor, permutation: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(code)
        ordered = code.gather(
            1, permutation[..., None].expand(batch, self.num_tokens, self.token_dim)
        )
        positions = self.position(permutation)
        sequence = condition.new_zeros(batch, 1 + 2 * self.num_tokens, condition.shape[-1])
        sequence[:, 0] = condition
        sequence[:, 1::2] = positions + self.instruction_type
        sequence[:, 2::2] = self.value_proj(ordered) + positions + self.value_type
        hidden = self.transformer(
            sequence, mask=self.causal_mask(sequence.shape[1], sequence.device)
        )
        prediction = self.output(hidden[:, 1::2])
        mean, log_std = prediction.chunk(2, dim=-1)
        return ordered, mean, log_std.clamp(-4.0, 1.0)

    def next_token(
        self,
        previous_values: torch.Tensor,
        previous_positions: torch.Tensor,
        target_position: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, previous_count = previous_values.shape[:2]
        length = 2 + 2 * previous_count
        sequence = condition.new_zeros(batch, length, condition.shape[-1])
        sequence[:, 0] = condition
        if previous_count:
            previous_pos_emb = self.position(previous_positions)
            sequence[:, 1:-1:2] = previous_pos_emb + self.instruction_type
            sequence[:, 2:-1:2] = (
                self.value_proj(previous_values) + previous_pos_emb + self.value_type
            )
        sequence[:, -1] = self.position(target_position) + self.instruction_type
        hidden = self.transformer(
            sequence, mask=self.causal_mask(length, sequence.device)
        )
        mean, log_std = self.output(hidden[:, -1]).chunk(2, dim=-1)
        return mean, log_std.clamp(-4.0, 1.0)


def fit_pca(
    residuals: torch.Tensor, latent_dim: int, max_samples: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = residuals.mean(0)
    centered = residuals - mean
    if len(centered) > max_samples:
        generator = torch.Generator().manual_seed(seed + 1709)
        indices = torch.randperm(len(centered), generator=generator)[:max_samples]
        fit = centered[indices]
    else:
        fit = centered
    _, _, components = torch.pca_lowrank(
        fit.to(device), q=latent_dim, center=False, niter=4
    )
    return mean, components.cpu()


def encode_residuals(
    features: torch.Tensor,
    labels: torch.Tensor,
    additive: torch.Tensor,
    residual_mean: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    residual = features - additive[labels]
    return (residual - residual_mean) @ components


def train_ar(
    model: AnyOrderResidualAR,
    code: torch.Tensor,
    labels: torch.Tensor,
    prompts: dict[str, torch.Tensor],
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[float]:
    counts = torch.bincount(labels).clamp_min(1)
    weights = counts[labels].float().reciprocal()
    sampler = WeightedRandomSampler(weights, len(labels), replacement=True)
    loader = DataLoader(
        torch.utils.data.TensorDataset(code, labels),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    attr_prompt = prompts["attr"].float().to(device)
    obj_prompt = prompts["obj"].float().to(device)
    pair_prompt = prompts["pair"].float().to(device)
    pair_attr = pair_attr.to(device)
    pair_obj = pair_obj.to(device)
    losses = []
    model.train()
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch_code, batch_labels in loader:
            batch_code = batch_code.float().to(device)
            batch_labels = batch_labels.to(device)
            if args.token_order == "random":
                permutation = torch.rand(
                    len(batch_code), model.num_tokens, device=device
                ).argsort(dim=1)
            else:
                permutation = torch.arange(
                    model.num_tokens, device=device
                ).expand(len(batch_code), -1)
            condition = model.condition(
                batch_labels,
                pair_attr,
                pair_obj,
                attr_prompt,
                obj_prompt,
                pair_prompt,
            )
            target, mean, log_std = model.teacher_forced(
                batch_code.view(len(batch_code), model.num_tokens, model.token_dim),
                permutation,
                condition,
            )
            inv_var = torch.exp(-2.0 * log_std)
            nll = 0.5 * (target - mean).square() * inv_var + log_std
            loss = nll.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)
        print(f"epoch={epoch + 1}/{args.epochs} ar_nll={epoch_loss:.6f}", flush=True)
    return losses


@torch.no_grad()
def generate_unseen(
    model: AnyOrderResidualAR,
    cache: dict[str, object],
    additive: torch.Tensor,
    counts: torch.Tensor,
    residual_mean: torch.Tensor,
    components: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    pair_attr, pair_obj = pair_maps(cache)
    pair_attr = pair_attr.to(device)
    pair_obj = pair_obj.to(device)
    prompts = cache["prompts"]
    attr_prompt = prompts["attr"].float().to(device)
    obj_prompt = prompts["obj"].float().to(device)
    pair_prompt = prompts["pair"].float().to(device)
    active = torch.tensor(
        cache["meta"].get("active", [True] * len(counts)), dtype=torch.bool
    )
    unseen = torch.where((counts == 0) & active)[0]
    all_features = []
    all_labels = []
    components = components.to(device)
    residual_mean = residual_mean.to(device)
    additive = additive.to(device)
    for pair_chunk in unseen.split(64):
        pair_ids = pair_chunk.repeat_interleave(args.samples_per_class).to(device)
        batch = len(pair_ids)
        if args.token_order == "random":
            permutation = torch.rand(
                batch, model.num_tokens, device=device
            ).argsort(dim=1)
        else:
            permutation = torch.arange(
                model.num_tokens, device=device
            ).expand(batch, -1)
        condition = model.condition(
            pair_ids,
            pair_attr,
            pair_obj,
            attr_prompt,
            obj_prompt,
            pair_prompt,
        )
        ordered_values = torch.empty(
            batch, model.num_tokens, model.token_dim, device=device
        )
        for step in range(model.num_tokens):
            previous_positions = permutation[:, :step]
            target_position = permutation[:, step]
            mean, log_std = model.next_token(
                ordered_values[:, :step], previous_positions, target_position, condition
            )
            if args.generation_mode == "sample":
                ordered_values[:, step] = mean + (
                    args.generation_temperature
                    * torch.exp(log_std)
                    * torch.randn_like(mean)
                )
            else:
                ordered_values[:, step] = mean
        code = torch.zeros_like(ordered_values)
        code.scatter_(
            1,
            permutation[..., None].expand_as(ordered_values),
            ordered_values,
        )
        code = code.flatten(1)
        residual = code @ components.T + residual_mean
        feature = F.normalize(
            additive[pair_ids] + args.residual_scale * residual, dim=-1
        )
        all_features.append(feature.cpu())
        all_labels.append(pair_ids.cpu())
    return torch.cat(all_features), torch.cat(all_labels)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    set_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    additive = additive_completion(cache, centroids, counts, 30)
    train = cache["train"]
    features = train["com"].float()
    labels = train["pair"].long()
    raw_residuals = features - additive[labels]
    residual_mean, components = fit_pca(
        raw_residuals, args.latent_dim, args.pca_max_samples, args.seed, device
    )
    code = encode_residuals(
        features, labels, additive, residual_mean, components
    ).float()
    pair_attr, pair_obj = pair_maps(cache)
    model = AnyOrderResidualAR(
        prompt_dim=cache["prompts"]["pair"].shape[1],
        token_dim=args.token_dim,
        num_tokens=args.latent_dim // args.token_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        pair_cond_scale=args.pair_cond_scale,
    ).to(device)
    train_started = time.perf_counter()
    train_losses = train_ar(
        model,
        code,
        labels,
        cache["prompts"],
        pair_attr,
        pair_obj,
        args,
        device,
    )
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_started
    torch.save(model.state_dict(), work_dir / "anyorder_ar.pt")
    torch.save(
        {
            "additive": additive.half(),
            "residual_mean": residual_mean.half(),
            "components": components.half(),
        },
        work_dir / "residual_pca.pt",
    )

    set_seed(args.seed + 1000)
    generation_started = time.perf_counter()
    unseen_features, unseen_labels = generate_unseen(
        model,
        cache,
        additive,
        counts,
        residual_mean,
        components,
        args,
        device,
    )
    seen_features, seen_labels = balanced_seen_samples(
        cache, counts, args.samples_per_class, args.seed
    )
    text = cache["prompts"]["pair"].float()
    classifier_args = argparse.Namespace(
        batch_size=args.classifier_batch_size,
        epochs=args.classifier_epochs,
        lr=args.classifier_lr,
        fixed_weight_mask=None,
    )
    classifier = train_classifier(
        torch.cat([seen_features, unseen_features]),
        torch.cat([seen_labels, unseen_labels]),
        text,
        classifier_args,
        device,
    )
    torch.save(classifier.state_dict(), work_dir / "classifier.pt")

    if cache["meta"].get("open_world", False):
        val_eval = FastOpenWorldEvaluator(cache, device)
        test_eval = None if args.validation_only else FastOpenWorldEvaluator(cache, device)
    else:
        val_eval = FastClosedWorldEvaluator(
            CompositionDataset(args.dataset_path, phase="val"), device
        )
        test_eval = None
        if not args.validation_only:
            test_eval = FastClosedWorldEvaluator(
                CompositionDataset(args.dataset_path, phase="test"), device
            )

    val = {key: value.to(device) for key, value in cache["val"].items()}
    val_features = cache["val"]["com"].float()
    val_clf = classifier_logits(classifier, val_features, device).to(device)
    text_device = F.normalize(text, dim=-1).to(device)
    val_text_z = zscore(val_features.to(device) @ text_device.T)
    active_mask = torch.tensor(
        cache["meta"].get("active", [True] * len(text)),
        dtype=torch.float32,
        device=device,
    )
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    val_clf_z = zscore(val_clf)
    trials = {}
    best = None
    for delta_name, val_delta in {
        "full": val_clf_z * active_mask[None],
        "update": (val_clf_z - val_text_z) * active_mask[None],
    }.items():
        for strength in (
            -2.0,
            -1.0,
            -0.5,
            -0.2,
            -0.1,
            -0.05,
            0.0,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
        ):
            stats = val_eval(
                val["logits"].float() + strength * val_delta,
                val["attr"],
                val["obj"],
                val["pair"],
            )
            trials[f"{delta_name}_{strength:.2f}"] = stats
            candidate = (stats["AUC"], stats["best_hm"], delta_name, strength)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    selection_source = None
    if args.shared_selection:
        shared_path = Path(args.shared_selection)
        shared = json.loads(shared_path.read_text(encoding="utf-8"))
        fixed = shared["shared_selection"]
        if not fixed["enabled"]:
            raise ValueError("shared validation gate is disabled; test evaluation is forbidden")
        delta_name = str(fixed["delta"])
        strength = float(fixed["strength"])
        fixed_key = f"{delta_name}_{strength:.2f}"
        if fixed_key not in trials:
            raise ValueError(f"shared trial {fixed_key!r} is absent from this run's grid")
        val_auc = float(trials[fixed_key]["AUC"])
        val_hm = float(trials[fixed_key]["best_hm"])
        val_gain = val_auc - base_val["AUC"]
        enabled = True
        selection_source = str(shared_path)
        raw_strength = strength
    else:
        val_auc, val_hm, delta_name, strength = best
        val_gain = val_auc - base_val["AUC"]
        enabled = val_gain >= args.min_val_auc_gain
        raw_strength = strength
        if not enabled:
            strength = 0.0

    result: dict[str, object] = {
        "method": {
            "name": "AOAR-Replay",
            "latent_dim": args.latent_dim,
            "token_dim": args.token_dim,
            "num_tokens": model.num_tokens,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
            "pair_cond_scale": args.pair_cond_scale,
            "token_order": args.token_order,
            "generation_mode": args.generation_mode,
            "generation_temperature": args.generation_temperature,
            "residual_scale": args.residual_scale,
        },
        "base_val": base_val,
        "classifier_val": val_eval(
            val_clf, val["attr"], val["obj"], val["pair"]
        ),
        "trials": trials,
        "selection": {
            "delta": delta_name,
            "strength": strength,
            "raw_strength": raw_strength,
            "source": selection_source or "independent validation diagnostic",
            "val_auc": val_auc,
            "val_hm": val_hm,
            "val_auc_gain": val_gain,
            "min_val_auc_gain": args.min_val_auc_gain,
            "enabled": enabled,
        },
        "train_losses": train_losses,
    }

    saved_logits = {"val": val_clf.half().cpu()}
    if not args.validation_only:
        assert test_eval is not None
        test = {key: value.to(device) for key, value in cache["test"].items()}
        test_features = cache["test"]["com"].float()
        test_clf = classifier_logits(classifier, test_features, device).to(device)
        saved_logits["test"] = test_clf.half().cpu()
        base_test = test_eval(
            test["logits"].float(), test["attr"], test["obj"], test["pair"]
        )
        test_text_z = zscore(test_features.to(device) @ text_device.T)
        test_clf_z = zscore(test_clf)
        test_delta = test_clf_z if delta_name == "full" else test_clf_z - test_text_z
        test_delta = test_delta * active_mask[None]
        fused = test["logits"].float() + strength * test_delta
        result["base_test"] = base_test
        result["classifier_test"] = test_eval(
            test_clf, test["attr"], test["obj"], test["pair"]
        )
        result["fused_test"] = test_eval(
            fused, test["attr"], test["obj"], test["pair"]
        )
    torch.save(saved_logits, work_dir / "classifier_logits.pt")
    torch.cuda.synchronize(device)
    result["efficiency"] = {
        "ar_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classifier_parameters": sum(
            parameter.numel() for parameter in classifier.parameters()
        ),
        "ar_train_seconds": train_seconds,
        "generation_classifier_eval_seconds": time.perf_counter() - generation_started,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "generated_samples_per_unseen_class": args.samples_per_class,
    }
    (work_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selection": result["selection"], "fused_test": result.get("fused_test")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
