"""Interaction-conditioned low-rank classifier trained on ReCoFlow distributions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    additive_completion,
    real_centroids,
    zscore,
)
from research.piflow_pilot import PrimitiveConditionalFlow  # noqa: E402
from research.warmstart_flow_classifier import (  # noqa: E402
    balanced_seen_samples,
    generate_unseen,
    pair_maps,
    residual_bank,
)
from utils import set_seed  # noqa: E402


class InteractionHyperClassifier(nn.Module):
    def __init__(
        self,
        text: torch.Tensor,
        attr_text: torch.Tensor,
        obj_text: torch.Tensor,
        pair_attr: torch.Tensor,
        pair_obj: torch.Tensor,
        rank: int,
    ) -> None:
        super().__init__()
        self.register_buffer("text", F.normalize(text, dim=-1))
        self.register_buffer("attr_text", F.normalize(attr_text, dim=-1))
        self.register_buffer("obj_text", F.normalize(obj_text, dim=-1))
        self.register_buffer("pair_attr", pair_attr)
        self.register_buffer("pair_obj", pair_obj)
        dim = text.shape[1]
        self.attr_down = nn.Linear(dim, rank, bias=False)
        self.obj_down = nn.Linear(dim, rank, bias=False)
        self.pair_down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def weights(
        self, pair_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_ids is None:
            pair_ids = torch.arange(len(self.text), device=self.text.device)
        attr_latent = self.attr_down(self.attr_text)[self.pair_attr[pair_ids]]
        obj_latent = self.obj_down(self.obj_text)[self.pair_obj[pair_ids]]
        pair_text = self.text[pair_ids]
        pair_latent = self.pair_down(pair_text)
        interaction = F.gelu(
            attr_latent + obj_latent + pair_latent + attr_latent * obj_latent
        )
        delta = self.up(interaction)
        gate = torch.sigmoid(self.gate(pair_text))
        return F.normalize(pair_text + gate * delta, dim=-1), delta

    def forward(
        self, features: torch.Tensor, pair_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, delta = self.weights(pair_ids)
        logits = (
            self.log_scale.exp().clamp(max=100.0)
            * F.normalize(features, dim=-1)
            @ weights.T
        )
        return logits, delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--transport-objective", choices=("flow_matching", "direct_endpoint"), default="flow_matching")
    parser.add_argument("--residual-distribution", choices=("empirical", "gaussian_diag", "gaussian_isotropic"), default="gaussian_isotropic")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--class-batch-size", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--delta-weight", type=float, default=1e-4)
    parser.add_argument("--pair-cond-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def classifier_logits(
    model: InteractionHyperClassifier,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    weights, _ = model.weights()
    scale = model.log_scale.exp().clamp(max=100.0)
    outputs = []
    for chunk in features.split(1024):
        outputs.append(
            (scale * F.normalize(chunk.float().to(device), dim=-1) @ weights.T).cpu()
        )
    return torch.cat(outputs)


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
    generated, generated_labels = generate_unseen(
        flow, cache, additive, residuals, counts, args.residual_scale,
        args, device, None,
    )
    del flow
    torch.cuda.empty_cache()
    seen, seen_labels = balanced_seen_samples(
        cache, counts, args.samples_per_class, args.seed
    )
    features = torch.cat([seen, generated])
    labels = torch.cat([seen_labels, generated_labels])
    loader = None
    class_bank = None
    if args.class_batch_size > 0:
        class_bank = torch.empty(
            len(counts), args.samples_per_class, features.shape[1], dtype=features.dtype
        )
        for class_id in range(len(counts)):
            class_features = features[labels == class_id]
            if len(class_features) != args.samples_per_class:
                raise ValueError(
                    f"class {class_id} has {len(class_features)} samples, "
                    f"expected {args.samples_per_class}"
                )
            class_bank[class_id] = class_features
    else:
        loader = DataLoader(
            TensorDataset(features, labels), batch_size=args.batch_size,
            shuffle=True, num_workers=0,
        )
    pair_attr, pair_obj = pair_maps(cache)
    model = InteractionHyperClassifier(
        cache["prompts"]["pair"].float().to(device),
        cache["prompts"]["attr"].float().to(device),
        cache["prompts"]["obj"].float().to(device),
        pair_attr.to(device), pair_obj.to(device), args.rank,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.train()
    sampling_generator = torch.Generator().manual_seed(args.seed + 3000)
    for epoch in range(args.epochs):
        losses = []
        if class_bank is not None:
            order = torch.randperm(len(counts), generator=sampling_generator)
            for class_ids_cpu in order.split(args.class_batch_size):
                sample_ids = torch.randint(
                    args.samples_per_class,
                    (len(class_ids_cpu),),
                    generator=sampling_generator,
                )
                batch_features = class_bank[class_ids_cpu, sample_ids].to(device)
                class_ids = class_ids_cpu.to(device)
                logits, delta = model(batch_features, class_ids)
                target = torch.arange(len(class_ids), device=device)
                loss = 0.5 * (
                    F.cross_entropy(logits, target)
                    + F.cross_entropy(logits.T, target)
                )
                loss = loss + args.delta_weight * delta.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.detach())
        else:
            assert loader is not None
            for batch_features, batch_labels in loader:
                logits, delta = model(batch_features.to(device))
                loss = F.cross_entropy(logits, batch_labels.to(device))
                loss = loss + args.delta_weight * delta.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.detach())
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"epoch={epoch + 1:03d} loss={torch.stack(losses).mean().item():.5f}",
                flush=True,
            )
    model.eval()
    val_classifier = classifier_logits(model, cache["val"]["com"], device)
    test_classifier = None if args.validation_only else classifier_logits(model, cache["test"]["com"], device)
    payload = {"val": val_classifier.half()}
    if test_classifier is not None:
        payload["test"] = test_classifier.half()
    torch.save(payload, work_dir / "classifier_logits.pt")
    torch.save(model.state_dict(), work_dir / "classifier.pt")

    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = None if args.validation_only else {key: value.to(device) for key, value in cache["test"].items()}
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_full = zscore(val_classifier.float().to(device))
    test_full = None if args.validation_only else zscore(test_classifier.float().to(device))
    val_text = zscore(val["com"].float() @ text.T)
    test_text = None if args.validation_only else zscore(test["com"].float() @ text.T)
    deltas = {
        "full": (val_full, test_full),
        "update": (val_full - val_text, None if args.validation_only else test_full - test_text),
    }
    val_eval = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), device
    )
    test_eval = None if args.validation_only else FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="test"), device
    )
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = None if args.validation_only else test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    trials, best = {}, None
    strengths = (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    for delta_name, (val_delta, test_delta) in deltas.items():
        for strength in strengths:
            stats = val_eval(
                val["logits"].float() + strength * val_delta,
                val["attr"], val["obj"], val["pair"],
            )
            trials[f"{delta_name}_{strength:.2f}"] = stats
            candidate = (
                stats["AUC"], stats["best_hm"], delta_name, strength, test_delta
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    val_gain = best[0] - base_val["AUC"]
    enabled = val_gain >= args.min_val_auc_gain
    strength = best[3] if enabled else 0.0
    raw_test = None
    selected_test = None
    if not args.validation_only:
        raw_test = test_eval(
            test["logits"].float() + best[3] * best[4],
            test["attr"], test["obj"], test["pair"],
        )
        selected_test = test_eval(
            test["logits"].float() + strength * best[4],
            test["attr"], test["obj"], test["pair"],
        )
    output = {
        "base_val": base_val, "base_test": base_test,
        "selection": {
            "delta": best[2], "strength": strength, "raw_strength": best[3],
            "val_auc": best[0], "val_hm": best[1], "val_auc_gain": val_gain,
            "min_val_auc_gain": args.min_val_auc_gain, "enabled": enabled,
        },
        "raw_test": raw_test, "selected_test": selected_test, "trials": trials,
        "validation_only": args.validation_only, "test_read": not args.validation_only,
    }
    (work_dir / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"selection": output["selection"], "raw_test": raw_test, "selected_test": selected_test}, indent=2))


if __name__ == "__main__":
    main()
