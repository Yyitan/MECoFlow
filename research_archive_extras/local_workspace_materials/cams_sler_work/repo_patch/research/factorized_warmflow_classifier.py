"""Factorized generated-data classifier for stable large-vocabulary CZSL."""

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


class FactorizedCosineClassifier(nn.Module):
    def __init__(
        self,
        text: torch.Tensor,
        pair_attr: torch.Tensor,
        pair_obj: torch.Tensor,
        num_attrs: int,
        num_objs: int,
    ) -> None:
        super().__init__()
        self.register_buffer("text", F.normalize(text, dim=-1))
        self.register_buffer("pair_attr", pair_attr)
        self.register_buffer("pair_obj", pair_obj)
        self.attr_delta = nn.Parameter(torch.zeros(num_attrs, text.shape[1]))
        self.obj_delta = nn.Parameter(torch.zeros(num_objs, text.shape[1]))
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def weights(self) -> torch.Tensor:
        return F.normalize(
            self.text + self.attr_delta[self.pair_attr] + self.obj_delta[self.pair_obj],
            dim=-1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * F.normalize(features, dim=-1) @ self.weights().T


class ResidualAdapterClassifier(nn.Module):
    def __init__(self, text: torch.Tensor, rank: int) -> None:
        super().__init__()
        self.register_buffer("text", F.normalize(text, dim=-1))
        self.down = nn.Linear(text.shape[1], rank, bias=False)
        self.up = nn.Linear(rank, text.shape[1], bias=False)
        nn.init.zeros_(self.up.weight)
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def adapted(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(features)))
        return F.normalize(features + residual, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * self.adapted(F.normalize(features, dim=-1)) @ self.text.T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--additive-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--head", choices=("factorized", "adapter"), default="factorized")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def logits(
    model: nn.Module, features: torch.Tensor, device: torch.device
) -> torch.Tensor:
    output = []
    with torch.no_grad():
        for chunk in features.split(1024):
            output.append(model(chunk.float().to(device)).cpu())
    return torch.cat(output)


def main() -> None:
    args = parse_args()
    set_seed(args.seed + 1000)
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    additive = torch.load(args.additive_path, map_location="cpu", weights_only=True).float()
    centroids, counts = real_centroids(cache)
    residuals = residual_bank(cache, additive)
    flow = PrimitiveConditionalFlow(
        cache["train"]["com"].shape[1], args.hidden_dim, args.cond_dim
    ).to(device)
    flow.load_state_dict(torch.load(args.flow_path, map_location="cpu", weights_only=True))
    flow.eval()
    unseen_features, unseen_labels = generate_unseen(
        flow, cache, additive, residuals, counts, args.residual_scale, args, device
    )
    seen_features, seen_labels = balanced_seen_samples(
        cache, counts, args.samples_per_class, args.seed
    )
    features = torch.cat([seen_features, unseen_features])
    labels = torch.cat([seen_labels, unseen_labels])
    loader = DataLoader(
        TensorDataset(features, labels), batch_size=args.batch_size, shuffle=True
    )
    pair_attr, pair_obj = pair_maps(cache)
    if args.head == "factorized":
        model = FactorizedCosineClassifier(
            cache["prompts"]["pair"].float().to(device),
            pair_attr.to(device),
            pair_obj.to(device),
            len(cache["meta"]["attrs"]),
            len(cache["meta"]["objs"]),
        ).to(device)
    else:
        model = ResidualAdapterClassifier(
            cache["prompts"]["pair"].float().to(device), args.rank
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for batch_features, batch_labels in loader:
            prediction = model(batch_features.to(device))
            loss = F.cross_entropy(prediction, batch_labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1:03d} loss={torch.stack(losses).mean().item():.5f}", flush=True)
    model.eval()
    val_classifier = logits(model, cache["val"]["com"], device)
    test_classifier = logits(model, cache["test"]["com"], device)
    torch.save(
        {"val": val_classifier.half(), "test": test_classifier.half()},
        work_dir / "classifier_logits.pt",
    )
    torch.save(model.state_dict(), work_dir / "classifier.pt")

    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_full, test_full = zscore(val_classifier.float().to(device)), zscore(test_classifier.float().to(device))
    val_text, test_text = zscore(val["com"].float() @ text.T), zscore(test["com"].float() @ text.T)
    deltas = {"full": (val_full, test_full), "update": (val_full - val_text, test_full - test_text)}
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    result: dict[str, object] = {
        "base_val": val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"]),
        "base_test": test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"]),
        "classifier_val": val_eval(val_full, val["attr"], val["obj"], val["pair"]),
        "classifier_test": test_eval(test_full, test["attr"], test["obj"], test["pair"]),
    }
    trials, best = {}, None
    for delta_name, (val_delta, test_delta) in deltas.items():
        for strength in (-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            stats = val_eval(
                val["logits"].float() + strength * val_delta,
                val["attr"], val["obj"], val["pair"],
            )
            trials[f"{delta_name}_{strength:.2f}"] = stats
            candidate = (
                stats["AUC"] + stats["best_hm"], stats["AUC"], stats["best_hm"],
                delta_name, strength, test_delta,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    assert best is not None
    fused_test = test_eval(
        test["logits"].float() + best[4] * best[5],
        test["attr"], test["obj"], test["pair"],
    )
    result.update({
        "trials": trials,
        "selection": {
            "delta": best[3], "strength": best[4], "selection_score": best[0],
            "val_auc": best[1], "val_hm": best[2],
        },
        "fused_test": fused_test,
    })
    (work_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selection": result["selection"], "fused_test": fused_test}, indent=2))


if __name__ == "__main__":
    main()
