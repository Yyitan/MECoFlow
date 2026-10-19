"""Cross-primitive rectified-flow generation for compositional feature completion.

An attribute feature and an object feature are sampled from different real images.
Their normalized sum is an empirical, factorized starting distribution.  A small
conditional rectified flow transports it to the real CAMS composition distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    evaluate_variant,
    real_centroids,
    zscore,
)
from research.generative_classifier_pilot import classifier_logits, train_classifier  # noqa: E402
from research.piflow_pilot import PrimitiveConditionalFlow  # noqa: E402
from research.warmstart_flow_classifier import balanced_seen_samples, pair_maps  # noqa: E402
from utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--primitive-cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--flow-path")
    parser.add_argument("--flow-epochs", type=int, default=20)
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--classifier-batch-size", type=int, default=2048)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--classifier-lr", type=float, default=3e-3)
    parser.add_argument("--aux-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pools(labels: torch.Tensor, size: int) -> list[torch.Tensor]:
    return [torch.where(labels == index)[0] for index in range(size)]


def sample_donors(labels: torch.Tensor, banks: list[torch.Tensor]) -> torch.Tensor:
    output = torch.empty_like(labels)
    for value in labels.unique().tolist():
        mask = labels == value
        bank = banks[value]
        output[mask] = bank[torch.randint(len(bank), (int(mask.sum()),))]
    return output


@torch.no_grad()
def primitive_prototypes(
    primitive: dict[str, torch.Tensor], cache: dict[str, object]
) -> torch.Tensor:
    pair_attr, pair_obj = pair_maps(cache)
    attr_feature = primitive["attr_feature"].float()
    obj_feature = primitive["obj_feature"].float()
    attr_label, obj_label = primitive["attr"].long(), primitive["obj"].long()
    attr_mean = torch.zeros(len(cache["meta"]["attrs"]), attr_feature.shape[1])
    obj_mean = torch.zeros(len(cache["meta"]["objs"]), obj_feature.shape[1])
    attr_count = torch.bincount(attr_label, minlength=len(attr_mean)).float().clamp_min(1)
    obj_count = torch.bincount(obj_label, minlength=len(obj_mean)).float().clamp_min(1)
    attr_mean.index_add_(0, attr_label, attr_feature)
    obj_mean.index_add_(0, obj_label, obj_feature)
    attr_mean = attr_mean / attr_count[:, None]
    obj_mean = obj_mean / obj_count[:, None]
    return F.normalize(attr_mean[pair_attr] + obj_mean[pair_obj], dim=-1)


def train_flow(
    cache: dict[str, object],
    primitive: dict[str, torch.Tensor],
    centroids: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> PrimitiveConditionalFlow:
    train = cache["train"]
    labels = train["pair"].long()
    attr_labels, obj_labels = train["attr"].long(), train["obj"].long()
    if not torch.equal(attr_labels, primitive["attr"].long()) or not torch.equal(
        obj_labels, primitive["obj"].long()
    ):
        raise ValueError("primitive cache order does not match the feature cache")
    pair_attr, pair_obj = pair_maps(cache)
    counts = torch.bincount(labels, minlength=len(centroids)).clamp_min(1)
    sampler = WeightedRandomSampler(counts[labels].float().reciprocal(), len(labels), replacement=True)
    loader = DataLoader(torch.arange(len(labels)), batch_size=args.batch_size, sampler=sampler)
    attr_banks = pools(attr_labels, len(cache["meta"]["attrs"]))
    obj_banks = pools(obj_labels, len(cache["meta"]["objs"]))

    target = train["com"].float().to(device)
    attr_feature = primitive["attr_feature"].float().to(device)
    obj_feature = primitive["obj_feature"].float().to(device)
    labels_device = labels.to(device)
    pair_attr, pair_obj = pair_attr.to(device), pair_obj.to(device)
    prompt = cache["prompts"]
    attr_prompt = prompt["attr"].float().to(device)
    obj_prompt = prompt["obj"].float().to(device)
    pair_prompt = prompt["pair"].float().to(device)
    seen_ids = torch.unique(labels_device).sort().values
    seen_lookup = torch.full((len(centroids),), -1, dtype=torch.long, device=device)
    seen_lookup[seen_ids] = torch.arange(len(seen_ids), device=device)
    seen_centroids = centroids[seen_ids.cpu()].to(device)

    model = PrimitiveConditionalFlow(target.shape[1], args.hidden_dim, args.cond_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.flow_epochs)
    model.train()
    for epoch in range(args.flow_epochs):
        epoch_loss = []
        for index in loader:
            batch_attr, batch_obj = attr_labels[index], obj_labels[index]
            attr_donor = sample_donors(batch_attr, attr_banks).to(device)
            obj_donor = sample_donors(batch_obj, obj_banks).to(device)
            index = index.to(device)
            label = labels_device[index]
            start = F.normalize(attr_feature[attr_donor] + obj_feature[obj_donor], dim=-1)
            endpoint = target[index]
            time = torch.rand(len(index), device=device)
            state = (1.0 - time[:, None]) * start + time[:, None] * endpoint
            velocity = model(
                state,
                time,
                attr_prompt[pair_attr[label]],
                obj_prompt[pair_obj[label]],
                pair_prompt[label],
            )
            flow_loss = (velocity - (endpoint - start)).square().sum(-1).mean()
            predicted_endpoint = F.normalize(state + (1.0 - time[:, None]) * velocity, dim=-1)
            aux_logits = 20.0 * predicted_endpoint @ seen_centroids.T
            aux_loss = F.cross_entropy(aux_logits, seen_lookup[label])
            loss = flow_loss + args.aux_weight * aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(torch.stack([loss.detach(), flow_loss.detach(), aux_loss.detach()]))
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            mean = torch.stack(epoch_loss).mean(0)
            print(
                f"epoch={epoch + 1:03d} loss={mean[0]:.5f} flow={mean[1]:.5f} aux={mean[2]:.5f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def generate_unseen(
    model: PrimitiveConditionalFlow,
    cache: dict[str, object],
    primitive: dict[str, torch.Tensor],
    counts: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_attr, pair_obj = pair_maps(cache)
    attr_labels, obj_labels = primitive["attr"].long(), primitive["obj"].long()
    attr_banks = pools(attr_labels, len(cache["meta"]["attrs"]))
    obj_banks = pools(obj_labels, len(cache["meta"]["objs"]))
    attr_feature = primitive["attr_feature"].float().to(device)
    obj_feature = primitive["obj_feature"].float().to(device)
    prompt = cache["prompts"]
    attr_prompt = prompt["attr"].float().to(device)
    obj_prompt = prompt["obj"].float().to(device)
    pair_prompt = prompt["pair"].float().to(device)
    pair_attr_device, pair_obj_device = pair_attr.to(device), pair_obj.to(device)
    samples, targets = [], []
    for ids in torch.where(counts == 0)[0].split(128):
        labels = ids.repeat_interleave(args.samples_per_class)
        attrs, objs = pair_attr[labels], pair_obj[labels]
        attr_donor = sample_donors(attrs, attr_banks).to(device)
        obj_donor = sample_donors(objs, obj_banks).to(device)
        labels_device = labels.to(device)
        state = F.normalize(attr_feature[attr_donor] + obj_feature[obj_donor], dim=-1)
        dt = 1.0 / args.sample_steps
        for step in range(args.sample_steps):
            time = torch.full((len(labels),), step * dt, device=device)
            state = state + dt * model(
                state,
                time,
                attr_prompt[pair_attr_device[labels_device]],
                obj_prompt[pair_obj_device[labels_device]],
                pair_prompt[labels_device],
            )
        samples.append(F.normalize(state, dim=-1).cpu())
        targets.append(labels)
    return torch.cat(samples), torch.cat(targets)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    primitive = torch.load(args.primitive_cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)

    prior = primitive_prototypes(primitive, cache)
    prior[counts > 0] = centroids[counts > 0]
    result: dict[str, object] = {
        "primitive_prior": evaluate_variant(
            "primitive_prior", prior, cache, val_eval, test_eval, device
        )
    }
    if args.flow_path:
        flow = PrimitiveConditionalFlow(
            cache["train"]["com"].shape[1], args.hidden_dim, args.cond_dim
        ).to(device)
        flow.load_state_dict(torch.load(args.flow_path, map_location="cpu", weights_only=True))
        flow.eval()
        print(f"loaded flow: {args.flow_path}", flush=True)
    else:
        flow = train_flow(cache, primitive, centroids, args, device)
    torch.save(flow.state_dict(), work_dir / "flow.pt")
    set_seed(args.seed + 1000)
    unseen_features, unseen_labels = generate_unseen(flow, cache, primitive, counts, args, device)
    seen_features, seen_labels = balanced_seen_samples(cache, counts, args.samples_per_class, args.seed)
    classifier_args = argparse.Namespace(
        batch_size=args.classifier_batch_size, epochs=args.classifier_epochs, lr=args.classifier_lr
    )
    classifier = train_classifier(
        torch.cat([seen_features, unseen_features]),
        torch.cat([seen_labels, unseen_labels]),
        cache["prompts"]["pair"].float(),
        classifier_args,
        device,
    )
    val_classifier = classifier_logits(classifier, cache["val"]["com"], device)
    test_classifier = classifier_logits(classifier, cache["test"]["com"], device)
    torch.save(
        {"val": val_classifier.half(), "test": test_classifier.half()},
        work_dir / "classifier_logits.pt",
    )
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    val_classifier_z = zscore(val_classifier.float().to(device))
    test_classifier_z = zscore(test_classifier.float().to(device))
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_text_z = zscore(val["com"].float() @ text.T)
    test_text_z = zscore(test["com"].float() @ text.T)
    # The classifier starts from CAMS' text weights.  Its most complementary signal
    # is therefore the generated-data update, rather than a duplicate text score.
    deltas = {
        "full": (val_classifier_z, test_classifier_z),
        "update": (val_classifier_z - val_text_z, test_classifier_z - test_text_z),
    }
    trials = {}
    best = None
    for delta_name, (val_delta, _test_delta) in deltas.items():
        for strength in (-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            stats = val_eval(
                val["logits"].float() + strength * val_delta,
                val["attr"], val["obj"], val["pair"],
            )
            trials[f"{delta_name}_{strength:.2f}"] = stats
            candidate = (
                stats["AUC"] + stats["best_hm"],
                stats["AUC"],
                stats["best_hm"],
                delta_name,
                strength,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    assert best is not None
    test_delta = deltas[best[3]][1]
    fused_test = test_eval(
        test["logits"].float() + best[4] * test_delta,
        test["attr"], test["obj"], test["pair"],
    )
    result["classifier_val"] = val_eval(
        val_classifier.float().to(device), val["attr"], val["obj"], val["pair"]
    )
    result["classifier_test"] = test_eval(
        test_classifier.float().to(device), test["attr"], test["obj"], test["pair"]
    )
    result["trials"] = trials
    result["selection"] = {
        "delta": best[3],
        "strength": best[4],
        "selection_score": best[0],
        "val_auc": best[1],
        "val_hm": best[2],
    }
    result["fused_test"] = fused_test
    (work_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selection": result["selection"], "fused_test": fused_test}, indent=2))


if __name__ == "__main__":
    main()
