"""Source-seed audit for a frozen instance-adaptive route blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_support(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    ai = {name: index for index, name in enumerate(meta["attrs"])}
    oi = {name: index for index, name in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    tp = cache["train"]["pair"].long()
    ac = torch.bincount(pa[tp], minlength=len(ai)).float()
    oc = torch.bincount(po[tp], minlength=len(oi)).float()
    raw = torch.log1p(torch.minimum(ac[pa], oc[po]))
    support = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)
    seen = torch.bincount(tp, minlength=len(meta["pairs"])) > 0
    return support, seen


def route(cache: dict[str, object], iso: torch.Tensor, diag: torch.Tensor, name: str, power: float) -> torch.Tensor:
    support, _ = pair_support(cache)
    weight = support.clamp_min(1e-6).pow(power)
    if name == "inverse_pair_support":
        weight = 1.0 - weight
    elif name != "pair_support":
        raise ValueError(name)
    return weight[None] * diag + (1.0 - weight)[None] * iso


def entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)


def normalized_test_entropy(val_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    val_raw = entropy(val_logits)
    raw = entropy(logits)
    return ((raw - val_raw.min()) / (val_raw.max() - val_raw.min()).clamp_min(1e-6)).clamp(0.0, 1.0)


def schedule_alpha(candidate: dict[str, object], uncertainty: torch.Tensor) -> torch.Tensor:
    name = str(candidate["schedule"])
    constant = re.fullmatch(r"constant_([0-9]+(?:\.[0-9]+)?)", name)
    if constant is not None:
        return torch.full_like(uncertainty, float(constant.group(1))).clamp(0.0, 1.0)
    match = re.fullmatch(r"unseen_entropy_([0-9]+(?:\.[0-9]+)?)", name)
    if match is None:
        raise ValueError(f"unsupported schedule: {candidate['schedule']}")
    return uncertainty.pow(float(match.group(1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pp = Path(args.protocol).resolve()
    protocol = json.loads(pp.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol not preregistered")
    cache = torch.load(Path(protocol["cache"]).resolve(), map_location="cpu", weights_only=False)
    paths = [Path(item["path"]).resolve() for item in protocol["sources"]]
    for item, path in zip(protocol["sources"], paths):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
    n = int(protocol["family_size"])
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    if any(set(item) != {"val", "test"} for item in saved):
        raise ValueError("source seed audit requires val/test source files")
    val, test = cache["val"], cache["test"]
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    support, pair_seen = pair_support(cache)
    scope = pair_seen.float() + (1.0 - pair_seen.float()) * float(protocol["candidate"]["route_b"]["seen_lambda"])
    val_schedule = schedule_alpha(protocol["candidate"], normalized_test_entropy(val["logits"], val["logits"]))
    test_schedule = schedule_alpha(protocol["candidate"], normalized_test_entropy(val["logits"], test["logits"]))
    val_alpha = pair_seen[val["pair"].long()].float() + (1.0 - pair_seen[val["pair"].long()].float()) * val_schedule
    test_alpha = pair_seen[test["pair"].long()].float() + (1.0 - pair_seen[test["pair"].long()].float()) * test_schedule
    a, b = protocol["candidate"]["route_a"], protocol["candidate"]["route_b"]
    ve = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    te = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_v = ve(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_t = te(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    rows = []
    for index in range(n):
        iso_v, diag_v = zscore(saved[index]["val"].float()), zscore(saved[n + index]["val"].float())
        iso_t, diag_t = zscore(saved[index]["test"].float()), zscore(saved[n + index]["test"].float())
        av, at = route(cache, iso_v, diag_v, a["route"], float(a["power"])), route(cache, iso_t, diag_t, a["route"], float(a["power"]))
        bv, bt = route(cache, iso_v, diag_v, b["route"], float(b["power"])), route(cache, iso_t, diag_t, b["route"], float(b["power"]))
        da_v, da_t = av - val_text, at - test_text
        db_v = bv * scope[val["pair"].long()][:, None] * float(b["strength"])
        db_t = bt * scope[test["pair"].long()][:, None] * float(b["strength"])
        dv = (1.0 - val_alpha[:, None]) * float(a["strength"]) * da_v + val_alpha[:, None] * db_v
        dt = (1.0 - test_alpha[:, None]) * float(a["strength"]) * da_t + test_alpha[:, None] * db_t
        fused_v = ve(val["logits"].float() + dv, val["attr"], val["obj"], val["pair"])
        fused_t = te(test["logits"].float() + dt, test["attr"], test["obj"], test["pair"])
        rows.append({"seed_index": index, "val_gain": {"AUC": fused_v["AUC"] - base_v["AUC"], "HM": fused_v["best_hm"] - base_v["best_hm"]}, "test_gain": {"AUC": fused_t["AUC"] - base_t["AUC"], "HM": fused_t["best_hm"] - base_t["best_hm"]}})
    result = {"status": "passed", "protocol": str(pp), "protocol_sha256": sha256(pp), "candidate": protocol["candidate"], "test_search_after_lock": False, "sources": [{"path": str(path), "sha256": sha256(path)} for path in paths], "runs": rows, "positive_test_auc_runs": sum(row["test_gain"]["AUC"] > 0 for row in rows), "positive_test_hm_runs": sum(row["test_gain"]["HM"] > 0 for row in rows)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
