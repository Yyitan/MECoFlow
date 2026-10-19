"""Validation-only fusion of composition- and global-space generated classifiers."""

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
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--com-logits", required=True)
    parser.add_argument("--glb-logits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    args = parser.parse_args()
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    generated = {
        "com": torch.load(args.com_logits, map_location="cpu", weights_only=True),
        "glb": torch.load(args.glb_logits, map_location="cpu", weights_only=True),
    }
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    delta: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for space in ("com", "glb"):
        val_clf = zscore(generated[space]["val"].float().to(device))
        test_clf = zscore(generated[space]["test"].float().to(device))
        val_text = zscore(val[space].float() @ text.T)
        test_text = zscore(test[space].float() @ text.T)
        delta[space] = {
            "full": (val_clf, test_clf),
            "update": (val_clf - val_text, test_clf - test_text),
        }
    strengths = (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    trials, best = {}, None
    for com_mode in ("full", "update"):
        for glb_mode in ("full", "update"):
            for com_strength in strengths:
                for glb_strength in strengths:
                    val_delta = (
                        com_strength * delta["com"][com_mode][0]
                        + glb_strength * delta["glb"][glb_mode][0]
                    )
                    stats = val_eval(
                        val["logits"].float() + val_delta,
                        val["attr"], val["obj"], val["pair"],
                    )
                    name = (
                        f"com_{com_mode}_{com_strength:g}__"
                        f"glb_{glb_mode}_{glb_strength:g}"
                    )
                    trials[name] = stats
                    candidate = (
                        stats["AUC"], stats["best_hm"], name,
                        com_mode, com_strength, glb_mode, glb_strength,
                    )
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
    assert best is not None
    val_gain = best[0] - base_val["AUC"]
    enabled = val_gain >= args.min_val_auc_gain
    com_strength = best[4] if enabled else 0.0
    glb_strength = best[6] if enabled else 0.0
    test_delta = (
        com_strength * delta["com"][best[3]][1]
        + glb_strength * delta["glb"][best[5]][1]
    )
    fused_test = test_eval(
        test["logits"].float() + test_delta,
        test["attr"], test["obj"], test["pair"],
    )
    output = {
        "base_val": base_val, "base_test": base_test,
        "selection": {
            "name": best[2], "com_mode": best[3], "com_strength": com_strength,
            "glb_mode": best[5], "glb_strength": glb_strength,
            "val_auc": best[0], "val_hm": best[1], "val_auc_gain": val_gain,
            "min_val_auc_gain": args.min_val_auc_gain, "enabled": enabled,
        },
        "fused_test": fused_test, "trials": trials,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"selection": output["selection"], "fused_test": fused_test}, indent=2))


if __name__ == "__main__":
    main()
