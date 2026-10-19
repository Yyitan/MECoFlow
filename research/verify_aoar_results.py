"""Verify and render the paper-facing AOAR-Replay results from archived JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(root: Path, name: str) -> dict[str, Any]:
    with (root / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> float:
    return 100.0 * value


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def val_row(label: str, result: dict[str, Any]) -> dict[str, Any]:
    base = result["base_val"]
    selected = result["shared_selection"]
    return {
        "label": label,
        "auc": pct(selected["auc_mean"]),
        "auc_std": pct(selected["auc_sample_std"]),
        "delta_auc": pct(selected["auc_mean"] - base["auc_mean"]),
        "hm": pct(selected["hm_mean"]),
        "hm_std": pct(selected["hm_sample_std"]),
        "delta_hm": pct(selected["hm_mean"] - base["hm_mean"]),
        "positive_auc_seeds": sum(
            value > base["auc_mean"] for value in selected["per_seed_auc"]
        ),
        "selection": f'{selected["delta"]} x {selected["strength"]:g}',
    }


def test_row(label: str, result: dict[str, Any]) -> dict[str, Any]:
    base = result["base_test"]
    fused = result["fused_test"]
    delta = result["delta_test"]
    return {
        "label": label,
        "base_auc": pct(base["AUC"]["mean"]),
        "auc": pct(fused["AUC"]["mean"]),
        "auc_std": pct(fused["AUC"]["sample_std"]),
        "delta_auc": pct(delta["AUC"]["mean"]),
        "base_hm": pct(base["best_hm"]["mean"]),
        "hm": pct(fused["best_hm"]["mean"]),
        "hm_std": pct(fused["best_hm"]["sample_std"]),
        "delta_hm": pct(delta["best_hm"]["mean"]),
        "positive_auc_seeds": sum(value > 0 for value in delta["AUC"]["per_seed"]),
        "positive_hm_seeds": sum(value > 0 for value in delta["best_hm"]["per_seed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sample_val = load(args.root, "shared_val_selection.json")
    raster_val = load(args.root, "raster_shared_val_selection.json")
    mean_val = load(args.root, "mean_shared_val_selection.json")
    sample_test = load(args.root, "shared_test_results.json")
    mean_test = load(args.root, "mean_shared_test_results.json")

    val_results = [sample_val, raster_val, mean_val]
    assert all(result["protocol"] == "shared validation-only selection across generator seeds" for result in val_results)
    assert all(result["num_seeds"] == 3 for result in val_results)
    assert all(result["shared_selection"]["enabled"] for result in val_results)
    assert sample_test["protocol"] == mean_test["protocol"] == "fixed shared validation selection evaluated on test"
    assert sample_test["selection"]["delta"] == sample_val["shared_selection"]["delta"]
    assert close(sample_test["selection"]["strength"], sample_val["shared_selection"]["strength"])
    assert mean_test["selection"]["delta"] == mean_val["shared_selection"]["delta"]
    assert close(mean_test["selection"]["strength"], mean_val["shared_selection"]["strength"])
    assert close(sample_test["base_test"]["AUC"]["mean"], mean_test["base_test"]["AUC"]["mean"])
    assert close(sample_test["base_test"]["best_hm"]["mean"], mean_test["base_test"]["best_hm"]["mean"])

    sample_val_row = val_row("random / Gaussian sample", sample_val)
    raster_val_row = val_row("raster / Gaussian sample", raster_val)
    mean_val_row = val_row("random / conditional mean", mean_val)
    sample_test_row = test_row("random / Gaussian sample", sample_test)
    mean_test_row = test_row("random / conditional mean", mean_test)

    assert sample_val_row["positive_auc_seeds"] == 3
    assert raster_val_row["positive_auc_seeds"] == 1
    assert mean_val_row["positive_auc_seeds"] == 3
    assert sample_val_row["auc_std"] < raster_val_row["auc_std"]
    assert mean_val_row["delta_auc"] > sample_val_row["delta_auc"]
    assert sample_test["delta_test"]["AUC"]["all_positive"]
    assert mean_test["delta_test"]["AUC"]["all_positive"]
    assert not sample_test["delta_test"]["best_hm"]["all_positive"]
    assert not mean_test["delta_test"]["best_hm"]["all_positive"]
    assert mean_test_row["delta_auc"] > sample_test_row["delta_auc"]
    assert abs(mean_test_row["delta_hm"]) < abs(sample_test_row["delta_hm"])

    lines = [
        "# AOAR-Replay verified results",
        "",
        "Generated directly from archived JSON files by `research/verify_aoar_results.py`.",
        "",
        "## Shared validation selection",
        "",
        "| Variant | AUC | Delta AUC | HM | Delta HM | Positive AUC seeds | Shared setting |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in [sample_val_row, raster_val_row, mean_val_row]:
        lines.append(
            f'| {row["label"]} | {row["auc"]:.3f} +/- {row["auc_std"]:.3f} | '
            f'{row["delta_auc"]:+.3f} | {row["hm"]:.3f} +/- {row["hm_std"]:.3f} | '
            f'{row["delta_hm"]:+.3f} | {row["positive_auc_seeds"]}/3 | {row["selection"]} |'
        )
    lines.extend([
        "",
        "## Fixed shared-selection test",
        "",
        "| Variant | CAMS AUC | AOAR AUC | Delta AUC | AUC-positive seeds | CAMS HM | AOAR HM | Delta HM | HM-positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in [sample_test_row, mean_test_row]:
        lines.append(
            f'| {row["label"]} | {row["base_auc"]:.3f} | {row["auc"]:.3f} +/- {row["auc_std"]:.3f} | '
            f'{row["delta_auc"]:+.3f} | {row["positive_auc_seeds"]}/3 | {row["base_hm"]:.3f} | '
            f'{row["hm"]:.3f} +/- {row["hm_std"]:.3f} | {row["delta_hm"]:+.3f} | {row["positive_hm_seeds"]}/3 |'
        )
    lines.extend([
        "",
        "Conditional-mean generation improves AUC and reduces the HM loss relative to Gaussian sampling, but does not produce a stable HM gain.",
        "",
        "All AOAR protocol and consistency assertions passed.",
    ])
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
