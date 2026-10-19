"""Audit the fixed validation-gain gate without changing any test result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SETTINGS = [
    ("MIT", "mit_mecoflow_shared.json", "closed main"),
    ("UT", "ut_mecoflow_shared.json", "closed main"),
    ("UT seed-1", "ut_seed1_mecoflow_gated.json", "closed checkpoint"),
    ("MIT seed-1", "mit_seed1_mecoflow_gated.json", "closed checkpoint"),
    ("C-GQA", "cgqa_mecoflow_seed0.json", "closed boundary"),
    ("UT 1-step", "ut_one_step_shared_raw.json", "ablation"),
    ("UT direct", "ut_direct_endpoint_shared_gated.json", "capacity control"),
    ("MIT OW", "mit_ow_mecoflow_shared_raw.json", "open boundary"),
    ("UT OW", "ut_ow_mecoflow_seed0.json", "open boundary"),
]

THRESHOLDS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.00)


def load_row(root: Path, label: str, filename: str, group: str) -> dict[str, object]:
    with (root / filename).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    output = result.get("mean_test") or result["fused_test"]
    base = result["base_test"]
    return {
        "label": label,
        "group": group,
        "val_gain": 100.0 * result["selection"]["val_auc_gain"],
        "test_auc_delta": 100.0 * (output["AUC"] - base["AUC"]),
        "test_hm_delta": 100.0 * (output["best_hm"] - base["best_hm"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [load_row(args.protocol_dir, *setting) for setting in SETTINGS]

    lines = [
        "# MECoFlow validation-gate sensitivity audit",
        "",
        "Generated from archived JSONs by `research/gate_sensitivity.py`. Thresholds and gains are in AUC percentage points.",
        "",
        "## Validation evidence and raw test outcome",
        "",
        "| Setting | Role | Validation gain | Raw test Delta AUC | Raw test Delta HM |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["label"]} | {row["group"]} | {row["val_gain"]:+.3f} | '
            f'{row["test_auc_delta"]:+.3f} | {row["test_hm_delta"]:+.3f} |'
        )

    lines.extend(
        [
            "",
            "## Decision sensitivity",
            "",
            "`pass` means only that the raw generator would be enabled at that validation threshold; the table does not reselect fusion hyperparameters.",
            "",
            "| Gate | " + " | ".join(row["label"] for row in rows) + " |",
            "|---:|" + "---:|" * len(rows),
        ]
    )
    for threshold in THRESHOLDS:
        decisions = ["pass" if row["val_gain"] >= threshold else "fallback" for row in rows]
        lines.append(f'| {threshold:.2f} | ' + " | ".join(decisions) + " |")

    main_positive = rows[:4]
    cgqa = rows[4]
    assert all(row["test_auc_delta"] > 0 for row in main_positive)
    assert cgqa["test_auc_delta"] < 0
    assert max(cgqa["val_gain"], 0.10) < min(row["val_gain"] for row in main_positive)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- On the closed-world main/checkpoint evidence, every threshold from 0.10 through 0.25 AUC points gives the same decision: keep all four positive settings and reject C-GQA. The chosen 0.20 is therefore not a knife-edge value.",
            "- The 0.20 gate deliberately rejects the positive but weak-validation UT 1-step diagnostic; it trades recall for conservative evidence.",
            "- UT open-world has a large positive validation gain but negative test transfer. No scalar validation-gain threshold that preserves the closed-world UT result can fix this domain shift, so the gate is not a universal reliability guarantee and open-world remains outside the claim.",
            "- This is a post-hoc sensitivity audit of one uniformly applied protocol, not evidence that 0.20 was theoretically derived or preregistered.",
            "",
        ]
    )
    text = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
