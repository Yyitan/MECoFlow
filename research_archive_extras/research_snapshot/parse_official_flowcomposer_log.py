#!/usr/bin/env python3
"""Parse FlowComposer's console-only evaluation records into auditable JSON.

The official Stage-1/Stage-2 scripts print metrics but do not write a metric
artifact.  This parser preserves the reported epoch and integration step; it
does not select a checkpoint or compute an "improved" result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(r"^epoch\s+(\d+)\s+train loss\s+([-+0-9.eE]+)\s*$")
STEP_RE = re.compile(
    r"^Evaluating with step\s+(?P<step>\d+)"
    r"(?:\s+compose\s+(?P<compose>True|False)\s+end_points\s+(?P<end>[-+0-9.eE]+))?\s*$"
)
METRIC_RE = re.compile(
    r"(best_seen|best_unseen|best_hm|AUC|attr_acc|obj_acc)\s+([-+0-9.eE]+)\s*\|"
)
EXIT_RE = re.compile(r"^\s*Exit status:\s*(\d+)\s*$")
WALL_RE = re.compile(
    r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)\s*$"
)
RSS_RE = re.compile(r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    current_epoch = None
    current_step = None
    current_compose = None
    current_end_points = None
    evaluations = []
    exit_status = None
    wall_clock = None
    max_rss_kib = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = EPOCH_RE.match(line)
        if match:
            current_epoch = int(match.group(1))
            continue
        match = STEP_RE.match(line)
        if match:
            current_step = int(match.group("step"))
            compose = match.group("compose")
            current_compose = None if compose is None else compose == "True"
            end_points = match.group("end")
            current_end_points = None if end_points is None else float(end_points)
            continue

        metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
        if metrics and "AUC" in metrics:
            evaluations.append(
                {
                    "reported_epoch": current_epoch,
                    "integration_steps": current_step,
                    "compose": current_compose,
                    "end_points": current_end_points,
                    **metrics,
                }
            )
            continue

        match = EXIT_RE.match(raw_line)
        if match:
            exit_status = int(match.group(1))
            continue
        match = WALL_RE.match(raw_line)
        if match:
            wall_clock = match.group(1)
            continue
        match = RSS_RE.match(raw_line)
        if match:
            max_rss_kib = int(match.group(1))

    return {
        "log": str(path),
        "completed_marker": "done!" in text,
        "exit_status": exit_status,
        "wall_clock": wall_clock,
        "max_rss_kib": max_rss_kib,
        "num_evaluations": len(evaluations),
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = parse_log(args.log)
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
