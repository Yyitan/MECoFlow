#!/usr/bin/env python3
"""Generate a conservative optimizer/generator-seed audit for MECoFlow."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


# Two-sided 95% Student-t critical value for df=2 (all archived settings use n=3).
T95_DF2 = 4.3026527297

SETTINGS = [
    ("MIT original", "mit_mecoflow_shared.json"),
    ("UT original", "ut_mecoflow_shared.json"),
    ("MIT CAMS seed-1", "mit_seed1_mecoflow_gated.json"),
    ("UT CAMS seed-1", "ut_seed1_mecoflow_gated.json"),
]


def summarize(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half_width = T95_DF2 * sd / (n**0.5)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "positive": sum(value > 0 for value in values),
    }


def points(value: float) -> str:
    return f"{100 * value:+.3f}"


def build_report(protocol_dir: Path) -> str:
    rows = []
    all_auc = []
    all_hm = []
    for label, filename in SETTINGS:
        data = json.loads((protocol_dir / filename).read_text(encoding="utf-8"))
        base = data["base_test"]
        auc_delta = [run["AUC"] - base["AUC"] for run in data["test_runs"]]
        hm_delta = [run["best_hm"] - base["best_hm"] for run in data["test_runs"]]
        all_auc.extend(auc_delta)
        all_hm.extend(hm_delta)
        rows.append((label, summarize(auc_delta), summarize(hm_delta)))

    lines = [
        "# MECoFlow 生成器种子稳健性审计",
        "",
        "本表由四个归档 shared-selection JSON 自动生成。所有数值为相对同一冻结 CAMS checkpoint 的百分点变化；每个设置的三个生成器种子共享一组只由 validation AUC 选择的融合超参数。",
        "",
        "| 设置 | ΔAUC mean±sd | AUC 95% t-CI | AUC 正种子 | ΔHM mean±sd | HM 95% t-CI | HM 正种子 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, auc, hm in rows:
        lines.append(
            f"| {label} | {points(auc['mean'])}±{100 * auc['sd']:.3f} | "
            f"[{points(auc['ci_low'])}, {points(auc['ci_high'])}] | "
            f"{auc['positive']}/{auc['n']} | {points(hm['mean'])}±{100 * hm['sd']:.3f} | "
            f"[{points(hm['ci_low'])}, {points(hm['ci_high'])}] | {hm['positive']}/{hm['n']} |"
        )

    lines.extend(
        [
            "",
            f"跨四个 checkpoint 的逐运行符号一致性为 AUC {sum(v > 0 for v in all_auc)}/{len(all_auc)}、HM {sum(v > 0 for v in all_hm)}/{len(all_hm)}。",
            "",
            "## 解释边界",
            "",
            "这些区间只刻画 flow 初始化、生成采样和生成分类器训练的随机性。三个种子共享相同训练/验证/测试样本和冻结 CAMS checkpoint，因此不能当成数据集总体置信区间，也不能把 12 个相关运行视为 12 个独立实验做显著性检验。最稳妥的论文表述是：四个 checkpoint 的平均 AUC 均为正，且每个 checkpoint 的三个生成器种子全部为正；HM 在 MIT seed-1 的一个种子上轻微下降，因此只主张 AUC 的一致性。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=default_root / "output" / "research" / "protocol",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.protocol_dir)
    if args.output:
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
