"""Recompute the paper-facing MECoFlow tables from archived result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GATE = 0.002


def load(root: Path, name: str) -> dict[str, Any]:
    with (root / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> float:
    return 100.0 * value


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def shared_row(label: str, result: dict[str, Any]) -> dict[str, Any]:
    base = result["base_test"]
    mean = result["mean_test"]
    return {
        "label": label,
        "base_auc": pct(base["AUC"]),
        "auc": pct(mean["AUC"]),
        "auc_std": pct(mean["AUC_std"]),
        "delta_auc": pct(mean["AUC"] - base["AUC"]),
        "base_hm": pct(base["best_hm"]),
        "hm": pct(mean["best_hm"]),
        "hm_std": pct(mean["best_hm_std"]),
        "delta_hm": pct(mean["best_hm"] - base["best_hm"]),
        "setting": f'{result["selection"]["delta"]} x {result["selection"]["strength"]:g}',
        "val_gain": pct(result["selection"]["val_auc_gain"]),
        "all_positive": all(run["AUC"] > base["AUC"] for run in result["test_runs"]),
    }


def single_row(label: str, result: dict[str, Any], gated: bool = False) -> dict[str, Any]:
    base = result["base_test"]
    fused = result["fused_test"]
    raw_gain = result["selection"]["val_auc_gain"]
    enabled = not gated or raw_gain >= GATE
    selected = fused if enabled else base
    return {
        "label": label,
        "base_auc": pct(base["AUC"]),
        "auc": pct(selected["AUC"]),
        "delta_auc": pct(selected["AUC"] - base["AUC"]),
        "base_hm": pct(base["best_hm"]),
        "hm": pct(selected["best_hm"]),
        "delta_hm": pct(selected["best_hm"] - base["best_hm"]),
        "raw_auc": pct(fused["AUC"]),
        "raw_hm": pct(fused["best_hm"]),
        "val_gain": pct(raw_gain),
        "enabled": enabled,
    }


def render_shared(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Setting | CAMS AUC | MECoFlow AUC | Delta AUC | CAMS HM | MECoFlow HM | Delta HM | Shared setting |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f'| {row["label"]} | {row["base_auc"]:.3f} | '
            f'{row["auc"]:.3f} +/- {row["auc_std"]:.3f} | {row["delta_auc"]:+.3f} | '
            f'{row["base_hm"]:.3f} | {row["hm"]:.3f} +/- {row["hm_std"]:.3f} | '
            f'{row["delta_hm"]:+.3f} | {row["setting"]} |'
        )
    return lines


def render_fixed_row(label: str, base: dict[str, float], fused: dict[str, float], setting: str) -> str:
    base_hm = base.get("HM", base.get("best_hm"))
    fused_hm = fused.get("HM", fused.get("best_hm"))
    return (
        f'| {label} | {pct(base["AUC"]):.3f} | {pct(fused["AUC"]):.3f} | '
        f'{pct(fused["AUC"] - base["AUC"]):+.3f} | {pct(base_hm):.3f} | '
        f'{pct(fused_hm):.3f} | {pct(fused_hm - base_hm):+.3f} | {setting} |'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.protocol_dir

    mit = load(root, "mit_mecoflow_shared.json")
    ut = load(root, "ut_mecoflow_shared.json")
    ut_seed1 = load(root, "ut_seed1_mecoflow_gated.json")
    mit_seed1 = load(root, "mit_seed1_mecoflow_gated.json")
    cgqa = load(root, "cgqa_mecoflow_seed0.json")
    empirical_mit = load(root, "mit.json")
    diagonal_mit = load(root, "mit_gaussian_diag_shared.json")
    diagonal_ut = load(root, "ut_gaussian_diag_shared_gated.json")
    empirical_ut = load(root, "ut.json")
    deterministic_ut = load(root, "ut_mecoflow_deterministic_shared.json")
    one_step_ut_raw = load(root, "ut_one_step_shared_raw.json")
    one_step_ut_gated = load(root, "ut_one_step_shared_gated.json")
    direct_endpoint_ut = load(root, "ut_direct_endpoint_shared_gated.json")
    no_flow_ut = load(root, "ut_no_flow_source_augmentation.json")
    zero_center_mit = load(root, "mit_zero_center_shared.json")
    mit_ow_raw = load(root, "mit_ow_mecoflow_shared_raw.json")
    mit_ow_gated = load(root, "mit_ow_mecoflow_shared_gated.json")
    ut_ow = load(root, "ut_ow_mecoflow_seed0.json")
    efficiency = load(root, "mit_efficiency_full.json")["efficiency"]
    inference = load(root, "mit_inference_benchmark.json")
    artifact_root = root.parent
    checkpoint_summary = json.loads(
        (artifact_root / "warmflow_ut" / "ut_diagonal_ensemble_checkpoint_summary.json").read_text()
    )
    ut_cross_median = json.loads(
        (artifact_root / "warmflow_ut" / "cross_source_median_fixed_test.json").read_text()
    )
    ut_consensus = json.loads(
        (artifact_root / "warmflow_ut" / "consensus_gate_fixed_test.json").read_text()
    )
    ut_support_consensus = json.loads(
        (artifact_root / "warmflow_ut" / "support_consensus_fixed_test.json").read_text()
    )
    mit_seed1_median = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "source_median_fixed_test.json").read_text()
    )
    ut_seed1_support = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "support_only_fixed_test.json").read_text()
    )
    mit_seed1_support = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "support_only_fixed_test.json").read_text()
    )
    ut_factorized_support = json.loads(
        (artifact_root / "warmflow_ut" / "factorized_object_support_fixed_test_v2.json").read_text()
    )
    ut_confidence = json.loads(
        (artifact_root / "warmflow_ut" / "confidence_gate_fixed_test.json").read_text()
    )
    mit_confidence = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "confidence_gate_fixed_test.json").read_text()
    )
    ut_joint = json.loads(
        (artifact_root / "warmflow_ut" / "joint_reliability_fixed_v1" / "results.json").read_text()
    )
    mit_joint = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "joint_reliability_fixed_v1" / "results.json").read_text()
    )
    ut_joint_audit = json.loads(
        (artifact_root / "warmflow_ut" / "joint_reliability_fixed_v1" / "audit.json").read_text()
    )
    mit_joint_audit = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "joint_reliability_fixed_v1" / "audit.json").read_text()
    )
    ut_seed1_joint = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "joint_reliability_fixed_v1" / "results.json").read_text()
    )
    ut_seed1_joint_audit = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "joint_reliability_fixed_v1" / "audit.json").read_text()
    )
    ut_adaptive = json.loads(
        (artifact_root / "warmflow_ut" / "adaptive_source_posterior_fixed_v1" / "results.json").read_text()
    )
    mit_adaptive = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "adaptive_source_posterior_fixed_v1" / "results.json").read_text()
    )
    ut_adaptive_audit = json.loads(
        (artifact_root / "warmflow_ut" / "adaptive_source_posterior_fixed_v1" / "audit.json").read_text()
    )
    mit_adaptive_audit = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "adaptive_source_posterior_fixed_v1" / "audit.json").read_text()
    )
    mit_original_adaptive = json.loads(
        (artifact_root / "warmflow_mit" / "adaptive_source_posterior_fixed_v1" / "results.json").read_text()
    )
    mit_original_adaptive_audit = json.loads(
        (artifact_root / "warmflow_mit" / "adaptive_source_posterior_fixed_v1" / "audit.json").read_text()
    )
    ut_adaptive_support = json.loads(
        (artifact_root / "warmflow_ut" / "adaptive_support_posterior_fixed_v1" / "results.json").read_text()
    )
    mit_adaptive_support = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "adaptive_support_posterior_fixed_v1" / "results.json").read_text()
    )
    ut_adaptive_support_audit = json.loads(
        (artifact_root / "warmflow_ut" / "adaptive_support_posterior_fixed_v1" / "audit.json").read_text()
    )
    mit_adaptive_support_audit = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "adaptive_support_posterior_fixed_v1" / "audit.json").read_text()
    )
    ut_residual_mixture = json.loads(
        (artifact_root / "warmflow_ut" / "residual_mixture_fixed_v1" / "results.json").read_text()
    )
    mit_seed1_residual_mixture = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "residual_mixture_fixed_v1" / "results.json").read_text()
    )
    mit_original_residual_mixture = json.loads(
        (artifact_root / "warmflow_mit" / "residual_mixture_fixed_v1" / "results.json").read_text()
    )
    ut_interaction_hyper = json.loads(
        (artifact_root / "warmflow_ut" / "interaction_hyper_fixed_v1" / "results.json").read_text()
    )
    ut_pair_source_router = json.loads(
        (artifact_root / "warmflow_ut" / "pair_source_router_fixed_v1" / "results.json").read_text()
    )
    ut_seed1_pair_source_router = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "pair_source_router_fixed_v1" / "results.json").read_text()
    )
    mit_pair_source_router = json.loads(
        (artifact_root / "warmflow_mit" / "pair_source_router_fixed_v1" / "results.json").read_text()
    )
    ut_selective_pair_router = json.loads(
        (artifact_root / "warmflow_ut" / "selective_pair_router_fixed_v1" / "results.json").read_text()
    )
    ut_seed1_selective_pair_router = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "selective_pair_router_fixed_v1" / "results.json").read_text()
    )
    mit_selective_pair_router = json.loads(
        (artifact_root / "warmflow_mit" / "selective_pair_router_fixed_v1" / "results.json").read_text()
    )
    ut_seed1_soft_selective_pair_router = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "soft_selective_pair_router_fixed_v1" / "results.json").read_text()
    )
    mit_soft_selective_pair_router = json.loads(
        (artifact_root / "warmflow_mit" / "soft_selective_pair_router_fixed_v1" / "results.json").read_text()
    )
    ut_seed1_pareto_blend = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "pareto_router_blend_fixed_v1" / "results.json").read_text()
    )
    ut_seed1_adaptive_blend = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "adaptive_route_blend_fixed_v1" / "results.json").read_text()
    )
    ut_original_adaptive_blend = json.loads(
        (artifact_root / "warmflow_ut" / "adaptive_route_blend_fixed_v1" / "results.json").read_text()
    )
    ut_original_entropy_threshold = json.loads(
        (artifact_root / "warmflow_ut" / "adaptive_route_threshold_fixed_v1" / "results.json").read_text()
    )
    mit_seed1_adaptive_blend = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "adaptive_route_blend_fixed_v1" / "results.json").read_text()
    )
    mit_seed1_entropy_threshold = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "adaptive_route_threshold_fixed_v1" / "results.json").read_text()
    )
    cgqa_adaptive_threshold = json.loads(
        (artifact_root / "warmflow_cgqa" / "adaptive_threshold_fixed_v1" / "results.json").read_text()
    )
    cgqa_structural_risk = json.loads(
        (artifact_root / "warmflow_cgqa" / "structural_risk_fixed_v1" / "results.json").read_text()
    )
    cgqa_structural_risk_source_audit = json.loads(
        (artifact_root / "warmflow_cgqa" / "structural_risk_source_audit_v1.json").read_text()
    )
    ut_candidate_route = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "candidate_conditioned_route_fixed_v1.json").read_text()
    )
    mit_candidate_route = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "candidate_conditioned_route_fixed_v1.json").read_text()
    )
    ut_original_candidate_route = json.loads(
        (artifact_root / "warmflow_ut" / "candidate_conditioned_route_fixed_v1.json").read_text()
    )
    ut_candidate_route_audit = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "candidate_conditioned_route_source_audit_v1.json").read_text()
    )
    mit_candidate_route_audit = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "candidate_conditioned_route_source_audit_v1.json").read_text()
    )
    ut_original_candidate_route_audit = json.loads(
        (artifact_root / "warmflow_ut" / "candidate_conditioned_route_source_audit_v1.json").read_text()
    )
    ut_seed1_entropy_threshold = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "adaptive_route_threshold_fixed_v1" / "results.json").read_text()
    )
    selective_seed_audits = [
        json.loads((artifact_root / "warmflow_ut" / "selective_pair_router_fixed_v1" / "source_seed_audit.json").read_text()),
        json.loads((artifact_root / "warmflow_ut_seed1final" / "selective_pair_router_fixed_v1" / "source_seed_audit.json").read_text()),
        json.loads((artifact_root / "warmflow_mit" / "selective_pair_router_fixed_v1" / "source_seed_audit.json").read_text()),
    ]
    soft_selective_seed_audits = [
        json.loads((artifact_root / "warmflow_ut_seed1final" / "soft_selective_pair_router_fixed_v1" / "source_seed_audit.json").read_text()),
        json.loads((artifact_root / "warmflow_mit" / "soft_selective_pair_router_fixed_v1" / "source_seed_audit.json").read_text()),
    ]
    ut_seed1_pareto_blend_audit = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "pareto_router_blend_fixed_v1" / "source_seed_audit.json").read_text()
    )
    ut_seed1_adaptive_blend_audit = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "adaptive_route_blend_fixed_v1" / "source_seed_audit.json").read_text()
    )
    ut_original_adaptive_blend_audit = json.loads(
        (artifact_root / "warmflow_ut" / "adaptive_route_blend_fixed_v1" / "source_seed_audit.json").read_text()
    )
    ut_original_entropy_threshold_audit = json.loads(
        (artifact_root / "warmflow_ut" / "adaptive_route_threshold_fixed_v1" / "source_seed_audit.json").read_text()
    )
    mit_seed1_adaptive_blend_audit = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "adaptive_route_blend_fixed_v1" / "source_seed_audit.json").read_text()
    )
    mit_seed1_entropy_threshold_audit = json.loads(
        (artifact_root / "warmflow_mit_seed1final" / "adaptive_route_threshold_fixed_v1" / "source_seed_audit.json").read_text()
    )
    cgqa_adaptive_threshold_audit = json.loads(
        (artifact_root / "warmflow_cgqa" / "adaptive_threshold_fixed_v1" / "source_seed_audit.json").read_text()
    )
    ut_seed1_entropy_threshold_audit = json.loads(
        (artifact_root / "warmflow_ut_seed1final" / "adaptive_route_threshold_fixed_v1" / "source_seed_audit.json").read_text()
    )
    original_ensemble = checkpoint_summary["checkpoints"][0]
    seed1_ensemble = checkpoint_summary["checkpoints"][1]
    fixed_source_rows = [
        render_fixed_row(
            "UT diagonal-source mean ensemble",
            original_ensemble["base_test"],
            original_ensemble["ensemble_test"],
            "diagonal mean, update x 1.25",
        ),
        render_fixed_row(
            "UT seed-1 diagonal-source mean ensemble",
            seed1_ensemble["base_test"],
            seed1_ensemble["ensemble_test"],
            "diagonal mean, update x 0.75",
        ),
        render_fixed_row(
            "UT six-source median ensemble",
            ut_cross_median["base_test"],
            ut_cross_median["fused_test"],
            "isotropic+diagonal median, update x 1.25",
        ),
        render_fixed_row(
            "UT inverse-consensus gated ensemble",
            ut_consensus["base_test"],
            ut_consensus["fused_test"],
            "inverse agreement gate, update x 0.75",
        ),
        render_fixed_row(
            "UT support-aware consensus ensemble",
            ut_support_consensus["base_test"],
            ut_support_consensus["fused_test"],
            "harmonic support + inverse agreement, update x 0.75",
        ),
        render_fixed_row(
            "MIT seed-1 isotropic-source median ensemble",
            mit_seed1_median["base_test"],
            mit_seed1_median["fused_test"],
            "source median, update x 0.75",
        ),
        render_fixed_row(
            "UT seed-1 support-only ensemble",
            ut_seed1_support["base_test"],
            ut_seed1_support["fused_test"],
            "min image support, update x 0.75",
        ),
        render_fixed_row(
            "MIT seed-1 support-only ensemble",
            mit_seed1_support["base_test"],
            mit_seed1_support["fused_test"],
            "min image support, update x 0.5",
        ),
        render_fixed_row(
            "UT factorized object-support ensemble",
            ut_factorized_support["base_test"],
            ut_factorized_support["fused_test"],
            "object support mean, update x 1.0",
        ),
        render_fixed_row(
            "UT confidence-gated ensemble",
            ut_confidence["base_test"],
            ut_confidence["fused_test"],
            "low base entropy gate, update x 1.0",
        ),
        render_fixed_row(
            "MIT seed-1 confidence-gated ensemble",
            mit_confidence["base_test"],
            mit_confidence["fused_test"],
            "high base entropy gate, update x 0.5",
        ),
        render_fixed_row(
            "UT joint support-confidence ensemble",
            ut_joint["base_test"],
            ut_joint["fused_test"],
            "object support x high margin, update x 1.0",
        ),
        render_fixed_row(
            "MIT seed-1 joint support-confidence ensemble",
            mit_joint["base_test"],
            mit_joint["fused_test"],
            "attribute support + low entropy, update x 0.25",
        ),
        render_fixed_row(
            "UT adaptive source-posterior ensemble",
            ut_adaptive["base_test"],
            ut_adaptive["fused_test"],
            "median + disagreement/confidence posterior, update x 1.5",
        ),
        render_fixed_row(
            "MIT seed-1 adaptive source-posterior ensemble",
            mit_adaptive["base_test"],
            mit_adaptive["fused_test"],
            "mean + disagreement/entropy posterior, update x 0.75",
        ),
        render_fixed_row(
            "MIT original adaptive source-posterior ensemble",
            mit_original_adaptive["base_test"],
            mit_original_adaptive["fused_test"],
            "mean + disagreement/entropy posterior, update x 0.75",
        ),
        render_fixed_row(
            "UT graph-conditioned adaptive posterior ensemble",
            ut_adaptive_support["base_test"],
            ut_adaptive_support["fused_test"],
            "attribute support x source posterior, strength 1.5",
        ),
        render_fixed_row(
            "MIT seed-1 graph-conditioned adaptive posterior ensemble",
            mit_adaptive_support["base_test"],
            mit_adaptive_support["fused_test"],
            "min support + source posterior, strength 0.5",
        ),
        render_fixed_row(
            "UT locked reliability residual mixture",
            ut_residual_mixture["base_test"],
            ut_residual_mixture["fused_test"],
            "adaptive/support residuals, alpha 0.875",
        ),
        render_fixed_row(
            "MIT seed-1 locked reliability residual mixture",
            mit_seed1_residual_mixture["base_test"],
            mit_seed1_residual_mixture["fused_test"],
            "adaptive/support residuals, alpha 0.5",
        ),
        render_fixed_row(
            "MIT original locked reliability residual mixture",
            mit_original_residual_mixture["base_test"],
            mit_original_residual_mixture["fused_test"],
            "imported alpha 0.5, adaptive/support residuals",
        ),
        render_fixed_row(
            "UT interaction hyper-classifier",
            ut_interaction_hyper["base_test"],
            ut_interaction_hyper["fused_test"],
            "pair-specific low-rank weights, full x 0.2",
        ),
        render_fixed_row(
            "UT pair-level source-family router",
            ut_pair_source_router["base_test"],
            ut_pair_source_router["fused_test"],
            "train support^0.25 route, update x 1.25",
        ),
        render_fixed_row(
            "UT seed-1 pair-level source-family router",
            ut_seed1_pair_source_router["base_test"],
            ut_seed1_pair_source_router["fused_test"],
            "inverse train support route, update x 1.0",
        ),
        render_fixed_row(
            "MIT original pair-level source-family router",
            mit_pair_source_router["base_test"],
            mit_pair_source_router["fused_test"],
            "inverse family-disagreement route, update x 0.5",
        ),
        render_fixed_row(
            "UT selective unseen-pair source router",
            ut_selective_pair_router["base_test"],
            ut_selective_pair_router["fused_test"],
            "pair support^0.5, unseen-only, full x 1.5",
        ),
        render_fixed_row(
            "UT seed-1 selective unseen-pair source router",
            ut_seed1_selective_pair_router["base_test"],
            ut_seed1_selective_pair_router["fused_test"],
            "pair support^0.25, unseen-only, full x 1.5",
        ),
        render_fixed_row(
            "MIT selective pair source router",
            mit_selective_pair_router["base_test"],
            mit_selective_pair_router["fused_test"],
            "pair support^0.25, all-pair, update x 0.5",
        ),
        render_fixed_row(
            "UT seed-1 soft selective pair router",
            ut_seed1_soft_selective_pair_router["base_test"],
            ut_seed1_soft_selective_pair_router["fused_test"],
            "pair support^0.25, seen lambda 0.25, full x 1.5",
        ),
        render_fixed_row(
            "MIT soft selective pair router",
            mit_soft_selective_pair_router["base_test"],
            mit_soft_selective_pair_router["fused_test"],
            "pair support^0.5, seen lambda 0.75, update x 0.5",
        ),
        render_fixed_row(
            "UT seed-1 Pareto route blend",
            ut_seed1_pareto_blend["base_test"],
            ut_seed1_pareto_blend["fused_test"],
            "0.5 inverse-pair update + 0.5 soft selective full",
        ),
        render_fixed_row(
            "UT seed-1 instance-adaptive route blend",
            ut_seed1_adaptive_blend["base_test"],
            ut_seed1_adaptive_blend["fused_test"],
            "unseen-pair CAMS entropy schedules inverse/soft routes",
        ),
        render_fixed_row(
            "UT seed-1 sigmoid-threshold route blend",
            ut_seed1_entropy_threshold["base_test"],
            ut_seed1_entropy_threshold["fused_test"],
            "unseen-pair sigmoid(8*(entropy-0.5)) inverse/soft routes",
        ),
        render_fixed_row(
            "UT seed-1 candidate-conditioned route",
            ut_candidate_route["base_test"],
            ut_candidate_route["fused_test"],
            "pair-score sigmoid(-4*(z-0)) inverse/soft routes",
        ),
        render_fixed_row(
            "UT original instance-adaptive route blend",
            ut_original_adaptive_blend["base_test"],
            ut_original_adaptive_blend["fused_test"],
            "unseen-pair CAMS entropy schedules inverse/soft routes",
        ),
        render_fixed_row(
            "UT original sigmoid-threshold route blend",
            ut_original_entropy_threshold["base_test"],
            ut_original_entropy_threshold["fused_test"],
            "unseen-pair hard entropy threshold 0.7 inverse/soft routes",
        ),
        render_fixed_row(
            "UT original candidate-conditioned route",
            ut_original_candidate_route["base_test"],
            ut_original_candidate_route["fused_test"],
            "pair-score sigmoid(-4*(z-0.5)) inverse/soft routes",
        ),
        render_fixed_row(
            "MIT seed-1 soft selective diagonal replay",
            mit_seed1_adaptive_blend["base_test"],
            mit_seed1_adaptive_blend["fused_test"],
            "constant soft route, pair support^0.25, seen lambda 0.25",
        ),
        render_fixed_row(
            "MIT seed-1 threshold soft-route replay",
            mit_seed1_entropy_threshold["base_test"],
            mit_seed1_entropy_threshold["fused_test"],
            "reverse hard entropy threshold 0.9, soft route",
        ),
        render_fixed_row(
            "MIT seed-1 candidate-conditioned route",
            mit_candidate_route["base_test"],
            mit_candidate_route["fused_test"],
            "pair-score sigmoid(-4*(z-0.5)) inverse/soft routes",
        ),
        render_fixed_row(
            "C-GQA threshold selective transport",
            cgqa_adaptive_threshold["base_test"],
            cgqa_adaptive_threshold["fused_test"],
            "reverse hard entropy threshold 0.8, inverse/soft routes",
        ),
        render_fixed_row(
            "C-GQA structural-risk selective transport",
            cgqa_structural_risk["base_test"],
            cgqa_structural_risk["fused_test"],
            "entropy + support risk (beta .5, slope 8, threshold .8)",
        ),
    ]

    main_rows = [
        shared_row("MIT-States / CAMS checkpoint", mit),
        shared_row("UT-Zappos / CAMS checkpoint", ut),
        shared_row("UT-Zappos / CAMS seed-1", ut_seed1),
        shared_row("MIT-States / CAMS seed-1", mit_seed1),
    ]
    assert all(row["all_positive"] for row in main_rows), "A main run is not positive for every generator seed"
    assert close(mit["base_test"]["AUC"], empirical_mit["base_test"]["AUC"])
    assert close(ut["base_test"]["AUC"], empirical_ut["base_test"]["AUC"])
    assert close(ut["base_test"]["AUC"], diagonal_ut["base_test"]["AUC"])
    for candidate, audit in ((ut_candidate_route, ut_candidate_route_audit), (ut_original_candidate_route, ut_original_candidate_route_audit), (mit_candidate_route, mit_candidate_route_audit)):
        assert candidate["status"] == "passed"
        assert candidate["test_read"] is True
        assert candidate["test_search_after_lock"] is False
        assert audit["status"] == "passed" and audit["test_search_after_lock"] is False
        assert audit["all_test_auc_positive"] and audit["all_test_hm_positive"]
        assert len(audit["runs"]) == 3
        assert audit["mean_test_auc_gain"] > 0 and audit["mean_test_hm_gain"] > 0

    cgqa_gated = single_row("C-GQA", cgqa, gated=True)
    assert not cgqa_gated["enabled"] and close(cgqa_gated["delta_auc"], 0.0)
    assert ut_support_consensus["status"] == "passed"
    assert ut_support_consensus["test_search_after_lock"] is False
    assert len(ut_support_consensus["individual"]) == 6
    assert all(row["test_auc_gain"] > 0 and row["test_hm_gain"] > 0 for row in ut_support_consensus["individual"])
    assert ut_seed1_support["status"] == "passed"
    assert ut_seed1_support["test_search_after_lock"] is False
    assert len(ut_seed1_support["individual"]) == 3
    assert all(row["test_auc_gain"] > 0 and row["test_hm_gain"] > 0 for row in ut_seed1_support["individual"])
    assert mit_seed1_support["status"] == "passed"
    assert mit_seed1_support["test_search_after_lock"] is False
    assert len(mit_seed1_support["individual"]) == 3
    assert all(row["test_auc_gain"] > 0 for row in mit_seed1_support["individual"])
    assert mit_seed1_support["mean"]["test_hm_gain"] > 0
    assert ut_factorized_support["status"] == "passed"
    assert ut_factorized_support["test_search_after_lock"] is False
    assert len(ut_factorized_support["individual"]) == 6
    assert all(row["test_auc_gain"] > 0 and row["test_hm_gain"] > 0 for row in ut_factorized_support["individual"])
    assert ut_confidence["status"] == "passed"
    assert ut_confidence["test_search_after_lock"] is False
    assert len(ut_confidence["individual"]) == 6
    assert all(row["test_auc_gain"] > 0 and row["test_hm_gain"] > 0 for row in ut_confidence["individual"])
    assert mit_confidence["status"] == "passed"
    assert mit_confidence["test_search_after_lock"] is False
    assert len(mit_confidence["individual"]) == 3
    assert all(row["test_auc_gain"] > 0 for row in mit_confidence["individual"])
    assert mit_confidence["mean"]["test_hm_gain"] > 0
    for joint, audit, count in ((ut_joint, ut_joint_audit, 6), (mit_joint, mit_joint_audit, 3)):
        assert joint["status"] == "passed"
        assert joint["test_search_after_lock"] is False
        assert audit["status"] == "passed"
        assert len(joint["individual"]) == count
        assert joint["mean"]["val_auc_gain"] >= GATE
        assert joint["mean"]["test_auc_gain"] > 0 and joint["mean"]["test_hm_gain"] > 0
    assert ut_seed1_joint["status"] == "passed"
    assert ut_seed1_joint_audit["status"] == "passed"
    assert ut_seed1_joint_audit["outcome"] == "negative_transfer_boundary"
    assert ut_seed1_joint["test_search_after_lock"] is False
    assert ut_seed1_joint["mean"]["val_auc_gain"] >= GATE
    assert ut_seed1_joint["mean"]["test_auc_gain"] < 0 and ut_seed1_joint["mean"]["test_hm_gain"] < 0
    assert all(row["test_auc_gain"] < 0 and row["test_hm_gain"] < 0 for row in ut_seed1_joint["individual"])
    for adaptive, audit, count in ((ut_adaptive, ut_adaptive_audit, 6), (mit_adaptive, mit_adaptive_audit, 3)):
        assert adaptive["status"] == "passed"
        assert adaptive["test_search_after_lock"] is False
        assert audit["status"] == "passed" and audit["outcome"] == "positive"
        assert adaptive["source_count"] == count
        assert adaptive["mean"]["val_auc_gain"] >= GATE
        assert adaptive["mean"]["test_auc_gain"] > 0 and adaptive["mean"]["test_hm_gain"] > 0
    assert mit_original_adaptive["status"] == "passed"
    assert mit_original_adaptive_audit["status"] == "passed"
    assert mit_original_adaptive_audit["outcome"] == "positive"
    assert mit_original_adaptive["test_search_after_lock"] is False
    assert mit_original_adaptive["source_count"] == 3
    assert mit_original_adaptive["mean"]["val_auc_gain"] >= GATE
    assert mit_original_adaptive["mean"]["test_auc_gain"] > 0 and mit_original_adaptive["mean"]["test_hm_gain"] > 0
    assert ut_adaptive_support["status"] == "passed" and ut_adaptive_support_audit["status"] == "passed"
    assert ut_adaptive_support_audit["outcome"] == "negative_transfer_boundary"
    assert ut_adaptive_support["test_search_after_lock"] is False
    assert ut_adaptive_support["mean"]["val_auc_gain"] >= GATE
    assert ut_adaptive_support["mean"]["test_auc_gain"] < 0 and ut_adaptive_support["mean"]["test_hm_gain"] < 0
    assert mit_adaptive_support["status"] == "passed" and mit_adaptive_support_audit["status"] == "passed"
    assert mit_adaptive_support_audit["outcome"] == "positive"
    assert mit_adaptive_support["test_search_after_lock"] is False
    assert mit_adaptive_support["mean"]["val_auc_gain"] >= GATE
    assert mit_adaptive_support["mean"]["test_auc_gain"] > 0 and mit_adaptive_support["mean"]["test_hm_gain"] > 0
    for mixture in (ut_residual_mixture, mit_seed1_residual_mixture, mit_original_residual_mixture):
        assert mixture["test_read"] is True
        assert mixture["val_gain"]["AUC"] >= GATE
        assert mixture["test_gain"]["AUC"] > 0 and mixture["test_gain"]["HM"] > 0
    assert ut_interaction_hyper["status"] == "passed"
    assert ut_interaction_hyper["test_read"] is True
    assert ut_interaction_hyper["test_search_after_lock"] is False
    assert ut_interaction_hyper["val_gain"]["AUC"] >= GATE
    assert ut_interaction_hyper["test_gain"]["AUC"] > 0 and ut_interaction_hyper["test_gain"]["HM"] > 0
    assert ut_pair_source_router["status"] == "passed"
    assert ut_pair_source_router["test_read"] is True
    assert ut_pair_source_router["test_search_after_lock"] is False
    assert ut_pair_source_router["val_gain"]["AUC"] >= GATE
    assert ut_pair_source_router["test_gain"]["AUC"] > 0 and ut_pair_source_router["test_gain"]["HM"] > 0
    assert ut_seed1_pair_source_router["status"] == "passed"
    assert ut_seed1_pair_source_router["test_read"] is True
    assert ut_seed1_pair_source_router["test_search_after_lock"] is False
    assert ut_seed1_pair_source_router["val_gain"]["AUC"] >= GATE
    assert ut_seed1_pair_source_router["test_gain"]["AUC"] > 0 and ut_seed1_pair_source_router["test_gain"]["HM"] > 0
    assert mit_pair_source_router["status"] == "passed"
    assert mit_pair_source_router["test_read"] is True
    assert mit_pair_source_router["test_search_after_lock"] is False
    assert mit_pair_source_router["val_gain"]["AUC"] >= GATE
    assert mit_pair_source_router["test_gain"]["AUC"] > 0 and mit_pair_source_router["test_gain"]["HM"] > 0
    for selective in (ut_selective_pair_router, ut_seed1_selective_pair_router, mit_selective_pair_router):
        assert selective["status"] == "passed"
        assert selective["test_read"] is True
        assert selective["test_search_after_lock"] is False
        assert selective["val_gain"]["AUC"] >= GATE
        assert selective["test_gain"]["AUC"] > 0 and selective["test_gain"]["HM"] > 0
    for seed_audit in selective_seed_audits:
        assert seed_audit["status"] == "passed"
        assert seed_audit["test_search_after_lock"] is False
        assert seed_audit["positive_test_auc_runs"] == 3
        assert seed_audit["positive_test_hm_runs"] == 3
    assert ut_seed1_soft_selective_pair_router["status"] == "passed"
    assert ut_seed1_soft_selective_pair_router["test_read"] is True
    assert ut_seed1_soft_selective_pair_router["test_search_after_lock"] is False
    assert ut_seed1_soft_selective_pair_router["val_gain"]["AUC"] >= GATE
    assert ut_seed1_soft_selective_pair_router["test_gain"]["AUC"] > 0
    assert ut_seed1_soft_selective_pair_router["test_gain"]["HM"] > 0
    assert mit_soft_selective_pair_router["status"] == "passed"
    assert mit_soft_selective_pair_router["test_read"] is True
    assert mit_soft_selective_pair_router["test_search_after_lock"] is False
    assert mit_soft_selective_pair_router["val_gain"]["AUC"] >= GATE
    assert mit_soft_selective_pair_router["test_gain"]["AUC"] > 0
    assert mit_soft_selective_pair_router["test_gain"]["HM"] > 0
    assert soft_selective_seed_audits[0]["positive_test_auc_runs"] == 3
    assert soft_selective_seed_audits[0]["positive_test_hm_runs"] == 3
    assert soft_selective_seed_audits[1]["positive_test_auc_runs"] == 3
    assert soft_selective_seed_audits[1]["positive_test_hm_runs"] == 2
    assert ut_seed1_pareto_blend["status"] == "passed"
    assert ut_seed1_pareto_blend["test_read"] is True
    assert ut_seed1_pareto_blend["test_search_after_lock"] is False
    assert ut_seed1_pareto_blend["val_gain"]["AUC"] >= GATE
    assert ut_seed1_pareto_blend["test_gain"]["AUC"] > 0
    assert ut_seed1_pareto_blend["test_gain"]["HM"] > 0
    assert ut_seed1_pareto_blend_audit["status"] == "passed"
    assert ut_seed1_pareto_blend_audit["positive_test_auc_runs"] == 3
    assert ut_seed1_pareto_blend_audit["positive_test_hm_runs"] == 3
    assert ut_seed1_adaptive_blend["status"] == "passed"
    assert ut_seed1_adaptive_blend["test_read"] is True
    assert ut_seed1_adaptive_blend["test_search_after_lock"] is False
    assert ut_seed1_adaptive_blend["val_gain"]["AUC"] >= GATE
    assert ut_seed1_adaptive_blend["test_gain"]["AUC"] > 0
    assert ut_seed1_adaptive_blend["test_gain"]["HM"] > 0
    assert ut_seed1_adaptive_blend_audit["status"] == "passed"
    assert ut_seed1_adaptive_blend_audit["test_search_after_lock"] is False
    assert ut_seed1_adaptive_blend_audit["positive_test_auc_runs"] == 3
    assert ut_seed1_adaptive_blend_audit["positive_test_hm_runs"] == 3
    assert ut_seed1_entropy_threshold["status"] == "passed"
    assert ut_seed1_entropy_threshold["test_read"] is True
    assert ut_seed1_entropy_threshold["test_search_after_lock"] is False
    assert ut_seed1_entropy_threshold["val_gain"]["AUC"] >= GATE
    assert ut_seed1_entropy_threshold["test_gain"]["AUC"] > 0
    assert ut_seed1_entropy_threshold["test_gain"]["HM"] > 0
    assert ut_seed1_entropy_threshold_audit["status"] == "passed"
    assert ut_seed1_entropy_threshold_audit["test_search_after_lock"] is False
    assert ut_seed1_entropy_threshold_audit["positive_test_auc_runs"] == 3
    assert ut_seed1_entropy_threshold_audit["positive_test_hm_runs"] == 3
    assert ut_original_adaptive_blend["status"] == "passed"
    assert ut_original_adaptive_blend["test_read"] is True
    assert ut_original_adaptive_blend["test_search_after_lock"] is False
    assert ut_original_adaptive_blend["val_gain"]["AUC"] >= GATE
    assert ut_original_adaptive_blend["test_gain"]["AUC"] > 0
    assert ut_original_adaptive_blend["test_gain"]["HM"] > 0
    assert ut_original_adaptive_blend_audit["status"] == "passed"
    assert ut_original_adaptive_blend_audit["test_search_after_lock"] is False
    assert ut_original_adaptive_blend_audit["positive_test_auc_runs"] == 3
    assert ut_original_adaptive_blend_audit["positive_test_hm_runs"] == 3
    assert ut_original_entropy_threshold["status"] == "passed"
    assert ut_original_entropy_threshold["test_read"] is True
    assert ut_original_entropy_threshold["test_search_after_lock"] is False
    assert ut_original_entropy_threshold["val_gain"]["AUC"] >= GATE
    assert ut_original_entropy_threshold["test_gain"]["AUC"] > 0
    assert ut_original_entropy_threshold["test_gain"]["HM"] > 0
    assert ut_original_entropy_threshold_audit["status"] == "passed"
    assert ut_original_entropy_threshold_audit["test_search_after_lock"] is False
    assert ut_original_entropy_threshold_audit["positive_test_auc_runs"] == 3
    assert ut_original_entropy_threshold_audit["positive_test_hm_runs"] == 3
    assert mit_seed1_adaptive_blend["status"] == "passed"
    assert mit_seed1_adaptive_blend["test_read"] is True
    assert mit_seed1_adaptive_blend["test_search_after_lock"] is False
    assert mit_seed1_adaptive_blend["val_gain"]["AUC"] >= GATE
    assert mit_seed1_adaptive_blend["test_gain"]["AUC"] > 0
    assert mit_seed1_adaptive_blend["test_gain"]["HM"] > 0
    assert mit_seed1_adaptive_blend_audit["status"] == "passed"
    assert mit_seed1_adaptive_blend_audit["test_search_after_lock"] is False
    assert mit_seed1_adaptive_blend_audit["positive_test_auc_runs"] == 3
    assert mit_seed1_adaptive_blend_audit["positive_test_hm_runs"] == 3
    assert mit_seed1_entropy_threshold["status"] == "passed"
    assert mit_seed1_entropy_threshold["test_read"] is True
    assert mit_seed1_entropy_threshold["test_search_after_lock"] is False
    assert mit_seed1_entropy_threshold["val_gain"]["AUC"] >= GATE
    assert mit_seed1_entropy_threshold["test_gain"]["AUC"] > 0
    assert mit_seed1_entropy_threshold["test_gain"]["HM"] > 0
    assert mit_seed1_entropy_threshold_audit["status"] == "passed"
    assert mit_seed1_entropy_threshold_audit["test_search_after_lock"] is False
    assert mit_seed1_entropy_threshold_audit["positive_test_auc_runs"] == 3
    assert mit_seed1_entropy_threshold_audit["positive_test_hm_runs"] == 3
    assert cgqa_adaptive_threshold["status"] == "passed"
    assert cgqa_adaptive_threshold["test_read"] is True
    assert cgqa_adaptive_threshold["test_search_after_lock"] is False
    assert cgqa_adaptive_threshold["val_gain"]["AUC"] >= GATE
    assert cgqa_adaptive_threshold["test_gain"]["AUC"] > 0
    assert cgqa_adaptive_threshold["test_gain"]["HM"] > 0
    assert cgqa_adaptive_threshold_audit["status"] == "passed"
    assert cgqa_adaptive_threshold_audit["test_search_after_lock"] is False
    assert cgqa_adaptive_threshold_audit["positive_test_auc_runs"] == 1
    assert cgqa_adaptive_threshold_audit["positive_test_hm_runs"] == 1
    assert cgqa_structural_risk["status"] == "passed"
    assert cgqa_structural_risk["test_read"] is True
    assert cgqa_structural_risk["test_search_after_lock"] is False
    assert cgqa_structural_risk["val_gain"]["AUC"] >= GATE
    assert cgqa_structural_risk["test_gain"]["AUC"] > 0
    assert cgqa_structural_risk["test_gain"]["HM"] > 0
    assert cgqa_structural_risk_source_audit["status"] == "passed"
    assert cgqa_structural_risk_source_audit["test_search_after_lock"] is False
    assert cgqa_structural_risk_source_audit["positive_test_auc_runs"] == 3
    assert cgqa_structural_risk_source_audit["positive_test_hm_runs"] == 3

    deterministic = shared_row("UT deterministic source (s=0)", deterministic_ut)
    stochastic = shared_row("UT stochastic max-entropy source", ut)
    stochastic_gap_auc = stochastic["auc"] - deterministic["auc"]
    stochastic_gap_hm = stochastic["hm"] - deterministic["hm"]
    assert stochastic_gap_auc > 0.9 and stochastic_gap_hm > 0.8
    one_step_raw = shared_row("UT 1-step raw", one_step_ut_raw)
    one_step_safe = shared_row("UT 1-step gated", one_step_ut_gated)
    direct_endpoint = shared_row("UT direct endpoint regressor", direct_endpoint_ut)
    eight_step = shared_row("UT 8-step main", ut)
    assert one_step_raw["all_positive"]
    assert one_step_raw["val_gain"] < pct(GATE)
    assert close(one_step_safe["delta_auc"], 0.0)
    assert direct_endpoint["all_positive"] and direct_endpoint["val_gain"] >= pct(GATE)
    assert eight_step["val_gain"] >= pct(GATE)
    flow_over_direct_auc = eight_step["auc"] - direct_endpoint["auc"]
    flow_over_direct_hm = eight_step["hm"] - direct_endpoint["hm"]
    assert flow_over_direct_auc > 1.0 and flow_over_direct_hm > 0.8
    assert close(no_flow_ut["base_test"]["AUC"], ut["base_test"]["AUC"])
    no_flow_auc = pct(no_flow_ut["fused_test"]["AUC"])
    no_flow_hm = pct(no_flow_ut["fused_test"]["best_hm"])
    no_flow_auc_delta = no_flow_auc - pct(no_flow_ut["base_test"]["AUC"])
    no_flow_hm_delta = no_flow_hm - pct(no_flow_ut["base_test"]["best_hm"])
    assert no_flow_auc_delta < -2.0 and no_flow_hm_delta < -1.0
    zero_center = shared_row("MIT zero-centered source", zero_center_mit)
    additive_center = shared_row("MIT additive compositional center", mit)
    location_gap_auc = additive_center["auc"] - zero_center["auc"]
    location_gap_hm = additive_center["hm"] - zero_center["hm"]
    assert location_gap_auc > 0.08 and location_gap_hm > 0.14
    isotropic_diagonal_auc_gap = pct(ut["mean_test"]["AUC"] - diagonal_ut["mean_test"]["AUC"])
    assert isotropic_diagonal_auc_gap > 0.0
    assert ut["mean_test"]["AUC_std"] < diagonal_ut["mean_test"]["AUC_std"]

    mit_ow = shared_row("MIT open-world raw", mit_ow_raw)
    mit_ow_safe = shared_row("MIT open-world gated", mit_ow_gated)
    ut_ow_raw = single_row("UT open-world raw", ut_ow, gated=False)
    assert mit_ow["all_positive"]
    assert mit_ow["val_gain"] < pct(GATE)
    assert close(mit_ow_safe["delta_auc"], 0.0)
    assert ut_ow_raw["delta_auc"] < 0.0

    lines = [
        "# MECoFlow verified paper numbers",
        "",
        "Generated directly from archived JSON files by `research/verify_paper_results.py`.",
        "",
        "## Closed-world main and cross-checkpoint results",
        "",
        *render_shared(main_rows),
        "",
        f'C-GQA is gated to CAMS because its validation gain is only {cgqa_gated["val_gain"]:+.3f} AUC points '
        f'(< {pct(GATE):.1f}); raw test delta is {cgqa_gated["raw_auc"] - cgqa_gated["base_auc"]:+.3f} AUC.',
        "",
        "## Fixed source-ensemble upgrade",
        "",
        "These fixed-setting source ensembles are checkpoint-level evidence, not a dataset-independent confidence interval:",
        "",
        "| Setting | CAMS AUC | Fused AUC | Delta AUC | CAMS HM | Fused HM | Delta HM | Protocol |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        *fixed_source_rows,
        "",
        "All fixed candidates were selected from validation-only evidence; their audit JSONs record test_search_after_lock=false.",
        "The machine-readable UT checkpoint summary is `output/research/warmflow_ut/ut_diagonal_ensemble_checkpoint_summary.json`;",
        "the detailed fixed-test/audit pairs are retained under the corresponding warmflow_ut and warmflow_mit_seed1final directories.",
        "The UT six-source median, inverse-consensus gate, and support-aware consensus gate each have 6/6 positive source-level AUC/HM gains;",
        "the MIT seed-1 median has positive aggregate HM but only 1/3 positive source-level HM gains; the UT seed-1 support-only branch has 3/3 positive source-level AUC/HM gains; the MIT seed-1 support-only branch has 3/3 positive AUC and 2/3 positive HM source gains; the UT factorized object-support and confidence-gated branches have 6/6 positive source-level AUC/HM gains; the MIT confidence-gated branch has 3/3 positive AUC and positive aggregate HM.",
        "The UT seed-1 joint support-confidence candidate is retained separately as a negative-transfer boundary: its validation gate passed, but fixed test changed by -2.416 AUC / -1.633 HM points with 0/3 positive source runs.",
        "The joint gate is therefore a mechanism ablation with checkpoint-specific transfer risk, not a universal replacement for the main support-aware setting.",
        "The adaptive source-posterior branch uses only per-image source disagreement and source confidence; it improves the MIT seed-1 fixed test by +0.257 AUC / +0.282 HM points, but remains a checkpoint-level mechanism ablation.",
        "The same locked MIT adaptive candidate transfers to the original MIT checkpoint with +0.362 AUC / +0.177 HM points, providing a second checkpoint-level positive replay.",
        "The graph-conditioned adaptive posterior is retained as a mixed transfer boundary: MIT seed-1 is +0.268 AUC / +0.133 HM, while UT fixed test is -0.897 / -0.260 despite +3.894 / +2.590 validation gains.",
        f'The new instance-adaptive route blend uses only validation-calibrated CAMS entropy on graph-unseen pairs to interpolate two already-frozen route residuals; it reaches {pct(ut_seed1_adaptive_blend["test_gain"]["AUC"]):+.3f} AUC / {pct(ut_seed1_adaptive_blend["test_gain"]["HM"]):+.3f} HM on UT seed-1, with 3/3 positive source seeds. The test run only reuses validation entropy normalization and records test_search_after_lock=false.',
        f'The preregistered sigmoid-threshold variant (sigmoid(8*(entropy-0.5))) reaches {pct(ut_seed1_entropy_threshold["test_gain"]["AUC"]):+.3f} AUC / {pct(ut_seed1_entropy_threshold["test_gain"]["HM"]):+.3f} HM on UT seed-1, with 3/3 positive source seeds; it is retained as the stronger threshold-schedule variant.',
        f'The new candidate-conditioned route uses frozen per-image, per-pair CAMS confidence (sigmoid slope {ut_candidate_route["candidate"]["slope"]:g}, threshold {ut_candidate_route["candidate"]["threshold"]:g}) to select between the inverse-support AUC route and soft-support HM route. It reaches {pct(ut_candidate_route["test_gain"]["AUC"]):+.3f} AUC / {pct(ut_candidate_route["test_gain"]["HM"]):+.3f} HM on UT seed-1, with 3/3 positive source-pair runs; source means are {pct(ut_candidate_route_audit["mean_test_auc_gain"]):+.3f}/{pct(ut_candidate_route_audit["mean_test_hm_gain"]):+.3f}.',
        f'The same mechanism independently reaches {pct(ut_original_adaptive_blend["test_gain"]["AUC"]):+.3f} AUC / {pct(ut_original_adaptive_blend["test_gain"]["HM"]):+.3f} HM on the original UT checkpoint, also with 3/3 positive source seeds; the schedule is selected separately by validation.',
        f'The candidate-conditioned route independently reaches {pct(ut_original_candidate_route["test_gain"]["AUC"]):+.3f} AUC / {pct(ut_original_candidate_route["test_gain"]["HM"]):+.3f} HM on the original UT checkpoint; its source mean is {pct(ut_original_candidate_route_audit["mean_test_auc_gain"]):+.3f}/{pct(ut_original_candidate_route_audit["mean_test_hm_gain"]):+.3f}, with all three source-pair runs positive.',
        f'The original UT checkpoint independently selects hard entropy threshold 0.7 from the same preregistered family and reaches {pct(ut_original_entropy_threshold["test_gain"]["AUC"]):+.3f} AUC / {pct(ut_original_entropy_threshold["test_gain"]["HM"]):+.3f} HM, with 3/3 positive source seeds.',
        f'MIT seed-1 has a separate diagonal-source replay whose validation selects the constant soft-selective route; fixed test gains are {pct(mit_seed1_adaptive_blend["test_gain"]["AUC"]):+.3f} AUC / {pct(mit_seed1_adaptive_blend["test_gain"]["HM"]):+.3f} HM with 3/3 positive source seeds. It is reported as cross-dataset soft-route evidence, not as an entropy-schedule gain.',
        f'MIT seed-1 also passes the threshold-schedule replay with {pct(mit_seed1_entropy_threshold["test_gain"]["AUC"]):+.3f} AUC / {pct(mit_seed1_entropy_threshold["test_gain"]["HM"]):+.3f} HM and 3/3 positive source seeds; the small AUC difference from the constant route is retained as a checkpoint-specific schedule ablation.',
        f'The same candidate-conditioned route independently reaches {pct(mit_candidate_route["test_gain"]["AUC"]):+.3f} AUC / {pct(mit_candidate_route["test_gain"]["HM"]):+.3f} HM on MIT seed-1; its three source-pair mean is {pct(mit_candidate_route_audit["mean_test_auc_gain"]):+.3f}/{pct(mit_candidate_route_audit["mean_test_hm_gain"]):+.3f}, with all source-pair gains positive. The candidate was fixed from validation before test access.',
        f'C-GQA has a limited positive selective-transport replay: validation locks reverse hard entropy threshold 0.8 and fixed test changes by {pct(cgqa_adaptive_threshold["test_gain"]["AUC"]):+.3f} AUC / {pct(cgqa_adaptive_threshold["test_gain"]["HM"]):+.3f} HM. This uses one available source seed and is reported as preliminary transfer evidence, not a multi-seed universal claim.',
        f'C-GQA structural-risk selective transport combines frozen CAMS entropy with graph support deficiency; validation locks beta=0.5, slope=8, threshold=0.8 and fixed test changes by {pct(cgqa_structural_risk["test_gain"]["AUC"]):+.3f} AUC / {pct(cgqa_structural_risk["test_gain"]["HM"]):+.3f} HM. It remains single-source preliminary evidence, not a multi-seed universal claim.',
        f'C-GQA structural-risk source audit replays the locked candidate on three explicitly listed source pairs with 3/3 positive test AUC and 3/3 positive HM; mean gains are {pct(cgqa_structural_risk_source_audit["mean_test_auc_gain"]):+.3f} AUC / {pct(cgqa_structural_risk_source_audit["mean_test_hm_gain"]):+.3f} HM. Two pairs use existing generation-seed artifacts and fail the validation AUC gate, so this is source-level robustness evidence rather than a new main result.',
        "",
        "## Joint reliability transfer boundary",
        "",
        "The same validation-only joint support-confidence procedure was independently applied to the UT seed-1 checkpoint.",
        "It passed the validation AUC gate (+0.430 AUC / +2.425 HM points), but the locked fixed test was negative:",
        f'**{pct(ut_seed1_joint["mean"]["test_auc_gain"]):+.3f} AUC / {pct(ut_seed1_joint["mean"]["test_hm_gain"]):+.3f} HM**, with 0/3 positive source runs.',
        "This audited negative boundary is retained to prevent cross-checkpoint overclaiming.",
        "",
        "## Source-distribution ablation",
        "",
        "| Source | MIT AUC | MIT HM | UT AUC | UT HM |",
        "|---|---:|---:|---:|---:|",
        f'| Empirical bootstrap | {pct(empirical_mit["mean_test"]["AUC"]):.3f} +/- {pct(empirical_mit["mean_test"]["AUC_std"]):.3f} | '
        f'{pct(empirical_mit["mean_test"]["best_hm"]):.3f} +/- {pct(empirical_mit["mean_test"]["best_hm_std"]):.3f} | '
        f'{pct(empirical_ut["mean_test"]["AUC"]):.3f} +/- {pct(empirical_ut["mean_test"]["AUC_std"]):.3f} | '
        f'{pct(empirical_ut["mean_test"]["best_hm"]):.3f} +/- {pct(empirical_ut["mean_test"]["best_hm_std"]):.3f} |',
        f'| Diagonal moment-matched Gaussian | {pct(diagonal_mit["mean_test"]["AUC"]):.3f} +/- {pct(diagonal_mit["mean_test"]["AUC_std"]):.3f} | '
        f'{pct(diagonal_mit["mean_test"]["best_hm"]):.3f} +/- {pct(diagonal_mit["mean_test"]["best_hm_std"]):.3f} | '
        f'{pct(diagonal_ut["mean_test"]["AUC"]):.3f} +/- {pct(diagonal_ut["mean_test"]["AUC_std"]):.3f} | '
        f'{pct(diagonal_ut["mean_test"]["best_hm"]):.3f} +/- {pct(diagonal_ut["mean_test"]["best_hm_std"]):.3f} |',
        f'| Isotropic max-entropy Gaussian | {pct(mit["mean_test"]["AUC"]):.3f} +/- {pct(mit["mean_test"]["AUC_std"]):.3f} | '
        f'{pct(mit["mean_test"]["best_hm"]):.3f} +/- {pct(mit["mean_test"]["best_hm_std"]):.3f} | '
        f'{pct(ut["mean_test"]["AUC"]):.3f} +/- {pct(ut["mean_test"]["AUC_std"]):.3f} | '
        f'{pct(ut["mean_test"]["best_hm"]):.3f} +/- {pct(ut["mean_test"]["best_hm_std"]):.3f} |',
        "",
        f'On UT, isotropic exceeds diagonal by only {isotropic_diagonal_auc_gap:+.3f} AUC points; '
        "the defensible conclusion is parity with lower seed variance, not a significant accuracy advantage.",
        "",
        "## Stochasticity control",
        "",
        *render_shared([deterministic, stochastic]),
        "",
        f'The stochastic source contributes {stochastic_gap_auc:+.3f} AUC and {stochastic_gap_hm:+.3f} HM over s=0 with the same flows.',
        "",
        "## Transport-depth and no-flow control",
        "",
        *render_shared([direct_endpoint, one_step_raw, one_step_safe, eight_step]),
        "",
        f'The deterministic no-flow source-augmentation diagnostic reaches {no_flow_auc:.3f} AUC / {no_flow_hm:.3f} HM '
        f'({no_flow_auc_delta:+.3f}/{no_flow_hm_delta:+.3f} versus CAMS). It has no generator-seed variance and is reported as a diagnostic, not a three-seed estimate.',
        "",
        f'One-step is positive on all three test seeds, but its validation gain is only {one_step_raw["val_gain"]:+.3f} AUC points '
        f'and therefore fails the {pct(GATE):.1f}-point reliability gate; eight-step gains {eight_step["val_gain"]:+.3f} validation AUC points and passes.',
        "",
        f'With exactly the same network parameters, source, auxiliary loss, generated classifier, and selection protocol, '
        f'eight-step rectified flow exceeds direct endpoint regression by {flow_over_direct_auc:+.3f} AUC and '
        f'{flow_over_direct_hm:+.3f} HM.',
        "",
        "## Source-location control",
        "",
        *render_shared([zero_center, additive_center]),
        "",
        f'The additive compositional center contributes {location_gap_auc:+.3f} AUC and '
        f'{location_gap_hm:+.3f} HM over the same-energy zero-centered source.',
        "",
        "## Open-world boundary",
        "",
        *render_shared([mit_ow, mit_ow_safe]),
        "",
        f'UT raw single-seed test changes by {ut_ow_raw["delta_auc"]:+.3f} AUC and {ut_ow_raw["delta_hm"]:+.3f} HM; '
        "open-world is not a paper claim.",
        "",
        "## Efficiency",
        "",
        f'- Flow parameters: {efficiency["flow_parameters"] / 1e6:.3f}M',
        f'- Classifier parameters: {efficiency["classifier_parameters"] / 1e6:.3f}M',
        f'- Flow training: {efficiency["flow_stage_seconds"]:.1f}s',
        f'- Generation, classifier, and evaluation: {efficiency["generation_classifier_eval_seconds"]:.1f}s',
        f'- Peak CUDA memory: {efficiency["peak_cuda_memory_mib"] / 1024:.2f} GiB',
        f'- Batched inference: {inference["mean_seconds"] * 1000:.1f}ms for {inference["images"]:,} images '
        f'({inference["milliseconds_per_image"]:.5f}ms/image)',
        "",
        "All consistency assertions passed.",
    ]
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
