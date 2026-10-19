"""Audit the protocol/SHA chain for support-aware consensus fixed test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    protocol_path = resolve(args.protocol, root)
    result_path = resolve(args.result, root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = protocol.get("candidate_set", [{}])[0]
    candidate_signatures = {
        ("support_aware_consensus_update", "median", "harmonic_image_support", "inverse", 2.0, 0.25, "linear", 0.75),
        ("support_aware_support_only_update", "median", "min_image_support", "none", 0.0, 0.5, "linear", 0.75),
        ("support_aware_support_only_update", "median", "min_image_support", "none", 0.0, 1.0, "linear", 0.5),
        ("support_aware_support_only_update", "mean", "object_image_support", "none", 0.0, 1.0, "support_only", 1.0),
    }
    candidate_signature = (
        candidate.get("name"), candidate.get("aggregation", "median"), candidate.get("support_mode"), candidate.get("consensus"),
        float(candidate.get("tau")), float(candidate.get("support_exponent")),
        candidate.get("blend"), float(candidate.get("strength")),
    )
    source_count = len(protocol.get("sources", []))
    checks = {
        "protocol_status": protocol.get("status") == "preregistered_before_fixed_test",
        "unique_candidate": len(protocol.get("candidate_set", [])) == 1,
        "candidate_locked": candidate_signature in candidate_signatures,
        "selection_rule_present": (
            ("support_exponent > 0" in protocol.get("validation_selection_rule", "")
             and "largest validation HM gain" in protocol.get("validation_selection_rule", ""))
            or ("maximize validation HM" in protocol.get("validation_selection_rule", ""))
        ),
        "test_rule_no_search": "do not search" in protocol.get("test_rule", ""),
        "validation_only_gate": protocol.get("validation_gate", {}).get("test_read") is False,
        "result_status": result.get("status") == "passed",
        "result_protocol_sha": result.get("protocol", {}).get("sha256") == sha256(protocol_path),
        "result_selection_locked": result.get("protocol", {}).get("selection_locked") is True,
        "result_test_read": result.get("protocol", {}).get("test_read") is True,
        "no_post_test_search": result.get("test_search_after_lock") is False,
        "source_count": source_count in (3, 6) and len(result.get("protocol", {}).get("sources", [])) == source_count,
        "validation_result_sha": sha256(resolve(protocol["validation_gate"]["result"], root)) == protocol["validation_gate"]["result_sha256"],
        "validation_script_sha": sha256(resolve(protocol["validation_gate"]["script"], root)) == protocol["validation_gate"]["script_sha256"],
        "cache_sha": sha256(resolve(protocol["cache"], root)) == protocol["cache_sha256"],
        "source_shas": all(sha256(resolve(item["path"], root)) == item["sha256"] for item in protocol["sources"]),
        "validation_source_shas": all(sha256(resolve(item["path"], root)) == item["sha256"] for item in protocol.get("validation_sources", [])),
        "individual_runs": len(result.get("individual", [])) == source_count,
        "all_individual_auc_positive": all(row.get("test_auc_gain", 0) > 0 for row in result.get("individual", [])),
        "aggregate_hm_positive": result.get("mean", {}).get("test_hm_gain", 0) > 0,
        "individual_hm_count_recorded": result.get("mean", {}).get("positive_individual_test_hm_runs") == sum(row.get("test_hm_gain", 0) > 0 for row in result.get("individual", [])),
    }
    audit = {
        "status": "passed" if all(checks.values()) else "failed",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "checks": checks,
        "test_search_after_lock": result.get("test_search_after_lock"),
    }
    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
