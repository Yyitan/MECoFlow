"""Audit the protocol, provenance, and leakage invariants of joint-gate tests."""

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


def contains_key(value: object, target: str) -> bool:
    if isinstance(value, dict):
        return any(key == target or contains_key(item, target) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_key(item, target) for item in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-negative", action="store_true", help="audit a valid negative-transfer boundary")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    protocol_path = resolve(args.protocol, root)
    result_path = resolve(args.result, root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation_path = resolve(protocol["validation_gate"]["result"], root)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checks = {
        "protocol_status": protocol.get("status") == "preregistered_before_fixed_test",
        "unique_candidate": len(protocol.get("candidate_set", [])) == 1,
        "selection_rule": "validation AUC gain" in protocol.get("validation_selection_rule", "") and "largest validation HM" in protocol.get("validation_selection_rule", ""),
        "test_rule": "do not search" in protocol.get("test_rule", ""),
        "validation_only": protocol.get("validation_gate", {}).get("test_read") is False,
        "validation_result_test_free": validation.get("test_read") is False and not contains_key(validation, "test"),
        "validation_candidate_match": validation.get("best_balanced") is not None and all(validation["best_balanced"].get(key) == protocol["candidate_set"][0].get(key) for key in protocol["candidate_set"][0]),
        "result_status": result.get("status") == "passed",
        "protocol_sha": result.get("protocol", {}).get("sha256") == sha256(protocol_path),
        "candidate_locked": result.get("protocol", {}).get("candidate") == protocol["candidate_set"][0],
        "selection_locked": result.get("protocol", {}).get("selection_locked") is True,
        "result_test_read": result.get("protocol", {}).get("test_read") is True,
        "no_post_test_search": result.get("test_search_after_lock") is False,
        "cache_sha": sha256(resolve(protocol["cache"], root)) == protocol["cache_sha256"],
        "source_shas": all(sha256(resolve(item["path"], root)) == item["sha256"] for item in protocol["sources"]),
        "result_source_shas": result.get("protocol", {}).get("sources") == protocol.get("sources"),
        "validation_result_exists": validation_path.exists(),
        "individual_count": len(result.get("individual", [])) == len(protocol.get("sources", [])),
        "validation_gain_match": abs(result.get("mean", {}).get("val_auc_gain", 999) - protocol["validation_gate"]["mean_auc_gain"]) < 1e-10 and abs(result.get("mean", {}).get("val_hm_gain", 999) - protocol["validation_gate"]["mean_hm_gain"]) < 1e-10,
        "aggregate_auc_positive": result.get("mean", {}).get("test_auc_gain", 0) > 0 or args.allow_negative,
        "aggregate_hm_positive": result.get("mean", {}).get("test_hm_gain", 0) > 0 or args.allow_negative,
    }
    audit = {
        "status": "passed" if all(checks.values()) else "failed",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "validation_result": str(validation_path),
        "validation_result_sha256": sha256(validation_path),
        "outcome": "positive" if result.get("mean", {}).get("test_auc_gain", 0) > 0 and result.get("mean", {}).get("test_hm_gain", 0) > 0 else "negative_transfer_boundary",
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
