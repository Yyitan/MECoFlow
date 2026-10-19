"""Audit graph-conditioned adaptive source-posterior provenance and leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(value: str, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def contains_key(value: object, target: str) -> bool:
    if isinstance(value, dict):
        return any(k == target or contains_key(v, target) for k, v in value.items())
    if isinstance(value, list):
        return any(contains_key(v, target) for v in value)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--allow-negative", action="store_true")
    args = ap.parse_args()
    root = Path.cwd().resolve()
    protocol_path = resolve(args.protocol, root)
    result_path = resolve(args.result, root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation_path = resolve(protocol["validation_gate"]["result"], root)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    candidate = protocol.get("candidate_set", [{}])[0]
    source_fields = {"center", "distance", "confidence", "tau", "confidence_power", "delta"}
    candidate_fields = source_fields | {"support", "support_power", "blend", "strength"}
    source_match = all(validation.get("source_posterior", {}).get(k) == candidate.get(k) for k in source_fields)
    support_match = validation.get("best_balanced") is not None and all(validation["best_balanced"].get(k) == candidate.get(k) for k in ("support", "support_power", "blend", "strength"))
    checks = {
        "protocol_status": protocol.get("status") == "preregistered_before_fixed_test",
        "unique_candidate": len(protocol.get("candidate_set", [])) == 1,
        "candidate_keys": set(candidate) == candidate_fields,
        "selection_rule": "validation AUC gain" in protocol.get("validation_selection_rule", "") and "largest validation HM" in protocol.get("validation_selection_rule", ""),
        "test_rule": "do not search" in protocol.get("test_rule", ""),
        "validation_only": protocol.get("validation_gate", {}).get("test_read") is False,
        "validation_result_test_free": validation.get("test_read") is False and not contains_key(validation, "test"),
        "validation_candidate_match": source_match and support_match,
        "result_status": result.get("status") == "passed",
        "protocol_sha": result.get("protocol", {}).get("sha256") == sha256(protocol_path),
        "candidate_locked": result.get("protocol", {}).get("candidate") == candidate,
        "selection_locked": result.get("protocol", {}).get("selection_locked") is True,
        "result_test_read": result.get("protocol", {}).get("test_read") is True,
        "no_post_test_search": result.get("test_search_after_lock") is False,
        "cache_sha": sha256(resolve(protocol["cache"], root)) == protocol["cache_sha256"],
        "source_shas": all(sha256(resolve(x["path"], root)) == x["sha256"] for x in protocol["sources"]),
        "result_source_shas": result.get("protocol", {}).get("sources") == protocol.get("sources"),
        "validation_result_exists": validation_path.exists(),
        "source_count": result.get("source_count") == len(protocol.get("sources", [])),
        "validation_gain_match": abs(result.get("mean", {}).get("val_auc_gain", 999) - protocol["validation_gate"]["mean_auc_gain"]) < 1e-10 and abs(result.get("mean", {}).get("val_hm_gain", 999) - protocol["validation_gate"]["mean_hm_gain"]) < 1e-10,
        "aggregate_auc_positive": result.get("mean", {}).get("test_auc_gain", 0) > 0 or args.allow_negative,
        "aggregate_hm_positive": result.get("mean", {}).get("test_hm_gain", 0) > 0 or args.allow_negative,
    }
    test_positive = result.get("mean", {}).get("test_auc_gain", 0) > 0 and result.get("mean", {}).get("test_hm_gain", 0) > 0
    audit = {"status": "passed" if all(checks.values()) else "failed", "outcome": "positive" if test_positive else "negative_transfer_boundary", "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path), "result": str(result_path), "result_sha256": sha256(result_path), "validation_result": str(validation_path), "validation_result_sha256": sha256(validation_path), "checks": checks, "test_search_after_lock": result.get("test_search_after_lock")}
    out = resolve(args.output, root); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__":
    main()
