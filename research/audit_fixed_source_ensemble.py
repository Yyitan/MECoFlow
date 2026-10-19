"""Audit the protocol and SHA chain for one fixed source-median test."""

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
    checks = {}

    checks["protocol_status"] = protocol.get("status") == "preregistered_before_fixed_test"
    checks["unique_candidate"] = len(protocol.get("candidate_set", [])) == 1
    candidate = protocol["candidate_set"][0]
    checks["candidate_locked"] = candidate.get("name") == "source_median_update" and float(candidate.get("strength")) == 0.75
    checks["test_rule_no_search"] = "do not search strength" in protocol.get("test_rule", "")
    checks["validation_only_gate"] = protocol.get("validation_gate", {}).get("test_read") is False
    checks["result_status"] = result.get("status") == "passed"
    checks["result_protocol_sha"] = result.get("protocol", {}).get("sha256") == sha256(protocol_path)
    checks["result_selection_locked"] = result.get("protocol", {}).get("selection_locked") is True
    checks["result_test_read"] = result.get("protocol", {}).get("test_read") is True
    checks["no_post_test_search"] = result.get("test_search_after_lock") is False
    checks["three_sources"] = len(result.get("protocol", {}).get("sources", [])) == 3
    checks["validation_result_sha"] = sha256(resolve(protocol["validation_gate"]["result"], root)) == protocol["validation_gate"]["result_sha256"]
    checks["cache_sha"] = sha256(resolve(protocol["cache"], root)) == protocol["cache_sha256"]
    checks["source_shas"] = all(
        sha256(resolve(item["path"], root)) == item["sha256"] for item in protocol["sources"]
    )
    checks["result_source_shas"] = all(
        item in result["protocol"]["sources"] for item in [
            {"path": str(resolve(source["path"], root)), "sha256": source["sha256"]}
            for source in protocol["sources"]
        ]
    )
    checks["individual_runs"] = len(result.get("individual", [])) == 3
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
