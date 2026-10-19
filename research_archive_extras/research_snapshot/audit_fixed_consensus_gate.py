"""Audit the protocol and SHA chain for the UT consensus-gated test."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    protocol_path, result_path = resolve(args.protocol, root), resolve(args.result, root)
    protocol, result = json.loads(protocol_path.read_text()), json.loads(result_path.read_text())
    candidate = protocol["candidate_set"][0]
    checks = {
        "protocol_status": protocol.get("status") == "preregistered_before_fixed_test",
        "unique_candidate": len(protocol.get("candidate_set", [])) == 1,
        "candidate_locked": candidate.get("name") == "inverse_consensus_gate_update" and float(candidate.get("tau")) == 1.0 and float(candidate.get("strength")) == 0.75,
        "selection_rule_present": "largest validation HM gain" in protocol.get("validation_selection_rule", ""),
        "test_rule_no_search": "do not search gate" in protocol.get("test_rule", ""),
        "validation_only_gate": protocol.get("validation_gate", {}).get("test_read") is False,
        "result_status": result.get("status") == "passed",
        "result_protocol_sha": result.get("protocol", {}).get("sha256") == sha256(protocol_path),
        "result_selection_locked": result.get("protocol", {}).get("selection_locked") is True,
        "result_test_read": result.get("protocol", {}).get("test_read") is True,
        "no_post_test_search": result.get("test_search_after_lock") is False,
        "six_sources": len(protocol.get("sources", [])) == 6 and len(result.get("protocol", {}).get("sources", [])) == 6,
        "validation_result_sha": sha256(resolve(protocol["validation_gate"]["result"], root)) == protocol["validation_gate"]["result_sha256"],
        "cache_sha": sha256(resolve(protocol["cache"], root)) == protocol["cache_sha256"],
        "source_shas": all(sha256(resolve(item["path"], root)) == item["sha256"] for item in protocol["sources"]),
        "individual_runs": len(result.get("individual", [])) == 6,
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
    output.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
