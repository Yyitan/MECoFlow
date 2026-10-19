"""Audit provenance and validation lock for interaction hyper-classifier."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(); result_path = args.result.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); result = json.loads(result_path.read_text(encoding="utf-8"))
    validation_path = Path(protocol["validation_result"]).resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checks = {
        "protocol_status": protocol.get("status") == "preregistered_before_fixed_test",
        "cache_sha": sha256(Path(protocol["cache"]).resolve()) == protocol["cache_sha256"],
        "model_sha": sha256(Path(protocol["model"]).resolve()) == protocol["model_sha256"],
        "validation_sha": sha256(validation_path) == protocol["validation_result_sha256"],
        "validation_test_free": validation.get("test_read") is False,
        "candidate_match": result.get("candidate") == protocol.get("candidate"),
        "result_test_read": result.get("test_read") is True,
        "protocol_sha": result.get("protocol_sha256") == sha256(protocol_path),
        "selection_locked": result.get("test_search_after_lock") is False,
        "val_gate": result.get("val_gain", {}).get("AUC", -1.0) >= 0.002,
        "test_positive": result.get("test_gain", {}).get("AUC", -1.0) > 0 and result.get("test_gain", {}).get("HM", -1.0) > 0,
    }
    audit = {"status": "passed" if all(checks.values()) else "failed", "outcome": "positive" if checks["test_positive"] else "negative_transfer_boundary", "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path), "result": str(result_path), "result_sha256": sha256(result_path), "validation_result": str(validation_path), "validation_result_sha256": sha256(validation_path), "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__":
    main()
