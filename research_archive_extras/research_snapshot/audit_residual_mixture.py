"""Audit protocol lock and validation/test separation for residual mixture."""

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


def has_key(value: object, target: str) -> bool:
    if isinstance(value, dict):
        return any(key == target or has_key(item, target) for key, item in value.items())
    if isinstance(value, list):
        return any(has_key(item, target) for item in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    result_path = args.result.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation_path = Path(protocol["validation_probe"]["result"]).resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checks = {
        "protocol_status": protocol.get("status") == "preregistered_before_fixed_test",
        "unique_candidate": abs(sum(protocol["candidate"].values()) - 1.0) < 1e-8,
        "validation_test_free": validation.get("test_read") is False and not has_key(validation, "test"),
        "validation_probe_sha": sha256(validation_path) == protocol["validation_probe"]["result_sha256"],
        "script_sha_recorded": len(protocol["validation_probe"].get("script_sha256", "")) == 64,
        "candidate_match": result.get("candidate") == protocol.get("candidate"),
        "result_test_read": result.get("test_read") is True,
        "protocol_sha": result.get("protocol_sha256") == sha256(protocol_path),
        "cache_sha": sha256(Path(protocol["cache"]).resolve()) == protocol["cache_sha256"],
        "source_shas": all(sha256(Path(item["path"]).resolve()) == item["sha256"] for item in protocol["sources"]),
        "result_source_shas": all(
            left.get("path") == right.get("path") and left.get("sha256") == right.get("sha256")
            for left, right in zip(result.get("sources", []), protocol.get("sources", []))
        ) and len(result.get("sources", [])) == len(protocol.get("sources", [])),
        "validation_auc_gate": result["val_gain"]["AUC"] >= 0.002,
        "test_auc_positive": result["test_gain"]["AUC"] > 0,
        "test_hm_positive": result["test_gain"]["HM"] > 0,
    }
    audit = {
        "status": "passed" if all(checks.values()) else "failed",
        "outcome": "positive" if checks["test_auc_positive"] and checks["test_hm_positive"] else "negative_transfer_boundary",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "validation_probe": str(validation_path),
        "validation_probe_sha256": sha256(validation_path),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
