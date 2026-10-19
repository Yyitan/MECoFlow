"""Audit fixed distribution-ensemble artifacts without rerunning evaluation."""

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


def check(protocol_path: Path, result_path: Path, expected_name: str) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError(f"{protocol_path}: protocol not preregistered")
    candidates = protocol.get("candidate_set")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError(f"{protocol_path}: candidate set is not unique")
    if candidates[0].get("name") != expected_name:
        raise ValueError(f"{protocol_path}: unexpected candidate")
    bound = result.get("protocol", {})
    if bound.get("sha256") != sha256(protocol_path):
        raise ValueError(f"{result_path}: protocol SHA mismatch")
    if bound.get("selection_locked") is not True or bound.get("test_read") is not True:
        raise ValueError(f"{result_path}: fixed-test lock missing")
    if "fused_test" not in result:
        raise ValueError(f"{result_path}: fused test output missing")
    return {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "candidate": candidates[0],
        "metrics": result.get("mean", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ut-protocol", required=True)
    parser.add_argument("--ut-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = {
        "status": "passed",
        "checks": [
            check(Path(args.ut_protocol).resolve(), Path(args.ut_result).resolve(), "diagonal_source_mean_update")
        ],
        "test_search_after_lock": False,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
