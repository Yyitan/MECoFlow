"""Audit the independent UT seed-1 fixed diagonal ensemble."""

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
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    result_path = Path(args.result).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    frozen = protocol.get("frozen_fusion", {})
    bound = result.get("protocol", {})
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol status mismatch")
    if frozen != bound.get("frozen_fusion"):
        raise ValueError("frozen fusion mismatch")
    if bound.get("sha256") != sha256(protocol_path):
        raise ValueError("protocol SHA mismatch")
    if result.get("test_search_after_lock") is not False:
        raise ValueError("post-test search flag missing")
    if len(result.get("individual", [])) != 3:
        raise ValueError("expected three generator records")
    output = {
        "status": "passed",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "metrics": result.get("mean", {}),
        "test_search_after_lock": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
