"""Verify validation logits are bitwise unchanged by test-enabled replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-logits", nargs="+", required=True)
    parser.add_argument("--test-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.validation_logits) != len(args.test_logits):
        raise ValueError("validation/test replay counts differ")
    rows = []
    for validation_path, test_path in zip(args.validation_logits, args.test_logits):
        va = torch.load(validation_path, map_location="cpu", weights_only=True)
        tb = torch.load(test_path, map_location="cpu", weights_only=True)
        if set(va) != {"val"} or set(tb) != {"val", "test"}:
            raise ValueError("unexpected logits key set")
        if va["val"].shape != tb["val"].shape:
            raise ValueError("validation shape changed")
        diff = (va["val"].float() - tb["val"].float()).abs()
        rows.append({
            "validation": str(Path(validation_path).resolve()),
            "validation_sha256": sha256(Path(validation_path).resolve()),
            "test_replay": str(Path(test_path).resolve()),
            "test_replay_sha256": sha256(Path(test_path).resolve()),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "exact_equal": bool(torch.equal(va["val"], tb["val"])),
        })
    output = {
        "status": "passed" if all(r["exact_equal"] for r in rows) else "failed",
        "rows": rows,
        "test_selection_changed_validation": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
