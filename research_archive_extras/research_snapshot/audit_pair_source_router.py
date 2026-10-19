"""Audit train-support pair-level source router provenance."""

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
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--result", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    pp=args.protocol.resolve(); rp=args.result.resolve(); p=json.loads(pp.read_text(encoding="utf-8")); r=json.loads(rp.read_text(encoding="utf-8")); vp=Path(p["validation_probe"]["result"]).resolve(); v=json.loads(vp.read_text(encoding="utf-8"))
    checks={"protocol_status":p.get("status")=="preregistered_before_fixed_test","cache_sha":sha256(Path(p["cache"]).resolve())==p["cache_sha256"],"probe_sha":sha256(vp)==p["validation_probe"]["result_sha256"],"probe_test_free":v.get("test_read") is False,"source_shas":all(sha256(Path(i["path"]).resolve())==i["sha256"] for i in p["sources"]),"candidate_match":r.get("candidate")==p.get("candidate"),"result_test_read":r.get("test_read") is True,"protocol_sha":r.get("protocol_sha256")==sha256(pp),"no_post_test_search":r.get("test_search_after_lock") is False,"val_gate":r.get("val_gain",{}).get("AUC",-1)>=0.002,"test_positive":r.get("test_gain",{}).get("AUC",-1)>0 and r.get("test_gain",{}).get("HM",-1)>0}
    a={"status":"passed" if all(checks.values()) else "failed","outcome":"positive" if checks["test_positive"] else "negative_transfer_boundary","protocol":str(pp),"protocol_sha256":sha256(pp),"result":str(rp),"result_sha256":sha256(rp),"validation_probe":str(vp),"validation_probe_sha256":sha256(vp),"checks":checks}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(a,indent=2),encoding="utf-8"); print(json.dumps(a,indent=2));
    if a["status"]!="passed": raise SystemExit(1)


if __name__ == "__main__": main()
