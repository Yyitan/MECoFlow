"""Audit a preregistered shared-fusion replay and its immutable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validation", "test"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--execution-log", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--generator-results", nargs="+")
    parser.add_argument("--cache")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--negative-test-log")
    parser.add_argument("--min-gate-conflict-log")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    aggregation_path = Path(args.aggregation).resolve()
    execution_log_path = Path(args.execution_log).resolve()
    protocol_record = file_record(protocol_path)
    aggregation_record = file_record(aggregation_path)
    execution_log_record = file_record(execution_log_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    aggregation = json.loads(aggregation_path.read_text(encoding="utf-8"))

    frozen = protocol["frozen_fusion"]
    expected_delta = str(frozen["delta"])
    expected_strength = float(frozen["strength"])
    expected_gain = float(frozen["min_val_auc_gain"])
    expected_candidate = f"{expected_delta}_{expected_strength:.2f}"
    selection = aggregation["selection"]
    source = selection["source"]
    assert source["type"] == "preregistered_protocol"
    assert Path(source["path"]).resolve() == protocol_path
    assert source["sha256"] == protocol_record["sha256"]
    assert source["status"] == protocol["status"]
    assert selection["delta"] == expected_delta
    assert float(selection["raw_strength"]) == expected_strength
    assert float(selection["min_val_auc_gain"]) == expected_gain
    assert set(aggregation["candidates"]) == {expected_candidate}

    cache_record = checkpoint_record = None
    cache_provenance = None
    if (args.cache is None) != (args.base_checkpoint is None):
        parser.error("--cache and --base-checkpoint must be supplied together")
    if args.cache:
        cache_path = Path(args.cache).resolve()
        checkpoint_path = Path(args.base_checkpoint).resolve()
        cache_record = file_record(cache_path)
        checkpoint_record = file_record(checkpoint_path)
        assert selection["input_cache"] == cache_record
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache_provenance = cache["meta"]["feature_cache_provenance"]
        assert cache_provenance["status"] == "amended_reproducible_pipeline"
        assert Path(cache_provenance["checkpoint"]["path"]).resolve() == checkpoint_path
        assert cache_provenance["checkpoint"]["sha256"] == checkpoint_record["sha256"]

    expected_logit_keys = {"val"} if args.mode == "validation" else {"val", "test"}
    logit_records = []
    for item in args.classifier_logits:
        path = Path(item).resolve()
        saved = torch.load(path, map_location="cpu", weights_only=True)
        keys = set(saved)
        assert keys == expected_logit_keys, (path, sorted(keys), sorted(expected_logit_keys))
        record = file_record(path)
        record["keys"] = sorted(keys)
        logit_records.append(record)

    generator_records = []
    for item in args.generator_results or []:
        path = Path(item).resolve()
        generator = json.loads(path.read_text(encoding="utf-8"))
        generator_selection = generator["selection"]
        generator_source = generator_selection["source"]
        assert generator_source["type"] == "preregistered_protocol"
        assert Path(generator_source["path"]).resolve() == protocol_path
        assert generator_source["sha256"] == protocol_record["sha256"]
        assert generator_source["status"] == protocol["status"]
        assert generator_selection["delta"] == expected_delta
        assert float(generator_selection["raw_strength"]) == expected_strength
        assert float(generator_selection["min_val_auc_gain"]) == expected_gain
        variant = generator_selection["variant"]
        assert set(generator[variant]["trials"]) == {expected_candidate}
        if cache_record is not None:
            generator_input = generator["input_provenance"]
            assert generator_input["cache"] == cache_record
            generator_cache_provenance = generator_input["feature_cache_provenance"]
            assert generator_cache_provenance == cache_provenance
        generator_records.append(file_record(path))

    test_fields = {"base_test", "mean_test", "test_runs"}
    if args.mode == "validation":
        assert not (test_fields & set(aggregation))
    else:
        assert bool(selection["enabled"])
        assert float(selection["strength"]) == expected_strength
        assert test_fields <= set(aggregation)

    negative_record = None
    if args.negative_test_log:
        negative_path = Path(args.negative_test_log).resolve()
        negative_text = negative_path.read_text(encoding="utf-8", errors="replace")
        expected_error = (
            "validation-only classifier logits must contain exactly the val key"
        )
        assert expected_error in negative_text
        negative_record = file_record(negative_path)
        negative_record["expected_error"] = expected_error

    min_gate_record = None
    if args.min_gate_conflict_log:
        min_gate_path = Path(args.min_gate_conflict_log).resolve()
        min_gate_text = min_gate_path.read_text(encoding="utf-8", errors="replace")
        expected_error = "--min-val-auc-gain conflicts with fusion protocol"
        assert expected_error in min_gate_text
        min_gate_record = file_record(min_gate_path)
        min_gate_record["expected_error"] = expected_error

    output = {
        "status": "passed",
        "mode": args.mode,
        "protocol_status": protocol["status"],
        "frozen_candidate": expected_candidate,
        "selection_enabled": bool(selection["enabled"]),
        "val_auc_gain": float(selection["val_auc_gain"]),
        "artifacts": {
            "protocol": protocol_record,
            "aggregation": aggregation_record,
            "execution_log": execution_log_record,
            "classifier_logits": logit_records,
            "generator_results": generator_records,
        },
        "invariants": {
            "selection_source": "preregistered_protocol",
            "unique_candidate": True,
            "classifier_logit_keys": sorted(expected_logit_keys),
            "aggregation_contains_test": bool(test_fields & set(aggregation)),
            "generator_results_bound_to_protocol": len(generator_records),
        },
    }
    if negative_record is not None:
        output["artifacts"]["negative_test_log"] = negative_record
    if min_gate_record is not None:
        output["artifacts"]["min_gate_conflict_log"] = min_gate_record
    if cache_record is not None:
        output["artifacts"]["cache"] = cache_record
        output["artifacts"]["base_checkpoint"] = checkpoint_record
        output["invariants"]["cache_status"] = cache_provenance["status"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
