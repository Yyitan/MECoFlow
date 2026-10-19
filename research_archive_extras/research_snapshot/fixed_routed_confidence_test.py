"""Protocol-locked fixed test for routed source plus image confidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402
from research.fixed_pair_source_router_test import routed  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True); parser.add_argument("--dataset-path", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    pp = Path(args.protocol).resolve(); p = json.loads(pp.read_text(encoding="utf-8"));
    if p.get("status") != "preregistered_before_fixed_test": raise ValueError("protocol not preregistered")
    cp = Path(p["cache"]).resolve(); vp = Path(p["validation_probe"]["result"]).resolve()
    if sha256(cp) != p["cache_sha256"] or sha256(vp) != p["validation_probe"]["result_sha256"]: raise ValueError("cache/probe SHA mismatch")
    if json.loads(vp.read_text(encoding="utf-8")).get("test_read") is not False: raise ValueError("validation probe not test-free")
    paths=[Path(i["path"]).resolve() for i in p["sources"]]
    for item,path in zip(p["sources"],paths):
        if sha256(path)!=item["sha256"]: raise ValueError(f"source SHA mismatch: {path}")
    c=torch.load(cp,map_location="cpu",weights_only=False); val=c["val"]; test=c["test"]; saved=[torch.load(path,map_location="cpu",weights_only=True) for path in paths]; n=int(p["family_size"])
    iso_val=torch.stack([zscore(x["val"].float()) for x in saved[:n]]); diag_val=torch.stack([zscore(x["val"].float()) for x in saved[n:]])
    iso_test=torch.stack([zscore(x["test"].float()) for x in saved[:n]]); diag_test=torch.stack([zscore(x["test"].float()) for x in saved[n:]])
    text=torch.nn.functional.normalize(c["prompts"]["pair"].float(),dim=-1); val_text=zscore(val["com"].float()@text.T); test_text=zscore(test["com"].float()@text.T)
    val_score=routed(c,iso_val,diag_val,p["route"]["name"],float(p["route"]["power"])); test_score=routed(c,iso_test,diag_test,p["route"]["name"],float(p["route"]["power"]))
    val_delta=val_score-val_text; test_delta=test_score-test_text
    def gate(logits: torch.Tensor) -> torch.Tensor:
        top2=torch.topk(zscore(logits.float()),k=2,dim=1).values; margin=top2[:,0]-top2[:,1]; g=normalize(margin).clamp_min(1e-6).pow(float(p["confidence"]["power"])); return g/g.mean().clamp_min(1e-6)
    val_gate=gate(val["logits"]); test_gate=gate(test["logits"]); strength=float(p["candidate"]["strength"])
    ev=FastClosedWorldEvaluator(CompositionDataset(args.dataset_path,phase="val"),torch.device("cpu")); et=FastClosedWorldEvaluator(CompositionDataset(args.dataset_path,phase="test"),torch.device("cpu"))
    bv=ev(val["logits"].float(),val["attr"],val["obj"],val["pair"]); fv=ev(val["logits"].float()+strength*val_gate[:,None]*val_delta,val["attr"],val["obj"],val["pair"]); bt=et(test["logits"].float(),test["attr"],test["obj"],test["pair"]); ft=et(test["logits"].float()+strength*test_gate[:,None]*test_delta,test["attr"],test["obj"],test["pair"])
    r={"status":"passed","test_read":True,"protocol":str(pp),"protocol_sha256":sha256(pp),"route":p["route"],"confidence":p["confidence"],"candidate":p["candidate"],"base_val":bv,"fused_val":fv,"val_gain":{"AUC":fv["AUC"]-bv["AUC"],"HM":fv["best_hm"]-bv["best_hm"]},"base_test":bt,"fused_test":ft,"test_gain":{"AUC":ft["AUC"]-bt["AUC"],"HM":ft["best_hm"]-bt["best_hm"]},"sources":[{"path":str(x),"sha256":sha256(x)} for x in paths],"test_search_after_lock":False}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(r,indent=2),encoding="utf-8"); print(json.dumps({"val_gain":r["val_gain"],"test_gain":r["test_gain"]},indent=2))


if __name__ == "__main__": main()
