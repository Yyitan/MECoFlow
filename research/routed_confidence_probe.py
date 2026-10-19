"""Validation-only image-confidence gating of a frozen pair-level source route."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True); parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--iso-logits", nargs=3, required=True); parser.add_argument("--diag-logits", nargs=3, required=True)
    parser.add_argument("--route", required=True); parser.add_argument("--route-power", type=float, required=True); parser.add_argument("--route-strength", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cp = Path(args.cache).resolve(); cache = torch.load(cp, map_location="cpu", weights_only=False); val = cache["val"]
    iso_paths=[Path(x).resolve() for x in args.iso_logits]; diag_paths=[Path(x).resolve() for x in args.diag_logits]
    iso=torch.stack([zscore(torch.load(p,map_location="cpu",weights_only=True)["val"].float()) for p in iso_paths]); diag=torch.stack([zscore(torch.load(p,map_location="cpu",weights_only=True)["val"].float()) for p in diag_paths])
    text=torch.nn.functional.normalize(cache["prompts"]["pair"].float(),dim=-1); text_logits=zscore(val["com"].float() @ text.T); base_logits=val["logits"].float(); base_z=zscore(base_logits)
    routed_score=routed(cache,iso,diag,args.route,args.route_power); routed_delta=routed_score-text_logits
    top2=torch.topk(base_z,k=2,dim=1).values; margin=top2[:,0]-top2[:,1]; probs=torch.softmax(base_z,dim=1); entropy=-(probs*probs.clamp_min(1e-8).log()).sum(1)
    confs={"base_entropy_low":1-normalize(entropy),"base_entropy_high":normalize(entropy),"base_margin_high":normalize(margin),"base_margin_low":1-normalize(margin)}
    evaluator=FastClosedWorldEvaluator(CompositionDataset(args.dataset_path,phase="val"),torch.device("cpu")); base=evaluator(base_logits,val["attr"],val["obj"],val["pair"])
    rows=[]
    for name,raw in confs.items():
        for power in (0.25,0.5,1.0,2.0):
            gate=raw.clamp_min(1e-6).pow(power); gate=gate/gate.mean().clamp_min(1e-6)
            for strength in (0.5,0.75,1.0,1.25,1.5):
                fused=evaluator(base_logits+strength*gate[:,None]*routed_delta,val["attr"],val["obj"],val["pair"])
                rows.append({"confidence":name,"power":power,"strength":strength,"auc_gain":fused["AUC"]-base["AUC"],"hm_gain":fused["best_hm"]-base["best_hm"],"fused_val":fused})
    eligible=[r for r in rows if r["auc_gain"]>=0.002]; best_balanced=max(eligible,key=lambda r:(r["hm_gain"],r["auc_gain"])) if eligible else None; best_auc=max(rows,key=lambda r:(r["auc_gain"],r["hm_gain"]))
    result={"mode":"validation_only_routed_source_confidence","test_read":False,"cache":{"path":str(cp),"sha256":sha256(cp)},"sources":[{"path":str(p),"sha256":sha256(p)} for p in iso_paths+diag_paths],"route":{"name":args.route,"power":args.route_power,"strength":args.route_strength},"base_val":base,"candidate_grid":{"confidence":list(confs),"powers":[0.25,0.5,1.0,2.0],"strengths":[0.5,0.75,1.0,1.25,1.5],"selection":"retain validation AUC gain >= 0.002 and maximize HM"},"best_balanced":best_balanced,"best_auc":best_auc,"rows":rows}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps({"best_balanced":best_balanced,"best_auc":best_auc},indent=2))


if __name__ == "__main__": main()
