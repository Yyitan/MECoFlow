#!/usr/bin/env bash
set -e

source ./anaconda3/bin/activate crdiff
cd ./lyt/czsl/CAMS-DiDiff

CUDA_VISIBLE_DEVICES=1 python scripts/eval_sweep.py \
  --config configs/utzappos-didiff.yml \
  --dataset_path data/ut-zap50k \
  --checkpoint output/utzappos_didiff/val_best.pt \
  --out_csv experiments/utzappos_v1_eval_sweep.csv \
  --steps 0,1,2,4 \
  --topks 64,128 \
  --step_sizes 0.25,0.5
