#!/usr/bin/env bash
set -e

source ./anaconda3/bin/activate crdiff
cd ./lyt/czsl/CAMS-DiDiff

CUDA_VISIBLE_DEVICES=1 python scripts/eval_mix_sweep.py \
  --config configs/utzappos-didiff-v2.yml \
  --dataset_path data/ut-zap50k \
  --checkpoint output/utzappos_didiff_v2/val_best.pt \
  --out_csv experiments/utzappos_v2_mix_sweep.csv \
  --mixes 0,0.03,0.05,0.1,0.2,0.3,0.5,1.0 \
  --delta_center
