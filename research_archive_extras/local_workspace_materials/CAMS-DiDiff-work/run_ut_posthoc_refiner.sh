#!/usr/bin/env bash
set -euo pipefail

cd ./lyt/czsl/CAMS-DiDiff
source ./anaconda3/bin/activate crdiff

CUDA_VISIBLE_DEVICES=1 python scripts/train_posthoc_refiner.py \
  --config configs/utzappos.yml \
  --dataset_path data/ut-zap50k \
  --cams_checkpoint ./lyt/czsl/CAMS-main/CAMS/output/utzappos_crdiff/val_best.pt \
  --cache_path output/utzappos_posthoc_refiner/cams_cache.pt \
  --save_dir output/utzappos_posthoc_refiner \
  --out_csv experiments/utzappos_posthoc_refiner.csv \
  --epochs 8 \
  --batch_size 256 \
  --lr 0.0001 \
  --weight_decay 0.0001 \
  --residual_reg 0.02 \
  --posthoc_diff_steps 2 \
  --posthoc_train_topk 64 \
  --posthoc_eval_topk 64 \
  --posthoc_step_size 0.5 \
  --posthoc_mix 0.2 \
  --posthoc_zero_mean_delta \
  --posthoc_use_gate \
  --posthoc_delta_center \
  --margin_thresholds none,0.02,0.05,0.1,0.2 \
  2>&1 | tee experiments/utzappos_posthoc_refiner.log
