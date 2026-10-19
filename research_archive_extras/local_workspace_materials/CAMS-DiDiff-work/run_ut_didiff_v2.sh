#!/usr/bin/env bash
set -e

source ./anaconda3/bin/activate crdiff
cd ./lyt/czsl/CAMS-DiDiff
mkdir -p output/utzappos_didiff_v2

CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/utzappos-didiff-v2.yml \
  --dataset_path data/ut-zap50k
