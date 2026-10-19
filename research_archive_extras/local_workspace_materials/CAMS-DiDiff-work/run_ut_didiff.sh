#!/usr/bin/env bash
set -e

source ./anaconda3/bin/activate crdiff
cd ./lyt/czsl/CAMS-DiDiff

CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/utzappos-didiff.yml \
  --dataset_path data/ut-zap50k
