#!/usr/bin/env bash
set -e

cd ./lyt/czsl/CAMS-DiDiff

CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/mit-states-didiff.yml \
  --dataset_path data/mit-states
