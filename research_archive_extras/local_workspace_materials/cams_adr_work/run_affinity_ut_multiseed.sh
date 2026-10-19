#!/bin/bash
set -euo pipefail

source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff

cd ./lyt/czsl/CAMS-ADR
mkdir -p output/affinity_multiseed
progress=output/affinity_multiseed/master.log
echo "START $(date)" > "$progress"

run_one() {
  local seed=$1
  local cfg="configs/utzappos_affinity_eps005_seed${seed}.yml"
  local out="output/ut_affinity_eps005_seed${seed}"

  mkdir -p "$out"
  echo ">> seed${seed} TRAIN start $(date)" >> "$progress"
  CUDA_VISIBLE_DEVICES=2 python -u train.py --config "$cfg" --save_every_n 999 > "$out/train_closed.log" 2>&1

  echo ">> seed${seed} OPEN start $(date)" >> "$progress"
  CUDA_VISIBLE_DEVICES=2 python -u test.py --config "$cfg" --open_world > "$out/test_open.log" 2>&1
  echo ">> seed${seed} DONE $(date)" >> "$progress"
}

run_one 1
run_one 2

echo "ALL DONE $(date)" >> "$progress"
