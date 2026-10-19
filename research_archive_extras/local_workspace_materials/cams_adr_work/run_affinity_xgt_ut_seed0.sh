#!/bin/bash
set -euo pipefail

source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff

cd ./lyt/czsl/CAMS-ADR
mkdir -p output/affinity_xgt_sweep
progress=output/affinity_xgt_sweep/master.log
echo "START $(date)" > "$progress"

run_one() {
  local tag=$1
  local cfg=$2
  local out=$3

  mkdir -p "$out"
  echo ">> ${tag} TRAIN start $(date)" >> "$progress"
  CUDA_VISIBLE_DEVICES=2 python -u train.py --config "$cfg" --save_every_n 999 > "$out/train_closed.log" 2>&1
  echo ">> ${tag} OPEN start $(date)" >> "$progress"
  CUDA_VISIBLE_DEVICES=2 python -u test.py --config "$cfg" --open_world > "$out/test_open.log" 2>&1
  echo ">> ${tag} DONE $(date)" >> "$progress"
}

run_one "xgt_eps0005" "configs/utzappos_affinity_xgt_eps0005.yml" "output/ut_affinity_xgt_eps0005_seed0"
run_one "xgt_eps001" "configs/utzappos_affinity_xgt_eps001.yml" "output/ut_affinity_xgt_eps001_seed0"
run_one "xgt_eps002" "configs/utzappos_affinity_xgt_eps002.yml" "output/ut_affinity_xgt_eps002_seed0"

echo "ALL DONE $(date)" >> "$progress"
