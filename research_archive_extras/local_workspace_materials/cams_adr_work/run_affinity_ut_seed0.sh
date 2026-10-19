#!/bin/bash
set -euo pipefail

source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff

cd ./lyt/czsl/CAMS-ADR
mkdir -p output/affinity_sweep
echo "START $(date)" > output/affinity_sweep/master.log

run_one() {
  local tag=$1
  local cfg=$2
  local out=$3

  mkdir -p "$out"
  echo ">> ${tag} TRAIN start $(date)" >> output/affinity_sweep/master.log
  CUDA_VISIBLE_DEVICES=2 python -u train.py --config "$cfg" --save_every_n 999 > "$out/train_closed.log" 2>&1

  echo ">> ${tag} OPEN start $(date)" >> output/affinity_sweep/master.log
  CUDA_VISIBLE_DEVICES=2 python -u test.py --config "$cfg" --open_world > "$out/test_open.log" 2>&1

  echo ">> ${tag} DONE $(date)" >> output/affinity_sweep/master.log
}

run_one "affinity_eps005" "configs/utzappos_affinity_eps005.yml" "output/ut_affinity_eps005_seed0"
run_one "affinity_eps010" "configs/utzappos_affinity_eps010.yml" "output/ut_affinity_eps010_seed0"
run_one "affinity_eps015" "configs/utzappos_affinity_eps015.yml" "output/ut_affinity_eps015_seed0"

echo "ALL DONE $(date)" >> output/affinity_sweep/master.log
