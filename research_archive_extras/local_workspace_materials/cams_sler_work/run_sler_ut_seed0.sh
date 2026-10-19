#!/bin/bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES after checking GPU occupancy}"

source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff

cd ./lyt/czsl/CAMS-ADR
out=output/ut_sler_seed0
mkdir -p "$out"

python -u train.py --config configs/utzappos_sler_seed0.yml --save_every_n 999 > "$out/train_closed.log" 2>&1
python -u test.py --config configs/utzappos_sler_seed0.yml --open_world > "$out/test_open.log" 2>&1

