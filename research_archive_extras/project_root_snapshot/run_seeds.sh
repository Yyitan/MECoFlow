#!/bin/bash
source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff
cd ./lyt/czsl/CAMS-main/CAMS
DATA=data/ut-zap50k
PROG=output/seeds_progress.log
echo "START $(date)" > $PROG

run_one() {
  s=$1; gpu=$2
  cfg=configs/ut_seed${s}.yml
  sed -e "s/seed: 0/seed: ${s}/" \
      -e "s|save_path: output/utzappos_crdiff|save_path: output/ut_seed${s}|" \
      -e "s/save_final_model: True/save_final_model: False/" \
      -e "s|load_model: weight/final_model.pt|load_model: output/ut_seed${s}/val_best.pt|" \
      configs/utzappos.yml > $cfg
  mkdir -p output/ut_seed${s}
  echo ">> seed ${s} TRAIN start $(date)" >> $PROG
  CUDA_VISIBLE_DEVICES=$gpu python -u train.py --config $cfg --dataset_path $DATA --save_every_n 999 > output/ut_seed${s}/train_closed.log 2>&1
  echo ">> seed ${s} OPEN start $(date)" >> $PROG
  CUDA_VISIBLE_DEVICES=$gpu python -u test.py --config $cfg --dataset_path $DATA --open_world > output/ut_seed${s}/test_open.log 2>&1
  rm -f output/ut_seed${s}/val_best.pt
  echo ">> seed ${s} DONE $(date)" >> $PROG
}

( run_one 0 0; run_one 4 0 ) &
run_one 1 1 &
run_one 2 2 &
run_one 3 3 &
wait
echo "ALL DONE $(date)" >> $PROG
