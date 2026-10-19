#!/bin/bash
source ./anaconda3/etc/profile.d/conda.sh
conda activate crdiff
cd ./lyt/czsl/CAMS-main/CAMS
DATA=./8222/all_data
PROG=output/all3_progress.log
echo "START $(date)" > $PROG
rm -rf output/ut_seed* output/mit_seed* output/cgqa_seed*

run_one() {
  tag=$1; cfg=$2; data=$3; s=$4; gpu=$5
  c=configs/${tag}_seed${s}.yml
  sed -E -e "s/^( *seed:) .*/\1 ${s}/" \
         -e "s|^( *save_path:) .*|\1 output/${tag}_seed${s}|" \
         -e "s/^( *save_final_model:) .*/\1 False/" \
         -e "s|^( *load_model:) .*|\1 output/${tag}_seed${s}/val_best.pt|" \
         configs/${cfg} > $c
  mkdir -p output/${tag}_seed${s}
  echo ">> ${tag} seed${s} TRAIN start $(date)" >> $PROG
  CUDA_VISIBLE_DEVICES=$gpu python -u train.py --config $c --dataset_path $data --save_every_n 999 > output/${tag}_seed${s}/train_closed.log 2>&1
  echo ">> ${tag} seed${s} OPEN start $(date)" >> $PROG
  CUDA_VISIBLE_DEVICES=$gpu python -u test.py --config $c --dataset_path $data --open_world > output/${tag}_seed${s}/test_open.log 2>&1
  rm -f output/${tag}_seed${s}/val_best.pt
  echo ">> ${tag} seed${s} DONE $(date)" >> $PROG
}

run_dataset() {
  tag=$1; cfg=$2; data=$3
  echo "=== DATASET ${tag} START $(date) ===" >> $PROG
  ( run_one $tag $cfg $data 0 0; run_one $tag $cfg $data 3 0 ) &
  ( run_one $tag $cfg $data 1 1; run_one $tag $cfg $data 4 1 ) &
  ( run_one $tag $cfg $data 2 2 ) &
  wait
  echo "=== DATASET ${tag} ALLDONE $(date) ===" >> $PROG
}

run_dataset ut   utzappos.yml   $DATA/ut-zap50k
run_dataset mit  mit-states.yml $DATA/mit-states
run_dataset cgqa cgqa.yml       $DATA/cgqa

echo "ALL3 DONE $(date)" >> $PROG
