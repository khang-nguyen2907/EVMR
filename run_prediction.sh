#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
dt=`date '+%Y%m%d_%H%M%S'`

dataset = "data/val.csv"
saved_model=""
shift
shift
args=$@




echo "***** PREDICTION hyperparameters *****"
echo "dataset: $dataset"
echo "saved_model: $save_model"
echo "******************************"


for seed in 0; do
    python3 -u prediction.py --saved_model $saved_model \
                        --val_dataset $dataset \
    >logs/val_${dt}.log.txt
done