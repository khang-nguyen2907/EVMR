#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
dt=`date '+%Y%m%d_%H%M%S'`

dataset = "data"
model=""
pooling_mode = "mean"
shift
shift
args=$@

lr = "1e-4"
bs = 4
epoch_nums = 5


echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: $lr"
echo "******************************"

output_dir = "saved_model"
log_dir = "logs"
mkdir -p $output_dir
mkdir -p $log_dir

for seed in 0; do
    python3 -u train.py --output_dir $output_dir \
                        --saved_model None \
                        --log_dir = $log_dir
                        --train_dataset $dataset/train.csv \
                        --val_dataset $dataset/test.csv \
                        --batch_size $bs \
                        --num_proc 4 \
                        --model_name $model \
                        --pooling_mode $pooling_mode \
                        --gradient_accumulation_steps 2 \
                        --epoch_nums $epoch_nums \
                        --fp16 True \
                        --save_steps 2 \
                        --learning_rate $lr \
    >logs/train_${dt}.log.txt
done