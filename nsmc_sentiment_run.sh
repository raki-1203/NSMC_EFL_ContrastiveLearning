#!/bin/bash

# temperature=0.05
# cl_weight=0

for lr in 0.00001 0.00005
do
for cl_weight in 0.9 0.6 0.3 0.1
do
for temperature in 0.25 0.5
do
python3 train.py \
  --is_train \
  --task sentiment \
  --use_amp \
  --write_summary \
  --device 0 \
  --method efl_scl \
  --output_path ./model/saved_model \
  --model_name_or_path ./model/checkpoint-2000000 \
  --vocab_path ./tokenizer/version_1.9 \
  --path_to_train_data nsmc_dataset_ver1 \
  --max_len 256 \
  --batch_size 32 \
  --lr ${lr} \
  --weight_decay 0.1 \
  --cl_weight ${cl_weight} \
  --epochs 10 \
  --pooler_option cls \
  --eval_steps 1000 \
  --tensorboard_dir tensorboard_logs \
  --warmup_ratio 0.05 \
  --temperature ${temperature} \
  --trial 0 \
  --seed 42
done
done
done
