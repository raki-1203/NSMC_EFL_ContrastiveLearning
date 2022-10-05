python3 train.py \
  --is_train \
  --task sentiment \
  --use_amp \
  --device 0 \
  --method efl_scl \
  --model_name_or_path ./model/checkpoint-2000000 \
  --vocab_path ./tokenizer/version_1.9 \
  --path_to_train_data ./data/preprocessed/sentiment_train_0921.csv \
  --path_to_valid_data ./data/preprocessed/sentiment_valid_0919.csv \
  --output_path ./model/saved_model \
  --max_len 256 \
  --batch_size 96 \
  --accumulation_steps 1 \
  --lr 0.00005 \
  --weight_decay 0.1 \
  --cl_weight 0.9 \
  --epochs 10 \
  --pooler_option cls \
  --eval_steps 100 \
  --tensorboard_dir tensorboard_logs \
  --warmup_ratio 0.05 \
  --temperature 0.05 \
  --trial 0 \
  --seed 26


  # --path_to_train_data ./data/preprocessed/train \
  # --path_to_valid_data ./data/preprocessed/valid \

  # --path_to_train_data ./data/preprocessed/sentiment_train_0921.csv \
  # --path_to_valid_data ./data/preprocessed/sentiment_valid_0919.csv \