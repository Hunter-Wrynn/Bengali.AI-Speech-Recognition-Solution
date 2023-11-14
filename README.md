# Bengali.AI-Speech-Recognition-Solution

dataset API command -- kaggle competitions download -c bengaliai-speech

# data_augmentation.py
!python preprocess.py \
  --model_name_or_path "openai/whisper-small" \
  --language "Bengali" \
  --output_dir "75k-samples" \
  --preprocessing_num_workers 90 \
  --preprocessing_only \
  --text_column_name "sentence" \
  --data_dir "data" \
  --min_duration_in_seconds 2 \
  --max_duration_in_seconds 30 \
  --max_train_samples 75000 \
  --max_eval_samples 5000 \
  --apply_spec_augment

  #train.py
  data_dir = "/kaggle/input/bengali-ai-asr-10k"

!torchrun --nproc_per_node 2 train.py \
 --model_name_or_path "bangla-speech-processing/BanglaASR" \
 --train_data_dir $data_dir \
 --validation_data_dir $data_dir \
 --language "Bengali" \
 --output_dir "whisper-base-bn" \
 --do_train \
 --do_eval \
 --fp16 \
 --group_by_length \
 --predict_with_generate \
 --dataloader_num_workers 1 \
 --overwrite_output_dir \
 --per_device_train_batch_size 4 \
 --length_column_name "input_length" \
 --report_to "none" \
 --metric_for_best_model "wer" \
 --greater_is_better False \
 --evaluation_strategy "epoch" \
 --save_strategy "epoch" \
 --save_total_limit 1 \
 --logging_steps 10 \
 --gradient_checkpointing \
 --warmup_steps 50 \
 --apply_spec_augment True \
 --num_train_epochs 3 \
 --learning_rate "1e-5"
