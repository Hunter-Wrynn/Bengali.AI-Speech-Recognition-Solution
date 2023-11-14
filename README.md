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
