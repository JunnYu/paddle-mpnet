python train.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --max_seq_length 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --logging_steps 50 \
    --save_steps 250 \
    --warmup_radio 0.1 \
    --weight_decay 0.1 \
    --output_dir outputs/ \
    --seed 42 \
    --num_workers 0 \
    --use_error