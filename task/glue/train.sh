# ["cola","sst-2","mrpc","sts-b","qqp","mnli", "rte", "qnli"]
# 请参考 logs/GLUE/task名字/args.json，然后配置参数！

unset CUDA_VISIBLE_DEVICES

# mnli
export TASK_NAME=mnli
cd task/glue
# 运行训练
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --task_name $TASK_NAM \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 7432 \
    --max_steps 123873 \
    --logging_steps 500 \
    --save_steps 2000 \
    --seed 42 \
    --output_dir $TASK_NAM/ \
    --device gpu