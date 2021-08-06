# mpnet_paddle
mpnet_paddle

使用PaddlePaddle2.x复现MPNet论文


########GLUE



####我的机器再跑！！！！！！！！！
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_proportion 0.06 \
    --num_train_epochs 10 \
    --logging_steps 500 \
    --save_steps 1000 \
    --seed 6666666 \
    --output_dir $TASK_NAME/ \
    --device gpu















#########SQUAD1
############################################### 第一次论文中的参数。

```bash
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --max_seq_length 512 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --logging_steps 200 \
    --save_steps 200 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir squad1.1/ \
    --device gpu \
    --do_train \
    --seed 42 \
    --do_predict
```
global step=21000
{
  "exact": 86.50898770104068,
  "f1": 92.71873272105263,
  "total": 10570,
  "HasAns_exact": 86.50898770104068,
  "HasAns_f1": 92.71873272105263,
  "HasAns_total": 10570
}
http://i-1.gpushare.com:7391/lab/tree/hy-tmp/task/squad1.1
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 第二次尝试！
```bash
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --max_seq_length 512 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --scheduler_type linear \
    --layer_lr_decay 0.8 \
    --logging_steps 500 \
    --save_steps 500 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir squad1.1/ \
    --device gpu \
    --do_train \
    --seed 42 \
    --do_predict 
```
global step 10500
{
  "exact": 86.15894039735099,
  "f1": 92.5312538070981,
  "total": 10570,
  "HasAns_exact": 86.15894039735099,
  "HasAns_f1": 92.5312538070981,
  "HasAns_total": 10570
}


#### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 第三次
```bash  （效果贼垃圾！！！！！！！！！！！！！）
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --max_seq_length 512 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --scheduler_type cosine \
    --layer_lr_decay 0.8 \
    --logging_steps 500 \
    --save_steps 500 \
    --warmup_proportion 0.1 \
    --weight_decay 0.1 \
    --output_dir squad1.1/ \
    --device gpu \
    --do_train \
    --seed 666666 \
    --do_predict 
```



################################################ SQUAD2
############################################### 第一次论文中的参数。
# global step 24000
{
  "exact": 81.78219489598249,
  "f1": 84.88699401134572,
  "total": 11873,
  "HasAns_exact": 78.64372469635627,
  "HasAns_f1": 84.86222670322341,
  "HasAns_total": 5928,
  "NoAns_exact": 84.9116904962153,
  "NoAns_f1": 84.9116904962153,
  "NoAns_total": 5945,
  "best_exact": 82.28754316516466,
  "best_exact_thresh": -1.2772836685180664,
  "best_f1": 85.32220781324624,
  "best_f1_thresh": -1.2772836685180664
}


######################################################  第二次尝试！
```bash
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --max_seq_length 512 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --scheduler_type cosine \
    --layer_lr_decay 0.8 \
    --logging_steps 500 \
    --save_steps 500 \
    --warmup_proportion 0.1 \
    --weight_decay 0.1 \
    --output_dir squad1.1/ \
    --device gpu \
    --do_train \
    --seed 666666 \
    --do_predict \
    --version_2_with_negative
```