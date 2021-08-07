# 使用MPNet复现论文：MPNet: Masked and Permuted Pre-training for Language Understanding

## MPNet

[MPNet: Masked and Permuted Pre-training for Language Understanding - Microsoft Research](https://www.microsoft.com/en-us/research/publication/mpnet-masked-and-permuted-pre-training-for-language-understanding/)

**Abstract：**
BERT adopts masked language modeling (MLM) for pre-training and is one of the most successful pre-training models. Since BERT neglects dependency among predicted tokens, XLNet introduces permuted language modeling (PLM) for pretraining to address this problem. However, XLNet does not leverage the full position information of a sentence and thus suffers from position discrepancy between pre-training and fine-tuning. In this paper, we propose MPNet, a novel pre-training method that inherits the advantages of BERT and XLNet and avoids their limitations. MPNet leverages the dependency among predicted tokens through permuted language modeling (vs. MLM in BERT), and takes auxiliary position information as input to make the model see a full sentence and thus reducing the position discrepancy (vs. PLM in XLNet). We pre-train MPNet on a large-scale dataset (over 160GB text corpora) and fine-tune on a variety of down-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNet outperforms MLM and PLM by a large margin, and achieves better results on these tasks compared with previous state-of-the-art pre-trained methods (e.g., BERT, XLNet, RoBERTa) under the same model setting. The code and the pre-trained models are available at: https://github.com/microsoft/MPNet.

本项目是 MPNet在 Paddle 2.x上的开源实现。

## 原论文效果
<p align="center">
    <img src="figure/QQP.png" width="100%" />
</p>
<p align="center">
    <img src="figure/SQuAD.png" width="100%" />
</p>


## 环境安装

| 名称   | 值             |
|--------|------------------|
| python | 3\.8             |
| GPU    | RTX3090          |
| 框架    | PaddlePaddle2\.1 |
| Cuda   | 11\.2            |
| Cudnn  | 8\.1\.1\.33\-1   |

或者使用本次复现使用的云平台：https://gpushare.com/
<p align="center">
    <img src="figure/yunfuwuqi.jpg" width="100%" />
</p>

```bash
# 克隆本仓库
git clone https://github.com/JunnYu/paddle_mpnet
# 进入paddlenlp目录
cd paddlenlp
# 本地安装
pip install -r requirements.txt
pip install -e .
# 返回初始目录
cd ..
```

## 快速开始

### （一）模型精度对齐
运行`python compare.py`，对比huggingface与paddle之间的精度，我们可以发现精度的平均误差在10^-6量级，最大误差在10^-5量级（更换不同的输入，误差会发生变化）。
```python
python compare.py
# meandif tensor(6.5154e-06)
# maxdif tensor(4.1485e-05)
```
### (二) 模型转换 or 下载
运行`python convert.py`,可将huggingface版本权重转化为padddle权重。（注意预先下载好hg的权重放入weights/hg/mpnet-base文件夹）
```python
python convert.py
# 会有如下的输出
# Converting: mpnet.embeddings.word_embeddings.weight => mpnet.embeddings.word_embeddings.weight | is_transpose False
# Converting: mpnet.embeddings.position_embeddings.weight => mpnet.embeddings.position_embeddings.weight | is_transpose False
# Converting: mpnet.encoder.relative_attention_bias.weight => mpnet.encoder.relative_attention_bias.weight | is_transpose False
# Converting: mpnet.embeddings.LayerNorm.weight => mpnet.embeddings.layer_norm.weight | is_transpose False
# Converting: mpnet.embeddings.LayerNorm.bias => mpnet.embeddings.layer_norm.bias | is_transpose False
# Converting: mpnet.encoder.layer.0.attention.attn.q.weight => mpnet.encoder.layer.0.attention.q.weight | is_transpose True
# Converting: mpnet.encoder.layer.0.attention.attn.q.bias => mpnet.encoder.layer.0.attention.q.bias | is_transpose False
# Converting: mpnet.encoder.layer.0.attention.attn.k.weight => mpnet.encoder.layer.0.attention.k.weight | is_transpose True
# Converting: mpnet.encoder.layer.0.attention.attn.k.bias => mpnet.encoder.layer.0.attention.k.bias | is_transpose False
# Converting: mpnet.encoder.layer.0.attention.attn.v.weight => mpnet.encoder.layer.0.attention.v.weight | is_transpose True
# Converting: mpnet.encoder.layer.0.attention.attn.v.bias => mpnet.encoder.layer.0.attention.v.bias | is_transpose False
# Converting: mpnet.encoder.layer.0.attention.attn.o.weight => mpnet.encoder.layer.0.attention.o.weight | is_transpose True
# Converting: mpnet.encoder.layer.0.attention.attn.o.bias => mpnet.encoder.layer.0.attention.o.bias | is_transpose False
# Converting: mpnet.encoder.layer.0.attention.LayerNorm.weight => mpnet.encoder.layer.0.attention.layer_norm.weight | is_transpose False
# Converting: mpnet.encoder.layer.0.attention.LayerNorm.bias => mpnet.encoder.layer.0.attention.layer_norm.bias | is_transpose False
# Converting: mpnet.encoder.layer.0.intermediate.dense.weight => mpnet.encoder.layer.0.ffn.weight | is_transpose True
# Converting: mpnet.encoder.layer.0.intermediate.dense.bias => mpnet.encoder.layer.0.ffn.bias | is_transpose False
# Converting: mpnet.encoder.layer.0.output.dense.weight => mpnet.encoder.layer.0.ffn_output.weight | is_transpose True
# Converting: mpnet.encoder.layer.0.output.dense.bias => mpnet.encoder.layer.0.ffn_output.bias | is_transpose False
# Converting: mpnet.encoder.layer.0.output.LayerNorm.weight => mpnet.encoder.layer.0.layer_norm.weight | is_transpose False
# Converting: mpnet.encoder.layer.0.output.LayerNorm.bias => mpnet.encoder.layer.0.layer_norm.bias | is_transpose False
# ......
# Converting: lm_head.dense.weight => lm_head.dense.weight | is_transpose True
# Converting: lm_head.dense.bias => lm_head.dense.bias | is_transpose False
# Converting: lm_head.decoder.bias => lm_head.decoder_bias | is_transpose False
# Converting: lm_head.layer_norm.weight => lm_head.layer_norm.weight | is_transpose False
# Converting: lm_head.layer_norm.bias => lm_head.layer_norm.bias | is_transpose False
```

转换好的模型链接：https://huggingface.co/junnyu/mpnet/tree/main/mpnet-base （记得把tokenizer_config.json也下载，不然本地调用tokenizer时候会报错缺少这个）


### （三）下游任务微调

#### 1、GLUE
以QQP数据集为例

- （对于其他GLUE任务，请参考logs/GLUE/对应task_name/args.json，该json有详细参数配置）

- （超参数遵循原论文的仓库 https://github.com/microsoft/MPNet/blob/master/MPNet/README.glue.md）

- (只有QQP数据集指定了warmup_steps和max_steps，而其他GLUE任务指定了训练的epoch和warmup的比率，查看args.json时候请注意)

##### （1）模型微调：
```shell
unset CUDA_VISIBLE_DEVICES
# 确保处在task/glue文件夹
cd task/glue
# 运行训练(其他的命令查看task/glue/train.sh)
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --task_name qqp \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 1e-5 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --weight_decay 0.1 \
    --warmup_steps 5666 \
    --max_steps 113272 \
    --logging_steps 500 \
    --save_steps 2000 \
    --seed 42 \
    --output_dir qqp/ \
    --device gpu
```
其中参数释义如下：
- `model_type` 指示了模型类型，当前支持BERT、ELECTRA、ERNIE、CONVBERT、MPNET模型。
- `model_name_or_path` 模型名称或者路径，其中mpnet模型当前仅支持mpnet-base几种规格。
- `task_name` 表示 Fine-tuning 的任务，当前支持CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE、 WNLI。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `scheduler_type` scheduler类型，可选linear和cosine
- `layer_lr_decay` 层学习率衰减，1.0表示不使用衰减。
- `warmup_steps` warmup步数。
- `max_steps` 表示最大训练步数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。

**模型链接**(这个链接包含所有GLUE任务微调后的权重)

链接：TODO
提取码：TODO

##### （2）模型预测：
```bash
# 确保处在glue文件夹
cd task/glue
# 运行预测，请指定模型权重文件夹
​```python
python run_predict.py --task_name qqp  --ckpt_path qqp/best-qqp_ft_model_106000.pdparams
```
# 完成后可以压缩template文件夹，然后提交到GLUE

##### （3）压缩template文件夹为zip文件，然后提交到[GLUE排行榜](https://gluebenchmark.com/leaderboard)：
###### GLUE排行榜结果：
<p align="center">
    <img src="figure/glue.jpg" width="100%" />
</p>


###### GLUE开发集结果：

| task                      | cola  | sst-2  | mrpc        | sts-b             | qqp         | mnli       | qnli | rte   | avg |
|--------------------------------|-------|-------|-------------|------------------|-------------|------|-------|-------|-------|
| **metric** | **mcc** | **acc** | **acc/f1** | **pearson/spearman** | **acc/f1**  | **acc(m/mm)**  | **acc** | **acc** |    |
| Paper | **65.0** | **95.5** | **91.8**/空 | 91.1/空 | **91.9**/空 | **88.5**/空 | 93.3 | 85.8 | **87.9** |
| Mine | 64.4 | 95.4 | 90.4/93.1 | **91.6**/91.3 | **91.9**/89.0 | 87.7/88.2 | **93.6** | **86.6** | 87.7 |

###### GLUE测试集结果对比：

| task                      | cola  | sst-2  | mrpc  | sts-b  | qqp | mnli-m | qnli  | rte   | avg      |
|--------------------------------|-------|-------|-------|-------|-----|-------|-------|-------|----------|
| **metric** | **mcc** | **acc** | **acc/f1** | **pearson/spearman** | **acc/f1**  | **acc(m/mm)**  | **acc** | **acc** |  |
| Paper | **64.0** | **96.0** | 89.1/空 | 90.7/空 | **89.9**/空 | **88\.5**/空 | 93\.1 | 81.0 | **86.5** |
| Mine | 60.5     | 95.9 | **91.6**/88.9 | **90.8**/90.3 | 89.7/72.5 | 87.6/86.6 | **93.3** | **82.4** | **86.5** |

#### 2、SQuAD v1.1

使用Paddle提供的预训练模型运行SQuAD v1.1数据集的Fine-tuning

```shell
unset CUDA_VISIBLE_DEVICES
# 确保处在squad1.1文件夹
cd task/squad1.1
# 开始训练
​```bash
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

训练过程中模型会自动对结果进行评估，其中最好的结果如下所示：（详细训练可查看logs文件夹）

```python
# global step=21000
{
  "exact": 86.50898770104068,
  "f1": 92.71873272105263,
  "total": 10570,
  "HasAns_exact": 86.50898770104068,
  "HasAns_f1": 92.71873272105263,
  "HasAns_total": 10570
}
```

##### 模型链接

链接：TODO
提取码：TODO

#### 3、SQuAD v2.0
对于 SQuAD v2.0,按如下方式启动 Fine-tuning:

```shell
unset CUDA_VISIBLE_DEVICES
# 确保处在squad2.0文件夹
cd task/squad2.0
# 开始训练
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type mpnet \
    --model_name_or_path mpnet-base \
    --max_seq_length 512 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --scheduler_type linear \
    --layer_lr_decay 1.0 \
    --logging_steps 500 \
    --save_steps 500 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir squad2/ \
    --device gpu \
    --do_train \
    --seed 42 \
    --do_predict \
    --version_2_with_negative
```

* `version_2_with_negative`: 使用squad2.0数据集和评价指标的标志。

训练过程中模型会自动对结果进行评估，其中最好的结果如下所示：（详细训练可查看logs文件夹）

```python
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
```

##### 模型链接

链接：TODO
提取码：TODO

# Tips:
- 对于SQUAD任务：根据这个[issues](https://github.com/microsoft/MPNet/issues/3)所说,论文中汇报的是`"best_exact"`和`"best_f1"`。
- 对于GLUE任务：根据这个[issues](https://github.com/microsoft/MPNet/issues/7)所说，部分任务使用热启动。

# Reference

```bibtex
@article{song2020mpnet,
    title={MPNet: Masked and Permuted Pre-training for Language Understanding},
    author={Song, Kaitao and Tan, Xu and Qin, Tao and Lu, Jianfeng and Liu, Tie-Yan},
    journal={arXiv preprint arXiv:2004.09297},
    year={2020}
}
```