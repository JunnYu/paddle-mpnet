# 介绍
本代码在`训练代码正确，预测错误`的基础上修改得到。
- `weight_decay` 从 `0.01` 变成了 `0.1`。
- 由于paddlenlp的tokenizer在分词的时候存在错误（offset_mapping错误），因此我使用了huggingface的tokenizer来代替！
- 将`train_ds.map`和`dev_ds.map`的方法从batch=True,改成了False，确保数据处理的顺序正确。
- 将所有的num_workers改成0，确保数据处理的顺序正确。


# 训练
```bash
bash train.sh
```

# 查看日志，最好的结果如下（`best_exact`和`best_f1`四舍五入达到论文效果。）：
```python
==================================================
global step 16850, epoch: 4, batch: 425, loss: 0.588575, speed: 0.55 step/s
Saving checkpoint to: squad1.1/model_16850
{
  "exact": 86.84957426679281,
  "f1": 92.82031917884066,
  "total": 10570,
  "HasAns_exact": 86.84957426679281,
  "HasAns_f1": 92.82031917884066,
  "HasAns_total": 10570
}
==================================================
```

# 验证(batch_size=16的时候预测。)
```bash
bash evaluate.sh
```
预期结果如下：
```python
{
  "exact": 86.84957426679281,
  "f1": 92.82031917884066,
  "total": 10570,
  "HasAns_exact": 86.84957426679281,
  "HasAns_f1": 92.82031917884066,
  "HasAns_total": 10570
}
```