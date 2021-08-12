# 介绍
本代码在之前第一个版本的基础上，修改了2个部分
- `weight_decay` 从 `0.01` 变成了 `0.1`。
- 由于paddlenlp的tokenizer在分词的时候存在错误（offset_mapping错误），因此我使用了huggingface的tokenizer来代替！

# 训练：
```bash
bash train.sh
```

# 查看日志，最好的结果如下（`best_exact`和`best_f1`四舍五入达到论文效果。）：
```python
==================================================
global step 29400, epoch: 4, batch: 4965, loss: 1.035067, speed: 2.51 step/s
Saving checkpoint to: squad2/model_29400
{
  "exact": 82.27912069401162,
  "f1": 85.2774124891565,
  "total": 11873,
  "HasAns_exact": 80.34750337381917,
  "HasAns_f1": 86.35268530427743,
  "HasAns_total": 5928,
  "NoAns_exact": 84.20521446593776,
  "NoAns_f1": 84.20521446593776,
  "NoAns_total": 5945,
  "best_exact": 82.86869367472417,
  "best_exact_thresh": -2.450321674346924,
  "best_f1": 85.67634263296013,
  "best_f1_thresh": -2.450321674346924
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
  "exact": 82.27912069401162,
  "f1": 85.2774124891565,
  "total": 11873,
  "HasAns_exact": 80.34750337381917,
  "HasAns_f1": 86.35268530427743,
  "HasAns_total": 5928,
  "NoAns_exact": 84.20521446593776,
  "NoAns_f1": 84.20521446593776,
  "NoAns_total": 5945,
  "best_exact": 82.86869367472417,
  "best_exact_thresh": -2.450321674346924,
  "best_f1": 85.67634263296013,
  "best_f1_thresh": -2.450321674346924
}
```

# Tips:
发现有趣的现象,batch_size=32的时候预测出下面这个结果- -|。效果降低了，应该跟paddding有关。
```bash
python evaluate.py --model_name_or_path ./best-model_29400 --version_2_with_negative --batch_size 32
```
```python
{
  "exact": 82.27912069401162,
  "f1": 85.27067451223407,
  "total": 11873,
  "HasAns_exact": 80.36437246963563,
  "HasAns_f1": 86.35605912344072,
  "HasAns_total": 5928,
  "NoAns_exact": 84.18839360807401,
  "NoAns_f1": 84.18839360807401,
  "NoAns_total": 5945,
  "best_exact": 82.86869367472417,
  "best_exact_thresh": -2.461650848388672,
  "best_f1": 85.66960465603769,
  "best_f1_thresh": -2.461650848388672
}
```