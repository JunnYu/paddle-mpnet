# 介绍
本代码在之前第一个版本的基础上，修改了2个部分
- `weight_decay` 从 `0.01` 变成了 `0.1`。
- 由于paddlenlp的tokenizer在分词的时候存在错误（offset_mapping错误），因此我使用了huggingface的tokenizer来代替！

# 注意
这个代码在预测的部分，处理数据集时候存在不对应的情况，因此该文件夹下的log所得到的结果不是最优的！因此我们需要额外进行预测。
个人分析原因：
    -  `dev_ds.map()`的时候采用了`batched=True`，可能导致了预测的结果不对应。

这个问题，是我通过观测`prediction.json`结果才发现到的(查看log/13000_prediction.json发现，竟然有这么多empty，发现预测出错了。)。
```json
{
    "5737a5931c456719005744ea": "empty",
    "5737a5931c456719005744eb": "empty",
    "5737a7351c456719005744f1": "empty",
    "5737a7351c456719005744f2": "empty",
    "5737a7351c456719005744f3": "empty",
    "5737a7351c456719005744f4": "empty",
    "5737a7351c456719005744f5": "empty",
    "5737a84dc3c5551400e51f59": "empty",
    "5737a84dc3c5551400e51f5a": "empty",
    "5737a84dc3c5551400e51f5b": "empty",
    "5737a84dc3c5551400e51f5c": "empty",
    "5737a9afc3c5551400e51f61": "empty",
    "5737a9afc3c5551400e51f62": "empty",
    "5737a9afc3c5551400e51f63": "empty",
    "5737a9afc3c5551400e51f64": "empty",
    "5737a9afc3c5551400e51f65": "empty",
    "5737aafd1c456719005744fb": "empty",
    "5737aafd1c456719005744fc": "empty",
    "5737aafd1c456719005744fd": "empty",
    "5737aafd1c456719005744fe": "empty",
    "5737aafd1c456719005744ff": "empty"
}
```

# 训练：
```bash
bash train.sh
```

# 验证(batch_size=16的时候预测。)
```bash
bash evaluate.sh
```
预期结果如下：
```python
{
  "exact": 86.90633869441817,
  "f1": 92.77206529725406,
  "total": 10570,
  "HasAns_exact": 86.90633869441817,
  "HasAns_f1": 92.77206529725406,
  "HasAns_total": 10570
}
```
