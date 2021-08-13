# 介绍
本代码主要将huggingface的tokenizer替换成paddlenlp的tokenizer。现在有关offset偏移的问题已经完全修复了。因此我们可以直接开始验证了！

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
