# 介绍
本代码主要将huggingface的tokenizer替换成paddlenlp的tokenizer。现在有关offset偏移的问题已经完全修复了。因此我们可以直接开始验证了！



# 验证(batch_size=16的时候预测。)
```bash
bash evaluate.sh
```
预期结果如下：
```python
{
  "exact": 86.89687795648061,
  "f1": 92.88753764866149,
  "total": 10570,
  "HasAns_exact": 86.89687795648061,
  "HasAns_f1": 92.88753764866149,
  "HasAns_total": 10570
}
```

# 注意：
使用huggingface的tokenizer的时候，如果context溢出，我只选择了前512个部分，溢出的部分没有进行预测，故该结果比之前要高！