# 介绍
本代码主要将huggingface的tokenizer替换成paddlenlp的tokenizer。现在有关offset偏移的问题已经完全修复了。因此我们可以直接开始验证了！

# 验证(batch_size=16的时候预测。)
```bash
bash evaluate.sh
```
预期结果如下：
```python
{
  "exact": 82.28754316516466,
  "f1": 85.29235824679081,
  "total": 11873,
  "HasAns_exact": 80.44871794871794,
  "HasAns_f1": 86.46696515926936,
  "HasAns_total": 5928,
  "NoAns_exact": 84.12111017661901,
  "NoAns_f1": 84.12111017661901,
  "NoAns_total": 5945,
  "best_exact": 82.90238355933631,
  "best_exact_thresh": -2.4589643478393555,
  "best_f1": 85.71655580405353,
  "best_f1_thresh": -2.4589643478393555
}
```

# Tips:
发现有趣的现象,batch_size=32的时候预测出下面这个结果- -|。效果提升了，应该跟paddding有关。
```bash
python evaluate.py --model_name_or_path ./best-model_29400 --version_2_with_negative --batch_size 32
```
```python
{
  "exact": 82.29596563631769,
  "f1": 85.29404274102141,
  "total": 11873,
  "HasAns_exact": 80.46558704453442,
  "HasAns_f1": 86.47033897843264,
  "HasAns_total": 5928,
  "NoAns_exact": 84.12111017661901,
  "NoAns_f1": 84.12111017661901,
  "NoAns_total": 5945,
  "best_exact": 82.91080603048934,
  "best_exact_thresh": -2.456880569458008,
  "best_f1": 85.71824029828416,
  "best_f1_thresh": -2.456880569458008
}
```