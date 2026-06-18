# tests/

单元测试，用 `pytest` 运行：

```bash
pytest tests/
```

建议覆盖：

- `src/<pkg>/voting_game.py` —— v_α 的边界 case
- `src/<pkg>/min_ppi.py` —— MinSS / MinBz 在小数据上与精确解一致
- `src/<pkg>/nsga2.py` —— 非支配排序、拥挤距离的正确性
- `src/<pkg>/cross_shapley.py` —— Lemma 1-4 的程序化验证

至少保证关键命题（Lemma 1 boundedness、Lemma 4 additive game collapse）有对应的 test case。
