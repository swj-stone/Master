<<<<<<< HEAD
# src/

可导入的源码包。按功能拆子模块。建议结构：

```
src/
└── <package_name>/          # 例如 minppi_mofs/
    ├── __init__.py
    ├── voting_game.py       # v_α, WVG formulation
    ├── min_ppi.py           # MinSS, MinBz 估计算法
    ├── nsga2.py             # MOFS-MinPPI NSGA-II 实现
    ├── shapg_integration.py # 调用 ShapG 库，构造引导信号
    ├── cross_shapley.py     # 毕业论文 bipartite 推广
    └── baselines/
        ├── shap_top_k.py
        ├── powershap.py
        ├── banzhaf_fs.py
        ├── min_shap.py
        └── ...
```

在 `pyproject.toml` 注册为 editable package 后即可 `from <package_name> import ...`：

```bash
pip install -e .
```
=======

>>>>>>> 7a28100fdd58d7ab28dd666fac46c7f5b27c700a
