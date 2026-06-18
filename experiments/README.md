# experiments/

实验入口脚本。每个脚本独立可运行：

- 从 `configs/` 读配置
- 从 `src/` import 模块
- 写结果到 `results/`

建议命名：

```
experiments/
├── run_main_table.py        # 主结果表
├── run_ablation.py          # 消融实验
├── run_alpha_sweep.py       # 配额敏感性
├── make_figures.py          # 生成所有论文图（PDF 格式）
└── preprocess_<dataset>.py  # 数据集下载与预处理
```

每个脚本接受 `--config <path>` 参数，从 YAML 加载配置。
