# configs/

实验配置（YAML 推荐）。建议按数据集 + 实验维度组织：

```
configs/
├── conference/
│   ├── spambase.yaml
│   ├── madelon.yaml
│   ├── isolet.yaml
│   ├── gisette.yaml
│   ├── credit_card_fraud.yaml
│   └── ablation/
│       ├── alpha_sweep.yaml
│       ├── beta_sweep.yaml
│       └── shapg_k_sweep.yaml
└── thesis/
    └── (Phase B 之后填充)
```

每份配置应包含：数据集路径、基学习器超参、配额 α、引导强度 β、NSGA-II 参数（种群规模、代数）、随机种子、输出路径。
