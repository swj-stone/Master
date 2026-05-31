# results/

实验产物存放目录。**gitignored**。

建议结构：

```
results/
├── figures/         # 论文图（PDF 格式）
├── tables/          # 论文表（CSV / Markdown）
└── logs/            # 实验运行日志
    └── <run_id>/    # 每次运行一个子目录
        ├── config.yaml      # 实际使用的配置
        ├── metadata.json    # seed, git commit, timestamp
        ├── metrics.csv
        └── output.log
```

每次运行用 timestamp 或 hash 作为 `<run_id>`，确保不同运行的结果不互相覆盖。
