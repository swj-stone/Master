# data/

数据集存放目录。**gitignored**（数据文件不入仓库）。

建议结构：

```
data/
├── raw/             # 原始数据（从 UCI / Kaggle / OpenML 等下载）
└── processed/       # 预处理后的数据（标准化、切分、特征工程）
```

每个数据集应有一份下载与预处理脚本（放 `experiments/preprocess_<dataset>.py` 或 `src/<pkg>/datasets/`）。

数据集来源：
- Spambase, Madelon, ISOLET, Gisette → UCI ML Repository
- Credit-Card-Fraud → Kaggle
