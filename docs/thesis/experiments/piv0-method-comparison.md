# PIv0 与历史版本及同列数基线的比较

六个数据集均使用 seed 42、123、456、789、1024 的冻结预处理划分，表中列数与测试分数为五个 seed 的平均值。Abalone 使用非负截断 R²，ISOLET 使用宏平均 OvR ROC-AUC，其他任务使用二分类 ROC-AUC。

| 数据集 | 会议 MinPI 保留列 | 会议 MinPI 分数 | PIv0 保留列 | PIv0 分数 | 同列数 Gain | 同列数 TreeSHAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Abalone | 5.8 | 0.53404 | 7.0 | 0.53793 | 0.53793 | 0.53493 |
| Spambase | 55.4 | 0.98575 | 25.6 | 0.98435 | 0.98483 | 0.98502 |
| Madelon | 56.4 | 0.79233 | 7.6 | 0.84329 | 0.84164 | 0.83487 |
| Gisette | 99.4 | 0.98862 | 54.2 | 0.98642 | 0.98749 | 0.98865 |
| Credit Card Fraud | 28.2 | 0.97262 | 10.8 | 0.97365 | 0.97513 | 0.97759 |
| ISOLET | 100.0 | 0.99348 | 76.2 | 0.99330 | 0.99320 | 0.99335 |

逐数据集、逐 seed 计算删除比例再对 30 个比例等权平均，会议 MinPI 为 12.53%，独立选列数的直接 Gain 为 30.33%，PIv0 为 48.56%。与会议 MinPI 比较，PIv0 的测试分数有 13 组较高、2 组相同、15 组较低。会议实验共享联盟缓存；原文的多目标前沿结果与上表的单一选列结果口径不同。[会议出版结果](../../../results/icomp2026_formal/icomp2026-formal-v1/tables/publication/aggregate_metrics.csv) · [五 seed 原始配对](../../../results/thesis_benchmark/PIv0-all-seeds-20261111-213840-398718/paired_comparison.csv)

旧硕士版本 guarded 的正式存档包含 11 个完整的数据集–seed 组合，使用 K=5。以固定 100 轮预测器的 `guarded_original_test` 对齐，PIv0 的测试分数在 3 组较高、2 组相同、6 组较低。此前这 11 组的整套流程耗时 4,504.92 秒（包含预测器比较和对照评估）；PIv0 对应的 11 组共耗时 474.81 秒。[guarded 原始日志汇总](../../../results/thesis_benchmark/guarded-rounds-formal-20261110-205751-632446/summary.csv)

在同列数补测中，每个划分沿用 PIv0 在验证集上确定的 k。直接 Gain 使用归档的全特征训练模型排序。TreeSHAP 用同配置的全特征模型，在最多 1,024 个训练样本上按行与模型输出计算平均绝对贡献。每组按各自排序取前 k 列；全部特征集合先冻结，再执行测试评分，重复的列集复用预测。该诊断使用此前已评估的测试划分，仅比较同压缩程度下的选列结果。

相对同列数 Gain，PIv0 为 9 胜、7 平、14 负；相对同列数 TreeSHAP，为 12 胜、3 平、15 负。若允许直接 Gain 自选 k，PIv0 变为 7 胜、7 平、16 负。只有自选 k 的完整配对运行用于比较端到端耗时。[同列数五 seed 汇总](../../../results/test/thesis_sampling/piv0-fixed-k-controls-20261111-231112-839839/dataset_means.csv) · [同列数配对明细](../../../results/test/thesis_sampling/piv0-fixed-k-controls-20261111-231112-839839/paired.csv) · [补测入口](../../../experiments/test/compare_piv0_fixed_k.py)
