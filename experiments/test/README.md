# 硕士论文限时方案试验

从 `Paper` 目录打开 PowerShell。单 seed 选列试验使用冻结的 seed 42 划分，
每组默认最多运行 900 秒，其中采样预算为 780 秒。例如在 Madelon 上试验
`guarded`：

```powershell
.\MastersThesis\thesis-python.ps1 experiments/test/run_thesis_sampling.py --selection-policy guarded --datasets madelon
```

固定特征集的预测器试验复用已冻结的列，原选列准备时间也计入每组上限：

```powershell
.\MastersThesis\thesis-python.ps1 experiments/test/run_thesis_predictor.py results/test/thesis_sampling/seed42-confirmed-20261110-191743-946395 --recipe rounds
```

| 试验入口 | 用途 |
| --- | --- |
| `run_thesis_benchmark.py` | 用限时时间预算试跑六数据集、五 seed 的批量配置，支持 `--plan` |
| `run_madelon_alpha.py` | 比较 Madelon seed 42 的互信息 Top-100 与原始 500 列。每组各运行一次 α=0.99、K=2，900 秒硬截止包含采样、选列、+1 复核和测试。 |
| `run_relative_voting.py` | 比较同规模相对投票与 Gain，支持 Madelon、Spambase 与五个正式 seed，每个方法和划分独立限时 900 秒。[方案与实测结果](../../docs/thesis/test/budget-relative-voting.md) |
| `replay_online_relative_voting.py` | 对已保存的验证贡献执行独立校准与单席序贯投票，记录每列失去保护资格、负证据过界与最终锁定的轮次。[规则和复现实例](../../docs/thesis/test/online-voting.md) |
| `run_thesis_sampling.py` | 比较单 seed 的特征选择策略 |
| `run_thesis_predictor.py` | 比较固定特征下的轮数或列采样方案，`--training-check` 在原训练分区内复核轮数 |
| `thesis_sampling_pilot.py` | Madelon 两种采样器的初步比较，每种最多 384 次拟合、约 240 秒，在下一次拟合前检查 |
| `compare_thesis_sign_aware.py`、`replay_thesis_guarded.py` | 比较不同提名方案并回放验证选择 |
| `compare_local_refinement.py`、`compare_fixed_predictor.py`、`compare_predictor_replication.py` | 核对已完成的特征调整、预测器及训练分区复核结果 |

新试验结果写入 `MastersThesis/results/test/`。单 seed 试验使用 `thesis_sampling/`，
批量试验使用 `thesis_benchmark/`。训练和硬截止的共用代码仍由批量入口和试验入口
分别传入本轮配置。原脚本路径保留兼容入口。

## 旧结果归档

- `results/test/thesis_sampling/` 收录先前 12 组 seed 42 试验的完整快照。归档时逐文件
  核对了 418 个文件的 SHA-256。原先的 `results/thesis_sampling/` 路径保留，供
  复查记录中的原始位置和哈希。实验标识及数据见
  [分值筛选](../../docs/thesis/test/sign-aware-seed42.md)、
  [稳定性试验](../../docs/thesis/test/stability-seed42.md)和
  [采样方法](../../docs/thesis/test/sampling-methods.md)。
- `results/test/thesis_benchmark/guarded-rounds-20261110-203136-404894/` 是更改配置前
  的 900 秒限时试跑，包含一个完成的任务和一个中断的任务。归档前后 17 个文件的
  SHA-256 一致。原始日志和协议保留当时使用的路径与 `formal` 标签；该目录归入
  试验记录。本次正式任务将生成自己的 30 组记录。

正式运行命令见[硕士实验说明](../../README.md)。
