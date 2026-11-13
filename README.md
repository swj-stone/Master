# Master's thesis workspace / 硕士论文工作区

硕士研究在 `MastersThesis/` 的 `thesis/masters` 分支进行，与 `../Master/` 共享 Git 仓库及 Python 环境。会议最终版在 `../Master/` 的 `main` 分支，供只读参考。工作树中的 `data/`、`results/` 和 `packages/` 是独立的本地文件，受 Git 忽略规则管理。

## 现有实验结果

| 版本 | 实验范围 | 查看结果 |
| --- | --- | --- |
| PIv0 | 六个预处理数据集、五个 seed，与直接 Gain 按划分配对 | [正式结果](docs/thesis/experiments/piv0-five-seeds.md) · [与会议版及其他基线的比较](docs/thesis/experiments/piv0-method-comparison.md) |
| PIv1 | 同一批 30 个划分。表中含分类 AUC 与准确率、回归 R² 与 MAE、未删列验证基线及耗时 | [数据集汇总](docs/thesis/archives/piv1/final_results_by_dataset.csv) · [逐 seed 结果](docs/thesis/archives/piv1/final_results_by_seed.csv) · [协议、对照与证据包](docs/thesis/archives/piv1/README.md) |
| PIv2 | seed 42 上，对比下游选择器单独使用和加 PI 预筛的结果 | [按方法汇总](results/test/thesis_sampling/PIv2-seed42-all-methods-summary/comparison.md) |
| PIv3 | seed 42 上的原始宽度实验。多后端试验保存了中止时已完成的配对，Lite 与 compact witness 各有独立报告 | [阶段性配对](results/test/thesis_sampling/PIv3-raw-six-seed42-tol358/paired.md) · [停止位置](results/test/thesis_sampling/PIv3-raw-six-seed42-tol358/stopped.json) · [Lite](results/test/thesis_sampling/PIv3-lite-raw-six-seed42/results.md) · [compact witness](results/test/thesis_sampling/PIv3-compact-witness-raw-six-seed42/comparison.md) |

## 运行 PIv0 或 PIv1 正式实验

在包含 `Master`、`MastersThesis` 和 `venv` 的 **`Paper` 目录**打开 PowerShell，按所需版本执行对应的一条命令。入口使用共享的 `venv`、导入硕士工作树的源码，并在 `MastersThesis/` 下解析实验路径。

**PIv0**

```powershell
.\MastersThesis\thesis-python.ps1 experiments/run_piv0_benchmark.py
```

**PIv1**

```powershell
.\MastersThesis\thesis-python.ps1 experiments/run_piv1_benchmark.py
```

正式实验使用 Abalone、Spambase、Madelon、Gisette、Credit Card Fraud、ISOLET，seed 为 42、123、456、789、1024。每组沿用对应 seed 的冻结数据划分和最多 100 列的预处理输入。每个版本有 30 组配对、60 条方法记录，新训练运行至完成。PIv0 复用已核验的 seed 42 结果。PIv1 使用已核验的直接 Gain 及 pilot 记录，补算其余 PIv1 划分。[PIv0 实验口径](docs/thesis/experiments/piv0-five-seeds.md) · [PIv1 实验口径](docs/thesis/experiments/piv1-five-seeds.md)

程序打印本次结果目录，并在终端和 `run.log` 中记录进度、拟合次数和用时。接续已有批次时，在对应启动命令后加 `--output` 和本次结果目录路径。要在另一个 PowerShell 窗口查看日志，在 `Paper` 目录运行下列命令，并将占位符替换为实际目录名。

```powershell
Get-Content -LiteralPath '.\MastersThesis\results\thesis_benchmark\<本次运行目录名>\run.log' -Tail 15 -Wait
```

完整批次结束后，使用对应版本的汇总脚本。

```powershell
.\MastersThesis\thesis-python.ps1 experiments/summarize_piv0.py 'results/thesis_benchmark/<PIv0本次运行目录名>'
.\MastersThesis\thesis-python.ps1 experiments/summarize_piv1.py 'results/thesis_benchmark/<PIv1本次运行目录名>'
```

PIv1 的封存标签为 `thesis/piv1-frozen`。含准确率、回归 MAE 和未删列基线的完整表及补算记录可从封存索引查阅。[PIv1 封存索引](docs/thesis/archives/piv1/README.md)

## 方案试验与历史材料

PIv0、PIv1 的开发对照、PIv2、PIv3 的前置筛选试验入口在 `experiments/test/`，单 seed 结果在 `results/test/thesis_sampling/`。各入口的用法、输入宽度及已有对照见[方案试验说明](experiments/test/README.md)。

PIv0 定名前的独立探索按清理清单移除了本地运行文件。硕士工作树仍保留论文比较引用的会议出版表、旧 guarded 汇总、PIv1 的 Abalone 联盟缓存，以及各版本复现实验用的原始数据、冻结划分和配置。[清理清单](docs/thesis/cleanup-inventory-2026-11-13.md) · [PIv0 历史比较](docs/thesis/experiments/piv0-method-comparison.md)

会议论文、海报和会议实验的使用说明在冻结参考版中可查阅。[会议项目说明](../Master/README.md) · [会议论文](conference-paper/paper/icomp2026_conference.pdf) · [海报](conference-paper/poster/ASCOMP_MinPI_Poster_Wenjie_SHI.pdf)

## 本地开发

在 `MastersThesis/` 中使用 `thesis-python.ps1`。该入口为当前进程设置源码路径，并将 Matplotlib、Numba 缓存放入本工作树的 `tmp/`。共享环境中已有的可编辑安装指向会议参考版。

```powershell
.\thesis-python.ps1 -c "import minppi_mofs; print(minppi_mofs.__file__)"
.\thesis-python.ps1 -m pytest tests -q -p no:cacheprovider --basetemp ".t$PID"
```

在本工作树提交后，如需推送硕士论文分支，可执行下列命令。

```powershell
git status --short --branch
git push -u origin thesis/masters
```
