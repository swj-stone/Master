# 单同伴投票、等宽随机列集和首次加载

在原 69 组、额外 57 组及论文 40 组三个面板中，共有 31 个低维配置、5 个不同的数据来源。每组固定训练行、列数、下游模型及测试集，将 minPI 的空同伴层独立用于排序，再与加入平均单同伴层后的排序比较。单同伴层改变了 7 组的列集。最终测试分数有 2 组上升、28 组持平、1 组下降。原论文 Breast Cancer 的两次 AUC 分别上升 0.001048、0.000839。US Macro GDP 一次 R² 下降 0.046616。[研究协议](../peer-protocol.md) · [逐项数据](cases.csv)

每组另抽取 32 个等宽随机列集。单个列集中的特征均匀无放回抽取，独立抽样之间可能出现同一组合。31 组中，26 组的 minPI 分数高于随机分布的中位数，5 组更低。下面的得分差先按配置用 minPI 减当组随机中位数，再在同一协议与同一指标内求平均。[按来源汇总](sources.csv) · [992 次随机抽样与 405 次枚举的分数核对](verification.json)

| 来源及协议 | 指标 | 组数 | 同伴改变列集 | 新分数相对空同伴 胜/平/负 | 高于 32 次随机的中位数 | 与随机中位数的平均分差 |
|---|---|---:|---:|---:|---:|---:|
| Abalone，原 69 | R² | 9 | 0 | 0/9/0 | 9 | +0.01164 |
| Diabetes，原 69 | R² | 3 | 0 | 0/3/0 | 3 | +0.05802 |
| Diabetes，论文 | R² | 5 | 0 | 0/5/0 | 5 | +0.05266 |
| Breast Cancer，额外 57 | 平衡准确率 | 3 | 2 | 0/3/0 | 1 | −0.00422 |
| Breast Cancer，论文 | AUC | 5 | 2 | 2/3/0 | 4 | −0.00004 |
| US Macro GDP，额外 57 | R² | 3 | 1 | 0/2/1 | 1 | −0.41056 |
| Wine，额外 57 | 平衡准确率 | 3 | 2 | 0/3/0 | 3 | +0.02315 |

## Abalone 的全部 45 种组合

在三个互异测试划分上，各以三个选择器种子复核，每一折在种子间的结果一致。minPI、空同伴及等宽 F 都留下七个连续测量和一个类别指示列，删去了 Sex 编码 0、2 对应的指示列。对 10 列中所有 8 列的组合实测测试 R²，minPI 在三个划分分别列第 4、第 1 和第 11。三个划分都高于同宽组合的中位数。筛选带来的这个收益在移去同伴层后全部保留。[逐组完整枚举排名](abalone-all-subsets.csv) · [数据编码](../../../experiments/stable_minpi_stress.py)

| 测试划分 | minPI R² | 45 种组合的中位 R² | 最高 R² | minPI 排名 |
|---|---:|---:|---:|---:|
| 1 | 0.525044 | 0.514087 | 0.525172 | 4/45 |
| 2 | 0.543710 | 0.533961 | 0.543710 | 1/45 |
| 3 | 0.521531 | 0.506999 | 0.534968 | 11/45 |

## 首次加载与暖选列

五个独立进程首次载入已编译的库，耗时中位数为 0.850 毫秒，范围 0.824–2.029 毫秒。在六种合成输入上，每个进程、每种输入及每种方法各预热一次再测 31 次，共 1,860 次独立选列调用。先在进程内求新旧方法的中位选列时间，再求时间比；六种输入上五个进程的中位时间比分别处于 0.089–0.597。此项计时隔离了选择器本身，实际部署还需要切列、拟合与预测。本机首次 GCC 构建的 0.478 秒单列为一次性离线准备成本。[五个进程的首次加载](cold-native.csv) · [暖选列时间](warm-selection.csv) · [主面板的完整流程成本](../core_evidence/README.md)

运行命令以 history 为工作目录。两个输入目录来自主面板和论文面板的正式第一进程。

```powershell
.\thesis-python.ps1 -m research.lowdim.ablate_peers results/minpi-study/lowdim-core-20261118a-pass1 results/minpi-study/lowdim-formal-20261118a-pass1 results/minpi-study/lowdim-peer-ablation-replay
$runs = 1..5 | ForEach-Object { 'results/minpi-study/lowdim-cold-replay-' + $_ }
foreach ($run in $runs) {
    .\thesis-python.ps1 -m research.lowdim.profile_selector $run
    if ($LASTEXITCODE -ne 0) { throw "Profile failed" }
}
.\thesis-python.ps1 -m research.lowdim.profile_summary results/minpi-study/lowdim-cold-summary-replay @runs
```
