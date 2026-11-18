# 最初六个二分类和回归来源的复核

原稿八个来源中有四个二分类和两个回归。每个来源采用原先五个 75%/25% 的随机行划分，各划分测试行可能重叠。新版二分类直接使用冻结的投票集中度选宽，回归用中心化的连续响应计算岭损失改善，其余投票和选宽公式沿用分类版。四种实测方法均在当前轮内重新选列、重训相同的标准化岭模型并预测。两进程各预热后测四轮，用各进程的计时中位数配对，取较大的耗时比。分数与耗时比在下表各取五次划分的中位数。计数逐划分要求真实删列及两进程均快于 Full。[冻结协议](paper-protocol.md) · [逐划分结果](paper-evidence/cases.csv)

| 来源 | 原始 → 自动保留列数 | 指标 | 全特征 | 新 minPI | 同宽 F | 同宽随机 | 新版耗时 / Full | 删列且快于 Full | 再要求主分差 ≥−0.02 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Breast Cancer | 30 → 9–13 | AUC | .9990 | .9985 | .9937 | .9975 | 1.108 | 0/5 | 0/5 |
| Madelon | 500 → 63–159 | AUC | .5718 | .5999 | .5726 | .5914 | .582 | 5/5 | 5/5 |
| Spambase | 57 → 20–23 | AUC | .9506 | .9395 | .9382 | .8720 | .913 | 5/5 | 5/5 |
| Gisette | 5,000 → 405–570 | AUC | .9675 | .9940 | .9946 | .9790 | .091 | 5/5 | 5/5 |
| Diabetes | 10 → 4–6 | R² | .4751 | .4586 | .4384 | .3779 | 1.175 | 0/5 | 0/5 |
| Communities and Crime | 102 → 11–32 | R² | .5434 | .5141 | .4942 | .4428 | .627 | 5/5 | 1/5 |

二分类的全部 20 折都有删列。相对 Full，15/20 折同时加速并将 AUC 损失控制在 0.02 以内。改用平衡准确率时为 10/20。差异来自 Spambase，其平衡准确率配对差的中位数为 −3.40 个百分点，五折均超出该容差。该来源的原行划分中有重复特征向量跨越训练与测试边界。另一个 69 组面板采用分组隔离划分。回归有 5/10 折删列且快于 Full，其中 1/10 保住 0.02 的 R² 容差。与同宽全样本 F 同时比较得分与总时间，二分类新版两项同胜为 9/20，回归为 3/10。与同宽单次均匀随机两项同胜均为零。[按来源](paper-evidence/sources.csv) · [按任务](paper-evidence/tasks.csv) · [分进程毫秒](paper-evidence/process-medians.csv)

## 相同固定预算下的原版与新版排名

用当前的训练投票排序选取原稿固定宽度 ⌈0.1p⌉，每折在相同数据、相同岭模型上评分。此项只比较选列质量。原稿的 F 在 Madelon 五个划分均出现单精度抵消，独立检查发现每折各有 7、7、11、9、8 列与稳定排名不一致。下表用双精度中心化 F，五个受影响的列集都重新训练评分。其余 25 个划分的原 F 列集均精确匹配。

| 来源 | k | 原 pivotal | 新 minPI 排名 | 稳定同宽 F | 新排名胜原 pivotal |
|---|---:|---:|---:|---:|---:|
| Breast Cancer | 3 | .9872 | .9889 | .9948 | 2/5 |
| Madelon | 50 | .6092 | .6046 | .5915 | 3/5 |
| Spambase | 6 | .9136 | .9000 | .8842 | 1/5 |
| Gisette | 500 | .9933 | .9953 | .9952 | 5/5 |
| Diabetes | 1 | .0212 | .3235 | .2984 | 5/5 |
| Communities and Crime | 11 | .2907 | .5101 | .4754 | 5/5 |

两项回归的新版排名均在五折胜原版，同预算下超过稳定 F 的折数分别为 1/5 和 4/5。两个来源的分数中位数分别计算，随后统计的胜负使用同一折配对。Gisette 的新版固定预算排名在 3/5 折胜稳定 F。自适应规则将宽度从 500 改至 405–570，得分超过自己同宽 F 的有 1/5 折。两种比较对应的保留宽度不同。[新旧单折分数](paper-evidence/cases.csv) · [F 边界复核](paper-evidence/f-audit.csv)

## 复现

在 history 目录运行，指定尚未使用的输出目录。

```powershell
.\thesis-python.ps1 -m research.conditional_dense.original_tasks.paper_splits results/minpi-study/paper-six-a --pass-id 1
.\thesis-python.ps1 -m research.conditional_dense.original_tasks.paper_splits results/minpi-study/paper-six-b --pass-id 2 --previous results/minpi-study/paper-six-a
.\thesis-python.ps1 -m research.conditional_dense.original_tasks.paper_summary results/minpi-study/paper-six-a results/minpi-study/paper-six-b results/minpi-study/paper-six-summary
```

正式测得 960 条计时，490 个数组跨进程逐元素相同。复算了 150 份新预测和 120 份历史预测的指标，每一折 Full 主分数与档案完全相同。当前自动宽度及固定宽度共 60 个 F 列集均与独立方差比或相关平方排序完全一致。[复核记录](paper-evidence/verification.json) · [原始计时一](../../../results/minpi-study/pivotal-original6-20261118a-1/times.csv) · [原始计时二](../../../results/minpi-study/pivotal-original6-20261118a-2/times.csv)
