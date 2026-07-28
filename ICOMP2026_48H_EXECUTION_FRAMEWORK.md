# ICOMP 2026 两天全流程执行框架

> 状态：代码冻结前最终候选，2026-07-28  
> 硬约束：1 张 8 GB GPU，P0 全流程 P90 不超过 48 小时  
> 截止日期：2026-08-05  
> 本文是当前正式执行的权威入口；旧 README 中逐 run 的 formal 命令仅作历史记录。

代码冻结验证：`1245 passed, 3 skipped`。三 shard 的真实 validate 创建、
原命令 resume 和 monitor 均已通过；该 smoke 未运行 estimator，也未访问 test。

## 1. 最终结论

当前最稳妥的方案不是降低模型轮数、改用 CPU 或将 NSGA-II 从 800 次评估
降到 400 次，而是：

1. 保持所有数据集和方法使用同一冻结 XGBoost 概率模型；
2. 使用每数据集 estimator 预算，ISOLET 最大 K=10，其余数据集最大 K=25；
3. 将五个种子拆成三个源代码完全一致的 shard 并发执行；
4. 复用同一 dataset/seed bundle 内的嵌套 K 前缀、coalition cache 和概率；
5. alpha 只运行全部七数据集的 seed 42；
6. ablation 只运行 Madelon、ISOLET、Credit、Abalone 的 seed 42；
7. removal 在测试集打开前执行验证非劣安全门；
8. NSGA-II 保持 40 population、800 evaluations 的完整证据强度；
9. 三个 shard 完成后只进行一次五种子统一汇总、统计和出图。

实测投影为：

| 项目 | P90 wall time |
|---|---:|
| Estimator 三 shard | 30.00 h |
| validate + preprocess | 1.00 h |
| safety/removal/controls | 2.00 h |
| paired NSGA-II | 3.00 h |
| ranking/statistics | 4.00 h |
| alpha/代表性 ablation | 2.00 h |
| summary/figures/exports | 1.00 h |
| **P0 合计** | **43.00 h** |
| **48 小时余量** | **5.00 h** |

这里的 estimator 账本来自真实 validation-only bundle；后处理的 13 小时是
基于旧阶段运行记录和新范围缩减后的保守 P90 预留，不是全部重新实测值。
因此 43 小时是可执行的时间门，不是完成承诺。若实际 P90 越过 48 小时，
停止 P1，不得削弱 P0 主证据或悄悄减少种子。

## 2. 冻结科学协议

### 2.1 模型

- backend：CUDA；
- `n_estimators=100`；
- `max_depth=4`；
- `learning_rate=0.1`；
- `n_jobs=1`；
- 分类任务统一 XGBoost；
- accuracy、ROC-AUC、OVR-AUC、PR-AUC 从同一预测概率投影；
- 所有方法在同一 dataset/seed 内共享相同模型参数和数据划分。

Calibration 表明低于 100 轮虽然常能保持 full-set utility，但不能稳定保持
局部 marginal sign 和 harmful recall。因此不能通过减少树轮数换取速度。

### 2.2 Estimator

| 数据集 | 嵌套 checkpoints | 正式最大 K |
|---|---|---:|
| Spambase | 2, 5, 10, 25 | 25 |
| Madelon | 2, 5, 10, 25 | 25 |
| ISOLET | 2, 5, 10 | 10 |
| Gisette | 2, 5, 10, 25 | 25 |
| Credit-Card-Fraud | 2, 5, 10, 25 | 25 |
| CrimeData | 2, 5, 10, 25 | 25 |
| Abalone | 2, 5, 10, 25；exact dispatch | exact |

K=2/5 是同一正式 bundle 的低成本收敛检查，不是独立随机实验。论文主结论只
使用每数据集的最大 checkpoint；不得把低 K 的方向性结果写成正式精度。

Credit 必须保留 K=25：K=10 没有召回 seed-42 中 K=25 的 `V25` witness。
ISOLET 的 K=25 seed-42 约需额外 9.04 小时，属于 P1，不得阻塞 P0。

### 2.3 Removal 安全门

Estimator 输出两个不同含义的集合：

- `raw_harmful_features`：存在 threshold-crossing coalition witness 的解释性候选；
- `safe_harmful_features`：验证集非劣门通过后才允许进入删除和 NSGA 预筛选的集合。

门限为：

- 分类：`selected_validation_utility - full_validation_utility >= -0.005`；
- 回归：`selected_validation_utility - full_validation_utility >= -0.010`。

未通过时：

- raw 集合和 witness 保留，用于报告反例；
- safe 集合置空；
- removal 主方法回退为 full features；
- `minppi_prefilter` NSGA-II 回退为完整搜索空间；
- 不允许根据测试集结果重新打开安全门。

该设计直接解决 Abalone：四个 exact contextual harmful witnesses 仍是解释证据，
但联合删除使 validation R² 下降约 0.0155，故安全门拒绝删除。论文必须将
“contextually harmful”与“safe to remove”分开陈述。

### 2.4 NSGA-II 与 controls

- variants：`full` 与 `minppi_prefilter`；
- population：40；
- evaluations：800；
- reference point：`[1.1, 1.1]`；
- 两个 variants 使用相同逻辑评估预算；
- random matched removal：30 trials。

没有可审计的 800→400 收敛证据，因此 P0 不降低 NSGA 预算。当前 48 小时
账本已为完整 paired NSGA 预留 3 小时。

### 2.5 范围缩减

- estimator、removal、NSGA-II：7 datasets × 5 seeds；
- alpha：7 datasets × seed 42；
- ablation：Madelon、ISOLET、Credit、Abalone × seed 42；
- ISOLET K25 sensitivity：P1，仅在 P0 完成且剩余时间充足时运行。

范围缩减发生在非核心 sensitivity，而不是主表、removal 或 paired NSGA。

## 3. 三 shard 执行结构

固定分片：

| Shard | Seeds |
|---|---|
| A | 42, 123 |
| B | 456, 789 |
| C | 1024 |

三 shard 使用完全相同的：

- 配置；
- model-policy digest；
- Python source fingerprint；
- dataset settings；
- estimator method/K mapping；
- removal 与 NSGA 设置。

只有 seed scope 和 protocol hash 可以不同。Campaign index 拒绝 seed 重叠、
缺失种子、模型策略变化、科学设置变化或源文件指纹变化。

当前冻结范围的逻辑 manifest 有 823 个 job，其中 estimator 405、removal
245、NSGA-II 70、preprocess 35、alpha 35、ablation 30，以及三个 run-level
job；但 estimator 只有 35 个物理 dataset/seed bundle。大量 method/K job 是
同一 bundle 的已验证投影，不是 823 次互相独立的长实验。

## 4. 为什么不启用 CPU 或旧 cache

ISOLET CPU 4×4 的吞吐约为 CUDA 三 worker 的 4.45 倍，但固定结构面板的
marginal-sign agreement 只有 0.725；CPU 与 CUDA 已不再是同一个局部 utility
landscape。正式结果固定 CUDA。

旧 decision/probe cache 不晋级到 P0：

- 正式协议、removal 安全门和汇总代码已经改变；
- cache identity 含 protocol/source authority；
- 跨源晋级节省有限，却会削弱可追溯性。

旧结果只用于运行时间估计和科学方向，不复制到正式 campaign。

## 5. 论文证据链

最终论文应按四层证据组织：

1. 理论：三值分类及 threshold-crossing 定义；
2. 算法：嵌套 schedule、共享 cache、物理拟合数与逻辑评估数；
3. 解释：raw harmful feature 的具体 coalition witness；
4. 行动：只有 validation safety gate 接受的集合才用于 removal/NSGA prefilter。

当前 validation-only 方向：

| 数据集 | 当前证据 |
|---|---|
| Madelon | strong support；15% raw removal，validation AUC +0.00615 |
| CrimeData | strong support；4% raw removal，validation R² +0.00062 |
| Credit | K25 找到 V25；3.45% compression，validation 非劣 |
| Spambase | neutral |
| ISOLET | neutral，且是主要时间瓶颈 |
| Gisette | neutral |
| Abalone | contextual-harmful counterexample；安全门必须回退 |

正式五种子结果出来前不得声称：

- Credit 压缩约 35%；
- Credit NSGA-II HV 提升 37%；
- 所有数据集删除 harmful 都安全；
- MinPPI 显著优于所有 baselines；
- 测试集 utility 或 HV 已改善。

统一汇总会生成：

- 五种子 raw/aggregate/paired/front 表；
- validation safety gate 通过率与 raw/safe harmful 数量；
- paired NSGA-II HV 与 convergence 统计；
- 六种 matched-removal 方法的 mean ranks；
- Friedman omnibus test；
- Nemenyi critical difference；
- MinPPI 对各方法的 Wilcoxon + Holm 结果；
- 三张论文 PDF 图；
- LaTeX macros 与来源证明。

排名以七个数据集为独立统计块，每个数据集先在五个种子上求均值；不会把
35 个 dataset/seed 伪装成 35 个独立 benchmark。

## 6. 从空结果开始的命令

以下命令全部在 `Master` 目录的 PowerShell 中运行。先关闭其他 GPU 程序。

```powershell
$PY = "..\venv\Scripts\python.exe"
$OUT = "results\icomp2026_48h"
$CID = "icomp2026-formal-20260728-v1"
$POLICY = "results\icomp2026_multifidelity\bootstrap-s42-20260728-v2\calibration\model_policy.json"
```

### 6.1 代码冻结门

```powershell
& $PY -m pytest tests -q -p no:cacheprovider
```

只有测试为 0 failures 后才创建正式 campaign。正式运行开始后不要修改 `.py`、
YAML 或 model policy；否则 source fingerprint 会拒绝 resume。

### 6.2 第一段：测试集密封的 estimator 主实验

```powershell
& $PY experiments\run_icomp2026_campaign.py `
  --config-dir configs `
  --output-root $OUT `
  --campaign-id $CID `
  --model-policy $POLICY `
  --through estimator `
  --max-parallel 3
```

此阶段执行 validate → preprocess → estimator，不访问 test。

### 6.3 中断后恢复

重新执行与 6.2 完全相同的命令。Runner 会识别已有 shard，并向底层 pipeline
传递 `--resume`；已完成的 bundle/checkpoint/coalition 不会重算。

不要删除部分结果，不要换 campaign ID，不要修改源码后强行续跑。

### 6.4 监控

一次性状态：

```powershell
& $PY experiments\monitor_icomp2026_campaign.py `
  --output-root $OUT `
  --campaign-id $CID
```

每 60 秒刷新：

```powershell
while ($true) {
  Clear-Host
  & $PY experiments\monitor_icomp2026_campaign.py `
    --output-root $OUT `
    --campaign-id $CID
  Start-Sleep -Seconds 60
}
```

停止监控用 `Ctrl+C`；不会停止后台实验进程。

### 6.5 Estimator 后检查 seed 42 方向

seed 42 位于 shard A：

```powershell
$SHARD42 = "$OUT\$CID-s42-123"
& $PY -m experiments.screen_icomp2026_results `
  --run-root $SHARD42 `
  --config-dir configs `
  --model-policy $POLICY `
  --backend cuda `
  --seed 42 `
  --output-dir "$OUT\$CID\screening"
```

检查：

- `screen_report.md`；
- 七个 claim cards；
- 每数据集最大 K 是否完整；
- Credit K25 与 ISOLET K10 是否存在；
- test audit 中是否没有 estimator 访问。

### 6.6 第二段：显式打开测试访问并完成全部 P0

只有在验证报告通过人工 freeze gate 后运行：

```powershell
& $PY experiments\run_icomp2026_campaign.py `
  --config-dir configs `
  --output-root $OUT `
  --campaign-id $CID `
  --model-policy $POLICY `
  --through summarize `
  --max-parallel 3 `
  --allow-test-access
```

该命令会 resume 已完成的 validate/preprocess/estimator，然后执行：

```text
removal → alpha(seed42) → paired NSGA-II → representative ablation
→ campaign-level five-seed summary → ranking/statistics → figures
```

相同命令也是第二段中断后的恢复命令。

## 7. 最终产物

Campaign 根目录：

```text
results/icomp2026_48h/<campaign-id>/
```

关键文件：

- `campaign_plan.json`：三 shard 计划；
- `campaign_index.json`：跨 shard 身份与种子联合；
- `campaign_summary.json`：最终发布状态与行数；
- `screening/screen_report.md`：seed-42 validation-only 方向；
- `tables/summary/raw_metrics.csv`；
- `tables/summary/aggregate_metrics.csv`；
- `tables/summary/paired_comparisons.csv`；
- `tables/summary/plot_sources.csv`；
- `tables/summary/latex_macros.tex`；
- `tables/ranking/removal_ranking.json`；
- `tables/ranking/removal_ranking.md`；
- `figures/paper/*.pdf`。

只有 `campaign_summary.json` 中 `publication_allowed=true`，且五个种子、全部 P0
stage、source fingerprints 和测试审计均通过时，结果才可进入论文正式表。

## 8. 48 小时运行中的停止规则

立即停止并保留现场：

- source/protocol/model-policy drift；
- 数据 SHA 不一致；
- test 在 estimator/screen 阶段被访问；
- 任一 shard 产生非可恢复失败；
- CUDA OOM 连续重现；
- P0 P90 预计超过 48 小时且无法通过 resume/调度恢复。

不得采用的“救火”方式：

- 临时把 ISOLET 切到 CPU；
- 减少正式种子；
- 将 800 NSGA evaluations 改成 400 后仍称为同一协议；
- 根据 test 结果调整安全门；
- 从旧 run 手工复制 scores/cache；
- 将 incomplete shards 混入五种子均值。

时间不足时只取消 P1 ISOLET K25 sensitivity。P0 五种子主表、safe removal、
paired NSGA 和统一统计不降级。

## 9. 接收导向的最终写法

论文的可接收性来自“主张与证据严格对齐”，不是把所有数据集写成正结果：

- 将 contextual detection 与 safe action 分开，是对 Abalone 反例的正面解决；
- 报告 neutral 数据集，证明方法不会为了压缩强行删除；
- 用具体 coalition witness 提供解释性；
- 用五种子 paired 结果回答 removal 与 NSGA 实用价值；
- 用 dataset-level Friedman/Nemenyi/Wilcoxon-Holm，避免伪重复；
- 报告逻辑 evaluations 与物理 model fits，证明加速没有减少科学预算；
- 删除旧稿中 35% Credit compression、+37% HV 等未复现数字。

若最终 paired NSGA 不显著，论文应把主要贡献定位为：

1. 可解释的 contextual harmful detection；
2. validation-gated safe prefilter；
3. 可审计、可恢复的 nested multi-fidelity estimator；
4. 多数据集上的支持、neutral 和 counterexample 边界。

这仍比保留无法复现的强效果量更可信，也更符合会议审稿对正确性、可复现性和
诚实消融的要求。
