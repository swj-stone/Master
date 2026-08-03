# MinPPI-MOFS：ICOMP 2026 可复现实验项目

本目录实现 Min-Marginal Power Indices（MinPPI）及其 ICOMP 2026 实验协议。MinPPI 在效用阈值投票博弈中考察特征的最坏边际贡献，将特征划分为有害、无影响或决定性，并验证其作为多目标特征选择预过滤器的作用。

## 当前状态

- 五种子 campaign 已完成全部注册阶段，原始七数据集 summary 作为不可变历史审计记录保留。
- 事后数据完整性审计发现 Communities & Crime 的目标可由保留的结果字段近乎直接重构，且还含全数据监督编码；该数据集的全部结果已隔离，不进入投稿证据。
- 论文统计由 canonical 行级记录重新聚合为六数据集 publication view，并与原 summary、派生表和三张 PDF 的 SHA-256 绑定。
- 当前正式配置已经改为六个通过完整性筛查的数据集；新的从头运行不会再调度被隔离的数据集。
- `conference-paper/figures/` 只接受通过 manifest 和哈希验证的正式图片；论文正文引用的三图均来自该证据链。

## 权威边界

| 内容 | 唯一真源 |
|---|---|
| 数据集、划分、效用 | `configs/<dataset>.yaml` |
| seeds、预算、alpha、消融、NSGA-II | `configs/icomp2026_pipeline.yaml` |
| 执行和恢复语义 | `experiments/run_icomp2026_pipeline.py`、`experiments/run_icomp2026_campaign.py` |
| 正式完成状态 | campaign 的 manifest、audit、canonical metrics 和 summary manifest |
| 当前论文证据 | 原 campaign 的六数据集 `tables/publication/` 及其来源绑定 |
| 论文图 | 对应 publication scope 的 `figures/paper/` |

`demos/`、`thesis/` 和 `configs_credit_probe/` 被保留为辅助研究材料，但不属于正式 ICOMP 证据链。旧式单脚本实验也不得替代 manifest 管线。

## 冻结协议摘要

- 正式数据集：Spambase、Madelon、ISOLET、Gisette、Credit-Card-Fraud、Abalone；Communities & Crime 的历史配置与 raw snapshot 仅用于完整性审计。
- 每个数据集 YAML 登记 raw CSV 的 SHA-256 与字节数；validate 在写出任何 authority 前必须精确匹配。
- seeds：`42, 123, 456, 789, 1024`。
- 划分：train/validation/test = `0.64/0.16/0.20`；分类任务分层划分。
- 预处理：只在训练集拟合标准化与 MI 过滤；最多保留 100 个特征。
- 模型：所有分类任务统一使用 XGBoost 概率输出，再由 `utility` 选择指标。
- 正式效用：二分类为 ROC-AUC，ISOLET 为 OVR-AUC，回归为 R²；accuracy 只用于 ISOLET utility ablation。
- 默认 `alpha=0.95`，扫描集合为 `{0.85, 0.90, 0.95, 0.99}`。
- alpha 和 ablation 阶段只运行 seed 42；estimator、removal 和 NSGA-II 使用五个 seeds。
- YAML 的可移植默认 backend 是 CPU；正式 campaign 入口当前显式使用 CUDA。

任何协议变更都必须先修改 `configs/`，然后以新 run ID 创建新实验。不要在已创建的 run 上修改 Python、YAML 或 model policy。

## 目录结构

```text
Master/
├── configs/                  正式协议
├── data/raw/                 六个正式 CSV 加一个隔离审计快照（不提交 Git）
├── experiments/              校准、管线、campaign、监测和汇总
├── src/minppi_mofs/          核心算法库
├── tests/                    自动化测试
├── conference-paper/         论文源文件与图同步工具
├── demos/                    非正式教学验证
├── thesis/                   非正式硕士论文研究草稿
├── configs_credit_probe/     非正式 Credit-Card 探测配置
└── results/                  可恢复实验产物（不提交 Git）
```

`data/processed` 和 `data/processed_v2` 不属于正式协议。正式 preprocess 阶段直接读取 `data/raw`，并把每次划分写入对应 run 目录。

## 使用现有虚拟环境

以下命令均在 `Master` 目录的 PowerShell 中执行，并且只使用项目相邻的现有 `venv` 解释器；不调用系统 Python、Conda base 或其他外部解释器。

```powershell
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY -c "import sys; print(sys.executable); print(sys.version)"
& $PY -m pip install --upgrade pip
& $PY -m pip install --no-build-isolation -r requirements.txt
```

支持 Python 3.12 及以上；推荐使用 Python 3.13。`pyproject.toml` 是依赖版本唯一真源，`requirements.txt` 只提供开发环境安装入口。

### 安装验收

```powershell
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY -m pip check
& $PY -c "import minppi_mofs; print('import OK')"
& $PY -m pytest tests -q -p no:cacheprovider --basetemp ".t$PID"
& $PY -m ruff check src experiments tests conference-paper/sync_figures.py
```

## 两组式完整执行

正式入口把五个 seeds 固定分为 `(42,123)`、`(456,789)` 和 `(1024)` 三个 shard。`--max-parallel 3` 是当前单 GPU、两天目标的候选配置，不是完成时长保证。一次只能启动一个 campaign 进程，不要同时手工启动其他 GPU/model 命令。

所有命令都从 `Master` 目录执行，并通过同一个相邻 `venv`。下面的函数会在任一 Python 子命令返回非零时立即停止，防止带着不完整的上游产物继续运行。

### 第一组：原始数据、校准、smoke 与密封 estimator

第一组入口显式运行六个正式数据获取脚本。它们只创建或逐字节复核 `data/raw/*.csv`，已有快照不会被清洗、去重或覆盖；下载先写同目录临时文件，只有 SHA-256 和字节数与 dataset YAML 完全一致才会发布。被隔离的 Communities & Crime 不参与正式获取、校准或调度。正式的 train/validation/test 划分仍由 manifest 管线在各 run 内完成。随后入口以 seed 42 校准模型、运行独立 smoke，最后让正式五种子 campaign 只运行到 estimator，held-out test 保持密封。

入口是单个 Python 编排进程，而不是一组彼此独立的 PowerShell 顶层语句。任一子命令非零时异常会立即向上传播，后续 preprocess、校准、smoke、campaign 或 screening 不会继续启动。

正式 provenance 必须对应一个可重建的 Git commit。一键入口会在开始时及正式 `validate` 前再次要求 `git status --short` 为空；请先提交本轮预期代码和文档。`data/`、`results/` 与测试临时目录已被忽略，不会妨碍该门禁。

```powershell
$ErrorActionPreference = "Stop"
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY experiments\run_icomp2026_group1.py --bootstrap-id bootstrap-s42-v2 --smoke-id smoke-s42-v2 --campaign-id icomp2026-formal-v2
if ($LASTEXITCODE -ne 0) { throw "Group 1 failed with exit $LASTEXITCODE" }
```

第一组中断后，原样重输这个代码块即可：bootstrap 与 smoke 会根据目录是否存在显式恢复，campaign 会对已有 shard 自动恢复。这里使用 `v2` ID，以免与保存七数据集历史证据的 `v1` 目录碰撞。最后检查 `results\icomp2026_formal\icomp2026-formal-v2\screening\screen_report.md`；它是 seed-42 的 validation-only 方向筛查，不是论文结果。不要删除成功 shard、替换 model policy、升级当前 venv/显卡驱动或复制其他 run 的 cache；科学环境摘要发生变化时 resume 会拒绝继续。

监测器读取 campaign plan，因此不能监测第一组前置的环境验收、数据获取、bootstrap、校准和 smoke；这些阶段直接看一键入口终端。等 `results\icomp2026_formal\icomp2026-formal-v2\campaign_plan.json` 出现后，再在另一个位于 `Master` 的 PowerShell 终端执行一次性 campaign 快照：

```powershell
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY experiments\monitor_icomp2026_campaign.py --output-root results\icomp2026_formal --campaign-id icomp2026-formal-v2 --group initial
```

### 第二组：held-out test、统一汇总、正式图与论文编译

只有确认第一组验证集方向合理后，才运行第二组。该命令显式开放 removal、alpha 与 NSGA-II 的注册 held-out-test 访问，完成 ablation、唯一 campaign 汇总以及三张正式图。同步器只接受同一 publication-eligible campaign 的精确 PDF/hash bundle。

```powershell
$ErrorActionPreference = "Stop"
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
$OUT = "results\icomp2026_formal"
$CID = "icomp2026-formal-v2"
$POLICY = "$OUT\bootstrap-s42-v2\calibration\model_policy.json"
$FIGURES = "$OUT\$CID\figures\paper"

function Invoke-ProjectPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    Write-Host "`n> $PY $($Arguments -join ' ')" -ForegroundColor Cyan
    & $PY @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "project Python failed with exit $LASTEXITCODE"
    }
}

Invoke-ProjectPython -Arguments @(
    "experiments\run_icomp2026_campaign.py", "--config-dir", "configs",
    "--output-root", $OUT, "--campaign-id", $CID,
    "--model-policy", $POLICY, "--through", "summarize",
    "--max-parallel", "3", "--allow-test-access"
)
Invoke-ProjectPython -Arguments @(
    "conference-paper\sync_figures.py", "--source", $FIGURES, "--check"
)
Invoke-ProjectPython -Arguments @(
    "conference-paper\sync_figures.py", "--source", $FIGURES,
    "--dest", "conference-paper\figures"
)

Push-Location conference-paper\paper
try {
    & pdflatex -halt-on-error icomp2026_conference.tex
    if ($LASTEXITCODE -ne 0) { throw "first pdflatex pass failed" }
    & bibtex icomp2026_conference
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed" }
    & pdflatex -halt-on-error icomp2026_conference.tex
    if ($LASTEXITCODE -ne 0) { throw "second pdflatex pass failed" }
    & pdflatex -halt-on-error icomp2026_conference.tex
    if ($LASTEXITCODE -ne 0) { throw "final pdflatex pass failed" }
} finally {
    Pop-Location
}
```

第二组中断后也原样重输该代码块；campaign 会继续未完成 job，已验证成功的 job 不会重算。第二组一次性快照为：

```powershell
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY experiments\monitor_icomp2026_campaign.py --output-root results\icomp2026_formal --campaign-id icomp2026-formal-v2 --group downstream
```

若需要持续监测，在对应一次性命令末尾增加 `--watch --interval-seconds 300`。按 `Ctrl+C` 只结束只读监测器，不会终止实验。

### 论文 Overview 图契约

论文排版中的 Figure 3 “Overview of MinPPI” 使用资产 `fig1_overview.pdf`。它使用 2 行 × 3 列布局：第一行依次为构造投票博弈的 MinSS 分类、与当前扰动核一致的条件性 Algorithm A/B 解析曲线、Madelon 与 Credit-Card-Fraud 的 seed-42 held-out Pareto fronts 和四个注册 alpha 点；第二行只放各自图例/代码说明。正式生成会从 `raw_metrics.csv` 验证 alpha 确实来自 seed 42、trial 0 和完整 `{0.85,0.90,0.95,0.99}` 网格。Smoke 不包含这两个目标数据集，因此第三个面板只显示 scope 占位说明，不能替代正式图。

面板 (b) 仅是明确标注 oracle-guidance 条件的构造博弈说明，不证明数据集无关的样本复杂度优势，也不使用旧草稿中不可复现的 `38×` 表述。`fig3_prefilter.pdf` 只是资产文件名，并不是论文 Figure 3；它在正文中承担独立的 1 行 × 2 列预过滤结果图职责，只比较 held-out hypervolume 与 convergence evaluations。

## 正式结果验收

只有同时满足以下条件，才能把结果写入论文：

1. 三个 shard 的 canonical seed 并集恰为五个 seeds，且没有重叠；正式数据集集合恰为六个。
2. 所有必需 job 成功；没有未解决的 failed、pending、running 或未批准 skipped job。
3. protocol evidence、source fingerprint、model-policy digest、raw-authority digest 和 scientific-environment digest 在全部 shard 中一致。
4. 每个 shard 的冻结元数据 marker、精确 artifact 集合及哈希、协议/环境绑定均有效。
5. campaign summary manifest 报告 `publication_allowed: true`。
6. `raw_metrics.csv`、`aggregate_metrics.csv`、`paired_comparisons.csv`、`plot_sources.csv`、LaTeX macros 与 provenance/manifest 属于同一个 summary bundle。
7. 三张论文图从该 bundle 重新生成；Overview 图的 alpha raw seed authority 与 seed-42 fronts 完整，并通过 campaign 摘要、manifest、PDF magic、长度和 SHA-256 同步检查。
8. 历史 `v1` 的投稿视图还必须记录 post-hoc 完整性排除、原七数据集 summary digest、六数据集派生表哈希，并从 raw/front 行级证据重新聚合；不得只在图片或现成 aggregate 表中隐藏被隔离数据集。

## 辅助材料

- `configs_credit_probe/`：单数据集方向性探测，配置不完整且不能替代六数据集 formal policy。
- `demos/`：算法性质和教学示例，不产生论文正式指标。
- `thesis/`：硕士论文未来研究草稿，包含尚未验证的计划性内容。
- `experiments/README.md`：正式执行器、恢复和输出契约的开发者说明。
- `configs/README.md`：正式配置字段及作用域说明。

## 引用

```bibtex
@inproceedings{minppi2026,
  title     = {Min-Marginal Power Indices for Multi-Objective Feature Selection},
  author    = {Anonymous},
  booktitle = {ICOMP},
  year      = {2026},
  note      = {Manuscript in preparation}
}
```
