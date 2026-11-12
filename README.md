# Master's thesis workspace / 硕士论文工作区

本工作树从最终会议论文版本 `9086ddc071c3e173f5ba3b1a0b61b123677b2605`
复制而来，专用于后续硕士论文撰写与实验。原 `../Master` 保持在 `main`
分支，仅作为最终版本参考；本工作树使用独立分支 `thesis/masters`，共享同一
Git 仓库和远程 `origin`，文件与暂存区彼此独立。

数据、结果和历史材料 `data/`、`results/`、`packages/` 已复制为独立文件，
继续按原规则被 Git 忽略；不是指向原目录的链接。Git 提交只保存未被忽略的文件。

## 共享 Python 环境

复用现有 `../venv`，无需安装依赖。其可编辑安装仍指向原 `Master/src`，
所以在本工作树中统一使用以下入口，确保加载本工作树的源码：

```powershell
cd 'C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\MastersThesis'
.\thesis-python.ps1 -c "import minppi_mofs; print(minppi_mofs.__file__)"
.\thesis-python.ps1 -m pytest tests -q -p no:cacheprovider --basetemp ".t$PID"
```

该入口仅在当前命令期间设置源码路径、禁用 Python 字节码写入，并将
Matplotlib / Numba 缓存放在本工作树的 `tmp/` 中；退出后恢复原环境和目录。
执行下方历史说明中的 Python 命令时，将 `& $PY` 换成 `./thesis-python.ps1`。
不要重复执行下方的环境安装步骤，也不要执行 `pip install -e .`，否则会改变
共享环境的源码指向。共享依赖的升级会同时影响两份项目。

## PIv1 与直接 Gain 的五个 seed 对照

在 `Paper` 目录运行以下 PowerShell 命令。第一条按冻结的 PIv1 普通投票配方汇集完整历史配对记录并补算缺少的划分，实时打印进度；第二条在全部 60 次运行完成后生成对比表。将第二条命令中的目录名替换为第一条打印的实际结果目录。需要接续中断的批次时，给第一条命令加 `--output <本次运行目录>`。[正式结果和实验口径](docs/thesis/experiments/piv1-five-seeds.md)

```powershell
.\MastersThesis\thesis-python.ps1 experiments/run_piv1_benchmark.py
.\MastersThesis\thesis-python.ps1 experiments/summarize_piv1.py 'results/thesis_benchmark/<本次运行目录名>'
```

## PIv0 与直接 Gain 的五个 seed 对照

在 `Paper` 目录运行下列命令，对六个正式预处理数据集逐一比较 PIv0 和直接 Gain。每个 seed 使用自己冻结的数据分区与预处理列，按整体验证效用选择保留列数。新训练的两种方法按配对顺序串行运行，每个方法独立计时。

```powershell
.\MastersThesis\thesis-python.ps1 experiments/run_piv0_benchmark.py
```

程序复用已完成并通过核验的 seed 42 配对结果，保留其原始耗时，再运行 seed 123、456、789 和 1024。进度写入本次运行目录的 `run.log`。重新启动时，可用 `--output <本次运行目录>` 接续已有结果。60 次运行全部完成后，用以下命令生成每数据集汇总、30 组配对明细和总耗时表；将引号内的路径替换为程序打印的实际目录。[五个 seed 的对照表与逐组数据](docs/thesis/experiments/piv0-five-seeds.md)

```powershell
.\MastersThesis\thesis-python.ps1 experiments/summarize_piv0.py 'results/thesis_benchmark/<本次运行目录名>'
```

## 硕士论文正式实验（六个数据集、五个 seed）

打开 PowerShell，在包含 `Master`、`MastersThesis` 和 `venv` 的 `Paper` 目录
运行下面这条命令。它使用共享环境，逐组启动 30 个任务，并实时输出进度。

```powershell
.\MastersThesis\thesis-python.ps1 experiments/run_thesis_benchmark.py
```

如需只核对数据、配置与任务顺序，在同一条命令末尾加 `--plan`。正式数据集为
Abalone、Spambase、Madelon、Gisette、Credit Card Fraud 和 ISOLET；seed 为
42、123、456、789、1024。启动时先检查冻结协议、模型参数和 30 组预处理分区，
只有三个分片除 seed 列表外的完整协议相符，且六个数据集的模型参数都与公共参数
相符，配置检查才会通过。每个任务加载相应数据时再校验 CSV 的 SHA-256。
本次核验的输入清单指纹为
`3fb4b9716759de8b3d4ebede5422a41a9a618e9b833f2437fa83bd29d252c0f3`。

30 组共用 CUDA XGBoost（100 轮、深度 4、学习率 0.1、`n_jobs=1`）。每组沿同一条
MinPI 抽样序列执行到 K=3，采样和整组任务均无时长上限。第 1、2、3 轮结束时分别用
截至当轮的累计证据进行 `guarded` 选列。得到 −1 见证的特征已经达到 MinPI 下界，
后续轮次可直接沿用该证据。每轮都从同一份完整特征表重新选列，因此上一轮删掉的列
可以在下一轮被保留。

每个 K 都在同一验证集上为全特征和所选子集分别决定使用 100 轮预测或 150 与 200 轮
预测的平均值。三个 K 的选集和预测方案全部确定后，每组读取一次测试集，给每个 K
独立计分，相同特征集复用模型。回归以 R² 计分，二分类以 ROC AUC 计分，ISOLET 以
宏平均 OvR AUC 计分。Spambase 的 seed 123 保留了 55 个预处理输入列，其余四个 seed 为 56 列。每组压缩率使用
本组的输入列数作分母。

每次启动都会打印一个新的 `results/thesis_benchmark/` 运行目录。可从 `Paper`
目录在另一个 PowerShell 窗口实时查看本次日志，将下方占位符替换成终端打印的目录名。

```powershell
Get-Content -LiteralPath '.\MastersThesis\results\thesis_benchmark\<本次运行目录名>\run.log' -Tail 15 -Wait
```

`config_audit.json` 保存本轮配置核对结果。终端与 `run.log` 每 10 秒更新当前阶段、
已用时间及可估计的剩余时间。每组的 `k-1/`、`k-2/`、`k-3/` 目录先写采样证据
（`sampling.json`）和验证集决定（`validation.json`），测试评分后再写 `result.json`。

| 运行目录中的文件 | 如何使用 |
| --- | --- |
| `jobs.csv` | 每组的运行状态和实际总耗时，完整批次共 30 组。 |
| `summary.csv` | 每组每个 K 的删列数、验证分数、测试分数和耗时，共 90 条。`selection_delta` 用固定 100 轮预测器比较选列收益，`predictor_delta` 衡量同一选集上调整预测轮数的收益。 |
| `dataset_summary.csv` | 每个数据集分 K 汇总已完成的 seed，给出均值与样本标准差。 |
| `marginal.csv` | 同一数据集、同一 seed 下 K=1→2 和 2→3 的差值，共 60 对。可查看多删几列、删除集合变化、新增 −1 见证、分数变化及新增采样耗时。 |
| `marginal_summary.csv` | 每个数据集分别汇总两次增量。比较五个 seed 上的平均分数增量、离散程度、改善次数及单位采样时间的收益。 |

例如，`guarded_original_test_delta_mean` 衡量在固定预测器下增加一轮带来的平均测试分数变化。
两段增量都为正且第二段较小时，表示第三轮仍有正收益，但分数增幅减小。
如果分数相近，还应结合删列数和新增耗时判断。对于验证集选定的预测器，
2→3 行的 `second_difference_guarded_chosen_test_mean` 给出同一 seed 上两次分数增量之差的均值。
`additional_sampling_s` 只计新增采样耗时，`checkpoint_selection_s` 与 `checkpoint_test_s`
分别记录该 K 的选列验证和测试评分耗时。各 K 共用拟合缓存，独立重跑的耗时可能不同。

## 限时试验（test）

试跑使用单独的入口和结果目录。从 `Paper` 目录运行下面的命令可在单个数据集和 seed
上测试整条流程。试跑默认采样时限 780 秒，单组硬截止 900 秒。

```powershell
.\MastersThesis\thesis-python.ps1 experiments/test/run_thesis_benchmark.py --datasets abalone --seeds 42
```

试跑结果写入 `results/test/thesis_benchmark/`。此前单 seed 方案比较的运行脚本、
实验记录和已核验结果分别集中在 `experiments/test/`、`docs/thesis/test/` 和
`results/test/`。[试验入口清单](experiments/test/README.md) · [采样与方案说明](docs/thesis/test/sampling-methods.md)

## Git 使用

```powershell
git status --short --branch
# 在本工作树提交后，将硕士论文分支推送到原仓库：
git push -u origin thesis/masters
```

所有硕士论文修改均在这里完成，使用新的实验 run ID；不向 `main` 合并。
下面保留会议论文原始项目说明，供研究和复现实验参考。

---

# MinPI — Min-Marginal Power Indices for Multi-Objective Feature Selection

Research code and experimental materials accompanying the manuscript by **Wenjie Shi and
Chi Zhao**, Saint Petersburg State University.

A mean marginal contribution describes whether a feature helps on average across coalitions. **MinPI asks
whether adding that feature can turn an acceptable model into an unacceptable one.** It
applies minimum marginal aggregation to a binary utility-threshold game. An observed
harmful label comes with an evaluated coalition that witnesses the threshold crossing.
These candidates support validation-gated feature deletion and prefiltering for
multi-objective feature selection.

The manuscript uses **MinPI** for the Min-Marginal Power Index. The distribution name
`minppi-mofs`, Python package `minppi_mofs`, configuration keys and `icomp2026_*` script
names are retained as implementation identifiers.

## Article and poster

This README follows the current final manuscript, including its six-dataset analysis.
The article's existing filename is retained; the ASCOMP 2026 poster is a separate artifact.

| Artifact | PDF | Editable LaTeX |
|---|---|---|
| Final manuscript | [Article](conference-paper/paper/icomp2026_conference.pdf) | [Article source](conference-paper/paper/icomp2026_conference.tex) |
| ASCOMP 2026 poster, 120 × 90 cm | [Poster](conference-paper/poster/ASCOMP_MinPI_Poster_Wenjie_SHI.pdf) | [Poster source](conference-paper/poster/latex/main.tex) |

The poster has a three-column layout, standard LaTeX mathematics and two upper-right QR
codes. [Print instructions](conference-paper/poster/PRINT_README.txt) accompany the source.
Historical articles, poster revisions and submission packages are kept separately in
`conference-paper/archive/` and `packages/`; compiler output and QA files live in `tmp/`.

## Method

Let $N$ be the analyzed feature set, $n=|N|$, and $U_{\mathrm{val}}(S)$ the validation
utility of a model fitted on training data using features $S$. With
$U_{\mathrm{val}}(N)>0$ and quota $\alpha\in(0,1]$, define

$$
v_\alpha(S)=\mathbf{1}\!\left[U_{\mathrm{val}}(S)\geq\alpha U_{\mathrm{val}}(N)\right],
\qquad
\Delta_i(S)=v_\alpha(S\cup\{i\})-v_\alpha(S).
$$

The exact minimum Shapley–Shubik and minimum Banzhaf analogues coincide because both
range over every coalition excluding $i$:

$$
\operatorname{MinPI}_i=\operatorname{MinSS}_i=\operatorname{MinBz}_i
=\min_{S\subseteq N\setminus\{i\}}\Delta_i(S)\in\{-1,0,+1\}.
$$

- **Harmful (−1):** at least one coalition changes from at or above the quota to below it when
  the feature is added. That evaluated coalition is the harmful witness.
- **Neutral (0):** all marginals are nonnegative and at least one is zero.
- **Decisive (+1):** every coalition changes from below the quota to at or above it.

These exact classes are relative to the model, data split, utility and quota. Finite-budget
estimators return the minimum observed marginal: a sampled −1 is witness-certified;
sampled 0 and +1 are estimates of the exact class.

### Estimation and selection

1. **Construct the game.** Fit preprocessing and coalition models on training data;
   evaluate the game on validation data. The empty coalition predicts training class
   priors for classification or the training response mean for regression.
2. **Estimate minimum marginals.** Algorithm A samples a coalition size uniformly, then
   a subset uniformly at that size. Algorithm B constructs a seed from a maximum-weight
   absolute-correlation tree and training-only mutual information, then independently
   drops seed members with probability 0.3 and adds up to five available features.
   Each trial restarts from the seed; both estimators stop when a −1 witness is found.
3. **Dispatch by analyzed dimension.** The removal experiments use exact enumeration for
   $n\leq20$, B for $20<n\leq75$ (A if correlation guidance is unavailable), and the
   featurewise minimum of A and B for $n>75$.
4. **Validate joint removal.** Collect $H=\{i:\widehat{\operatorname{MinPI}}_i=-1\}$.
   Remove all of $H$ when
   $U_{\mathrm{val}}(N\setminus H)\geq U_{\mathrm{val}}(N)-\varepsilon$; otherwise
   remove none. Use $\varepsilon=0.005$ for classification and $0.01$ for regression.
5. **Evaluate removal and search.** Compare retained subsets with matched-count controls
   on held-out data. The optional NSGA-II experiment compares full-space and
   MinPI-prefiltered search, selecting fronts on validation and evaluating them on test.

NSGA-II minimizes feature fraction and clipped validation utility loss:

$$
\left(\frac{|S|}{n},\;1-\operatorname{clip}(U_{\mathrm{val}}(S),0,1)\right).
$$

Both variants use the same analyzed dimension $n$ before prefiltering, population 40,
800 evaluations and hypervolume reference point $(1.1,1.1)$.

## Experimental design reported in the article

| Component | Setting |
|---|---|
| Benchmarks | Spambase, Madelon, ISOLET, Gisette, Credit-Card-Fraud, Abalone |
| Seeds | 42, 123, 456, 789, 1024 |
| Split | 64% training / 16% validation / 20% test; stratified for classification |
| Preprocessing | Training-only variance filtering, MI selection to at most 100 features, standardization |
| Learner | XGBoost; 100 boosting rounds, depth 4, learning rate 0.1, one CPU worker per fit, CUDA tree backend |
| Utility | ROC AUC for binary tasks; one-vs-rest AUC for ISOLET; $R^2$ for Abalone |
| Quota | Default $\alpha=0.95$; sweep 0.85, 0.90, 0.95, 0.99 at seed 42 |
| Sampling budget | 2, 5, 10, 25 draws per feature per estimator; ISOLET uses 2, 5, 10; removal uses the largest budget |
| Controls | Full model, SHAP-bottom, MinShap-bottom, Banzhaf-bottom, random-matched removal |
| Random baseline | 30 matched-count trials averaged within each seed |

The 100-round model setting was selected by training/validation-only calibration from
10, 20, 40 and 100 rounds. SHAP-bottom ranks mean absolute training TreeSHAP values;
MinShap-bottom and Banzhaf-bottom rank the minimum and mean continuous utility marginal
from shared sampled coalitions. All deletion controls use the accepted MinPI removal count.

The formal exact-reference estimator comparison is capped at 16 features and therefore
uses eight-feature Abalone. This benchmark cap is distinct from the dispatcher's
$n\leq20$ exact branch. Guidance/dispatcher ablations use seed 42 on Madelon, ISOLET,
Credit-Card-Fraud and Abalone; the utility ablation uses ISOLET at seed 42.

The original campaign included seven datasets. Communities & Crime was excluded from
the final analysis after outcome-component leakage was identified, and all reported
aggregates were recomputed for the six retained datasets. Current dataset configurations
list those six benchmarks; historical campaign records remain available for provenance.

## Main results

At $\alpha=0.95$, across 30 dataset–seed units, MinPI removed **12.53% of analyzed
features** with a mean held-out utility change of **−0.000313**. The table reports
five-seed means ± sample standard deviations; compression is relative to the analyzed
features after preprocessing. Here $\Delta U=U_{\mathrm{test}}(S_{\mathrm{ret}})-U_{\mathrm{test}}(N)$.

| Dataset | Removed features (%) | Held-out $\Delta U$ |
|---|---:|---:|
| Madelon | 43.60 ± 31.96 | +0.00469 ± 0.00497 |
| Abalone | 27.50 ± 25.62 | −0.00565 ± 0.01077 |
| Credit-Card-Fraud | 2.76 ± 2.89 | −0.00078 ± 0.00335 |
| Spambase | 0.72 ± 0.99 | −0.00003 ± 0.00007 |
| Gisette | 0.60 ± 0.89 | −0.00011 ± 0.00016 |
| ISOLET | 0.00 ± 0.00 | 0.00000 ± 0.00000 |

- **Matched removal:** Madelon achieved $\Delta U=+0.0047$ versus −0.0732 for random
  deletion of the same size; Abalone achieved −0.0056 versus −0.0370.
- **Prefiltered search:** the dataset-averaged held-out hypervolume change was +0.01584,
  with positive dataset-averaged differences in all five seeds (Holm-adjusted $p=0.4375$).
  Madelon contributed the largest change, +0.09515; outcomes varied across datasets.
- **Estimator comparison:** on Abalone at budget 25, mean harmful-set $F_1$ was 0.443
  for A and 0.313 for B. A exceeded B at every tested finite budget; the hybrid achieved
  $F_1=1$ at each budget by selecting exact enumeration.

The clearest gains occur where many features can be removed, especially Madelon.
Exact enumeration is the article's preferred option when feasible; beyond that, its
estimator analysis favors size-uniform A as the default. The hybrid routing above is
the fixed policy used in the reported removal experiments.

## Repository layout

```text
Master/
├── configs/                  dataset and experiment settings
├── data/                     raw datasets and local data artifacts
├── experiments/              runners, calibration, statistics and figure generation
├── src/minppi_mofs/           core implementation (legacy package name)
├── tests/                    implementation tests
├── conference-paper/
│   ├── paper/                final article PDF, LaTeX and bibliography
│   ├── figures/              article figures
│   ├── poster/               final poster, LaTeX, assets and print instructions
│   ├── presentation/         presentation materials
│   └── archive/              previous article and poster versions
├── results/                  experiment records and resumable outputs
├── packages/                 historical submission and review bundles
├── docs/                     design and revision history
├── tmp/                      compiler output, visual checks and scratch files
└── demos/                    teaching examples
```

## Requirements

- Python 3.12+ (3.13 recommended).
- A CUDA-capable NVIDIA GPU and compatible driver, with a CUDA-enabled XGBoost wheel —
  **required for the formal campaign** (Group 1 stages and the campaign run with `--backend cuda`).
- Article build: `latexmk`, `pdflatex` and `bibtex`. Poster build: XeLaTeX, Windows Arial
  and the included Latin Modern Math font. These are separate from experiment execution.
- `git` — required for the formal provenance gate (the run verifies a clean working tree).
- Network access to download the six raw datasets during Group 1 acquisition (for example, the
  Credit-Card-Fraud raw snapshot is ~149 MB).
- Sufficient disk space for the raw snapshots and the resumable `results/` outputs.
- A clean git working tree before the formal run.

## Environment Setup

From a fresh clone, create the project-adjacent virtual environment first, then install and
verify. On Windows the interpreter is `venv\Scripts\python.exe`; on Linux/macOS it is
`venv/bin/python`.

```powershell
# one-time setup, run from the Master directory
python -m venv ../venv
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY -m pip install --upgrade pip
& $PY -m pip install --no-build-isolation -r requirements.txt
& $PY -m pip check
& $PY -c "import minppi_mofs; print('import OK')"
```

## Experiment execution

Prerequisites: complete the Environment Setup above (create and install the `../venv`), ensure a
CUDA environment is available, and keep the git working tree clean. All commands run from the
`Master` directory in PowerShell with the project venv. Only one campaign process may run at a
time; do not launch other GPU/model jobs concurrently. The commands below run the current
six-dataset configuration. Use a new set of run IDs for a new protocol or source revision;
reuse IDs only to resume their matching run. The article's reported results are the
historical six-dataset publication analysis described above.

### Group 1 — Acquisition, Calibration, Sealed Estimators, Screening

Runs environment acceptance, raw-data acquisition/verification, bootstrap, calibration, smoke,
the sealed estimator campaign, and validation-only screening. It fails fast if any sub-command
fails. The git working tree must be clean (`git status --short` empty) before the formal run;
re-run the same block after an interruption to resume.

```powershell
$ErrorActionPreference = "Stop"
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY experiments\run_icomp2026_group1.py --bootstrap-id bootstrap-s42-v2 --smoke-id smoke-s42-v2 --campaign-id icomp2026-formal-v2
if ($LASTEXITCODE -ne 0) { throw "Group 1 failed with exit $LASTEXITCODE" }
```

### Group 2 — Held-out Tests, Summarization and Verified Figures

Only run after confirming the Group 1 validation direction. Opens the registered held-out-test
access, completes ablation and the single campaign summary, regenerates the three formal figures,
and exports a verified figure bundle. Article compilation is a separate step below.

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
    if ($LASTEXITCODE -ne 0) { throw "project Python failed with exit $LASTEXITCODE" }
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
    "--dest", "tmp\verified-paper-figures\$CID"
)
```

Re-run the same block after an interruption; the campaign resumes unfinished jobs and skips
already-verified ones.

The exported figures stay under `tmp/` for review. Updating the final manuscript's
`conference-paper/figures/` is an explicit publication update, separate from running
another experiment.

## Compile the article or poster

These commands use the local LaTeX sources and keep the resulting PDFs and compiler
intermediates under `tmp/`. They leave the final PDFs linked above unchanged.

```powershell
# Run from Master/
Push-Location conference-paper/paper
try {
    & latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../../tmp/pdfs/article-final-build icomp2026_conference.tex
    if ($LASTEXITCODE -ne 0) { throw "Article build failed" }
} finally { Pop-Location }

New-Item -ItemType Directory -Force tmp/pdfs/ascomp-poster/latex-build | Out-Null
Push-Location conference-paper/poster/latex
try {
    & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=../../../tmp/pdfs/ascomp-poster/latex-build main.tex
    if ($LASTEXITCODE -ne 0) { throw "First poster pass failed" }
    & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=../../../tmp/pdfs/ascomp-poster/latex-build main.tex
    if ($LASTEXITCODE -ne 0) { throw "Second poster pass failed" }
} finally { Pop-Location }
```

Outputs: `tmp/pdfs/article-final-build/icomp2026_conference.pdf` and
`tmp/pdfs/ascomp-poster/latex-build/main.pdf`. The poster project includes its text,
formulas, background and math font; compiling it does not require rerunning experiments.

## Experiment Scripts and Recommended Order

All experiment scripts live under `experiments/`. The manifest pipeline is the only formal
execution path; run scripts directly only when the README instructs it.

### Recommended running order (two groups)

| Step | Script | Purpose |
|---|---|---|
| **Group 1** | `run_icomp2026_group1.py` | One-click entry: environment acceptance → data acquisition → bootstrap → calibration → smoke → sealed estimators → validation-only screening. Runs the `preprocess_*.py` and `calibrate_icomp2026_models.py` internally. |
| **Group 2** | `run_icomp2026_campaign.py` | Opens held-out-test access, runs removal/alpha/NSGA-II/ablation, and produces the single campaign summary (`--through summarize`). |
| | `make_paper_figures.py` | Generates the three formal figures from the canonical summary tables. |
| | `conference-paper/sync_figures.py` | Validates and copies the figure bundle to the specified destination. |
| Separate artifact build | `latexmk` / `xelatex` | Compiles the article / poster sources. |

### All scripts and their functions

**Formal entries (recommended to run, in order):**

- `run_icomp2026_group1.py` — complete Group 1 workflow (data → calibration → smoke → sealed estimators → screening).
- `run_icomp2026_campaign.py` — run the frozen protocol as three seed shards and summarize (Group 2).
- `make_paper_figures.py` — generates paper figures from canonical tables without changing experiment records.

**Supporting / monitoring:**

- `run_icomp2026_pipeline.py` — authoritative CLI controller for a single pipeline run (used by the entries; resumable with `--resume`).
- `monitor_icomp2026_campaign.py` — read-only campaign status snapshots (`--group initial` / `--group downstream`).
- `calibrate_icomp2026_models.py` — calibrate the frozen per-dataset XGBoost round policy on validation only.
- `screen_icomp2026_results.py` — provisional validation-only screening from a seed-42 run.
- `summarize_multifidelity_campaign.py` — validate shard provenance and build the campaign summary.

**Pipeline internals (imported, not run directly):**

- `icomp2026_stages.py` — manifest-authoritative stage execution engine.
- `icomp2026_estimator_bundle.py` — estimator bundle identities and schemas.
- `icomp2026_progress.py` — strict progress state and run logging.
- `native_process_guard.py` — authenticated completion receipts for worker processes.
- `summarize_icomp2026.py` — pure statistics and publication tables.

**Data acquisition (`experiments/preprocess_data/`):**

- `preprocess_abalone.py`, `preprocess_Spambase.py`, `preprocess_Madelon.py`, `preprocess_Gisette.py`, `preprocess_ISOLET.py`, `preprocess_Credit-Card-Fraud.py` — download and verify each formal raw CSV (SHA-256 fail-closed).
- `preprocess_crimedata.py` — quarantined Communities & Crime audit snapshot only (not formal).
- `raw_snapshot.py` — fail-closed raw snapshot helpers (imported).

**Diagnostics / auxiliary (optional, not part of the formal run):**

- `benchmark_icomp2026_backends.py` — validation-only throughput benchmark.
- `validate_grouped_acceleration.py` — training/validation-only acceptance gates for grouped acceleration.
- `forecast_icomp2026_48h.py` — measured runtime forecast ledger.
- `run_isolet_utility_pair.py` — ISOLET seed-42 paired utility ablation.
- `build_leakage_screened_publication.py` — historical compatibility; rebuilds the six-dataset view from the seven-dataset record (not needed for a new six-dataset campaign).

## Authority Boundaries

| Content | Source of truth |
|---|---|
| Data / split / utility | `configs/<dataset>.yaml` |
| Seeds / budget / alpha / ablations / NSGA-II | `configs/icomp2026_pipeline.yaml` |
| Execution and resume | `experiments/run_icomp2026_*.py` |
| Formal completion | campaign manifest, audit, canonical metrics, summary manifest |
| Paper evidence | six-dataset `tables/publication/` |
| Paper figures | `figures/paper/` (manifest- and hash-verified) |
| Current manuscript definitions and reported results | `conference-paper/paper/icomp2026_conference.tex` |

The pipeline YAML has a portable CPU default; the formal campaign entry selects CUDA.
Dataset-specific estimator budgets override the generic budget list. Change configurations
under a new run ID when changing the protocol. `demos/` and archived packages are auxiliary
or historical material rather than the current article's result tables.

## Tests and Lint

```powershell
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
& $PY -m pytest tests -q -p no:cacheprovider --basetemp ".t$PID"
& $PY -m ruff check src experiments tests conference-paper/sync_figures.py
```

## Citation

```bibtex
@misc{shi2026minpi,
  title     = {Min-Marginal Power Indices for Multi-Objective Feature Selection},
  author    = {Shi, Wenjie and Zhao, Chi},
  year      = {2026},
  note      = {Manuscript}
}
```
