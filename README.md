# 项目入口

## 研究方向

**会议论文 (ICOMP 2026)**: MinPPI-MOFS —— 把 voting game + 最小边际算子 + SHAP 引导的 NSGA-II 整合为多目标特征选择方法，ShapG 作为重要性 baseline 与组件。

**硕士论文**: 在会议方法基础上推广到 **bipartite (feature × architecture) factored game**，引入 **cross-Shapley** 交互值 + 现代 DARTS 进行特征选择与神经架构联合优化。

## 关键截止日期

| 日期 | 事件 |
|------|------|
| 2026-06-21 | ICOMP 摘要截止 |
| 2026-07-21 | ICOMP 全文截止（建议 07-20 提交，留 1 天缓冲） |
| 2026-08-上旬 | 论文工作启动 |
| 2026-11-11~13 | ICOMP 会议（卡萨布兰卡，摩洛哥） |
| 2027-05 | 硕士论文提交 |

## 文件结构

```
to_student/
├── README.md                            ← 本文件
├── conference-paper/
│   ├── spec.md                          会议论文规格（数据集、baseline、消融）
│   └── method-skeleton.md               §4 数学骨架（LaTeX 起稿大纲）
├── thesis/
│   ├── overview.md                      论文整体规划与时间表
│   └── cross-shapley.md                 论文理论核心：cross-Shapley 框架
└── demos/
    ├── minppi_demo.py                   会议 §4 Lemma 1 + Theorem 1 实证
    ├── algorithm_b_demo.py              会议 §4.3 算法 A vs B 对比
    └── cross_shapley_demo.py            论文 bipartite cross-Shapley 实证
```

## 阅读顺序

1. 本文件 (README.md)
2. `conference-paper/spec.md` —— 知道论文要做什么
3. `conference-paper/method-skeleton.md` —— 知道论文 §4 怎么写
4. 跑一遍 `demos/minppi_demo.py` 和 `demos/algorithm_b_demo.py` —— 看实证
5. `thesis/overview.md` —— 知道毕业论文怎么接上
6. `thesis/cross-shapley.md` —— 知道毕业论文的理论核心
7. 跑 `demos/cross_shapley_demo.py` —— 看 bipartite 推广

## 会议论文写作时间表

| 周 | 任务 |
|----|------|
| W1 (05/28-06/03) | 跑 baseline (ShapG, MinShap, Powershap, Karavas-Banzhaf) 在 5 数据集上 |
| W2 (06/04-06/10) | 主实验全表 (5 数据集 × 5 seeds × 7 方法)；起 §1-§3 写作 |
| W3 (06/11-06/17) | preliminary results 整理；起 §4 + §5；摘要写作 |
| W3 末 | **06-21 摘要提交** |
| W4 (06/22-06/28) | 完整消融实验；§6 实验章节 |
| W5 (06/29-07/05) | §2 Related Work；§7 Conclusion；所有 figures |
| W6 (07/06-07/12) | Full draft；internal review |
| W7 (07/13-07/19) | 修订；格式检查 |
| **07-20** | **提交（不要拖到 07-21）** |

## 已经完成的实证（直接进会议论文 §4 + §5）

会议论文的核心实证已经全部跑通，**不需要再做新实验**就能写 §4 + §5：

| 命题 | 实证文件 | 结论 |
|------|---------|------|
| Lemma 1 (Boundedness) | `demos/minppi_demo.py` Examples 1-3 | MinSS ∈ {-1, 0, +1} 全程成立 |
| Theorem 1 (3-way classification) | `demos/minppi_demo.py` Examples 1-2 | Decisive / Harmful / Neutral 三分类成立 |
| Theorem 1 caveat | `demos/minppi_demo.py` Example 2 | "在每个 winning 中 ≠ MinSS=+1" 反例展示 |
| Proposition 1 (Algorithm A Hoeffding bound) | `demos/minppi_demo.py` Example 4 | 理论与实测吻合 |
| Algorithm B vs A 对比 | `demos/algorithm_b_demo.py` | 38× sample efficiency @ K=1 |
| Algorithm B 鲁棒性 | `demos/algorithm_b_demo.py` | σ 噪声下 graceful degradation |

## 必读论文（按优先级）

**P0 (核心理论靠山)**:
1. **Rabitti & Borgonovo (2019)** "A Shapley-Owen Index for Interaction Quantification." *SIAM/ASA JUQ* 7(3): 1060–1075.
2. **Grabisch & Roubens (1999)** "An axiomatic approach to the concept of interaction among players in cooperative games." *Int. J. Game Theory* 28: 547–565.

**P1 (会议 §4.3 近似算法)**:
3. **Muschalik et al. (2024)** "KernelSHAP-IQ: Weighted Least-Square Optimization for Shapley Interactions" — arXiv:2405.10852
4. **Tsai et al. (2023)** "Faith-Shap: The Faithful Shapley Interaction Index" — *JMLR* 24
5. **Song, Nelson, Staum (2016)** "Shapley Effects for Global Sensitivity Analysis" — *SIAM/ASA JUQ* 4

**P2 (会议 baseline 定位)**:
6. Zheng & Raskutti (2026) "MinShap" — arXiv:2604.15107
7. Karavas (2017) "Feature importance scores and lossless feature pruning using Banzhaf power indices" — arXiv:1711.04992
8. Verhaeghe et al. (2022) "Powershap" — *ECML PKDD*
9. **Zhao, Liu, Parilina (2025)** "ShapG" — *EAAI* 148

**P3 (论文 NAS+FS, 8 月开始)**:
10. EL-DARTS (2026) —— 推荐的 DARTS 现代稳定版
11. DASH —— 快速 architecture-only 搜索
12. PC-DARTS / DARTS- / FairDARTS —— 标准稳定化变体

## 写作硬注意事项

1. **WVG-FS 的加性近似缺陷** —— 在会议论文 §4.5 显式说明如何规避（用 NSGA-II fitness 替代加性近似）
2. **§4.6 spurious interactions** —— Rabitti-Borgonovo §7.3 警告必须正面回应，审稿人必问
3. **Theorem 1 的 caveat** —— "在每个 winning 中 ≠ MinSS=+1"，必须在论文中明确说明（demo Example 2 已经实证）
4. **MinShap 是近期预印本** —— 引用为 inspiration，不依赖其结论
5. **DARTS 用现代稳定版** —— 不要用 2019 vanilla DARTS，要用 PC-DARTS + DARTS- 或 EL-DARTS 或 DASH
6. **所有论文图片统一使用 PDF 格式** —— matplotlib 用 `savefig(..., format='pdf')` 或 `.pdf` 后缀；不要用 PNG/JPG 等位图格式（会议论文与毕业论文都遵守）

## 代码标准与风格

**语言与环境**:
- Python 3.10+，所有公开 API（函数、方法、类）必须有 type hints
- 自行配置开发环境（conda / venv / uv 等任选），依赖锁定在 `requirements.txt` 或 `environment.yml` 中
- 所有 demo 与实验脚本必须可在干净 Python 环境下通过 `python <script>.py` 复现

**项目目录结构** —— 顶层目录平级：

```
to_student/
├── README.md
├── pyproject.toml         # 项目元数据 + 可安装包定义
├── requirements.txt       # 运行时依赖
├── conference-paper/      # 会议论文写作
├── thesis/                # 毕业论文写作
├── demos/                 # 教学性 demo（已有 3 个）
├── src/                   # 源代码（importable package）
├── configs/               # 实验配置（YAML / JSON）
├── data/                  # 数据集（gitignored）
├── experiments/           # 实验入口脚本
├── tests/                 # 单元测试
└── results/               # 实验产物（gitignored）
```

**各目录职责**：
- `src/` —— 可导入的源码包，按功能拆子模块（如 `src/<pkg>/voting_game.py`, `nsga2.py`, `shapg_integration.py`, `cross_shapley.py`, `baselines/`）。建议用 `pyproject.toml` 注册为 editable package：`pip install -e .`
- `configs/` —— 实验配置（YAML / JSON），**不是代码**，按数据集 / 实验维度组织
- `data/` —— 数据集（raw + processed），gitignored，README 文档化下载与预处理方式
- `experiments/` —— 实验入口脚本，`import` `src/` 的模块，读 `configs/` 的配置，写结果到 `results/`
- `demos/` —— 教学性 demo（现有 3 个保留），新 demo 沿用现有结构
- `tests/` —— 单元测试
- `results/` —— 实验输出（figures / tables / logs），gitignored

**代码组织规则**：
- 代码必须**模块化**：单一职责，函数与类粒度合理，避免几百行的大函数
- 代码必须**可读性强**：清晰命名、合理注释、避免魔法数与隐式状态
- **不使用 Jupyter notebook** —— 所有探索、实验、出图统一用 `.py` 脚本；交互式调试用 IPython 或 IDE 调试器，但提交的代码必须是 `.py`

**格式化与 lint**:
- 格式化用 `ruff format`（或 `black`，二选一项目内统一）
- Lint 用 `ruff check`，提交前必须 0 warning
- import 顺序：stdlib → 第三方 → 本项目，每组内按字母排序（`ruff` 默认）

**文档**:
- 所有公开函数/类必须有 docstring（NumPy 风格或 Google 风格，项目内统一）
- 模块顶部必须有模块级 docstring，简短说明用途 + 运行命令（参考 `demos/minppi_demo.py` 头部）
- 数学符号在 docstring 中尽量与论文一致（如 `phi_{i,o}`, `MinSS`, `v_alpha`）

**随机性与可复现**:
- 所有随机性必须通过显式 seed 控制：`rng = np.random.default_rng(seed)`，不要用全局 `np.random.*`
- 实验脚本必须接受 `--seed` 参数或在文件顶部固定 seed
- 用到的库版本写进 `requirements.txt` 或 `environment.yml`

**Demo / 实验代码**:
- 新 demo 沿用 `demos/minppi_demo.py` / `demos/algorithm_b_demo.py` 的结构：
  - 顶部 module docstring + 运行命令
  - 用 `banner()` 之类的辅助函数分节输出
  - 关键性质用 `assert` 显式验证（如 `assert min_ss in {-1, 0, 1}, "Lemma 1 violated"`）
  - 输出末尾给一句"Observation / Implication"说明结果对论文的含义
- 数据规模优先小而可复现，不要为了规模牺牲调试可读性

**实验结果管理**（W2 之后）:
- 实验配置外置（YAML 或 argparse），不要硬编码超参
- 结果保存时附带 metadata：配置 hash、seed、git commit、时间戳
- 大规模实验产物（模型权重、中间结果）不进 git，进 `.gitignore` 并文档化存储位置

### ICOMP 2026 estimator execution and accounting

- `execution.mode: legacy` runs every estimator projection independently and
  retains the historical ResultRecord/table schema. The production protocol
  uses `execution.mode: grouped_nested_v1`: one immutable bundle is built for
  each dataset/seed authority and canonical method/budget jobs are projections
  of its validated checkpoints.
- Grouped budgets are nested prefixes of the frozen maximum-budget schedule.
  A smaller budget means the deterministic prefix ending at that checkpoint;
  it is not a separately randomized or independently fitted estimator run.
- `model_fit_count` remains the backward-compatible logical count of unique
  nonempty coalitions, and equals `logical_unique_coalition_count`.
  `logical_evaluation_count` counts scheduled algorithm evaluations.
  `physical_model_fit_count`, `cache_hit_count`, and `runtime_s` are
  method/budget projection counters and component wall time.
- Complete-bundle cost is reported separately as
  `group_physical_model_fit_count`, `group_cache_hit_count`, and
  `group_runtime_s`. Summary tables emit that scope once under
  `__deduplicated_group_physical__` for each `group_request_digest`; totals
  must sum those digest-deduplicated rows, never every projected job.
- The source configuration keeps its CPU default. Every smoke or formal GPU
  invocation must explicitly pass `--backend cuda`; do not infer or silently
  select CUDA.
- A run freezes its resolved protocol and source identity. After any source
  change, create a new run ID instead of resuming, deleting, or overwriting an
  existing run directory.
- `icomp2026-formal-20260723` is a stopped, read-only legacy run. Never open,
  resume, mutate, or mix its artifacts/tables with grouped results. A grouped
  formal run requires a new run ID and its own complete publication lineage.

**图片**:
- 所有论文图片用 PDF 格式（见上一节第 6 条）
- matplotlib 出图统一字体大小，与论文正文协调
- 出图脚本必须可独立运行重生成图片，不依赖 notebook 状态

## 算力

- **3× H100** 可用
- 会议论文主要 CPU-bound (XGBoost fitness)，H100 用来**并行扫描** (data × seed × method 矩阵)
- 论文 Phase B 起 GPU-bound (DARTS supernet 训练)，3 张 H100 用得满
- 会议阶段可以留 1 张 H100 做 FT-Transformer + DeepSHAP 作为重要性替代实验

## 遇到困难时的 fallback

- **数学卡住**：与导师讨论，特别是 cross-Shapley 的 A4 公理 reformulation
- **DARTS 不收敛**：切到论文 cross-Shapley 文档底部的 Topic 4-lite (NSGA-II 搜架构超参，no DARTS)
- **会议实验跑不完**：砍数据集到 3 个 (Spambase, Madelon, Credit-Card-Fraud)，baseline 砍到 5 个核心
- **Theorem 1 (cross-Shapley uniqueness) 证不出**：fallback 为 "well-definedness only"，论文照常发，理论一致性留 future work

## ICOMP 2026 多保真实验流程（2026-07-28）

所有命令都在 `Master` 目录的 PowerShell 中运行。Seed 42 筛选阶段只允许
使用训练集和验证集；在人工检查 `screen_report.md` 以前，不要启动
`removal`、`alpha`、`nsga2` 或其他会访问测试集的阶段。

```powershell
$PY = "..\venv\Scripts\python.exe"
$ROOT = "results\icomp2026_multifidelity"
$BOOT = "bootstrap-s42-v1"
$SMOKE = "policy-smoke-s42-v1"
$SCREEN = "screen-s42-v1"
$POLICY = "$ROOT\$BOOT\calibration\model_policy.json"
```

先完成预处理并校准每个数据集的 XGBoost 轮数：

```powershell
& $PY -m experiments.run_icomp2026_pipeline --stage validate --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $BOOT
& $PY -m experiments.run_icomp2026_pipeline --stage preprocess --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $BOOT --resume
& $PY -m experiments.calibrate_icomp2026_models --run-root "$ROOT\$BOOT" --config-dir configs --backend cuda --seed 42 --candidate-rounds 10 20 40 100 --output $POLICY
```

对两个 smoke 数据集验证冻结策略，全程不访问测试集：

```powershell
& $PY -m experiments.run_icomp2026_pipeline --stage validate --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $SMOKE --smoke --model-policy $POLICY
& $PY -m experiments.run_icomp2026_pipeline --stage preprocess --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $SMOKE --smoke --model-policy $POLICY --resume
& $PY -m experiments.run_icomp2026_pipeline --stage estimator --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $SMOKE --smoke --model-policy $POLICY --resume
```

运行七个数据集的 seed 42 筛选实验并生成方向性报告：

```powershell
& $PY -m experiments.run_icomp2026_pipeline --stage validate --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $SCREEN --model-policy $POLICY --estimator-budgets 2 5
& $PY -m experiments.run_icomp2026_pipeline --stage preprocess --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $SCREEN --model-policy $POLICY --estimator-budgets 2 5 --resume
& $PY -m experiments.run_icomp2026_pipeline --stage estimator --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $SCREEN --model-policy $POLICY --estimator-budgets 2 5 --resume
& $PY -m experiments.screen_icomp2026_results --run-root "$ROOT\$SCREEN" --config-dir configs --model-policy $POLICY --backend cuda --seed 42 --estimator-budgets 2 5
```

估计器中断后只需执行以下命令。分组 SQLite 缓存与进度文件会跳过已经完成的
物理拟合：

```powershell
& $PY -m experiments.run_icomp2026_pipeline --stage estimator --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $SCREEN --model-policy $POLICY --estimator-budgets 2 5 --resume
```

持续查看日志，或列出每个数据集最新的进度文件：

```powershell
Get-Content "$ROOT\$SCREEN\output.log" -Tail 30 -Wait
```

```powershell
Get-ChildItem "$ROOT\$SCREEN\estimator\_bundles" -Recurse -Filter progress.json | Sort-Object LastWriteTime -Descending | Select-Object -First 7 FullName,LastWriteTime
```

通过验证集冻结门后，方向性 `$SCREEN` 不进入最终主表。使用完整
`K=10,25,50` 重新运行正式 seed 42，再使用同一模型策略和未改动的 Python
源码运行剩余四个 seed：

```powershell
$FORMAL42 = "formal-s42-v1"
$CONFIRM = "confirm-s123-456-789-1024-v1"
& $PY -m experiments.run_icomp2026_pipeline --stage validate --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $FORMAL42 --model-policy $POLICY
& $PY -m experiments.run_icomp2026_pipeline --stage preprocess --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $FORMAL42 --model-policy $POLICY --resume
& $PY -m experiments.run_icomp2026_pipeline --stage estimator --config-dir configs --output-root $ROOT --seeds 42 --backend cuda --run-id $FORMAL42 --model-policy $POLICY --resume
& $PY -m experiments.run_icomp2026_pipeline --stage validate --config-dir configs --output-root $ROOT --seeds 123 456 789 1024 --backend cuda --run-id $CONFIRM --model-policy $POLICY
& $PY -m experiments.run_icomp2026_pipeline --stage preprocess --config-dir configs --output-root $ROOT --seeds 123 456 789 1024 --backend cuda --run-id $CONFIRM --model-policy $POLICY --resume
& $PY -m experiments.run_icomp2026_pipeline --stage estimator --config-dir configs --output-root $ROOT --seeds 123 456 789 1024 --backend cuda --run-id $CONFIRM --model-policy $POLICY --resume
```

最终索引会拒绝 seed 重叠、模型策略变化、实验设置变化和源代码指纹变化：

```powershell
& $PY -m experiments.summarize_multifidelity_campaign --shard "$ROOT\$FORMAL42" --shard "$ROOT\$CONFIRM" --output "$ROOT\campaign_index.json"
```

### 2026-07-28：48 小时正式执行入口

上面的逐 run 命令是旧多保真流程说明。当前 1-GPU、48 小时正式协议和全部
PowerShell 命令以
[`ICOMP2026_48H_EXECUTION_FRAMEWORK.md`](ICOMP2026_48H_EXECUTION_FRAMEWORK.md)
为准。

测试集密封的第一段：

```powershell
$PY = "..\venv\Scripts\python.exe"
$OUT = "results\icomp2026_48h"
$CID = "icomp2026-formal-20260728-v1"
$POLICY = "results\icomp2026_multifidelity\bootstrap-s42-20260728-v2\calibration\model_policy.json"
& $PY experiments\run_icomp2026_campaign.py --config-dir configs --output-root $OUT --campaign-id $CID --model-policy $POLICY --through estimator --max-parallel 3
```

人工检查 seed-42 validation report 后，显式完成全部 P0、统一五种子汇总、排名
统计和论文图：

```powershell
& $PY experiments\run_icomp2026_campaign.py --config-dir configs --output-root $OUT --campaign-id $CID --model-policy $POLICY --through summarize --max-parallel 3 --allow-test-access
```

两条命令都可在中断后原样重输以安全恢复。不要在正式 campaign 创建后修改
Python/YAML/model policy，也不要复制旧 run 的 cache。
