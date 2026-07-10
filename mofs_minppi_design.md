# MOFS-MinPPI 分层重构方案：从代码清理到可发表论文

**日期**：2026-07-10
**语言**：中文
**目标**：按照标准软件工程分层架构，彻底清理项目、新增 NSGA-II baseline 方法、重新规划实验流程，最终产出可发表的 MOFS-MinPPI 论文。

---

# Part A：当前代码审计与清理

## A1. 必须删除的文件（Plan A/B 残留 + 重复实验）

```
# ====== 源码层：从未存在或被删除的模块引用 ======
tests/test_mofs_minppi.py
    → 导入 from minppi_mofs.mofs_minppi import ...
    → 该模块已在 Plan C 中被删除。test 文件本身就是孤儿代码。

# ====== baselines 层：已删除的 NSGA-II 方法 ======
# （不需要删除文件——baselines/ 目录当前已是干净的 5 个方法，
#  旧 NSGA-II baseline 代码早已移除。仅需确认目录下无残留。）

# ====== experiments 层：Plan A/B 残留 ======
experiments/run_ablation.py
    → 70% dead code。调用已删除的 mofs_minppi_search()。
      消融 1（min vs mean）和消融 3（β sweep）完全不可执行。
      消融 2（binary vs continuous）已被 E3（run_pareto_eval.py）覆盖。
      消融 4（ShapG vs SHAP）单 seed 分析，价值有限。

experiments/run_alpha_sweep.py
    → 功能被 E3（run_pareto_eval.py）完全覆盖。
      E3 跨 7 数据集 × 5 seeds × 4α 的扫参比这个单 seed 42 脚本更完整。

# ====== experiments 层：功能重复 ======
experiments/minppi_filter_eval.py
    → "MinPPI 过滤 → 训练一个 XGBoost → 提取 gain importance"。
      与 E4（eval_harmful_utility.py）的 c1 removal test 高度重叠。
      c1 removal test 更完整（含 SHAP/MinShap/Random 对照组）。

experiments/eval_pipeline_comparison.py
    → MinPPI/MinShap/Random 预处理 → SHAP-top-k 对比。
      数据集上 MinPPI 过滤几乎总为 0（因为有害特征极少），
      因此所有方法的 AUC@k 都相同——无信息量。
      被 E5（MOFS verification）替代。

experiments/madelon_alpha_recall.py
    → 单数据集（Madelon）× 单 seed 42 的 α-recall 曲线。
      E4 的 c2（ground-truth）已跨 5 seeds 计算 Madelon precision/recall/F1。

experiments/harmful_vs_dimension.py
    → 3 数据集 × 3 cutoffs × 单 seed 的分析。
      未做统计检验，仅目视判断趋势。对双栏论文而言边际价值低。

experiments/sensitivity_friedman.py
    → Friedman leave-one-out。E6 阶段的 make_cd_diagram.py 已输出
      Friedman χ²/p 值 + Nemenyi CD。留一法分析可在需要时手动运行，
      不需要独立脚本。

experiments/d1_semisynthetic_harmful.py
    → 半合成实验仍在推进中，不包含在本轮论文范围。

# ====== configs 层：NSGA-II 时代配置 ======
configs/ablation/  （整个目录）
    → alpha_sweep.yaml, beta_sweep.yaml, shapg_k_sweep.yaml
      均引用 nsga2.pop_size, nsga2.generations 等 Plan A/B 参数。

# ====== results 层：旧运行数据 ======
results/tables/2026-06-28_07-44-10/   (NSGA-II 时代)
results/tables/2026-07-01_*/
results/tables/2026-07-04_11-02-28/
results/tables/2026-07-07_*/
results/tables/2026-07-08_*/
results/tables/2026-07-09_*/
results/logs/2026-06-*/
results/logs/2026-07-01_*/
results/logs/2026-07-04_*/
results/logs/2026-07-05_*/
results/logs/2026-07-07_*/
results/logs/2026-07-08_*/
results/logs/2026-07-09_*/
results/REAMME.md                        (typo duplicate of README.md)
```

## A2. 清理后保留的目录结构

```
Master/
├── src/minppi_mofs/
│   ├── __init__.py
│   ├── voting_game.py           # v_α 二元博弈公式
│   ├── min_ppi.py               # MinSS/MinBz 估计 (exact/AlgoA/AlgoB/Hybrid)
│   ├── shapg_integration.py     # 快速 MI 引导 + 相关邻居
│   └── baselines/
│       ├── shap_top_k.py        # SHAP-Top-k
│       ├── shapg.py             # ShapG
│       ├── min_shap.py          # MinShap
│       ├── banzhaf_fs.py        # Banzhaf
│       ├── powershap.py         # Powershap
│       └── nsga2_fs.py          # [NEW] NSGA-II Feature Selection baseline
│
├── configs/
│   ├── spambase.yaml
│   ├── madelon.yaml
│   ├── isolet.yaml
│   ├── gisette.yaml
│   ├── credit_card_fraud.yaml
│   ├── crimedata.yaml
│   ├── abalone.yaml
│   └── nsga2.yaml               # [NEW] NSGA-II 参数配置
│
├── experiments/
│   ├── preprocess_data/         # 7 个数据集预处理脚本（不变）
│   ├── run_main_table.py        # 5 baseline AUC@k 排名（修改：添加 nsga2_fs baseline）
│   ├── run_pareto_eval.py       # α 扫参操作点
│   ├── eval_harmful_utility.py  # c1 删除测试 + c2 Madelon GT + c8 MinShap 对比
│   ├── run_mofs_verify.py       # [NEW] MinPPI 预筛 + NSGA-II 前端验证
│   ├── make_cd_diagram.py       # CD 图生成（不变）
│   └── make_figures.py          # Demo figures 1-3（不变）
│
├── tests/
│   ├── test_voting_game.py
│   ├── test_min_ppi.py
│   └── test_shapg_integration.py
│
├── demos/                       # 3 个 demo（不变）
├── thesis/                      # 硕士论文规划（不变）
├── conference-paper/            # LaTeX + figures
├── results/                     # 全部清空，等新实验产出
├── data/                        # 预处理数据（不变）
├── pyproject.toml
├── requirements.txt
└── mofs_minppi_design.md        # 本文档
```

---

# Part B：分层新增内容 —— 详细设计

## B1. 新增 Baseline 方法：`src/minppi_mofs/baselines/nsga2_fs.py`

### 目的

实现标准 NSGA-II 特征选择，作为多目标 baseline。MinPPI 本身不调用它——它是实验中与 MinPPI 独立的对照组。**关键设计原则**：NSGA-II 完全不知道 MinPPI 的存在。它只接受一个特征 DataFrame，返回 Pareto 前沿。MinPPI 和 NSGA-II 之间的唯一联系在**实验脚本**层面——即实验脚本先调用 MinPPI 做预筛，再把过滤后的数据传给 NSGA-II。

### 模块接口

```python
"""
NSGA-II Feature Selection baseline.

Standard multi-objective feature selection using NSGA-II.
This is a standalone method — it does NOT import from or depend on
the MinPPI package.  The only connection to MinPPI is at the
experiment-script level: the experiment script may optionally
pre-filter features using MinPPI before passing data to this module.

Usage (standalone):
    from minppi_mofs.baselines.nsga2_fs import nsga2_feature_selection
    result = nsga2_feature_selection(X_train, y_train, X_val, y_val, task="binary", seed=42)
    # result.pareto_front: list of (mask, fitness) pairs
    # result.hv: hypervolume
    # result.igd_plus: IGD+ against reference front
"""

# 核心函数签名
def nsga2_feature_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    task: str = "binary",
    seed: int = 42,
    pop_size: int = 20,
    max_gen: int = 20,
    p_crossover: float = 0.5,
    p_mutation: Optional[float] = None,  # default: 1/n_features
    ref_front: Optional[np.ndarray] = None,  # for IGD+ computation
    verbose: bool = False,
) -> NSGA2Result:
    """
    在给定的特征空间上运行 NSGA-II 多目标特征选择。

    Args:
        X_train, y_train: 训练数据。
        X_val, y_val: 验证数据（评估每个个体的目标值）。
        task: "binary" / "multiclass" / "regression"。
        seed: 随机种子。
        pop_size: 种群大小。
        max_gen: 最大代数。
        p_crossover: 均匀交叉的概率。
        p_mutation: Bit-flip 变异概率。默认为 1/n_features。
        ref_front: 参考 Pareto 前端，用于计算 IGD+。若为 None，IGD+ 返回 NaN。
        verbose: 是否打印进度。

    Returns:
        NSGA2Result(
            pareto_front: list[tuple[np.ndarray, tuple[float, float]]],
            hv: float,
            igd_plus: float | None,
            runtime_s: float,
            converged_gen: int,
            hv_history: list[float],
        )
    """

# 返回数据类型
@dataclass
class NSGA2Result:
    pareto_front: list[tuple[np.ndarray, tuple[float, float]]]
    hv: float
    igd_plus: Optional[float]
    runtime_s: float
    converged_gen: int
    hv_history: list[float]
```

### 内部实现结构（伪代码层）

```python
# ── Individual evaluation ──
def _evaluate_individual(mask, X_tr, y_tr, X_val, y_val, task, seed):
    """
    编码: mask ∈ {0,1}^n, mask[j]=1 表示选择特征 j。
    目标 1 (f₁): 基数比例 = |mask|₀ / n  （最小化 → 更少特征）
    目标 2 (f₂): 1 − AUC(M_mask)           （最小化 → 更高 AUC）
    """
    selected_cols = [X_tr.columns[j] for j in range(n) if mask[j]]
    if not selected_cols:
        return (0.0, 1.0)
    # 训练 XGBoost(n_estimators=100, max_depth=4) on selected_cols
    # 在 validation set 上计算 AUC 或 accuracy
    return (cardinality_ratio, 1.0 - utility)

# ── NSGA-II 核心 ──
def _nsga2_core(X_tr, y_tr, X_val, y_val, task, seed, pop_size, max_gen, p_cx, p_mut, verbose):
    """
    标准 NSGA-II 循环:
    1. 初始化种群（随机二元掩码）
    2. 评估所有个体
    3. For generation = 1..max_gen:
       a. 二元锦标赛选择（非支配排序 → 拥挤距离）
       b. 均匀交叉 (p_crossover)
       c. Bit-flip 变异 (p_mutation)
       d. 评估后代
       e. 合并父代+后代 → 非支配排序 + 拥挤距离 → 截断到 pop_size
       f. 记录 HV
    4. 返回最终 Pareto 前端 + HV 曲线
    """

# ── 辅助函数 ──
def _non_dominated_sort(fitness): ...
def _crowding_distance(fitness, front): ...
def _compute_hv(front, ref_point): ...
def _compute_igd_plus(front, ref_front): ...
```

### 示例用法（在实验脚本中）

```python
# ── 用法 1: 标准 NSGA-II on full features ──
from minppi_mofs.baselines.nsga2_fs import nsga2_feature_selection
result_full = nsga2_feature_selection(
    X_train, y_train, X_val, y_val, task="binary", seed=42,
)
print(f"HV (full space, {X_train.shape[1]} features): {result_full.hv:.4f}")

# ── 用法 2: NSGA-II on MinPPI-filtered features ──
from minppi_mofs.min_ppi import estimate_minppi
minppi = estimate_minppi(X_train, X_val, y_train, y_val, alpha=0.95, method="shapg_guided", ...)
harmful = {f for f, v in minppi.items() if v == -1}
keep = [c for c in X_train.columns if c not in harmful]
X_train_f, X_val_f = X_train[keep], X_val[keep]
result_filt = nsga2_feature_selection(
    X_train_f, y_train, X_val_f, y_val, task="binary", seed=42,
)
print(f"HV (filtered space, {X_train_f.shape[1]} features): {result_filt.hv:.4f}")
```

---

## B2. 新增配置文件：`configs/nsga2.yaml`

```yaml
# NSGA-II 参数配置
# 在 experiments/run_mofs_verify.py 中加载，传给 nsga2_feature_selection()

nsga2:
  pop_size: 20          # 种群大小
  max_gen: 20           # 最大代数
  p_crossover: 0.5      # 均匀交叉概率
  p_mutation: null      # null = auto (1/n_features)
  seed: null            # null = 从命令行 --seed 继承

  # HV reference point
  hv_ref: [1.0, 1.0]

  # 早停
  early_stop:
    enabled: true
    patience: 5          # 若 HV 在 5 代内无改善则早停
    tolerance: 1e-4

  # 评估模型
  evaluator:
    model: "xgboost"
    n_estimators: 100
    max_depth: 4
    learning_rate: 0.1
```

---

## B3. 修改现有 Baseline：`run_main_table.py` 中的方法注册

当前 `run_main_table.py` 的 `BASELINES` dict 只包含 5 个排名方法。需要在其中**添加 nsga2_fs**：

```python
# 在 run_main_table.py 中新增:

from minppi_mofs.baselines.nsga2_fs import nsga2_feature_selection

def _compute_nsga2_ranking(X, y, task="binary", seed=42):
    """
    NSGA-II 不直接产生排名——它产生 Pareto 前端。
    为与其他 ranking baselines 一致，取 Pareto 前端中 AUC 最高的个体
    的 feature importance（XGBoost gain），并按 gain 降序排列。
    """
    result = nsga2_feature_selection(
        X, y, X, y, task=task, seed=seed,
    )
    if not result.pareto_front:
        return list(X.columns)  # fallback
    # 找 Pareto 前端中 AUC 最高的解
    best_mask = max(result.pareto_front, key=lambda x: 1 - x[1][1])[0]
    selected = [c for j, c in enumerate(X.columns) if best_mask[j]]
    # 对选定特征训练一个 XGBoost，提取 gain
    m = XGBClassifier(...) if task != "regression" else XGBRegressor(...)
    m.fit(X[selected].values, y)
    imp = dict(zip(selected, m.feature_importances_))
    ranking = sorted(imp, key=imp.get, reverse=True)
    # 未被选中的特征排在末尾
    unselected = [c for c in X.columns if c not in selected]
    return ranking + unselected

BASELINES["nsga2"] = {
    "compute": _compute_nsga2_ranking,
    "rank": lambda scores: scores,
}
```

**影响**：CD 图从 5 方法变成 6 方法（新增 NSGA-II），论文的 baseline 排名对比更加完整。

---

## B4. 新实验脚本：`experiments/run_mofs_verify.py`

### 数据流

```
For each (dataset, seed):
  ┌─────────────────────────────────────────────────────────┐
  │ 1. 加载 train/val/test splits                           │
  │ 2. 计算 MinPPI (α=0.95) → harmful set H                 │
  │ 3. 构造 filtered = all_features \ H                     │
  │                                                          │
  │ 4. NSGA-II-full: 在所有 n 个特征上运行 nsga2_fs          │
  │    → HV, IGD+, search_dim, runtime, converged_gen        │
  │                                                          │
  │ 5. NSGA-II-filtered: 在过滤后的 k 个特征上运行 nsga2_fs  │
  │    → HV, IGD+, search_dim, runtime, converged_gen        │
  └─────────────────────────────────────────────────────────┘
```

### 参考前端构造（用于 IGD+ 计算）

对所有方法 × seed 的非支配联合前端：
```python
all_fronts = []
for method in ["nsga2_full", "nsga2_filtered"]:
    for seed in SEEDS:
        all_fronts.append(method_results[(dataset, seed, method)].front)
ref_front = non_dominated_union(all_fronts)
```

### 产出

| 文件 | 每行 | 列 |
|------|------|----|
| `mofs_verify.csv` | (dataset, seed, method) | hv, igd_plus, search_dim, runtime_s, converged_gen |
| `mofs_verify_agg.csv` | (dataset, method) | mean ± std 聚合 |

---

# Part C：完整执行顺序（6 步，约 8 小时）

## 步骤 0：清理（10 分钟，一次性）

```powershell
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master

# ====== 删除残留实验脚本 ======
Remove-Item experiments/run_ablation.py -ErrorAction SilentlyContinue
Remove-Item experiments/run_alpha_sweep.py -ErrorAction SilentlyContinue
Remove-Item experiments/minppi_filter_eval.py -ErrorAction SilentlyContinue
Remove-Item experiments/eval_pipeline_comparison.py -ErrorAction SilentlyContinue
Remove-Item experiments/madelon_alpha_recall.py -ErrorAction SilentlyContinue
Remove-Item experiments/harmful_vs_dimension.py -ErrorAction SilentlyContinue
Remove-Item experiments/sensitivity_friedman.py -ErrorAction SilentlyContinue
Remove-Item experiments/d1_semisynthetic_harmful.py -ErrorAction SilentlyContinue

# ====== 删除孤儿测试 ======
Remove-Item tests/test_mofs_minppi.py -ErrorAction SilentlyContinue

# ====== 删除旧配置 ======
Remove-Item configs/ablation -Recurse -ErrorAction SilentlyContinue

# ====== 删除旧结果 ======
Remove-Item results/tables/2026-06-28_07-44-10 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-01_13-47-28 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-01_17-45-15 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-04_11-02-28 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-07_09-25-53 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-07_11-08-15 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-07_12-26-30 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-08_10-05-11 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-09_13-20-20 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-09_14-12-09 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/tables/2026-07-09_14-15-43 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-06-27_19-51-48 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-06-28_07-19-46 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-06-28_07-44-10 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-01_14-39-54 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-01_20-01-18 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-04_07-22-21 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-04_14-19-54 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-05_14-14-40 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-08_11-46-33 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-09_14-12-09 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-09_14-12-10 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-09_14-15-43 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/logs/2026-07-09_14-45-25 -Recurse -ErrorAction SilentlyContinue
Remove-Item results/REAMME.md -ErrorAction SilentlyContinue

Write-Host "Cleanup complete."
```

## 步骤 1：新增 `nsga2_fs.py`（30 分钟，一次性）

在 `src/minppi_mofs/baselines/` 下创建新的 NSGA-II 特征选择模块。

## 步骤 2：新增 `configs/nsga2.yaml`（5 分钟，一次性）

## 步骤 3：修改 `run_main_table.py`（15 分钟，一次性）

在 BASELINES dict 中注册 `"nsga2"` 方法。

## 步骤 4：新增 `experiments/run_mofs_verify.py`（1 小时，一次性）

## 步骤 5：运行实验（5-8 小时，可并行）

打开 **3 个 PowerShell 终端**同时运行：

```powershell
# 终端 1
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master
..\venv\Scripts\python -u experiments/run_main_table.py

# 终端 2
..\venv\Scripts\python -u experiments/run_pareto_eval.py

# 终端 3
..\venv\Scripts\python -u experiments/eval_harmful_utility.py
```

等终端 1-3 全部完成后，运行：

```powershell
# 终端 1（等上面三个都跑完）
..\venv\Scripts\python -u experiments/run_mofs_verify.py
```

或者**4 个终端同时启动**（如果算力允许）。

## 步骤 6：出图 + 填表 + 编译（2-3 小时）

```powershell
$VENV = "..\venv\Scripts\python"
$R0 = (ls -dt results/tables/*/ | Where-Object { Test-Path "$_\main_metrics.csv" } | Select-Object -First 1).Name
$R1 = (ls -dt results/tables/*/ | Where-Object { Test-Path "$_\pareto_front_agg.csv" } | Select-Object -First 1).Name
$R5 = (ls -dt results/tables/*/ | Where-Object { Test-Path "$_\mofs_verify.csv" } | Select-Object -First 1).Name

# CD 图
$VENV experiments/make_cd_diagram.py results/tables/$R0/main_metrics.csv --metric AUC@10 --higher
$VENV experiments/make_cd_diagram.py results/tables/$R0/main_metrics.csv --metric AUC@20 --higher
$VENV experiments/make_cd_diagram.py results/tables/$R5/mofs_verify.csv --metric hv --higher
$VENV experiments/make_cd_diagram.py results/tables/$R5/mofs_verify.csv --metric igd_plus --lower
$VENV experiments/make_figures.py

# 复制 PDF
Copy-Item results/tables/$R0/cd_diagram_AUC@10.pdf conference-paper/figures/ -Force
Copy-Item results/tables/$R0/cd_diagram_AUC@20.pdf conference-paper/figures/ -Force
Copy-Item results/tables/$R1/pareto_front_cls.pdf conference-paper/figures/ -Force
Copy-Item results/tables/$R1/pareto_front_reg.pdf conference-paper/figures/ -Force
Copy-Item results/tables/$R5/cd_diagram_hv.pdf conference-paper/figures/ -Force
Copy-Item results/tables/$R5/cd_diagram_igd_plus.pdf conference-paper/figures/ -Force
Copy-Item results/figures/figure_1_minppi_trichotomy.pdf conference-paper/figures/ -Force
Copy-Item results/figures/figure_2_prop1_detection.pdf conference-paper/figures/ -Force
Copy-Item results/figures/figure_3_algorithm_b_vs_a.pdf conference-paper/figures/ -Force

# 编译
cd conference-paper/paper
pdflatex -interaction=nonstopmode icomp2026_conference.tex
bibtex icomp2026_conference
pdflatex -interaction=nonstopmode icomp2026_conference.tex
pdflatex -interaction=nonstopmode icomp2026_conference.tex
```

---

# Part D：最终项目结构（清理后）

```
Master/
├── src/minppi_mofs/
│   ├── __init__.py
│   ├── voting_game.py
│   ├── min_ppi.py
│   ├── shapg_integration.py
│   └── baselines/
│       ├── shap_top_k.py
│       ├── shapg.py
│       ├── min_shap.py
│       ├── banzhaf_fs.py
│       ├── powershap.py
│       └── nsga2_fs.py              ← NEW
│
├── configs/
│   ├── spambase.yaml
│   ├── madelon.yaml
│   ├── isolet.yaml
│   ├── gisette.yaml
│   ├── credit_card_fraud.yaml
│   ├── crimedata.yaml
│   ├── abalone.yaml
│   └── nsga2.yaml                   ← NEW
│
├── experiments/
│   ├── preprocess_data/   (×7)
│   ├── run_main_table.py           ← MODIFIED (added nsga2)
│   ├── run_pareto_eval.py
│   ├── eval_harmful_utility.py
│   ├── run_mofs_verify.py          ← NEW
│   ├── make_cd_diagram.py
│   └── make_figures.py
│
├── tests/
│   ├── test_voting_game.py
│   ├── test_min_ppi.py
│   └── test_shapg_integration.py
│
├── demos/   (×3)
├── thesis/
├── conference-paper/
├── results/   (clean — only new runs)
├── data/      (unchanged)
├── pyproject.toml
├── requirements.txt
└── mofs_minppi_design.md
```

---

# Part E：论文内容变化

| 位置 | 变化 |
|------|------|
| §6.1 Setup | "six methods" → 含 NSGA-II-full |
| §6.2 Main Results | 新增 CD 图 ×1（6 methods AUC@10），新增 `tab:mofs_prefilter`（NSGA-II-full vs NSGA-II-filtered HV/IGD+） |
| §6.2 | 删除 "MinPPI does not appear in the ranking comparison" — NSGA-II baseline 代替了这一角色 |
| §6.4 | 新增 MOFS 验证段落 |
| Appendix | 新增 `app:mofs_protocol`，列出 NSGA-II 参数 |

---

# Part F：总时间表

| 步骤 | 内容 | 时间 | 并行性 |
|:--:|------|:--:|:--:|
| 0 | 清理 | 10min | — |
| 1 | 新建 `nsga2_fs.py` | 30min | — |
| 2 | 新建 `configs/nsga2.yaml` | 5min | — |
| 3 | 修改 `run_main_table.py` | 15min | — |
| 4 | 新建 `run_mofs_verify.py` | 1h | — |
| 5 | 运行 4 个实验 | 5-8h | ✓ 全部可并行 |
| 6 | 出图 + 编译 | 2h | — |
| **总计** | | **~10h 墙钟** | |
