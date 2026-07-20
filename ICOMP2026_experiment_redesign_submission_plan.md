# ICOMP 2026 实验重设计与投稿接收行动计划

> **执行说明：** 本文档是截至 2026-07-20 的实验审计、重设计方案和投稿倒排计划。执行代码修改时，建议逐任务使用 `superpowers:test-driven-development`；执行完整计划时使用 `superpowers:executing-plans`。本文档本身不要求也不包含 Git 提交。

**目标：** 在用户确认的 2026-08-05 最终截止时间前，修复当前实验中的数据泄漏、错误指标和不可验证结论，重新运行一套可复现、可审计且与论文主张一致的实验，并把会议论文正文压缩至 ICOMP 规定的 9 页。

**总体路线：** 采用“证据优先的完整修复”方案。先修实验协议和指标实现，再冻结配置并重跑；所有模型选择、特征筛选和 NSGA-II 搜索只访问训练集与验证集，测试集只用于最终一次评估。无法由新结果支持的结论必须删除或降级，不允许通过选择种子、数据集或指标来保住旧结论。

**技术栈：** Python 3.13.9、NumPy 2.4.6、pandas 3.0.3、scikit-learn 1.9.0、XGBoost 3.3.0、SHAP 0.52.0、SciPy 1.18.0、statsmodels 0.14.6、pytest 8.4.2、ruff 0.15.15、LaTeX。

## 全局约束

- 投稿目标：ICOMP 2026，用户确认最终截止为 2026-08-05；内部目标必须设为 2026-08-03 完成可提交 PDF。
- ICOMP 格式：双盲、单个 PDF、正文最多 9 页，参考文献和附录不限；会议官网为 https://icomp.cc/ 。
- 计算资源：24 个逻辑 CPU；NVIDIA GeForce RTX 5070 Laptop GPU，显存 8151 MiB；XGBoost 3.3.0 已确认 `USE_CUDA=True`。
- 随机种子固定为 `[42, 123, 456, 789, 1024]`；任何新增主实验必须至少覆盖这 5 个种子。
- 旧结果目录只读保留，不覆盖 `results/tables/2026-*` 和 `results/logs/2026-*`；新结果单独写入不覆盖。
- 任何监督式预处理只能在训练集上拟合；验证集用于算法决策；测试集只允许在最终特征集合冻结后评估。
- 所有方法必须使用同一数据划分、同一预测模型、同一模型超参数和同一评价函数；方法特有的计算预算必须单独报告。
- 论文中的每个数值必须能追溯到一个 CSV 行、配置文件、随机种子和运行日志。
- 不提交 Git；但执行前后都要保存 `git status --short` 输出，防止无意覆盖其他工作。

---

## 1. 执行结论

当前项目的核心思想具有投稿价值：用非单调简单投票博弈定义特征子集效用，并用最小边际贡献寻找“存在有害上下文”的特征。题目与 ICOMP 的 feature/model selection、computational optimization 和 efficient learning 方向匹配。

但是，当前实验还不能支撑现有摘要中的全部强结论。最严重的问题不是模型性能不足，而是评价协议存在泄漏、Hypervolume 实现错误、Madelon ground truth 定义错误，以及论文数字与 CSV 不一致。若直接投稿，熟悉特征选择或多目标优化的审稿人很可能据此给出 Reject。

接收机会最高的策略不是继续增加结论，而是把论文收缩为三个可验证贡献：

1. **定义贡献：** MinPPI 用最小边际而非平均边际识别“存在上下文使特征有害”的情况。
2. **估计贡献：** 精确算法、随机估计和引导估计之间具有可测量的准确率—预算权衡。
3. **应用贡献：** MinPPI 作为保守预筛选器，在冻结的无泄漏协议下能够减少部分特征，同时保持测试效用；对 NSGA-II 的影响应按数据集报告为正、负或中性，而不是预设为普遍提升。

### 投稿前的硬性判定

以下任一项未完成，都不应在论文中保留相应强结论：

- Hypervolume 单元测试未通过：删除全部 HV 改进结论。
- NSGA-II 仍使用测试集搜索：删除全部 NSGA-II 预筛选结论。
- 监督式 MI 仍在全数据上拟合：受影响数据集的结果全部作废。
- Madelon 无官方特征身份映射：删除 probe precision/recall/F1 结论。
- 正文仍超过 9 页：存在 desk rejection 风险，不能提交。
- 摘要数字与新 CSV 无自动一致性检查：不得手工复制数字到摘要。

---

## 2. 已有实验与结果审计

### 2.1 当前结果资产

| 实验块 | 结果位置 | 规模 | 当前用途 |
|---|---|---:|---|
| 六种基线的 Top-k 排名 | `results/tables/2026-07-10_12-20-33/` | 7 数据集 × 5 种子 | 描述基线排序能力；不包含 MinPPI |
| α 扫描与删除后效用 | `results/tables/2026-07-10_16-23-55/` | 7 数据集 × 4 α × 5 种子 | 描述压缩率和效用保持 |
| 删除效用、matched controls | `results/logs/2026-07-11_09-18-01/c1_removal/` | 7 数据集 × 5 种子 | 比较 MinPPI、SHAP-bottom、MinShap、随机删除 |
| Madelon ground truth | `results/logs/2026-07-11_09-18-01/c2_ground_truth/` | 5 种子 | 当前定义无效，不能继续引用 |
| MinPPI 与 MinShap 集合一致性 | `results/logs/2026-07-11_09-18-01/c8_minshap_vs_minppi/` | 3 数据集 × 5 种子 | 说明两者识别集合差异很大 |
| MinPPI 预筛选 + NSGA-II | `results/tables/2026-07-12_08-02-42/` | 7 数据集 × 5 种子 | 当前存在测试泄漏、HV 错误和收敛指标错误 |
| ISOLET utility 消融 | `results/tables/2026-07-14_07-09-55/` | 1 数据集 × 1 种子 | 当前混淆了效用函数和模型，不能形成因果结论 |

### 2.2 当前基线排名结果实际说明什么

`avg_ranks_AUC@10.csv` 的平均排名为：

| 方法 | AUC@10 平均排名 |
|---|---:|
| SHAP-Top-k | 1.8 |
| PowerSHAP | 2.0 |
| NSGA-II | 3.2 |
| Banzhaf | 4.2 |
| MinShap | 4.8 |
| ShapG | 5.0 |

Friedman 检验在 AUC@10 和 AUC@20 上分别得到 `p=0.0169` 和 `p=0.0402`，但所有 Nemenyi 两两比较均为不显著。因此可以写“整体方法差异达到 omnibus 显著”，不能写某一对方法显著优于另一方法。更重要的是，这个表里没有 MinPPI，所以它不能证明 MinPPI 的排名能力或 superiority。

### 2.3 当前 α=0.95 删除结果

| 数据集 | 平均压缩率 | 完整模型效用 | 删除后效用 | 保持率 | 解释 |
|---|---:|---:|---:|---:|---|
| Abalone | 0.00% | 0.5410 | 0.5410 | 1.0000 | 没有删除任何特征 |
| Credit-Card-Fraud | 32.41% | 0.9783 | 0.9634 | 0.9847 | 压缩明显，但效用下降约 0.0149 |
| Crimedata | 0.20% | 0.9624 | 0.9624 | 1.0000 | 基本没有压缩 |
| Gisette | 0.00% | 0.9885 | 0.9885 | 1.0000 | 没有删除任何特征 |
| ISOLET | 2.80% | 0.9931 | 0.9930 | 0.9999 | 小幅压缩、效用稳定 |
| Madelon | 16.80% | 0.7763 | 0.7707 | 0.9930 | 有压缩，效用略降 |
| Spambase | 1.79% | 0.9867 | 0.9865 | 0.9997 | 小幅压缩、效用稳定 |

这组结果支持“MinPPI 多数情况下保守”，不支持“在所有数据集上都有显著压缩”，也不支持把 `-1` 解释为无条件的“必须删除”。Credit-Card-Fraud 是必须正面讨论的反例。

### 2.4 当前删除效用实验

| 数据集 | 平均 harmful 数 | 平均压缩率 | 平均测试效用变化 | 最差种子变化 |
|---|---:|---:|---:|---:|
| Abalone | 0.0 | 0.00% | 0.0000 | 0.0000 |
| Credit-Card-Fraud | 9.4 | 32.42% | -0.0177 | -0.0301 |
| Crimedata | 0.2 | 0.20% | +0.0002 | 0.0000 |
| Gisette | 0.0 | 0.00% | 0.0000 | 0.0000 |
| ISOLET | 2.0 | 2.00% | -0.0001 | -0.0004 |
| Madelon | 15.0 | 15.00% | +0.0105 | -0.0058 |
| Spambase | 0.6 | 1.07% | -0.0013 | -0.0049 |

当前结果中 Madelon 的平均改善较有吸引力，但种子间方向不一致；Credit-Card-Fraud 的下降不是“安全保持”。新论文应把 preservation 写成连续量和置信区间，不使用绝对的 safe/unsafe 二分措辞。

### 2.5 当前 NSGA-II 结果实际说明什么

| 数据集 | Full HV | Filtered HV | HV 相对变化 | 搜索维度变化 | 结论 |
|---|---:|---:|---:|---:|---|
| Credit-Card-Fraud | 0.3243 | 0.4452 | +37.3% | 29 → 18.8，-35.2% | 旧指标下唯一强正结果 |
| Madelon | 0.2513 | 0.1925 | -23.4% | 100 → 81.4，-18.6% | HV 明显下降 |
| ISOLET | 0.3771 | 0.3671 | -2.7% | 100 → 99.6，-0.4% | 基本无压缩且略降 |
| 其余四个数据集 | 相同 | 相同 | 0% | 基本相同 | 没有证明预筛选收益 |

因此，摘要中的“HV +37%、搜索空间缩小 35%”是 Credit-Card-Fraud 的单数据集结果，不能写成总体结论。修正 HV 和数据泄漏后，数值还会变化。

### 2.6 当前 Madelon 与 ISOLET 消融

- 当前 CSV 中 Madelon 的 MinPPI precision 为 `0.9479±0.0965`，正文中的 `0.978` 与结果不一致。
- UCI 官方说明 Madelon 特征顺序已随机化。当前代码按 `f1–f20` 为真实特征、`f21–f500` 为 probe 的规则计算 precision/recall，没有有效依据。
- ISOLET 消融只有种子 42；accuracy 和 OVR-AUC 两条路径使用不同的模型及训练方式，而且两者都检测到 0 个 harmful 特征。它既不能证明效用选择等价，也不能解释运行时间差异。

---

## 3. 问题清单、影响和修复决策

### 3.1 P0：不修复就会使核心结论失效

| 编号 | 问题 | 证据位置 | 对结论的影响 | 必须采取的动作 |
|---|---|---|---|---|
| P0-1 | Madelon ground truth 按列号定义错误 | `experiments/eval_harmful_utility.py:92` | precision/recall/F1 无效 | 删除现有 ground-truth 结果；Madelon 只作为预测基准；用解析投票游戏验证 estimator |
| P0-2 | MI 特征预选在全数据上进行 | `experiments/preprocess_data/preprocess_Madelon.py:160` 及 ISOLET/Gisette/Crimedata 同类代码 | 测试标签参与特征选择，所有下游结果泄漏 | 先划分，再在训练集拟合 MI/Variance/Scaler，向 val/test 变换 |
| P0-3 | MinPPI 把训练集同时作为验证集 | `experiments/run_mofs_verify.py:199-218` | 联盟效用是训练性能，容易过拟合 | 使用已有 `*_val.csv`，调用必须为 `(X_train, X_val, y_train, y_val)` |
| P0-4 | NSGA-II 直接用测试集搜索 | `experiments/run_mofs_verify.py:240-245, 288-293` | 测试集参与进化，最终前沿偏乐观 | 搜索只用 val；冻结 validation Pareto set 后在 test 上重新训练/评估 |
| P0-5 | 2D Hypervolume 公式错误 | `src/minppi_mofs/baselines/nsga2_fs.py:210-224` | “+37% HV” 数学上不可用 | 实现标准二维最小化 HV，并加入已知答案单元测试 |
| P0-6 | 收敛代数定义几乎必然在早期触发 | `src/minppi_mofs/baselines/nsga2_fs.py:360-363` | “收敛更快”结论不可解释 | 运行结束后计算首次达到最终最佳 HV 的 95% 的评价次数 |
| P0-7 | 正文数字与 CSV 不一致 | Madelon precision、NSGA 汇总、页数等 | 可复现性和可信度受损 | 由汇总脚本生成 LaTeX 宏/表格，禁止手工复制关键数字 |

### 3.2 P1：会削弱方法解释或公平性

| 编号 | 问题 | 风险 | 修复方式 |
|---|---|---|---|
| P1-1 | ISOLET accuracy 与 AUC 消融同时更换模型 | 无法归因于效用函数 | 同一个模型、同一预测概率，同时计算 accuracy 与 macro OVR-AUC |
| P1-2 | Algorithm B 的 guidance score 实际是 mutual information，却称 ShapG-guided | 方法命名与实现不符 | 截止前优先重命名为 MI/correlation-guided；只有真正调用 ShapG 才保留 ShapG 名称 |
| P1-3 | “minimal connected graph”实现可能保留 cycle edge | 图不是严格 MST，理论描述失配 | 只有 union 成功后才写入邻接矩阵；测试边数必须为 `n-1` |
| P1-4 | 论文 hybrid dispatch 与实际 runner 不一致 | 无法复现论文方法 | 把 dispatcher 规则写进配置并由所有 runner 调用同一入口 |
| P1-5 | 论文称 Abalone exact，runner 实际使用 Algorithm A | 方法和结果对不上 | n=8 必须执行 exact；同时用它校验 A/B 估计误差 |
| P1-6 | Credit-Card-Fraud 只报告 ROC-AUC | 极端类别不平衡下可能掩盖少数类性能 | ROC-AUC 保持与全文一致，同时必须增加 PR-AUC 和 recall-at-fixed-FPR |
| P1-7 | MinPPI 不是排序方法，却与 Top-k ranking 叙事混合 | 贡献边界模糊 | 主实验改为 matched-budget deletion；Top-k 排名降到补充材料并明确不含 MinPPI |
| P1-8 | Full/filtered NSGA-II 只对齐代数，没有严格对齐 objective evaluations | 计算预算不公平 | 固定 population × generations 或直接固定评价次数；同时报告总 wall time |
| P1-9 | 预筛选运行时间未计入 filtered 总成本 | 可能夸大加速 | 报告 `MinPPI time + NSGA time` 和单独分量 |
| P1-10 | 只有 5 个分类数据集做 Friedman/Nemenyi | 统计功效低，pairwise 全部不显著 | 报效应量、配对差和置信区间；排名只作描述，不写显著 superiority |

### 3.3 P2：投稿表达和工程质量问题

- `-1 = must remove` 应改为“发现一个负边际见证上下文”；是否删除由独立验证集和测试效用决定。
- “threshold-free”不准确，因为 α 本身就是效用阈值；可改成“无需额外的 score cutoff”。
- α 扫描产生的是 operating points，不保证每个点都 Pareto-optimal；报告前必须做 dominance filtering。
- “压缩率随 α 单调增加”不是理论保证；当前 Abalone 结果已经非单调。
- “所有数据集都存在 harmful features”与 Gisette、Abalone 的 0 harmful 冲突。
- “analytically solved/no optimization”不准确；MinPPI 需要随机估计和大量模型重训练。
- 当前 PDF 共 15 页，其中正文约 10 页、参考文献 1 页、附录 4 页；正文必须压缩 1 页。
- 作者仍是模板占位信息；双盲稿可以匿名，但 camera-ready 前必须替换。
- 项目 README 中的 ICOMP 日期已过时；投稿操作只以用户确认的 8 月 5 日和 OpenReview 实际界面为准。

---

## 4. 新的无泄漏实验协议

### 4.1 数据划分

每个种子固定一次 `64% train / 16% validation / 20% test` 划分。分类数据使用 stratified split；回归数据使用普通随机划分。每个方法共享完全相同的行索引。

正确的数据流：

```text
raw data
  └─ split(seed) ──> train_raw / validation_raw / test_raw
                       │
                       ├─ fit variance filter, MI selector, scaler on train_raw only
                       │
                       └─ transform train / validation / test with frozen objects
                              │
                              ├─ train: fit coalition models and final models
                              ├─ validation: MinPPI value, α decision, NSGA-II objective
                              └─ test: one-time final evaluation after feature set freezes
```

必须保存：

- 每个种子的 train/val/test 行索引及 SHA-256；
- train-only 预处理器参数；
- MI 选择后的原始列名；
- 数据源 URL/OpenML ID、下载日期、原始文件 SHA-256；
- 每个 CSV 的列数、行数、目标分布和缺失值数量。

### 4.2 特征预处理

建议新增 `src/minppi_mofs/data_protocol.py`，统一替代 7 个预处理脚本中重复且容易泄漏的逻辑。

```python
@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    selected_features: tuple[str, ...]
    seed: int


def prepare_split(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task: str,
    seed: int,
    max_features: int | None,
) -> DataSplit:
    """Split first; fit every supervised and unsupervised transform on train only."""
```

约束：

- `VarianceThreshold.fit`、`mutual_info_classif/regression`、`StandardScaler.fit` 只允许接收 `X_train`。
- MI 的随机状态等于当前外层 seed，不能固定为对所有 seed 相同的 42。
- 选中的列必须按同一顺序应用到 validation/test。
- 若某个数据集无需降至 100 维，保留全部通过无监督清洗的特征，并在论文中说明。

### 4.3 联盟效用与最终评价

| 任务 | 游戏效用 `U` | 必报测试指标 | 推荐模型 |
|---|---|---|---|
| 二分类 | ROC-AUC | ROC-AUC、PR-AUC；Credit 另报 recall@FPR=1% | 固定参数 XGBClassifier |
| 多分类 ISOLET | macro OVR ROC-AUC | macro OVR ROC-AUC、accuracy | 同一个固定分类器，不允许因指标改变模型 |
| 回归 | R² | R²、RMSE | 固定参数 XGBRegressor |

对每个特征子集 `S`：模型只在 `X_train[S]` 上拟合，`U(S)` 只在 `X_val[S]` 上计算。完整模型阈值为 `alpha * U_full_validation`。确定最终保留特征后，在 train+validation 上重新拟合一次模型，并在 test 上评估一次。

测试效用不能反馈到 α、K、K_adv、图阈值、模型超参数或删除策略。

### 4.4 GPU/CPU 后端选择

当前 XGBoost 支持 CUDA，但 MinPPI 会重复训练大量小模型，GPU 数据传输可能比 CPU 更慢。正式重跑前必须执行一个固定的 100 coalition backend benchmark：

```powershell
..\venv\Scripts\python.exe experiments\benchmark_backend.py --dataset spambase --coalitions 100 --seed 42 --backend cpu
..\venv\Scripts\python.exe experiments\benchmark_backend.py --dataset spambase --coalitions 100 --seed 42 --backend cuda
```

选择规则：若 CUDA 中位运行时间至少快 20%，且同一子集的 AUC 差异小于 `1e-6`，所有 XGBoost 主实验统一使用 `device="cuda", tree_method="hist"`；否则使用 CPU。不能按方法分别选择更有利的后端。GPU 运行时只启动一个重型实验进程；CPU-only 汇总和 LaTeX 可并行运行。

---

## 5. 必须重做的实验

### E0：实现正确性与无泄漏测试

**目的：** 在消耗数天算力前证明核心指标和数据流正确。

**必须新增的测试：**

1. `tests/test_nsga2_hv.py`

```python
def test_hv_two_point_front():
    front = np.array([[0.2, 0.8], [0.8, 0.2]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.28)


def test_hv_single_point():
    front = np.array([[0.25, 0.40]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.45)


def test_hv_ignores_duplicate_and_dominated_points():
    front = np.array([[0.2, 0.8], [0.2, 0.8], [0.9, 0.9], [0.8, 0.2]])
    ref = np.array([1.0, 1.0])
    assert _compute_hv(front, ref) == pytest.approx(0.28)
```

2. `tests/test_data_protocol.py`

- 构造一个只有 test 标签能揭示的特征；断言训练集 MI 选择器不会访问或选择它。
- 断言 scaler 的 `mean_` 等于训练集均值而不是全数据均值。
- 断言三个 split 行索引不重叠且并集等于全部样本。
- 断言相同 seed 完全复现、不同 seed 至少一个 split 不同。

3. `tests/test_feature_graph.py`

- 相关图在 `n>1` 时恰有 `n-1` 条无向边；
- 图连通；
- 重复运行得到同一图；
- 只有 union 成功的边才能进入邻接矩阵。

4. `tests/test_experiment_protocol.py`

- monkeypatch 一个 test 访问即抛异常的数据对象；搜索和特征选择阶段必须能够完整运行；
- 只有 `final_test_evaluation` 函数允许读取 `X_test/y_test`。

**通过标准：** 全部新测试通过；原有 65 个测试继续通过；`ruff check src tests experiments demos` 通过。

### E1：Estimator 准确性与预算曲线

**目的：** 直接证明 Algorithm A、MI/correlation-guided B 和 hybrid 是否能逼近 exact MinPPI，而不是仅展示一个有利 toy example。

**数据与重复：**

- 解析 simple games：加权投票、非单调 XOR、含一个已知 harmful feature、无 harmful feature、多个稀有负见证；每类 20 个随机实例。
- Abalone：8 个特征，使用 exact enumeration 获得 ML value function 下的真实 MinPPI。
- 小型受控 ML 数据：`n_features ∈ {8, 12, 16}`，只在可精确枚举的规模运行。

**方法：** Exact、Algorithm A、MI/correlation-guided B、Hybrid。

**预算：** `K ∈ {10, 25, 50, 100, 200}`；B 使用相同 coalition evaluation 数，不按循环次数对齐。

**指标：** harmful precision/recall/F1、false-harmful rate、exact score agreement、每个特征的 miss probability、模型拟合次数、wall time。

**期望但不预设的结果：** B/Hybrid 在稀有负见证游戏上以更少评价达到与 A 相同的 recall；在无 harmful 游戏上 false-harmful rate 必须为 0。若 B 没有稳定优势，论文应保留 A 作为主估计器，把 B 降为启发式补充。

### E2：主删除效用实验

**目的：** 回答“MinPPI 找到的特征删除后是否比同预算对照更少损害泛化性能”。这是论文最重要的经验问题。

**数据集：** Spambase、Madelon、ISOLET、Gisette、Credit-Card-Fraud、Crimedata、Abalone。

**方法：**

- MinPPI harmful set；
- SHAP-bottom，删除数量与 MinPPI 完全相同；
- MinShap-bottom，删除数量与 MinPPI 完全相同；
- Banzhaf-bottom，删除数量与 MinPPI 完全相同；
- Random matched deletion，至少 30 次随机重复；
- Full model，不删除特征。

不要把“所有 MinShap 负分特征”与 MinPPI matched-budget 结果混在同一公平性比较中；可在附录单独报告 MinShap-native cutoff。

**主要指标：**

- `ΔU_test = U_filtered_test - U_full_test`；
- compression ratio；
- paired win/tie/loss against each matched control；
- 95% bootstrap CI；
- total selection time 和 model-fit count。

**预注册式成功阈值：**

- 主要可接受结果：7 个数据集中至少 5 个的平均 `ΔU_test ≥ -0.005`；
- 任何数据集若平均下降超过 `0.01`，必须在正文点名并停止使用“safe deletion”；
- 若 MinPPI 对 matched random/SHAP 的配对差没有优势，可将贡献改写为“可解释的保守筛选”，不宣称性能优越；
- 0 harmful 是合法结果，必须报告，不能视为实验失败或从数据集中删除。

### E3：α 敏感性与稳定性

**α 网格：** `{0.85, 0.90, 0.95, 0.99}`，5 个种子，7 个数据集。

**报告：** harmful 数、压缩率、validation utility、test utility、preservation、跨 seed harmful-set Jaccard、运行时间。

**分析规则：**

- α 点先按 `(compression↑, preservation↑)` 做 nondominance filtering，再使用“observed Pareto-like trade-off”措辞；
- 不假设压缩率随 α 单调；若非单调，解释为估计方差与非单调 value function 的共同结果；
- α=0.95 只有在 validation 预先确定后才能称 default；不能根据 test 结果重新选择 α；
- 若 Credit-Card-Fraud 在 α=0.95 仍下降明显，可在训练/验证阶段选择更保守 α，但必须把选择规则预先固定，例如选择满足 validation preservation ≥0.995 的最大压缩点。

### E4：MinPPI 预筛选 + NSGA-II

**目的：** 公平判断预筛选是否改善多目标搜索，而不是保证得到正结果。

**协议：**

1. Full 和 Filtered 使用相同 train/val/test split、population、objective evaluation budget 和模型。
2. NSGA-II 只在 validation 上优化 `(selected feature ratio, 1-validation utility)`。
3. reference point 对所有方法和种子固定，并严格支配所有可行点；建议把两个目标归一化到 `[0,1]` 后使用 `(1.1, 1.1)`。
4. validation Pareto set 冻结后，每个点在 train+validation 上重新拟合并在 test 上评估。
5. 报告 validation HV、held-out test HV、IGD 或 epsilon indicator、evaluation count、总时间。
6. Filtered 总成本必须包含 MinPPI 预筛选时间。
7. “收敛评价次数”定义为累计最佳 HV 首次达到本次运行最终最佳 HV 的 95%，不能在运行中用当前最大值触发。

**公平预算：** 5 个种子；每个方法至少 `population=40, evaluations=800`。若 smoke run 显示单数据集超过 12 小时，统一降为 `population=30, evaluations=600`，并在所有数据集和方法上保持一致。

**可接受结论门槛：**

- 若至少 2 个数据集维度降低 ≥15%，且 held-out test HV 没有超过 2% 的相对下降，可写“在部分高维任务中降低搜索维度而基本保持前沿质量”；
- 只有多数数据集 test HV 改善且配对 CI 不跨 0，才能写总体 HV 提升；
- 若 Credit 正、Madelon 负的混合模式仍存在，应报告为 dataset-dependent trade-off，而不是失败后隐藏。

### E5：效用、引导和 dispatcher 消融

**消融 A：同模型不同效用。** ISOLET 使用同一模型产生概率；分别以 accuracy 和 macro OVR-AUC 定义联盟是否获胜。5 个种子，比较 harmful 数、集合 Jaccard、删除后两种测试指标。这样差异才能归因于 utility。

**消融 B：引导来源。** Algorithm B 至少比较：无引导随机、MI score、correlation-neighborhood、MI+correlation。若完整 ShapG 在时间内可运行，再加入 Full ShapG；否则论文中删除 ShapG-guided 名称。

**消融 C：dispatcher。** exact、A、B、hybrid 在同一个小规模 exact benchmark 上比较；大规模数据只比较 A/B/hybrid 的效用和成本，不冒充 exact accuracy。

### E6：统计分析与复现性

每个数据集和 seed 是配对单元。报告 mean、std、median、95% bootstrap CI 和原始 5 个 seed 值。对多个方法的成对检验使用 paired Wilcoxon 或 permutation test，并用 Holm 校正；同时给出效应量。

限制：5 个 seed 的显著性检验功效低，因此结论主要依赖效应大小、方向一致性和置信区间。跨数据集平均排名可以保留为描述图，但不能把 Nemenyi 全部不显著的结果写成成对 superiority。

---

## 6. 运行架构与命令

建议新增统一入口 `experiments/run_icomp2026_pipeline.py`，避免人工拼接不同日期的结果。接口固定为：

```text
--stage {validate,preprocess,estimator,removal,alpha,nsga2,ablation,summarize,figures,all}
--config-dir configs
--output-root results/icomp2026_rerun
--seeds 42 123 456 789 1024
--backend {cpu,cuda}
--resume
--smoke
```

### 6.1 环境验证

```powershell
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master
..\venv\Scripts\python.exe --version
..\venv\Scripts\python.exe -c "import xgboost as xgb; print(xgb.__version__); print(xgb.build_info())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
..\venv\Scripts\python.exe -m pytest tests -q
..\venv\Scripts\python.exe -m ruff check src tests experiments demos
```

预期：Python 3.13.9；XGBoost 3.3.0 且 `USE_CUDA=True`；65 个旧测试加新增测试全部通过；ruff 无错误。

### 6.2 Smoke run

```powershell
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage validate --config-dir configs --output-root results\icomp2026_rerun
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage all --config-dir configs --output-root results\icomp2026_rerun --seeds 42 --backend cuda --smoke
```

Smoke run 限制为 1 个 seed、2 个数据集、低预算，必须验证：结果目录、metadata、断点恢复、测试集访问审计、GPU 峰值显存、异常退出后的 resume。

### 6.3 正式运行顺序

```powershell
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage preprocess --config-dir configs --output-root results\icomp2026_rerun --seeds 42 123 456 789 1024 --backend cuda --resume
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage estimator --config-dir configs --output-root results\icomp2026_rerun --seeds 42 123 456 789 1024 --backend cuda --resume
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage removal --config-dir configs --output-root results\icomp2026_rerun --seeds 42 123 456 789 1024 --backend cuda --resume
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage alpha --config-dir configs --output-root results\icomp2026_rerun --seeds 42 123 456 789 1024 --backend cuda --resume
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage nsga2 --config-dir configs --output-root results\icomp2026_rerun --seeds 42 123 456 789 1024 --backend cuda --resume
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage ablation --config-dir configs --output-root results\icomp2026_rerun --seeds 42 123 456 789 1024 --backend cuda --resume
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage summarize --config-dir configs --output-root results\icomp2026_rerun --seeds 42 123 456 789 1024 --backend cuda --resume
..\venv\Scripts\python.exe experiments\run_icomp2026_pipeline.py --stage figures --config-dir configs --output-root results\icomp2026_rerun --seeds 42 123 456 789 1024 --backend cuda --resume
```

每个 stage 必须创建独立 `status.json`，记录成功、失败、跳过的 dataset/seed/method 组合。汇总阶段若发现任何组合缺失，必须失败退出，而不是对不完整数据静默取平均。

### 6.4 结果目录结构

```text
results/icomp2026_rerun/<run_id>/
├── environment.json
├── configs_frozen/
├── data_manifest.csv
├── protocol_audit.json
├── raw_metrics/
├── estimator/
├── removal/
├── alpha/
├── nsga2/
├── ablation/
├── tables/
├── figures/
├── latex_macros.tex
├── status.json
└── output.log
```

---

## 7. 代码修改任务与验收顺序

### Task 1：修正 Hypervolume 与收敛指标

**文件：**

- 修改：`src/minppi_mofs/baselines/nsga2_fs.py`
- 新建：`tests/test_nsga2_hv.py`

**正确的二维最小化 HV 核心实现：**

```python
def _compute_hv(front: np.ndarray, ref: np.ndarray) -> float:
    points = np.asarray(front, dtype=float)
    if points.size == 0:
        return 0.0
    points = points[np.all(points <= ref, axis=1)]
    if len(points) == 0:
        return 0.0
    points = np.unique(points, axis=0)
    points = points[_non_dominated_sort(points)[0]]
    points = points[np.argsort(points[:, 0])]
    hv = 0.0
    best_y = float(ref[1])
    for x, y in points:
        if y < best_y:
            hv += (float(ref[0]) - float(x)) * (best_y - float(y))
            best_y = float(y)
    return max(hv, 0.0)
```

收敛指标改为 post-hoc `generation_to_95pct_best(hv_history)`，使用 cumulative-best 序列计算。运行测试后，已知两点 front 必须从当前错误的 `0.52` 变为 `0.28`。

### Task 2：建立统一数据协议

**文件：**

- 新建：`src/minppi_mofs/data_protocol.py`
- 新建：`tests/test_data_protocol.py`
- 修改：`experiments/preprocess_data/preprocess_*.py`

先写泄漏测试，再实现 split-first pipeline。所有原预处理器只负责下载/清洗 raw data，监督式选择和缩放统一由 `prepare_split` 完成。

### Task 3：统一模型和效用

**文件：**

- 修改：`src/minppi_mofs/voting_game.py`
- 新建：`src/minppi_mofs/utility.py`
- 修改：`experiments/e4_isolet_utility_ablation.py`
- 新建：`tests/test_utility.py`

接口必须允许同一组预测同时计算 ROC-AUC、macro OVR-AUC、accuracy、PR-AUC、R² 和 RMSE。任务类型、metric 和 model 不能隐式绑定到不同实现。

### Task 4：修复 feature graph 与 guidance 命名

**文件：**

- 修改：`src/minppi_mofs/shapg_integration.py`
- 修改：`src/minppi_mofs/min_ppi.py`
- 新建：`tests/test_feature_graph.py`

如果截止前没有完成真正的 ShapG guidance，公共方法名和论文统一改为 `mi_correlation_guided`，避免方法名夸大。

### Task 5：建立单一正式 runner

**文件：**

- 新建：`experiments/run_icomp2026_pipeline.py`
- 新建：`tests/test_experiment_protocol.py`
- 修改：`configs/*.yaml`

runner 必须实现 stage、resume、smoke、manifest、test-access audit 和缺失组合检查。旧 runner 保留作历史参考，但不得用于新论文数字。

### Task 6：重跑、汇总和生成论文数字

**文件：**

- 新建：`experiments/summarize_icomp2026.py`
- 修改：`experiments/make_paper_figures.py`
- 生成：`results/icomp2026_rerun/<run_id>/latex_macros.tex`

摘要中的数据集数、平均效用变化、维度减少和 HV 变化必须引用宏，例如 `\newcommand{\CreditCompression}{...}`，避免正文与 CSV 漂移。

### Task 7：重写论文并压缩到 9 页

**文件：**

- 修改：`conference-paper/paper/icomp2026_conference.tex`
- 修改：`conference-paper/paper/icomp2026_conference.bib`

正文优先保留定义、核心性质、估计算法、主删除实验和正确的 NSGA-II 结果。详细证明、数据集表、完整 α 表、所有消融和统计表移入附录。删除重复的 Related Work 叙述和未经证明的 Pareto/安全措辞。

---

## 8. 结果期望与论文改写决策树

### 情形 A：理想结果

- MinPPI 在至少 5/7 数据集保持 `ΔU_test ≥ -0.005`；
- estimator 引导方法在相同评价预算下提高 harmful recall；
- 至少两个高维数据集预筛选 ≥15%，test HV 基本不下降；
- 结果跨种子方向稳定。

论文可以主张：MinPPI 是一种可解释、保守、计算可控的特征预筛选器，并在部分高维任务中改善后续多目标搜索效率。

### 情形 B：最可能的混合结果

- ISOLET/Spambase 稳定但压缩小；
- Madelon 有压缩但种子方差大；
- Credit 压缩大但效用下降；
- NSGA-II 只在部分数据集受益。

论文应主张：MinPPI 发现的是 contextual harm，而不是通用噪声；它提供可解释的 operating-point family，效果依赖数据集和 α。强调负结果和局限性反而比夸大普适安全更可信。

### 情形 C：修复后主要性能结论消失

- matched controls 与 MinPPI 无差异；
- NSGA-II test HV 没有改善；
- harmful 集合不稳定。

仍可投稿，但必须转为方法与估计论文：突出非单调 simple game 定义、exact-vs-estimator 基准、sample-efficiency 分析和完整负结果。删除“improves NSGA-II”“safe deletion”“noise detection”三类表述。

### 不允许的应对方式

- 只保留表现好的 seed；
- 重复试 α 后用 test 选择最佳 α；
- 删除 Credit/Madelon 等反例而不说明；
- 改 reference point 直到 HV 变正；
- 把 0 harmful 的数据集从平均中排除；
- 用相对百分比掩盖很小的绝对变化；
- 将 validation 结果写成 test 结果。

---

## 9. 如何提高 ICOMP 接收概率

### 9.1 审稿人最可能认可的部分

- 问题与会议主题高度一致：feature selection + computational optimization。
- 最小边际与平均边际的差异直观且可解释。
- 非单调 value function 比加性 weighted voting approximation 更符合模型重训练现实。
- exact、小样本估计和引导搜索形成清晰的算法层级。
- 若完整开放配置、seed、模型拟合次数和日志，复现性会明显优于许多短会议论文。

### 9.2 审稿人最可能拒绝的原因

1. 把 existential harmful witness 过度解释为必须删除。
2. 将 α operating points 宣称为有保证的 Pareto front。
3. 实验协议泄漏或方法计算预算不公平。
4. Madelon ground truth 无依据。
5. 只有单数据集或单 seed 支撑强结论。
6. 论文正文塞入太多松散贡献，主线不清。
7. 与 MinShap、Banzhaf、SHAP 的差异只停留在措辞，没有 controlled experiment。

### 9.3 推荐的 9 页正文结构

| 页数预算 | 内容 |
|---:|---|
| 0.75 页 | Abstract + Introduction：问题、三项贡献、诚实结果摘要 |
| 1.00 页 | Related Work + 定位：平均权力指数、最小边际、MOFS |
| 1.25 页 | Problem formulation：value function、MinPPI、三值性质 |
| 1.50 页 | Exact/A/B/Hybrid 算法与复杂度 |
| 0.75 页 | 理论：upward bias、miss probability、正确的解释边界 |
| 0.75 页 | 无泄漏实验协议、数据集、指标、预算 |
| 1.75 页 | 主结果：estimator + matched deletion + α |
| 0.75 页 | NSGA-II 预筛选与消融 |
| 0.50 页 | Limitations + Conclusion |

证明、完整数据集表、所有 seed 原始值、完整 α 表、统计检验和更多图全部进入附录。

### 9.4 摘要应满足的证据规则

摘要最多保留三个数字：一个 estimator 预算结果、一个删除效用结果、一个 NSGA-II 或压缩结果。每个数字必须是跨 seed 汇总，并在正文表格中出现。若某结论只在单数据集成立，摘要必须直接写数据集名。

### 9.5 预期接收标准

无法保证会议接收，但在本项目上，达到以下状态才具有合理竞争力：

- 实验协议无已知泄漏；
- 所有 headline metric 有单元测试；
- 结果与主张一一对应，没有数字漂移；
- 至少一个经验结论跨多个数据集/种子稳定；
- 对 Credit、Madelon、0 harmful 和 NSGA 负结果进行正面讨论；
- 方法命名与实现一致；
- 正文不超过 9 页且主线集中；
- 附录足以复现所有配置和统计；
- 投稿前由未参与实现的人按命令从至少一个 seed 完整复跑。

---

## 10. 2026-07-20 至 2026-08-05 倒排计划

| 日期 | 必须完成 | 退出条件 |
|---|---|---|
| 07-20 | 冻结旧结果；完成本文档；确认最终协议 | 不再使用旧 CSV 更新论文数字 |
| 07-21 | 修 HV、收敛指标、graph；新增单元测试 | 所有 correctness tests 通过 |
| 07-22 | 重构 split-first preprocessing；测试泄漏 | protocol audit 显示 test access=0 |
| 07-23 | 完成统一 utility、runner、metadata、resume | smoke run 可断点恢复 |
| 07-24 | CPU/CUDA benchmark；启动 E1 与 E2 | 后端和全部超参数冻结 |
| 07-25—07-27 | 完成 E1、E2、E3 | 7 数据集 × 5 seed 无缺失 |
| 07-28—07-30 | 完成 E4 NSGA-II | Full/Filtered 预算完全一致 |
| 07-30—07-31 | 完成 E5、E6、汇总和图 | CSV、图、LaTeX macros 自动一致 |
| 08-01 | 按新结果重写正文和摘要 | 所有旧 headline 数字已移除 |
| 08-02 | 压缩至 9 页；附录和引用检查 | 正文 ≤9 页，PDF 无溢出 |
| 08-03 | 独立复跑 1 seed；全文一致性审计 | 复跑数值在预期容差内 |
| 08-04 | 上传 OpenReview 草稿并检查匿名性 | PDF、metadata、作者匿名检查通过 |
| 08-05 | 只做提交缓冲，不再改实验协议 | 在截止前完成最终提交 |

若进度落后，删减顺序固定为：完整 ShapG guidance → 额外 utility 消融 → 次要 Top-k ranking 图。不能删减 P0 修复、主删除实验、正确 HV、5 seeds 或 9 页合规。

---

## 11. 投稿前最终检查表

### 代码和数据

- [ ] 旧 65 个测试和所有新增测试通过。
- [ ] Ruff 静态检查通过。
- [ ] 每个数据集 5 个 seed 完整，无静默缺失。
- [ ] train/val/test 行索引互斥；所有预处理器只在 train 拟合。
- [ ] 搜索阶段 test access 次数为 0。
- [ ] Hypervolume 已用已知 front 和独立实现交叉验证。
- [ ] Full/Filtered NSGA-II evaluation budget 相同。
- [ ] 新结果目录包含环境、配置、hash、日志和 status。

### 统计和图表

- [ ] 表格同时给出均值、标准差或 95% CI，并保留原始 seed 值。
- [ ] 每个显著性结论注明检验、配对单位和多重比较校正。
- [ ] 不显著结果没有被写成 superiority。
- [ ] α 点经过 dominance check，未把全部点自动称为 Pareto-optimal。
- [ ] Credit 同时报告 ROC-AUC 和 PR-AUC。
- [ ] 运行时间包含 MinPPI 预筛选成本。

### 论文

- [ ] 标题、摘要、贡献列表只包含新实验支持的主张。
- [ ] 删除 Madelon `f1–f20` ground truth precision/recall/F1。
- [ ] 删除或改写 `must remove`、`threshold-free`、`guaranteed Pareto`、`safe on all datasets`。
- [ ] Algorithm B 名称与实际 guidance 一致。
- [ ] 数据集数、seed、模型、指标、预算与配置一致。
- [ ] 所有正文数字来自生成的 LaTeX macros 或表格。
- [ ] 正文最多 9 页；参考文献和附录页数单独检查。
- [ ] 双盲 PDF 不含作者、机构、文件路径、GitHub 身份或 PDF metadata 泄漏。
- [ ] 引用完整，所有相关工作表述可由文献支持。
- [ ] Limitations 明确说明 existential harm、估计误差、计算成本和数据集依赖性。

### 提交

- [ ] OpenReview 中实际 deadline 与时区再次确认。
- [ ] 标题、摘要、关键词与 PDF 完全一致。
- [ ] 上传后下载平台生成的 PDF 再检查一次。
- [ ] 在截止前至少 24 小时保存一版可提交版本。
- [ ] 最终提交状态和确认邮件已保存。

---

## 12. 最终建议

接下来第一优先级不是立即重跑，而是用 1—2 天把 P0 问题全部修好并让测试证明它们已修好。只有协议冻结后，单 GPU 连续运行才有价值。当前最值得保留的经验主线是 matched-budget deletion；NSGA-II 应作为条件性应用案例，不能再承担“普遍 +37% HV”的 headline。

如果新结果弱于旧结果，应降低结论强度而不是调整协议寻找更好数字。对于 ICOMP，这种诚实、可复现、与 optimization 主题紧密对应的论文，比包含更多但不可审计的正结果更容易通过技术审查。
