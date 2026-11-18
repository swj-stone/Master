# 现有实验的对照方法及核验

正式比较应给出方法全名、对应实现、保留列数、评价划分和求解器。本页覆盖最近一次稠密成本审计及四个文本开发折。基准论文、统计量定义、软件接口和项目自行规定的预算分别列出；每个结论只适用于表中范围。

| 实验标签 | 方法与出处 | 项目中的选择 | 核验状态 |
|---|---|---|---|
| `F50`、`F_match` | 单因素方差分析 F 统计量，scikit-learn `f_classif` | 在训练行逐列计分，按分数降序取前 k 列，同分优先原索引较小者，NaN 分数排末；固定半宽取 ⌈p/2⌉ | NIST 原算例、二值 CSR、三分类、零方差、完美区分、并列及实际配置通过 |
| `random50`、`random_match` | 同宽均匀无放回随机子集。用 NumPy 随机排列的前 k 项，再按原索引排列 | 选列独立于标签，种子、抽取次数与 k 属于实验协议。每份支持概率为 1/C(p,k) | 两个稠密开发配置及四个文本开发折复算一致 |
| `full` | 全特征岭分类器。岭回归依据 Hoerl 与 Kennard；二类/多类编码及预测规则使用 scikit-learn RidgeClassifier | 全部列按原序保留，截距不受惩罚，α=1。稠密矩阵在训练行上中心化并标准化，稀疏输入按训练列标准差缩放，求解器处理截距。LSQR 容差 0.001、上限 250 步；同一稠密数据另测 Cholesky | 手工类别编码、增广最小二乘和直接 LSQR 对照通过。稠密真实配置逐项重现决策值与预测 |
| `pooled_match` | **项目定义**的词项出现边际岭收益。对二元词项与二元标签的中心化独热响应，逐列拟合带单位惩罚的一变量回归，以相对仅截距模型的惩罚平方误差减少值排序 | 在全部训练行或明确标识的 512 个提案行求分，切回原列索引 | 直接比较两个目标函数、四个开发折两种行范围的分数及支持均通过。输入限二分类和规范二值 CSR |
| `F_bestcal` | 官方 ANOVA F 排序加**项目定义**的训练内选宽 | 仅用训练内排名行评分，在独立校准行选 BA 最高的真正删列前缀；并列取更窄者，整列参照容差 0.02，失败回退全宽 | 两个稠密配置重算 F 排序的全部候选前缀、BA 及最终支持，使用与 minPI 一致的校准费用 |
| `representatives_bestcal` | minPI 的投票消融 | 保留项目的候选代表列及其相关性/相关强度筛选顺序，关闭枢轴投票，复用同一训练内选宽与岭拟合协议 | 核对关闭投票、代表列顺序、BA 选宽及最终支持 |
| `grouped50`、`shared_c`、`pivotal_bestcal` | minPI 旧版本和自适应候选，用于版本对比 | 每次计时重新选择、校准，再切列、拟合、预测 | 候选算法的单独机制与测试详见[组合实验报告](../../docs/minpi-study/combination-exploration-results.md) |

ANOVA 的定义为组间均方除以组内均方，各自自由度为类别数减一、样本数减类别数。NIST 例子公布的 F 为 9.59，本次独立按组中心化计算得到 9.591107036442805。项目采用官方逐列统计量，前 k 列的预算和并列处理由实验规定。文中应使用“单因素 ANOVA F”，避免和其他名为 Fisher Score 的方法混用。[NIST/SEMATECH 公式和公开算例](https://www.itl.nist.gov/div898/handbook/prc/section4/prc433.htm) · [scikit-learn f_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html) · [项目选列实现](../../experiments/compare_predictive_minpi.py)

岭原论文讨论回归估计。分类实验另使用 −1/+1 的目标编码，二分类看决策值是否大于零，多分类取最大决策值。LSQR 的实现求解阻尼最小二乘问题，直接分解与迭代停止可能导致不同预测，因此每种求解器单独报告实测精度。当前复算环境为 scikit-learn 1.9.0、SciPy 1.18.0。[Hoerl 与 Kennard 原论文](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634) · [RidgeClassifier 官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html) · [scikit-learn 1.9.0 求解器源码](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.0/sklearn/linear_model/_ridge.py) · [Paige 与 Saunders 的 LSQR](https://web.stanford.edu/group/SOL/software/lsqr/)

固定种子随机法在两个稠密开发配置各抽一次。文本同宽比较还按 `[108391, fold, selector_seed, draw]` 生成 20 份排列；同一配置的不同预算复用每份排列的前缀。单次抽样和随机支持的分布回答不同问题，不能把共享排列或同一配置的重复预算计作独立数据集。[NumPy 2.4 permutation](https://numpy.org/doc/2.4/reference/random/generated/numpy.random.Generator.permutation.html) · [文本每次抽样协议](../../experiments/compare_calibrated_same_width.py)

对于词项列 x 和独热目标 Y，令 xᶜ、Yᶜ 为训练行中心化结果。pooled 收益为 ‖(xᶜ)ᵀYᶜ‖²/(‖xᶜ‖²+1)。快速实现通过 CSR 的存储索引计数。换成重复索引、显式零、原始词频或 TF-IDF 矩阵会改变这种计数的含义。已核对四份文本训练输入均采用训练词表、规范 CSR、无显式零、所有存储值为 1。后续若采用词频或多分类，应使用对应输入定义下的评分器并重新验算。[快速实现](../../experiments/fast_document_vote_benchmark.py) · [实际向量化与划分](../../experiments/document_vote_benchmark.py)

## 复验与结果

从 `history` 执行下列命令。它读取本地四个文本折、两个稠密配置和存档数组，使用共享依赖，并打印独立计算结果和源码摘要。[核验程序](verify_existing.py) · [本次运行输出](verification.json)

```powershell
.\thesis-python.ps1 -m research.baselines.verify_existing
```

- 稠密配置复验 Full、半宽 F、半宽随机、F 自选宽、无投票代表列自选宽，分别配两个求解器，共 20 个结果。两类和四类构造上的决策值相对直接求解最大绝对差小于 2×10⁻¹⁵。这里的比较对象是同一求解器在两种实现下的输出。
- 文本 A2/B2 实验共核对 184 份同宽对照支持，包含 160 份带抽样编号的随机支持；对 F、pooled、固定随机重算 24 份预测。组合交换实验另外对 16 个“F / pooled × 提案行 / 全训练行”结果核对支持和预测。在各实际行集上，还用每类出现次数独立重算了 F 与 pooled 的全部分数。
- 原稠密计时从共同训练/测试矩阵开始，含列选择及其校准、矩阵切片、训练标准化、学习器拟合和测试预测。文本开发的同宽对照只回答选列质量。不同范围的耗时不能拼接。固定规模获得的 k 若用于同宽质量对照，注明 k 的来源；宣称端到端自适应加速时，要计入确定 k 的时间。[计时实现](../../experiments/grouped_solver_cost_audit.py) · [稠密原始记录](../../results/minpi-study/grouped-solver-cost-audit-20261117a/paired-costs.csv) · [文本同宽原始记录](../../results/minpi-study/calibrated-same-width-dev-20261117a/summary.csv)

旧的成本审计驱动按固定八项清单分发方法，内部 `select` 的兜底会将未知的固定宽度标签送入随机选择。因此它只适合重放原清单，新方法不能仅通过向此清单追加名称接入。固定宽度须满足 1≤k≤p。历史两例的十个稠密对照组合/例均已对照明确名称及支持复算。程序中的 `F` 适用于完整、有限的分类输入；若原任务是连续目标，应单独核查其回归测度及评分器。新增数据还需按其训练/测试分组或时间协议重新判断可用性。

历史扩展脚本另含 ReliefF、L1 逻辑回归系数筛选、mRMR 等不同协议。它们的论文对应关系、候选宽度和实现必须分别审计，不从本页的六个开发配置外推。新论文对照的核验另见[论文对照目录](../comparators/)。[外部分类实验](../../experiments/minpi_study_external.py) · [早期 mRMR 实现](../../experiments/test/piv2_conventional_filters.py)

历史 ShapG 有多个入口。`baselines/shapg.py` 的 `select(context)` 记录了它所对应的图方法和效用设置，可以用小图测试核对边覆盖和局部分数。另一个基于互信息的启发式入口，以及参考论文 2510.24982v2 中的冻结模型、余弦图实验，需要各自检查特征图、收益和采样约定。本轮正式 Digits 对照采用已核验的离散 mRMR–MID；历史 ShapG 的全数据结果未在此轮复验。[历史论文实现](../../src/minppi_mofs/baselines/shapg.py) · [小图测试](../../tests/test_game_graph_banzhaf.py) · [历史审计脚本](../../experiments/audit_literature_methods.py) · [参考论文笔记](../theory/README.md)

