# 训练数据、计时与复现

在 `history` 目录运行以下 PowerShell 命令。先构建低维内核，使用新的输出目录可以保留本轮的原始计时。正式实验每种方法端到端计时，包括选列、切分训练与测试矩阵的列、训练集预处理、拟合最终岭模型与预测。F 和随机基线直接使用当折新版选中的列数。预热、导入、数据文件读取、音频特征提取、测试集评分及保存结果在计时之外。各方法共用同批训练和测试矩阵。部署如果从音频开始，应另加所有方法共同的声学特征提取时间。

| 研究协议 | 配置 | 训练及预测 | 比较依据 |
|---|---:|---|---|
| 原始面板和额外面板 | 69 + 57 | 原归档的标准化 Ridge 或 RidgeClassifier | 原输入指纹、Full 逐样本预测与历史记录一致 |
| 原论文正式实验 | 40 | 原正式协议下的 LSQR Ridge | 八个来源各五个种子，核对原始列映射、训练与测试索引及历史 Full 分数 |
| 四阶段数据 | 30 | 原四阶段数据的单一 Ridge 配置，训练内均值处理和标准化，Cholesky | 10 个数据各三种划分，核对历史 Full 预测 |
| 稠密与语音 | 19 | 原版 Cholesky RidgeClassifier | 保留说话人或设备单位划分、分组指标，核对历史 Full 预测 |
| 近红外 | 36 | 原光谱实验的 LSQR Ridge，浓度等连续响应 | 12 个数据任务，各固定训练/测试集使用三种选列种子，核对历史 Full 预测 |
| 冗余或分散信号模拟 | 64 | 原生成器、标准化与 Cholesky | 两种机制、两种任务、两种宽度、八次独立数据生成 |
| 调查、广告与原稿模拟 | 33 | 原协议的岭模型 | 原稿模拟 15，调查原始及加噪 12，广告嵌套训练规模 6 |
| 稀疏文本与 Farm Ads | 38 | 保持稀疏的原始 Ridge | 分块 CSR 训练输入，宽矩阵新旧投票数值等价 |
| 信用卡欺诈历史扩展 | 1 | 同一冻结划分、同一 29 列，标准化后的 LSQR Ridge | 此处比较五种选列方法，原会议实验使用的预测器为 XGBoost |

F 为单变量得分。分类使用 ANOVA F，回归使用单特征线性回归 F。每个样本划分先独立计算，再按确定顺序取前 k 列。随机用固定随机种子，均匀无放回抽取 k 个不同列号。低维的单同伴消融还测量了每折 32 个等宽随机子集的质量分布。[分类 F 的官方定义](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html) · [回归 F 的官方定义](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)

相对于某个对照的分差为新版减对照，耗时差为新版减对照（毫秒）。每进程内取各方法重复部署的中位数。报告中的单一毫秒值为两个进程中位数的平均，再在同一来源取配置中位数。删列加速要求 k 小于 p，且两个进程中新版总时间都严格更低。保守时间比采用两进程内配对比值的较大值，不能用表中两个边际中位数相除重建。得分变好要求绝对分差大于 10⁻¹²，平分加速单独保留。成功数不设精度下限，另列下降最多 0.02 的子集。所有这些计数都是描述性配对结果。

原论文保留分类 AUC、回归 R²。其余分类沿用各实验的 BA（平衡准确率）或原分组 BA 聚合。信用卡另报告平均精确率和欺诈召回。不同指标分别展示。卷积声学预处理、文本向量化等共用输入构造在时间范围以外。数据集均已由旧研究或本轮比较使用，现有计数描述选定基准上的表现。

```powershell
.\thesis-python.ps1 -m research.lowdim.build_native
.\thesis-python.ps1 -m pytest research/lowdim -q
```

核心 126 组、正式 40 组、调查/广告、稀疏 CSR 和信用卡的命令行参数由各自的 `--help` 给出。实验脚本默认以新目录保存，防止覆盖原结果。四组补充面板的完整命令如下。

```powershell
$runPath = 'results/minpi-study/lowdim-breadth-reproduction'
foreach ($panel in @('four_stage_ridge','dense','spectra','mechanism')) {
    foreach ($runId in @(1,2)) {
        .\thesis-python.ps1 -m research.lowdim.extra_study $runPath --cohort $panel --process $runId
        if ($LASTEXITCODE -ne 0) { throw "Failed $panel $runId" }
    }
}
.\thesis-python.ps1 -m research.lowdim.extra_study $runPath --summary research/lowdim/extra_evidence
.\thesis-python.ps1 -m research.lowdim.audit_breadth $runPath research/lowdim/extra_evidence
.\thesis-python.ps1 -m research.lowdim.combined
.\thesis-python.ps1 -m research.lowdim.verify_combined
```

[原 69 组与额外 57 组协议](core-protocol.md) · [原论文正式协议](formal-protocol.md) · [调查与广告协议](survey-protocol.md) · [稀疏文本协议](text-protocol.md) · [信用卡扩展协议](fraud-protocol.md)

