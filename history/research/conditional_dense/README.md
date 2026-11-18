# 给定同伴后的重拟合岭投票

新算法关注一个可直接复算的问题：已有若干列参与判别时，加入候选列并重新拟合整个小型岭模型，目标函数能改善多少。每个候选列在不同同伴数下收到各类的贡献票，取两个同伴层评分中较小者。这个 minPI 变体改变了票的效用和投票人。原分组版在近邻三元组上投多数票，这里在分类器的小型岭问题上计分，实际比较同时保留原分组版和去掉同伴的消融。

**模型、评价样本、投票和重拟合。** 只从外层训练集取最多 768 行，先为每类随机留一行，再从剩余行均匀补齐。第一阶段初模是这些行上的标准化岭分类目标，采用中心化的独热编码，岭罚为样本数平方根。对每个候选分别比较给定同伴的模型与加入该候选的模型，各类别的训练平方损失及惩罚的变化构成贡献票。内部的票来自拟合集，泛化精度由外层测试集评价。同伴来自随机排列，同一候选有较小和较大两种同伴数，全部候选共享分解与投票计算。最后用选中列和全部外层训练行重新拟合标准化、单位正则的 Cholesky RidgeClassifier，再对外层测试行预测。[冻结规则](protocol.md) · [v2 实现](shared_method.py) · [v1 开发结果](development.csv) · [v2 开发结果](development-v2.csv)

设训练子样本的标准化矩阵为 X，中心化目标为 Y。对同伴集合 R 定义 G_R=X_RᵀX_R+λI，h=X_Rᵀx_j。单列加入后全部系数均允许重拟合，目标改善量为：

```
||x_jᵀY − hᵀ G_R⁻¹ X_RᵀY||² / (λ + x_jᵀx_j − hᵀ G_R⁻¹ h)
```

该式由分块消元直接得到。若 j 已经处于共享排列前 s 项中，改用前 s+1 项剔除 j 作为其 s 个同伴。此时同一加入收益是扩展系统中第 j 行系数的平方范数除以逆矩阵的第 j 个对角元。每个候选获得规定数目的其他列，单独看它时同伴集合均匀分布。v1 每个规模重新求解，并用两份排列平均票值。v2 采用一个排列、同一最大前缀的 Cholesky 因子与投影，复用两个层以及候选替换的全部子系统。两个版本的同伴数、选宽和精度格式也不同，开发成绩变化无法只归因于提速。公式已在小矩阵与真正的逐候选重拟合核对，也验证了遍历全部排列的平均量等于按同伴集合直接枚举的结果。[独立数值测试](test_method.py)

自动宽度等于较大同伴数，加上其余标准化列在这些同伴上剩余尺度比的和，向上取整并限于输入宽度。相比对投票分数求累计质量，这个规则用输入冗余决定预算；剩余尺度由带岭惩罚消元后的 Schur 对角估计，是启发式量。模型没有将目标函数下降换算成测试 BA 损失的保证。相同宽度的零同伴消融复用相同行、缩放、独热编码；对照可免费获得当次宽度。

**实测。** 主实验在两个 ISOLET 随机划分、HAR 跨受试者、Gas 跨批次、Multiple Features 重新分层三折和 Digits 小矩阵压力测试上运行。同一组须同时真实删列、对全特征模型损失不超过 0.02、两次进程总时间都更短，才算 Full 联合达标。最后一列还要求同一组的新算法分数同时严格高于同宽 F 与 20 个同宽随机支持的中位数。各来源的分数按协议先按测试单位再按配置等权。独立语音确认采用冻结规则和 UCI 官方数据。[逐组分数和配对比](evidence/pairs.csv) · [语音预案](speech-protocol.md) · [语音逐组结果](speech-evidence/pairs.csv)

| 来源 | 保留列数 | 平均 BA | 较同宽 F 变化 | Full 联合 | Full 联合且同时胜 F 和随机中位数 |
|---|---:|---:|---:|---:|---:|
| ISOLET，另外两个外层划分 | 355–360 / 617 | 91.56% | +2.23 个百分点 | 2/2 | 2/2 |
| HAR，同一跨人测试的三个选择种子 | 267–274 / 561 | 95.00% | +2.52 | 3/3 | 3/3 |
| Gas，同一前推测试的三个选择种子 | 51–56 / 128 | 75.03% | −8.77 | 2/3 | 0/3 |
| Multiple Features，一个来源三折 | 349–352 / 649 | 98.49% | −0.15 | 3/3 | 1/3 |
| Digits，两个额外外层折 | 52–53 / 64 | 94.20% | +0.25 | 0/2 | 0/2 |
| Spoken Arabic Digit，官方划分三种子 | 154–155 / 208 | 93.73% | +0.50 | 3/3 | 2/3 |

在 ISOLET、HAR 和阿拉伯语数字的这 8 个配置，新方法均实际删列、在容差内快于 Full，同时比同宽 F 更准且更快，并比同宽零同伴消融更准。若还要求比零同伴消融更快，这些配置均未满足。相同宽度下随机选择的总时间也更低。使用独立半宽 F 时，HAR 在 3/3 个配置以更少列数、更高分数和更低时间胜出。ISOLET 相比半宽 F 多留 46–51 列，一个配置更快、一个略慢。阿拉伯语料相比半宽 F 多 50–51 列，分数高 1.45–1.55 个百分点，总时间慢 14.2%–23.9%。这些对照说明同伴票确实能提供额外预测信息，同时仍然有计算和预算代价。[同行交集](intersections.csv) · [复算程序](intersections.py) · [主实验两次进程中位时间](evidence/medians.csv) · [新增语音逐项耗时](speech-evidence/medians.csv)

预设的宽稠密多分类类别在 Gas 和 Multiple Features 未通过。Gas 分布漂移下一次比 Full 下降 8.58 个百分点，说明训练内小岭模型的边际增益对未来批次可能不可靠。Multiple Features 的 F 已取得约 98%–99% 的分数，新票未稳定超越。Digits 的 Full 训练本身便宜，删列时间收益不足。ISOLET 与独立阿拉伯语音的成功支持了在这两个任务上的有限主张，同一官方测试集的三个种子仅用于检测算法对抽样种子的敏感性。

**复现。** 以下命令在 `history` 运行。两个进程应依次独占计时资源。out 路径须为新目录。原始结果在 `results/minpi-study/conditional-dense-{quality,timing1,timing2}-20261118a` 和 `conditional-speech-{quality,timing1,timing2}-20261118a`。额外语音 ZIP 来自 UCI，保留原官方训练与测试分区；每条语音的 13 个 MFCC 通道各按固定 16 个相对时刻插值得到 208 个输入。共用这次特征转换的代价另存 preparation.json，各方法统一从该矩阵开始计时。[UCI 数据、许可与引用](https://archive.ics.uci.edu/dataset/195/spoken+arabic+digit)

```powershell
.\thesis-python.ps1 -m pytest research/conditional_dense/test_method.py -q
.\thesis-python.ps1 -m research.conditional_dense.confirmation --pass-id 0 --out tmp/dense-quality
.\thesis-python.ps1 -m research.conditional_dense.confirmation --pass-id 1 --out tmp/dense-first
.\thesis-python.ps1 -m research.conditional_dense.confirmation --pass-id 2 --out tmp/dense-second
.\thesis-python.ps1 -m research.conditional_dense.audit tmp/dense-quality tmp/dense-first tmp/dense-second tmp/dense-audit
```

语音验证使用 `research.conditional_dense.speech` 替换上述三条测量入口，输出到独立目录后使用同一审核程序。若缺少原始文件，先运行 `.\thesis-python.ps1 -m research.conditional_dense.fetch_speech`。主审计检查了 636 次计时和每次进程 331 个数组，额外语音检查了 144 次计时和每次进程 75 个数组；两次进程的结果与独立质量运行逐项一致。见[主审计](evidence/verification.json)与[语音审计](speech-evidence/verification.json)。

在相同的 HAR、ISOLET 和阿拉伯语数字支持上，四种最终学习器的迁移复核共保存 192 条流水线结果。同宽 F 的分数在 24/32 个配对中被超过，实际删列且比 Full 更快的有 29/32；阿拉伯语数字加 PLE 后，同宽 F 在所有种子上都没有被严格超过。[跨学习器分数、实际耗时及支持宽度](transfer/README.md)
