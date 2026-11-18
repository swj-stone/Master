# 上一版审稿修订与复跑索引

最新主稿的研究设计、证据、可复跑入口和独立审稿决定见[当前研究指南](README.md)和[当前审稿记录](current-review.md)。

审稿所指的是上一份 15 页 ICOMP 稿。当前稿将方法范围限定为给定数目特征的逐列最弱层排名，保留原投票和取最小值的定义。完整结论和剩余异议见[独立复审](revision-review.md)，原文见[最初的审稿意见](../../paper/minpi/README.md)。

| 审稿关注 | 此次如何回答 | 可核对的材料 |
| --- | --- | --- |
| 五个真实换列的后果 | 同一见证、预算和学习器重放六种打分，逐列记录跨层扣减，逐折计算两后端 R² 与 MAE | [列名和配对质量](../../results/minpi-study/matched/regression_changes.csv) · [边界差](../../results/minpi-study/matched/swap_contrasts.csv) · [分项耗时](../../results/minpi-study/matched/stage_timings.csv) |
| 训练见证的波动 | 对 Gisette、ISOLET 交叉五个外部训练划分和五个见证种子，分别比较固定训练集和固定随机种子的支持重合度 | [逐项 Jaccard](../../results/minpi-study/matched/stability.csv) |
| 为什么使用 voting 和 min | 三个先后设计的受控实验，在各轮生成新世界前固定方案。独立见证衡量逐列目标，独立有标签样本检验预测。补充同样本上的中间层、层均值、随机筛选和有监督的 F 对照 | [设计、结果和复跑命令](../../results/minpi-study/mechanism/findings.md) |
| 能否外推 | 筛选冻结后在气体传感器的较晚批次、无组成重叠的材料组和腕踝设备上的新受试者上评测；所有三项均报告简单对照 | [气体与材料](new-cohorts-protocol.md) · [MHEALTH](../../results/minpi-study/device/findings.md) |
| 何时值得精算 | 同一浮点内核比较三个预算，再以穷举整数为基准检验有理数支持。单独展示真正近并列和低采样精度的边界情形 | [预算实验](frontier-protocol.md) · [最终的公平整数计时及反例](../../results/minpi-study/integer-final/findings.md) |

材料实验中，测试 R² 差值的 95% 区间以材料组为单位回抽，并固定已拟合模型的预测。重复的算法种子、同一人的多个时间窗、同一世界的不同采集条件均按其共享的来源解释。实验协议在本地保存，后续的模拟设置依据前一轮观察制定。

### 使用与复现

全层浮点分数可通过 `ExactMinPI` 查看。只需要数学上正确的所选列时，可在 scikit-learn 流水线中使用以下接口。特征选择应在每次训练划分内部拟合。

```python
from minppi_mofs.support_selector import PivotalSupportSelector

screen = PivotalSupportSelector(task="classification", fraction=0.1,
                               random_state=1129, arithmetic="rational")
screen.fit(X_train, y_train)
X_test_selected = screen.transform(X_test)
```

以 `MastersThesis/history` 为当前目录。新输出路径保留已有运行的原始记录。三个模拟与 MHEALTH 各自的复跑步骤见上表。

```powershell
.\thesis-python.ps1 -m experiments.minpi_study_matched --output tmp/replay-matched
.\thesis-python.ps1 -m experiments.minpi_study_new_cohorts --output tmp/replay-cohorts
.\thesis-python.ps1 -m experiments.minpi_study_cohort_checks --input tmp/replay-cohorts
.\thesis-python.ps1 -m experiments.minpi_study_frontier --output tmp/replay-frontier
.\thesis-python.ps1 -m experiments.minpi_study_integer --output tmp/replay-integer
.\thesis-python.ps1 -m experiments.audit_minpi_study_mechanism
.\thesis-python.ps1 -m experiments.minpi_study_cost_figure
.\thesis-python.ps1 -m pytest tests/test_integer_minpi.py tests/test_certified_minpi.py -q
lean paper/minpi/formal/Pivotal.lean
latexmk -cd -pdf -interaction=nonstopmode paper/minpi/main.tex
```

气体与材料的两个原始 ZIP 随包附上。MHEALTH 随包提供 40 个窗口特征、原始录制下载地址、哈希及提取程序。数据来源与作者分别为 Alexander Vergara 的传感器漂移数据、Kamran Hamidieh 的超导临界温度数据，以及 Oresti Banos、Rafael Garcia、Alejandro Saez 的活动数据。[Gas Sensor Array Drift](https://doi.org/10.24432/C5RP6W) · [Superconductivty Data](https://doi.org/10.24432/C53P47) · [MHEALTH](https://doi.org/10.24432/C5TW22)

在历史工作树运行完整测试，共 1,482 项通过，4 项依赖 Windows 符号链接权限而跳过。首次运行使用较长的临时路径，沙箱也拒绝了部分硬链接操作；短路径重跑涵盖全部 1,486 项。[最终测试日志](../../results/minpi-study/revision-final-tests.log) · [机器可读结果](../../results/minpi-study/revision-final-tests.xml)

本稿按机器学习方法论文的要求核对了理论适用条件、负结果、复现材料和主要对照。审稿标准还涉及工作的原创性、实际用途和读者覆盖面。当前采用作者指定的 ICOMP 版式，尚未指定投稿场地。[JMLR 作者指南](https://www.jmlr.org/author-info.html) · [NeurIPS 论文检查表](https://neurips.cc/public/guides/PaperChecklist)
