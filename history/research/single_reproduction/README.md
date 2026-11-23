# 论文最终版本与完整复现

最终算法是 **Sampled Single-peer**，实现为 `research.fast_single.selector.select_fast`，基准提交为 `thesis/masters@56f47fa`。论文数据表里的 `fast` 对应该算法，`single` 对应穷举单同伴对照，`F_fast` 和 `random_fast` 分别是按最终算法的 K 选择的单变量 F 和均匀随机子集。分类和回归使用各自对应的 F 统计量。原始维度不超过 32 时使用所有单同伴，超过 32 时在最多 384 条训练样本上抽取 8 个同伴。该提交另外保存了少数类抽样及稀疏计算的可选实现 `select_retained`，论文所报结果使用 `select_fast`。[论文正本](../../paper/minpi-lowdim/main.pdf) · [算法实现](../fast_single/selector.py)

## 保存的实验

| 内容 | 数量 | 核验依据 |
|---|---:|---|
| 主实验 | 原 387 ＋ 扩展 18 ＝ 405 个配置，74 个低维、331 个较宽 | 训练/测试特征与标签指纹、支持列、预测及可用的决策值、各方法得分、两轮原始计时 |
| 宽数据消融 | 331 个配置 | 固定宽度移除同伴票、固定排名移除宽度修正、38 个稀疏配置的非零数 |
| 低维子集与图 2 | 31 个配置，各 32 次随机抽样 | 992 次随机预测；Abalone 的三个拆分各有 45 个不同的八列子集，三个选择器种子共记录 405 个枚举分数 |
| 补充实验 | Gasoline、Yarn、Olive、Real-sim，各 3 个种子 | 12 个配置独立于主 405 表，保留原划分、7 个方法标签及配对时钟 |
| 论文 | 三张图、英文 LaTeX、正文 PDF | 保留出图数据、生成脚本、原图和排版后的正本 |

原论文归档中，405 组里有 320 组同时删列并快于全特征，加入原指标绝对损失不超过 0.02 后剩 215 组。一次复现中重新测得的毫秒数随主机及负载变化。要复原论文图表所用的计时，应使用原来两个进程的记录。[主结果](../fast_single/evidence-extension/all-405.csv) · [补充结果](../fast_single/evidence/transfer.csv)

## 使用复现包

工作区归档为 `history/packages/single-peer-paper-56f47fa.zip`，其旁边的 `.sha256` 文件记录整包校验值。ZIP 解压到新的空目录后应有同级的 `history/`、`data/`、`results/`、`environment-data/`。从其中的 `history/` 运行下面命令。比如在含共享 `venv/` 的 `Paper/` 下创建 `SingleReplay/` 再解压，则 `SingleReplay/history/thesis-python.ps1` 可直接定位现有环境。实验依靠相对路径，也可还原为原来的 `MastersThesis/` 布局。

```powershell
.\thesis-python.ps1 -m research.single_reproduction.bundle verify
.\thesis-python.ps1 -m research.single_reproduction.replay archived
.\thesis-python.ps1 -m research.single_reproduction.paper_evidence --render
```

第一条逐文件验证 SHA-256，同时检查环境中八份内置数据。第二条从两个原始计时进程重建主表与补充表，并逐项比对论文归档。第三条核对论文数值，在临时目录重建全部图并运行两遍 XeLaTeX。校验比较三图 PDF/PNG 的文件内容、论文 PDF 的全文和 96 dpi 的逐页渲染。编译 PDF 的文档元数据可能随构建变化。

要重新选列、拟合、预测和计时，执行以下命令。每个计时过程独立启动，数值线程限制为 1。新测量及新表写入 `results/minpi-study/single-paper-replays/` 下的新目录，屏幕上会给出路径。重新拟合的列、预测与得分对照原记录；新测时长单独记录。若只需验证质量，最后一条命令会重放主实验、两项消融及补充数据并输出读取轨迹。指定的审计输出目录专用于这次核验。

```powershell
.\thesis-python.ps1 -m research.single_reproduction.replay fresh
.\thesis-python.ps1 -m research.single_reproduction.inputs_trace --out tmp/single-quality-check
```

原机器使用 Windows x86-64、Python 3.13.9、NumPy 2.4.6、SciPy 1.18.0 和 scikit-learn 1.9.0。复现包保存了 Python 代码、三个原生库及 C 源码；在相同 Windows 环境可使用归档二进制。重新编译需要 C11 编译器。图文核验还使用 XeLaTeX 和 Poppler 的 `pdftotext`、`pdftoppm`。在另一台机器应先配置相应解释器、依赖和工具。[环境版本](environment.json) · [项目依赖](../../pyproject.toml)

## 哪些路径必须保留

[keep.csv](keep.csv) 每行是一份应保留的工作区文件，记录以 `history/` 为基准的相对路径、字节数、SHA-256 及用途。该文件自身由归档整体校验，八份 Python 环境数据另有清单，在 ZIP 内也留有副本。[实际读取清单](inputs.csv) · [上游来源清单](inputs-origins.csv) · [环境内置数据](inputs-environment.csv)

- `../data/raw/credit_card_fraud.csv` 和 `../results/icomp2026_formal/.../credit_card_fraud/s42/` 中的拆分元数据和两份数据表是论文输入。仅复制 `history/` 会漏掉它们。
- `tmp/text-peer-panels-20261118b/` 下的 91 个文件是论文用的文本特征矩阵、标签及索引。历史实验目录中的部分 NPZ 也直接用于校验数据、对照支持或预测。
- 30 个已保存的文本面板通过原语料独立重建；MHEALTH 的 3,355 个窗口和两张 NIR 药片数据表也已从上游数据复核。Real-sim 的原始稀疏行、标签及跨集分组隔离均已核对。目前需要保留冻存的训练/测试 CSR、标签及 manifest 才能恢复完全相同的归属，现有资料未找到最初生成该划分的随机脚本。[输入复核](inputs-verified.json) · [上游复核](inputs-origins-verified.json)

共享 Python 环境、参考论文与原会议稿各自保留在原位置。工作区的清理候选和审批范围独立记录；打包及校验命令只处理复现材料。
