# 稀疏文本的数值 minPI 检验

本面板保留此前登记的 30 个文档切分、20 Newsgroups 官方时间划分下的两个二分类任务、Farm Ads 同三折的两个嵌套训练规模，共 38 组。训练集、测试集及列均沿用已登记数据。Farm Ads 两种规模是同一来源。20 Newsgroups 两个任务也是同一来源。分类指标为测试集平衡准确率。

所有矩阵宽于 32 列，新旧数值 minPI 使用相同训练行、同伴、岭收益及集中度宽度。CSR 仅将所抽取的最多 768 个训练行转为连续 float32，避免将完整文本矩阵稠密化。此处不运行另一种文本专用投票。F 在训练 CSR 双精度矩阵上调用 scikit-learn 的 f_classif，用独立类别平方和核验选列边界。并列时按原列号排序。随机对照沿用每个切分的固定种子和均匀无放回列置换。F 和随机免费获得同宽预算。

五臂均用原始稀疏 LSQR 岭分类器。计时从收到训练和测试矩阵后开始，依次选列、切片、标准化、拟合和预测。Full 的输入矩阵可直接使用。每个进程每臂预热一次，每臂正式独立重选三次，随机排列方法后循环换位。两个全新单线程进程逐项核对支持、标签及预测。各自三次总时间中位数都低于相应对照才计加速，删列是成功的必要条件。保存全部逐次计时，以及全列历史预测的逐样本核对结果。

在 history 目录运行，一次只启动一个进程。

```powershell
.\thesis-python.ps1 -m research.lowdim.text_study results/minpi-study/lowdim-wide-text --process 1
.\thesis-python.ps1 -m research.lowdim.text_study results/minpi-study/lowdim-wide-text --process 2
.\thesis-python.ps1 -m research.lowdim.text_study results/minpi-study/lowdim-wide-text --summary research/lowdim/text_evidence
```
