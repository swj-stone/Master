# 语音来源外部复核预案

前述训练规则、768行上限、两层同伴数、岭惩罚和选宽均已在 ISOLET、Digits 首折开发后冻结。为检验 ISOLET 上可见的语音分类现象，额外选定 UCI Spoken Arabic Digit（Bedda、Hammami；[UCI 官方资料](https://archive.ics.uci.edu/dataset/195/spoken+arabic+digit)，[数据说明](https://www.archive.ics.uci.edu/ml/machine-learning-databases/00195/documentation.html)，CC BY 4.0，DOI 10.24432/C52C9Q）。在读取新来源的特征、标签及结果之前写下此方案。所有同伴、选宽规则保持不变。该来源补充检验语音输入，不能抵消先前 Gas 和 Multiple Features 的失败。

源文件的每条语音是若干帧，每帧13个MFCC。对每个样本沿相对时间 0 到 1 的16个等距位置取各通道线性插值，再按时间、通道顺序展开为208列。源官方训练/测试两文件分别有每类660/220条，类别从0到9按文件顺序给出。此共同处理不看标签，无标准化外泄，作为输入准备另计；计时从所有方法接到相同208列矩阵开始。三种选择器种子1129、2027、3469共享官方测试集，并非三个独立来源。评价 pooled BA，主准则与 conditional_dense/protocol.md 相同，包括同宽F与20随机中位数、较全特征损失≤0.02与两进程实际加速。F_half、Random_half 与旧grouped均保留。
