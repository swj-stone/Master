# PIv1 封存记录

2026-11-13 在硕士分支 `thesis/masters` 封存，Git 标签为 `thesis/piv1-frozen`。算法源码来自提交 `dee8f4257580c7571bba5ba186b47e56bf07419f`。标签同时收录随后完成的指标补算、采样基线和实验核验。

## 冻结的方法

正式配方为 `vote`。在最多 512 条验证样本上，用全模型 TreeSHAP 构建 32 条排列路径。相对投票按小、中、大联盟统计净票率，取三层最低值。综合排序把该值乘以 0.30，加到训练集 Gain 和平均绝对 TreeSHAP 的归一化名次均值上。选列阶段在由输入列数决定的预算内验证不同子集。100 列时预算为 24 个候选，包含全特征模型。按完整验证集的主指标选列，冻结后再评估测试集。

[投票源码](../../../../src/minppi_mofs/piv1.py) · [子集搜索源码](../../../../src/minppi_mofs/piv1_search.py) · [正式运行协议与源码 SHA-256](protocol.json) · [开发消融](../../test/piv1-pilot.md)

## 结果与恢复

六个正式数据集、五个 seed 共 30 组。PIv1 的 580 次拟合共耗时 689.46 秒，直接 Gain 全前缀扫描的 1,994 次拟合共耗时 2,070.02 秒。追加的轻量子采样 PFI 基线共拟合 85 次，耗时 199.73 秒。PIv1 与 Gain 使用正式归档计时，PFI 使用独立冷启动实测。Abalone 的全空间枚举与排列 Shapley 五个 seed 均可回放，三种方法同次冷启动的测速限于 seed 42。

- [正式主表](final_results_by_dataset.csv)和[逐 seed 数据](final_results_by_seed.csv)含分类 AUC、准确率、回归 R²、MAE、运行时间与未删列验证基线。
- [PFI、PIv1 与 Gain 的 30 组配对](pfi-paired.csv)、[Abalone 五 seed 的精确与采样对照](abalone-exact-five-seeds.csv)、[Abalone seed 42 的冷启动计时](abalone-cold-seed42.csv)收录后续比较。
- [正式实验审计](independent_audit.json)记录数据划分、选列及预测复核。下列命令可在硕士工作树内查阅标签和复算本地证据包的 SHA-256。

```powershell
git show thesis/piv1-frozen --stat
Get-FileHash -Algorithm SHA256 .\packages\thesis-piv1-evidence.zip
```

[本地证据包](../../../../packages/thesis-piv1-evidence.zip)另存了正式及开发运行、对照实验、训练/验证/测试的 90 个预处理分区和 Abalone 的联盟缓存。包内按硕士工作树的相对路径存放文件，根目录的 `SHA256SUMS` 给出逐文件摘要。对照 [ZIP 的 SHA-256](evidence.sha256) 后，可解压到一个空的标签检出目录，读取归档数据与预测。PFI seed 42 的首次有效脚本已按当时协议中的哈希还原并随原始运行存放。共享虚拟环境由现有 `thesis-python.ps1` 使用。

## 后续版本

后续 PI 保留相对投票和跨联盟规模取 min 的机制，作为独立版本研究可插拔的前置筛选。比较同一个后端单独运行和加入 PI 后的端到端耗时、预测分数及保留列数，同时报告前置耗时和下游节省的工作量。正式 PIv1 的已有加速结果对应整套算法。
