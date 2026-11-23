# Sampled Single-peer 论文与实验

本目录保存最终算法、论文和复现 405 个主实验配置所需的文件。论文结果表用 `fast` 表示 Sampled Single-peer，用 `single` 表示枚举所有单同伴的对照。

| 内容 | 入口 |
| --- | --- |
| 最终算法 | [Python 选列入口 `select_fast`](research/fast_single/selector.py) · [C 计算内核](research/fast_single/panel_votes.c) |
| 对照方法与实验 | [穷举 Single-peer](research/peer_crossover/single.py) · [含同宽 F、同宽随机和全特征的配对实验](research/fast_single/runner.py) |
| 源稿、三张图与实验结果 | [论文 PDF](paper/minpi-lowdim/main.pdf) · [图表与来源](paper/minpi-lowdim/README.md) · [405 组逐项结果](research/fast_single/evidence-extension/all-405.csv) |
| 复现 | [环境和运行命令](research/single_reproduction/README.md) · [逐文件 SHA-256 清单](research/single_reproduction/keep.csv) · [完整复现包](packages/single-peer-paper-56f47fa.zip) |

从本目录运行以下命令。第一条执行最终算法的单元测试，第二条校验本地复现材料，第三条用原始两轮计时重建论文结果表。

```powershell
.\thesis-python.ps1 -m pytest -q
.\thesis-python.ps1 -m research.single_reproduction.bundle verify
.\thesis-python.ps1 -m research.single_reproduction.replay archived
```

`experiments/`、`src/` 和 `research/` 中还保留了正式实验实际调用的数据、模型和对照模块。文本面板依赖 `tmp/text-peer-panels-20261118b/` 中的矩阵，信用卡实验依赖与 `history/` 同级的 `data/` 和 `results/`。

本次从工作目录移出的旧源码、测试和实验资料存于硕士工作树的本地归档，详见[逐文件恢复说明](../packages/history-cleanup-56f47fa/README.md)。
