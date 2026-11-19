# 复现运行

从 MastersThesis/history 执行，使用这个目录的 thesis-python.ps1 和已有依赖。本次扩展使用 Windows x86-64、Intel Core Ultra 9 275HX、Python 3.13.9、NumPy 2.4.6、SciPy 1.18.0、scikit-learn 1.9.0。数值线程池在计时进程中限制为单线程。预先编译原生库。原始数据、历史预测和先前的 74 个低维对照均来自同一历史工作区。

~~~powershell
.\thesis-python.ps1 -m research.lowdim.build_native
.\thesis-python.ps1 -m research.wide_blocks.build_native
.\thesis-python.ps1 -m pytest research/wide_blocks/test_method.py -q
$runPath = 'results/minpi-study/wide-blocks-replication'
.\thesis-python.ps1 -m research.wide_blocks.benchmark $runPath --process 1
if ($LASTEXITCODE -ne 0) { throw 'First pass failed' }
.\thesis-python.ps1 -m research.wide_blocks.benchmark $runPath --process 2
if ($LASTEXITCODE -ne 0) { throw 'Second pass failed' }
.\thesis-python.ps1 -m research.wide_blocks.benchmark $runPath --summary
.\thesis-python.ps1 -m research.wide_blocks.summarize
.\thesis-python.ps1 -m research.wide_blocks.pool_quality $runPath
~~~

每次计时请使用新目录。汇总命令刷新 research/wide_blocks/evidence 下的结果。检查两次运行的 complete.json 可获得源代码和原生库哈希。各次耗时、各配置精度与列号/预测摘要位于对应 pass 目录中。第一个进程还保存新宽度下的列集与预测数组，第二个进程逐项核对。旧版与全特征的原始预测取自历史存档，两次都作验证。

主表中 331 个新跑配置的 F 和随机都按提前停止版的实际列数选列。F 用分类方差分析或单变量回归 F 检验，同时以独立的方差分解核验边界。同宽随机用该配置的登记种子，对列号作均匀随机排列、无放回选取前 K 列。对照不承担确定 K 的费用。74 个低维配置直接并入上批在相同定义下的配对结果；低维的强制补足对照留空。整个工作区的来源与索引见[完整数据覆盖](coverage/README.md)。

全流程计时不包括数据读取、一次性编译/加载、音频与文本的公共特征提取和测试评分。候选集上限 256 及使用整个候选集拟合的补充实验只比较得分。
