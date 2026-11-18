# 归档与目录导航

从当前研究首页可以进入新的算法与配对实验。下表按复现时所需的对象组织此前积累的材料。已有运行清单记录了源码位置和 SHA-256，重放程序还会按目录名读取原始支持、预测和数据划分，因此保留原目录结构。

| 对象 | 所在位置与用途 |
| --- | --- |
| 近邻 pivotal minPI 原始定义与精确求解 | [稿件](../../paper/minpi/main.tex)、[形式化证明](../../paper/minpi/formal/README.md)、[整数选择器](../../src/minppi_mofs/word_minpi.py)、[对外筛选接口](../../src/minppi_mofs/support_selector.py) |
| 当前改进的起点 | [按相关性分组并建立投票](../../src/minppi_mofs/grouped_minpi.py)、[最后一轮组合与成本实验](../../docs/minpi-study/combination-exploration-results.md) |
| 累积的数值表、光谱与文本实验 | [按研究问题索引](../../docs/minpi-study/README.md)、[跨数据源比较](../../docs/minpi-study/next-vote-results.md)、[文本种子和迁移复核](../../docs/minpi-study/reliability-document-results.md) |
| PIv0–PIv3、会议材料 | [PIv0 历史对照](../../docs/thesis/experiments/piv0-method-comparison.md)、[PIv1 冻结标签和归档](../../docs/thesis/archives/piv1/README.md)、[后续预筛试验](../../experiments/test/README.md)、[会议资料副本](../../conference-paper/) |
| 运行入口和依赖 | [实验程序](../../experiments/)、[算法库](../../src/minppi_mofs/)、[配置](../../configs/)、[回归测试](../../tests/)、[共享环境启动器](../../thesis-python.ps1) |
| 本地证据与交付物 | [输入数据](../../data/)、[按运行编号保存的证据](../../results/minpi-study/)、[固定预算研究复现包](../../packages/pivotal-minpi-study.zip)、[压缩包校验值](../../packages/pivotal-minpi-study.zip.sha256) |
| 本轮之前的首页 | [原始字节快照](workspace-readme-before-reorganization.txt)。里面的历史路径按原位置解释。 |

本次清理范围限于 `history/tmp/` 中三个已结束测试运行生成的用例目录（b2、c4、f4），以及两个带 pytest 缓存标识的旧缓存（c9rc、c9rtcache）。清理前核对了 Git 跟踪状态、目录结构、实际路径和重解析点。每个删除项的文件数与字节数见[清单](cleaned-scratch.csv)。其他临时目录里含有审稿证据、调试脚本、本机内核和未归类结果，应按其内容判断用途。
