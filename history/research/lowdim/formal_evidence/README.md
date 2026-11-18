# 原稿八来源复算

每个来源五个原稿随机留出划分，共 40 个配置。新版每折均删列，38 折的两次独立计时均快于 Full。其中 32 折的原稿指标相对 Full 下降最多 0.02。新版比分配相同列数的 F 更快 34 折，分数严格更高 22 折，两项同时 17 折。相对同宽随机的三个计数分别是 3、36 和 2 折。分类指标为 ROC AUC，回归指标为 R²。

Diabetes 五折均选 6/10 列，与同宽 F 五折得分均相同。相对 Full 的配对 R² 差中位数是 −0.01743，两个进程均较快的有 3 折，其中 1 折同时守住 0.02。Wisconsin Breast Cancer 五折均选 19/30 列，AUC 配对差的中位数为 0，五折均在两次测量中快于 Full。

`cases.csv` 提供逐折得分、各臂两进程的选列与完整时间。`sources.csv` 中的分数、分差和比值是逐折中位数。`process-medians.csv` 保留每个进程及方法的基础计时。`input-fingerprints.csv` 记录每折四个矩阵/向量的 SHA-256 以及训练与测试行数。`verification.json` 记录被检查的 200 个方法配置、2,000 次计时及 670 份跨进程数组。

在 `history` 中调用 `thesis-python.ps1 -m research.lowdim.formal_study run <run1>`、`thesis-python.ps1 -m research.lowdim.formal_study run <run2> --previous <run1>`、`thesis-python.ps1 -m research.lowdim.formal_study summarize <run1> <run2> <out>`。两个归档分别是 `results/minpi-study/lowdim-formal-20261118a-pass1` 和 `results/minpi-study/lowdim-formal-20261118a-pass2`。恢复输入摘要可调用 `thesis-python.ps1 -m research.lowdim.formal_fingerprints <run1> <out>/input-fingerprints.csv`。
