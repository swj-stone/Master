# 调查、广告与原稿模拟

33 个配置全部删列。ANES96 和 Fair 的两个原始调查各三折，两个扩列版本各三折，Internet Advertisements 的两个训练规模各三折，共 18 个配置均比 Full 快且掉分最多 0.02。原稿三个模拟各五次随机留出，稀疏信号有 2/5 达标，XOR 和随机标签各 0/5。新版比同宽 F 同时更快且分数更高的共 6/33，比同宽随机同时更快且更高的为 0/33。扩列和原始调查复用相同测试行，广告的两种训练规模固定为 256、768 个样本且分层嵌套。

`cases.csv` 含逐配置五臂得分和各进程时间。`process-medians.csv` 为进程级长表。用 pandas 读两表时指定 `keep_default_na=False` 以保留数据集名 `null`。`verification.json` 登记的 1,650 次计时和 624 份跨进程数组覆盖全部 33 组。

归档目录为 `results/minpi-study/lowdim-survey-20261118a-pass1` 与 `lowdim-survey-20261118a-pass2`。调用 `thesis-python.ps1 -m research.lowdim.survey_study run <run1>`、`thesis-python.ps1 -m research.lowdim.survey_study run <run2> --previous <run1>`、`thesis-python.ps1 -m research.lowdim.survey_study summarize <run1> <run2> <out>` 可重放。
