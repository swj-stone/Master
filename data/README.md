# Data contract

正式 ICOMP 管线只读取当前 pipeline 配置列出的六个 dataset YAML 所登记的 `raw_path`，不会扫描并自动纳入 `data/raw/*.csv`。数据文件被 Git 忽略；dataset YAML 中的 `raw_path`、target、task、utility 和 split 是权威定义。

## 获取 raw cache

在项目根目录安装环境后运行：

```powershell
$ErrorActionPreference = "Stop"
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
$SCRIPTS = @(
  "experiments\preprocess_data\preprocess_abalone.py",
  "experiments\preprocess_data\preprocess_Credit-Card-Fraud.py",
  "experiments\preprocess_data\preprocess_Gisette.py",
  "experiments\preprocess_data\preprocess_ISOLET.py",
  "experiments\preprocess_data\preprocess_Madelon.py",
  "experiments\preprocess_data\preprocess_Spambase.py"
)
foreach ($SCRIPT in $SCRIPTS) {
  & $PY $SCRIPT
  if ($LASTEXITCODE -ne 0) { throw "data acquisition failed: $SCRIPT" }
}
```

脚本从 UCI、OpenML 或脚本中声明的公开镜像下载数据，并只写相应的 `data/raw/<dataset>.csv`。已有文件只会按照 dataset YAML 的 SHA-256 与字节数进行复核，不会被清洗、去重或重写；不匹配时脚本保留原文件并失败关闭。缺失文件先写同目录临时文件，完整性检查通过后才原子发布。`preprocess_crimedata.py` 仅用于复核已隔离的历史快照，不得加入上述正式脚本列表。

## 防止数据泄漏

这些 acquisition 脚本不负责正式 split、标准化或特征选择。每次 run 的 `preprocess` 阶段从 raw CSV 开始，并在对应 run 目录中执行：

1. 固定 seed 的 `0.64/0.16/0.20` train/validation/test 划分；
2. 分类任务分层划分；
3. 只在训练集拟合方差过滤、标准化和 mutual-information 排名；
4. 使用同一训练期变换处理 validation/test；
5. 保存 data manifest、选中特征和审计信息。

训练集内拟合无法识别 raw 表中语义上属于目标组成、GOAL 或全数据监督编码的字段。因此数据集进入正式协议前还必须完成人工字段角色审计和目标重构检查；CrimeData 正是因这一检查被隔离。

`data/processed` 与 `data/processed_v2` 是旧式全局缓存，不属于正式协议，也不应重新创建。

## 当前 raw 文件校验

当前文件的 SHA-256 记录在 `data/raw/README.md`。若重新下载得到不同字节，应先检查上游版本、列名和 target 映射；不要更新 YAML 来迁就不同快照，也不要把不同数据快照混入同一 campaign。正式协议登记的是原始行级快照，acquisition 阶段不执行去重。
