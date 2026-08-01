# Data contract

正式 ICOMP 管线只读取 `data/raw/*.csv`。数据文件被 Git 忽略；dataset YAML 中的 `raw_path`、target、task、utility 和 split 是权威定义。

## 获取 raw cache

在项目根目录安装环境后运行：

```powershell
$PY = (Resolve-Path "../venv/Scripts/python.exe").Path
Get-ChildItem experiments\preprocess_data\preprocess_*.py |
  Sort-Object Name |
  ForEach-Object { & $PY $_.FullName; if ($LASTEXITCODE -ne 0) { throw "data acquisition failed: $($_.Name)" } }
```

脚本从 UCI、OpenML 或脚本中声明的公开镜像下载数据，并只写相应的 `data/raw/<dataset>.csv`。已有文件会作为本地 cache 使用。

## 防止数据泄漏

这些 acquisition 脚本不负责正式 split、标准化或特征选择。每次 run 的 `preprocess` 阶段从 raw CSV 开始，并在对应 run 目录中执行：

1. 固定 seed 的 `0.64/0.16/0.20` train/validation/test 划分；
2. 分类任务分层划分；
3. 只在训练集拟合方差过滤、标准化和 mutual-information 排名；
4. 使用同一训练期变换处理 validation/test；
5. 保存 data manifest、选中特征和审计信息。

`data/processed` 与 `data/processed_v2` 是旧式全局缓存，不属于正式协议，也不应重新创建。

## 当前 raw 文件校验

当前文件的 SHA-256 记录在 `data/raw/README.md`。若重新下载得到不同字节，应先检查上游版本、列名、target 映射和去重规则；不要把不同数据快照混入同一 campaign。
