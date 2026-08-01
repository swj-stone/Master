# Raw data snapshot

以下校验和描述 2026-08-01 当前本地 raw snapshot。CSV 本身被 Git 忽略。

```text
81c45a0cf2d6244fc158cc18ccddf7f133509fe9820cfa85514699d83c006e40  abalone.csv
5ac0db239c79b8234914ac04242b5a864cd34cd9c4e73526ed3808185285e335  credit_card_fraud.csv
6fb4c4f378a5e2ea98eb3bfe9effb7d750ef888e5975db10fa174b30ef8edcab  crimedata.csv
02b8d81c00ed14e65310e9c57ec8ec34deaed237eb04117ba545c41ceef4ed6a  gisette.csv
7113075d4802471d595540ea5161978fd76cd892b58d96ce26d535e837dda200  isolet.csv
ac9cb79fe194a9032496d92160a679bf2b2dedf6a0d946962116864abdb77357  madelon.csv
9c4af29aa3b36519a8686fb9dc9b8dc0784d9686079c2945efeb298829d727e1  spambase.csv
```

从项目根目录复核：

```powershell
Get-FileHash -Algorithm SHA256 data\raw\*.csv | Sort-Object Path
```

数据来源由 `experiments/preprocess_data/preprocess_*.py` 的模块说明和下载逻辑记录；正式协议通过 run-level data manifest 固定实际输入。
