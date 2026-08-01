# Formal ICOMP 2026 configuration

`configs/` 是正式静态协议的唯一真源。

- `icomp2026_pipeline.yaml`：seeds、backend 默认值、模型、alpha、估计预算、removal、NSGA-II、消融和 stage scopes。
- 七个 dataset YAML：数据身份、raw path、受版本控制的原始 CSV SHA-256/字节数、target、task、utility、训练集预处理和 split。

严格加载器会拒绝未知字段、缺失字段、重复 seeds、非法路径和不完整的协议；validate 会在写出分片 authority 前核对原始 CSV 的 SHA-256 与字节数。CLI override 在新 run 创建前解析，随后完整配置、model policy、source fingerprint 和稳定的科学环境摘要被冻结；resume 时源代码、协议或科学环境漂移都会被拒绝。

## 冻结事实

- 数据集：`spambase`, `madelon`, `isolet`, `gisette`, `credit_card_fraud`, `crimedata`, `abalone`。
- canonical seeds：`[42, 123, 456, 789, 1024]`。
- split：每个数据集均为 train `0.64`、validation `0.16`、test `0.20`。
- 最大特征数：100；方差过滤与 MI 排序只在训练集拟合。
- 分类模型统一产生概率；效用由 dataset config 决定。
- ISOLET 正式效用为 `ovr_auc`；`accuracy` 只属于 utility ablation。
- 默认 alpha：`0.95`；grid：`[0.85, 0.90, 0.95, 0.99]`。
- alpha scope：七个数据集、seed 42。
- ablation scope：Madelon、ISOLET、Credit-Card-Fraud、Abalone，seed 42。
- `backend: cpu` 是便携默认值；正式 campaign 入口显式使用 CUDA。

## Smoke

`--smoke` 将范围投影到 Spambase、Abalone 和 seed 42，并使用低预算。它会冻结 `run_kind: smoke`；其 artifacts 不能改名或复制为 formal artifacts。

## 非正式配置

`../configs_credit_probe/` 是单数据集方向性探测材料，不满足正式七数据集 schema，也不能提供 formal model policy。正式 runner 一律使用本目录。
