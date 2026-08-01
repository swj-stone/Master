# ICOMP 2026 experiment executors

本目录中的 manifest 管线是新 ICOMP 2026 证据的唯一执行入口。所有命令从项目根目录 `Master` 运行；完整可复制命令见根目录 `README.md`。

## 入口

- `run_icomp2026_pipeline.py`：执行一个 shard。新 run 创建冻结协议；恢复时必须显式传入 `--resume`。
- `run_icomp2026_campaign.py`：把 canonical 五个 seeds 编排为三个 shard，并在 campaign 级别统一汇总。已有 shard 会自动恢复。
- `calibrate_icomp2026_models.py`：只使用训练集和验证集，为七个数据集生成冻结 model policy。
- `screen_icomp2026_results.py`：在正式 estimator 后读取 seed 42 checkpoint，生成 validation-only 方向筛查；它不是论文结果，也不访问正式 test。
- `monitor_icomp2026_campaign.py`：只读状态监测；`--group initial` 与 `--group downstream` 分别查看两段实验，省略 `--watch` 时只输出一次；`--watch --interval-seconds 300` 每五分钟刷新并即时写出快照。
- `summarize_multifidelity_campaign.py`：校验 shard 并集、协议和 provenance 后生成唯一正式汇总。
- `make_paper_figures.py`：只从 canonical summary tables 生成三张论文图；正式 Overview 图还会用 raw canonical table 验证 Madelon/Credit-Card-Fraud 的完整 seed-42 alpha authority，smoke 只显示目标面板不在其 scope 内的占位说明。

## 阶段顺序

```text
validate -> preprocess -> estimator -> removal -> alpha -> nsga2 -> ablation -> summarize -> figures
```

- `alpha` 和 `ablation` 的实际数据集/seed 范围由 `configs/icomp2026_pipeline.yaml::stage_scopes` 决定。
- `removal`、`alpha` 和 `nsga2` 会访问 held-out test；campaign 必须显式带 `--allow-test-access`。
- 新 run 不能直接从 `estimator` 开始。应先顺序完成 `validate` 和 `preprocess`，或在 smoke 中使用 `--stage all`。
- Smoke 与 formal 必须使用不同 run ID；smoke 的表格、宏和图片不能进入论文。

## 单 GPU 并发边界

正式 campaign 当前固定使用 CUDA，并支持最多三个内部 shard worker。`--max-parallel 3` 是当前两天目标配置，不是未实测的完成时长保证；并发只应由一个 campaign 进程管理。运行 campaign 时不得再手工启动第二个 campaign、pipeline、校准或其他 GPU 模型任务。

若 smoke 暴露显存不足，应停止并为一个全新的 campaign 选择较小的 `--max-parallel`；不要在同一 campaign 运行中途改变执行设置。

`native_process_guard.py` 隔离 pipeline worker 的解释器生命周期并验证完成 receipt；它不等价于显存调度器。

## 恢复语义

- Pipeline：复用原命令并增加 `--resume`；run ID、源代码、配置和 model policy 必须完全一致。
- Campaign：原样重输同一 campaign 命令；编排器对已存在 shard 自动使用 resume。
- 成功 job 只有在 canonical record 和声明的 artifacts 均通过校验时才会跳过。
- 不要删除、覆盖或复制其他 run 的 cache。源文件或协议漂移会 fail closed。

## 输出契约

每个 run 保存：

- 冻结的 resolved protocol、稳定科学环境摘要、source fingerprint 与注册 raw authority；
- `run_manifest.csv`、根状态和阶段状态；
- canonical `raw_metrics/` records；
- test-access audit 和 protocol audit；
- 可恢复的 estimator bundle/cache/progress；
- summary tables、LaTeX macros、bundle manifest 和 figures。

正式发布前，campaign summary 必须验证五种子范围、跨 shard 的 raw/environment/source/model-policy 摘要和全部上游 artifacts，并明确写出 `publication_allowed: true`。当前 `results/` 为空，因此仓库内没有已完成的正式 campaign。

## 辅助脚本

Benchmark、forecast、screening、加速验证、ISOLET utility pair 和 `preprocess_data/` 是诊断或数据获取工具。它们可以帮助选择正式设置，但不能单独产生论文结果，也不能替代 manifest 管线。
