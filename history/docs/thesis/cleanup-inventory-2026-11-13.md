# 硕士工作树实验清理清单

2026-11-13 已按确认的 A+B+C 范围完成清理，位置为 `MastersThesis/` 的 `thesis/masters` 分支。首次盘点的提交为 `f806010`，执行时为 `8e5e52b`。下列均为相对硕士工作树的路径。容量按文件长度计算，1 MiB = 1,048,576 字节。

| 组 | 内容 | 文件数 | MiB |
|---|---|---:|---:|
| A | 会议执行入口、测试和演示 | 21 | 0.31 |
| A | 会议运行产物 | 1,768 | 365.33 |
| A | 会议测试临时输出 | 2,564 | 6.92 |
| B | 重复快照和三次早期 guarded 试跑 | 464 | 28.12 |
| C | PIv0 定名前的历史运行 | 909 | 33.83 |
| C | 早期硕士专用代码和测试 | 35 | 0.20 |
| C | 早期硕士测试临时输出 | 9,158 | 14.67 |

实际清除 212 个清单项目，包括 14,919 个文件，合计 471,209,422 字节（449.38 MiB）。A+B+C 同时包含了早期 12 组试验的原件和重复快照；这些试验现在可查阅历史文字与表格。以下按原清单保留路径，便于核对。

## 运行所需及用于论文比较的保留项

- PIv0 和 PIv1 两份六集五 seed 正式目录全留，包括直接 Gain、逐次预测、验证指标与计时。PIv1 的协议引用 PIv0 的直接 Gain 运行和 PIv1 pilot。两个正式路径分别是 `results/thesis_benchmark/PIv0-all-seeds-20261111-213840-398718/`、`results/thesis_benchmark/PIv1-all-seeds-20261112-003506-318341/`。
- PIv1 跟踪归档 `docs/thesis/archives/piv1/`、本地包 `packages/thesis-piv1-evidence.zip` 和标签 `thesis/piv1-frozen`。证据包的 SHA-256 与封存值一致；包中包含 90 个预处理分区文件，PIv3 的原始数据流程还会读取分区元数据。
- 全部 `data/raw/`、`configs/`、`experiments/preprocess_data/`、`src/` 以及共享环境入口。六个正式数据集沿用原配置，另一个原始数据集维持审计用途。三个会议正式分片的 `preprocess/`（包括划分元数据）、`data_manifest.csv`、`metadata_complete.json`、`resolved_protocol.yaml`、`environment.json` 和中央 `campaign_plan.json` 保留原路径。
- PIv1 的 Abalone 精确/采样对照读取各分片 `cache/selection_validation/abalone/s{seed}/coalitions.sqlite3`。保留所有 `cache/selection_validation/`。早期 seed 42 回放另会读取 `icomp2026-formal-v1-s42-123/estimator/` 与 `removal/`，两目录均保留。ISOLET 对照的 `ablation/isolet/s42/utility/paired/b10/scores.json` 也保留。
- 中央 `results/icomp2026_formal/icomp2026-formal-v1/tables/publication/` 整体保留。用其中 `raw_metrics.csv` 的 30 条会议记录复核 PIv0 为 13 胜、2 平、15 负；`aggregate_metrics.csv` 是方法比较文档引用的汇总表。保留 `results/thesis_benchmark/guarded-rounds-formal-20261110-205751-632446/` 整个目录；用其中 `summary.csv` 可复核 PIv0 对旧 guarded 的 11 组为 3 胜、2 平、6 负。
- 保留所有新增的 59 个 PI 实验脚本、源码和测试（盘点开始时未被 Git 跟踪），以及 `tmp/` 下 PIv2/PIv3 的辅助分析脚本、工作簿和生成素材。保留共享的 `experiments/calibrate_icomp2026_models.py`、`icomp2026_stages.py`、`run_thesis_sampling.py`、`thesis_prepared_catalog.py`、`thesis_sampling_worker.py` 等。PIv0 的试验直接使用旧数据读取函数，PIv3 的原始宽度入口直接使用旧目标编码函数。保留 `experiments/test/replay_online_relative_voting.py` 和 `thesis_madelon_alpha_data.py`，后者也是新版本的原始 Madelon 数据准备依赖。
- `conference-paper/`、会议投稿与复核包、`demos/cross_shapley_demo.py` 保留。本清单不涉及冻结工作树、共享虚拟环境和 Git 引用。

### 32 个 PI 试验、输入或对照目录

其中包含 8 个 PIv2 与 8 个 PIv3 目录，以及未在目录名开头写 PI 的采样排列对照、精确值、剪枝验证与 Madelon 100/500 列消融。`PIv3-six-raw-seed42-input` 保存六组全维数据输入。32 个目录合计 10,241 个文件，约 940.69 MiB。

```text
results/test/thesis_sampling/PIv0-seed42-20261111-173806-672616
results/test/thesis_sampling/PIv1-pilot-20261111-234033-780781
results/test/thesis_sampling/PIv2-light-prefilter-seed42-v1
results/test/thesis_sampling/PIv2-madelon100-validation-k-seed42
results/test/thesis_sampling/PIv2-madelon500-seed42
results/test/thesis_sampling/PIv2-madelon500-validation-k-seed42
results/test/thesis_sampling/PIv2-reference-tolerance-madelon500-20261113-171647-837174
results/test/thesis_sampling/PIv2-seed42-all-methods-summary
results/test/thesis_sampling/PIv2-seed42-expanded-all-six
results/test/thesis_sampling/PIv2-seed42-summary-to-date
results/test/thesis_sampling/PIv3-compact-raw-six-seed42
results/test/thesis_sampling/PIv3-compact-witness-raw-six-seed42
results/test/thesis_sampling/PIv3-frontend-madelon500-seed42-3-5-8
results/test/thesis_sampling/PIv3-lite-peer-test-raw-six-seed42
results/test/thesis_sampling/PIv3-lite-raw-six-seed42
results/test/thesis_sampling/PIv3-madelon500-adaptive-seed42
results/test/thesis_sampling/PIv3-raw-six-seed42-tol358
results/test/thesis_sampling/PIv3-six-raw-seed42-input
results/test/thesis_sampling/abalone-piv1-exact-mc32-20261113-113752-644906
results/test/thesis_sampling/adaptive-pruning-validation-20261113-175433-248
results/test/thesis_sampling/madelon-all-sizes-20261111-162002-353446
results/test/thesis_sampling/madelon-fixed13-widths-20261111-160627-204395
results/test/thesis_sampling/madelon-fixed13-widths-20261111-161028-408176
results/test/thesis_sampling/madelon-s42-binary-widths-20261111-160617-502663
results/test/thesis_sampling/piv0-fixed-k-controls-20261111-231112-839839
results/test/thesis_sampling/sampled-permutation-comparison-20261113-114110-846514
results/test/thesis_sampling/sampled-permutation-seed1024-20261113-114003-572256
results/test/thesis_sampling/sampled-permutation-seed123-20261113-113759-515968
results/test/thesis_sampling/sampled-permutation-seed42-20261113-113543-094228
results/test/thesis_sampling/sampled-permutation-seed42-20261113-113612-912266
results/test/thesis_sampling/sampled-permutation-seed456-20261113-113841-914284
results/test/thesis_sampling/sampled-permutation-seed789-20261113-113922-555460
```

## A　会议专用项目

### 10 个执行或汇总脚本、9 个测试和 2 个教学演示

```text
experiments/benchmark_icomp2026_backends.py
experiments/build_leakage_screened_publication.py
experiments/forecast_icomp2026_48h.py
experiments/monitor_icomp2026_campaign.py
experiments/run_icomp2026_campaign.py
experiments/run_icomp2026_group1.py
experiments/run_isolet_utility_pair.py
experiments/screen_icomp2026_results.py
experiments/summarize_multifidelity_campaign.py
experiments/validate_grouped_acceleration.py
tests/test_campaign_monitor_groups.py
tests/test_group1_runner.py
tests/test_icomp2026_backend_benchmark.py
tests/test_icomp2026_screening.py
tests/test_isolet_utility_pair_scope.py
tests/test_leakage_screened_publication.py
tests/test_multifidelity_campaign.py
tests/test_validate_grouped_acceleration.py
tests/test_sync_figures.py
demos/algorithm_b_demo.py
demos/minppi_demo.py
```

已核查其余实验与测试的导入。与数据读取、校准、建模、论文比较共用的旧模块和相关测试保留。硕士 README 中的会议说明已指向冻结参考版。

### 24 个会议结果目录

其中 4 个预检、失败或试跑目录占 349.10 MiB。其余 20 个目录限于中央会议日志及指定正式分片中的旧输出。三个分片的 `figures/` 为空；中央 `figures/` 保留，用于历史图包的校验。这些目录中现存的 1,768 个文件在冻结 `Master/` 中全部有相同相对路径、相同 SHA-256 的独立副本。

```text
results/icomp2026_formal/bootstrap-s42-v1
results/icomp2026_formal/smoke-s42-v1
results/icomp2026_preflight_only
results/failed_icomp2026_formal_20260801-1715_raw_snapshot_mismatch
results/icomp2026_formal/icomp2026-formal-v1/logs
results/icomp2026_formal/icomp2026-formal-v1-s42-123/raw_metrics
results/icomp2026_formal/icomp2026-formal-v1-s42-123/nsga2
results/icomp2026_formal/icomp2026-formal-v1-s42-123/figures
results/icomp2026_formal/icomp2026-formal-v1-s42-123/cache/baseline_internal
results/icomp2026_formal/icomp2026-formal-v1-s42-123/cache/final_test
results/icomp2026_formal/icomp2026-formal-v1-s456-789/raw_metrics
results/icomp2026_formal/icomp2026-formal-v1-s456-789/nsga2
results/icomp2026_formal/icomp2026-formal-v1-s456-789/figures
results/icomp2026_formal/icomp2026-formal-v1-s456-789/cache/baseline_internal
results/icomp2026_formal/icomp2026-formal-v1-s456-789/cache/final_test
results/icomp2026_formal/icomp2026-formal-v1-s456-789/estimator
results/icomp2026_formal/icomp2026-formal-v1-s456-789/removal
results/icomp2026_formal/icomp2026-formal-v1-s1024/raw_metrics
results/icomp2026_formal/icomp2026-formal-v1-s1024/nsga2
results/icomp2026_formal/icomp2026-formal-v1-s1024/figures
results/icomp2026_formal/icomp2026-formal-v1-s1024/cache/baseline_internal
results/icomp2026_formal/icomp2026-formal-v1-s1024/cache/final_test
results/icomp2026_formal/icomp2026-formal-v1-s1024/estimator
results/icomp2026_formal/icomp2026-formal-v1-s1024/removal
```

### 11 个会议测试的临时输出目录

逐个检查了目录中的测试名，均能对应到会议基线提交 `9086ddc` 的测试。这里列的是临时测试输出，包含最小化计算、配对消融与会议流程测试产生的样本文件。

```text
.t-check-fast
.t-fv2-check
.t-group-verify
.t-nest-final
.t-nest-publish
.t-nest-v2
.tqareplace
tmp/minimum-final-b0f3228b5b454d55bd9672d5f5f66f71
tmp/minimum-tests-41b8b61e2df64cb999f3d5781532fa58
tmp/minpair_f2e6d271
tmp/paired-minimum-tests-c958de67ab7f49cb9c677e5c5ec3a84a
```

## B　重复快照和三个旧硕士试跑

```text
results/thesis_sampling
results/test/thesis_benchmark/guarded-rounds-20261110-203136-404894
results/test/thesis_benchmark/test-guarded-rounds-20261110-205227-427571
results/thesis_benchmark/guarded-rounds-formal-20261110-204444-825720
```

第一项原有 12 个运行子目录、418 个文件，删除前与 `results/test/thesis_sampling/` 中的副本逐文件 SHA-256 一致（27.27 MiB）。这两套记录分别按 B、C 清理。另三项合计 46 个文件，均是正式 guarded 对照完成前的试跑或中断记录。

## C　PIv0 定名前的独立硕士试验

### 41 个运行目录

12 个是 B 中早期 seed 42 目录的另一份，29 个是后续 alpha、采样、投票或两阶段筛选的前期探索。这些历史结果被 Git 忽略，删除后部分比较仅存文字记录。与 PIv0 同次提交引入的 Madelon 100/500 列消融保留。经检查保留的 PI 入口、运行协议及 PIv1 证据包索引，未发现对以下 41 个路径的运行依赖。

```text
results/test/thesis_sampling/abalone-stratified-exact-20261111-134548-337267
results/test/thesis_sampling/abalone-witness-design-20261111-012433-971008
results/test/thesis_sampling/binary-hard-then-gain-20261111-153403-927107
results/test/thesis_sampling/binary-hard-then-gain-extension-20261111-153700-791073
results/test/thesis_sampling/binary-negative-replay-20261111-152949-894864
results/test/thesis_sampling/binary-nomination-20261111-152743-057454
results/test/thesis_sampling/hybrid-C-gain-exact-pruning-20261111-145342-768139
results/test/thesis_sampling/hybrid-class-subsets-20261111-144839-251033
results/test/thesis_sampling/madelon-contribution-vote-20261111-134723-075837
results/test/thesis_sampling/madelon-halfheldout-exchange-20261111-140125-788619
results/test/thesis_sampling/madelon-multianchor128-20261111-133237-157945
results/test/thesis_sampling/madelon-relative-votes-20261111-135137-410847
results/test/thesis_sampling/madelon-s123-sampling128-20261111-132852-442843
results/test/thesis_sampling/madelon-s42-20261110-160113-546839
results/test/thesis_sampling/madelon-s42-alpha095-k2-20261111-003923-100995
results/test/thesis_sampling/madelon-s42-alpha099-k2-20261111-002123-979739
results/test/thesis_sampling/madelon-s42-boundary128-20261111-012733
results/test/thesis_sampling/madelon-s42-boundary128-gate-20261111-013100
results/test/thesis_sampling/madelon-s42-cold-controls-20261111-012618-705975
results/test/thesis_sampling/madelon-s42-independent128-clock-20261111-132608-337652
results/test/thesis_sampling/madelon-s42-zero-review-20261111-132827-567300
results/test/thesis_sampling/madelon-spambase-exact-pruning-20261111-135742-554359
results/test/thesis_sampling/online-relative-evidence-20261111-144106-439953
results/test/thesis_sampling/online-relative-replay-20261111-145534-154460
results/test/thesis_sampling/race-class-subsets-20261111-144408-043834
results/test/thesis_sampling/relative-vote-20261111-140347-136542
results/test/thesis_sampling/seed42-20261110-163831-426000
results/test/thesis_sampling/seed42-confirmed-20261110-191743-946395
results/test/thesis_sampling/seed42-fold_consensus-20261110-194334-355192
results/test/thesis_sampling/seed42-guarded-20261110-192134-444036
results/test/thesis_sampling/seed42-local_refine-20261110-193726-790162
results/test/thesis_sampling/seed42-predictor-20261110-194615-238818
results/test/thesis_sampling/seed42-predictor-rounds-20261110-194923-538963
results/test/thesis_sampling/seed42-sign_aware-20261110-191152-485309
results/test/thesis_sampling/seed42-structured-20261110-174333-085284
results/test/thesis_sampling/seed42-training-check-rounds-20261110-195502-056614
results/test/thesis_sampling/seed42-voting-ablation-replay-20261111-005644-068289
results/test/thesis_sampling/seed42-wide-20261110-173236-268848
results/test/thesis_sampling/spambase-relative-votes-20261111-135503-499433
results/test/thesis_sampling/top1-calibrated-majority-e-20261111-144820-286110
results/test/thesis_sampling/top1-majority-race-20261111-144234-016850
```

### 35 个早期独立脚本或对应测试

```text
experiments/compare_fixed_predictor.py
experiments/compare_local_refinement.py
experiments/compare_predictor_replication.py
experiments/compare_thesis_sign_aware.py
experiments/replay_thesis_guarded.py
experiments/replicate_fixed_rounds.py
experiments/run_thesis_benchmark.py
experiments/run_thesis_predictor.py
experiments/thesis_benchmark_summary.py
experiments/thesis_k_summary.py
experiments/thesis_nested_k_worker.py
experiments/thesis_sampling_pilot.py
experiments/test/compare_fixed_predictor.py
experiments/test/compare_local_refinement.py
experiments/test/compare_predictor_replication.py
experiments/test/compare_thesis_sign_aware.py
experiments/test/replay_thesis_guarded.py
experiments/test/replicate_fixed_rounds.py
experiments/test/run_madelon_alpha.py
experiments/test/run_relative_voting.py
experiments/test/run_thesis_benchmark.py
experiments/test/run_thesis_predictor.py
experiments/test/run_thesis_sampling.py
experiments/test/thesis_madelon_alpha_noise.py
experiments/test/thesis_madelon_alpha_worker.py
experiments/test/thesis_sampling_pilot.py
tests/test_compare_fixed_predictor.py
tests/test_compare_local_refinement.py
tests/test_compare_thesis_sign_aware.py
tests/test_predictor_budget.py
tests/test_replicate_fixed_rounds.py
tests/test_run_thesis_benchmark.py
tests/test_thesis_benchmark_summary.py
tests/test_thesis_k_summary.py
tests/test_thesis_nested_k_worker.py
```

### 76 个早期硕士测试临时目录

```text
.t-thesis-blocking-log-green
.t-thesis-blocking-log-red
.t-thesis-deadline-green
.tqa610b
.twiderun8
tmp/bench-runner-verify2
tmp/bench-sched-tests
tmp/catalog-audit-green
tmp/checkpoints-ready
tmp/complete-v-bounded-preflight
tmp/confirmed-official-worker-check
tmp/confirmed-supervisor-post
tmp/confirmed-supervisor-pre
tmp/fixed-predict-audit-both
tmp/fixed-predict-audit-both-again
tmp/fixed-predictor-tests
tmp/fold-before-run-final
tmp/fold-cli-before
tmp/formal_evidence_green
tmp/formal_evidence_red
tmp/formal_final_stage_green
tmp/formal_final_stage_red
tmp/formal_mode_green
tmp/guarded-cli-before
tmp/guarded-real-run-qa
tmp/guarded-rounds-effects-test
tmp/guarded-rounds-worker-test
tmp/inner-rounds-ready
tmp/inventory-complete
tmp/inventory-green
tmp/k-aggregate-test
tmp/k-final-report
tmp/k-prefix-worker-tests
tmp/k-report-green
tmp/k-report-qa
tmp/local-protocol-ready
tmp/monitor-retry-regression
tmp/nested-default-red
tmp/nested-k-integration
tmp/nestedworker-green
tmp/nestedworker-parity-final
tmp/predictor-preflight
tmp/progress-fix-all-callers
tmp/pytest-comparison-confirm-191800
tmp/refine-preflight-green
tmp/refine-preflight-red
tmp/rounds-before
tmp/rounds-ready
tmp/sign-v3-integration-root
tmp/summary-interrupted-red
tmp/supervisor-green-v3
tmp/supervisor-red-v3
tmp/synthetic-alpha-worker-993b2fc919814ea2823c68063cabb597
tmp/test-archive-modules
tmp/test-contention-before
tmp/test-contention-fixed
tmp/test-contention-red2
tmp/test_mode_red
tmp/thesis-summary-tdd
tmp/thesis-unlimited-worker
tmp/trainonly-local
tmp/unlimited-before
tmp/worker-consensus-before
tmp/worker-consensus-ready
tmp/worker-integration-20261110a
tmp/worker-local-final
tmp/worker-local-red
tmp/worker-observability-20261110b
tmp/worker-structured-no-shap-final
tmp/worker-structured-only-green
tmp/worker-structured-only-red
tmp/worker-structured-rescue-frozen
tmp/worker-wide-green-20261110
tmp/worker-wide-red-20261110
tmp/worker-wide-shap-20261110
tmp/worker-wide-union-20261110
```

## 清理后核验

- 32 个 PI 试验目录及两份正式目录中的 11,513 个文件，路径和 SHA-256 与清理前一致。PIv0 和 PIv1 各自保留 30 组与直接 Gain 的配对结果。
- 六个正式数据集、五个 seed 的 90 个训练、验证、测试分区均与冻结清单一致。PIv3 的 54 个全维输入数组、Abalone 五个联盟缓存及 PIv1 封存证据包已核验。
- 盘点时的 59 个新增 PI 实验、源码及测试文件已在 `8e5e52b` 中受版本控制，清理前后其内容一致。
- 11 份 PIv0–PIv3 测试文件和数据目录测试共 58 项，全部通过。文档列出的 7 个实验及汇总入口均能从该工作树导入。历史训练结果沿用原始运行记录。

硕士 README、`experiments/test/README.md` 和早期研究说明已同步更新，保留历史表格并注明原始记录的可用性。
