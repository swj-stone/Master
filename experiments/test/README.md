# PI 方案试验

以下 PowerShell 示例均从 `Paper` 目录执行。实验输入使用各自协议冻结的数据和 seed。新运行的日志与结果写入硕士工作树的 `results/test/thesis_sampling/`。

## 运行示例

**PIv0** 在 Madelon 的预处理输入上比较负向投票后筛选与直接 Gain。两臂独立计时，各有 900 秒上限。

```powershell
.\MastersThesis\thesis-python.ps1 experiments/test/run_piv0_seed42.py --datasets madelon
```

**PIv2** 在 Madelon、seed 42 上对同一 Gain 后端比较有无 PI 预筛。删去 `--datasets madelon` 可覆盖六个预处理数据集。

```powershell
.\MastersThesis\thesis-python.ps1 experiments/test/run_piv2_prefilter.py --datasets madelon --methods gain
```

**PIv3** 在 Madelon 原始宽度输入上运行多后端配对。显式指定新目录，已有阶段性报告可继续单独查看。数据来自 `PIv3-six-raw-seed42-input/` 中保存的原始宽度划分。

```powershell
$piv3Run = 'results/test/thesis_sampling/PIv3-madelon-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
.\MastersThesis\thesis-python.ps1 experiments/test/run_piv3_raw_six.py --run $piv3Run --datasets madelon
```

## 其他保留试验

| 入口 | 用途 |
| --- | --- |
| [run_madelon_binary_widths.py](run_madelon_binary_widths.py)、[run_madelon_all_sizes.py](run_madelon_all_sizes.py) | Madelon、seed 42 的 100 列和 500 列输入，比较负向投票后 Gain 与直接 Gain。全列数试验沿每种排序逐个验证可用的保留列数。 |
| [run_madelon_fixed13.py](run_madelon_fixed13.py) | 传入 `--primary` 及宽度对照的四臂运行目录，在各自冻结的排序上取前 13 列，比较 AUC 和原 Top-100 之外的入选列。 |
| [compare_piv0_fixed_k.py](compare_piv0_fixed_k.py) | 在与 PIv0 相同的保留列数下比较 Gain 和 TreeSHAP。 |
| [run_piv1_pilot.py](run_piv1_pilot.py) | 用 `develop` 开发候选方案，再用 `final --run <开发运行目录>` 锁定并比较 PIv1 与等预算对照。[开发实验记录](../../docs/thesis/test/piv1-pilot.md) |
| [run_sampled_permutation_seed42.py](run_sampled_permutation_seed42.py)、[compare_sampled_permutation.py](compare_sampled_permutation.py) | PIv1 的轻量采样排列重要性（PFI）基线与配对汇总。 |
| [run_madelon500_piv2.py](run_madelon500_piv2.py) | PIv2 在 Madelon 原始 500 列上的对照。 |
| [run_piv3_lite_six.py](run_piv3_lite_six.py)、[run_piv3_lite_peers.py](run_piv3_lite_peers.py) | 六集原始宽度的 PIv3-Lite 及其横向对照。 |
| [run_compact_minpi.py](run_compact_minpi.py)、[run_compact_witness.py](run_compact_witness.py) | PIv3 的紧凑子集与负向见证筛选试验。 |

[PIv0 定义与 seed 42 对照](../../docs/thesis/test/piv0-seed42.md) · [PIv2 按方法汇总](../../results/test/thesis_sampling/PIv2-seed42-all-methods-summary/comparison.md) · [PIv3 原始宽度阶段性配对](../../results/test/thesis_sampling/PIv3-raw-six-seed42-tol358/paired.md)

## 历史记录

PIv0 定名前的 12 组旧单 seed 快照、三次早期 guarded 试跑和清单所列的其他前期运行已清理。早期采样及稳定性试验的文字说明仍可查阅。PIv0 正式历史比较所用的 11 组 guarded 对照和会议出版表另有保留。[早期采样说明](../../docs/thesis/test/sampling-methods.md) · [历史比较与来源](../../docs/thesis/experiments/piv0-method-comparison.md) · [清理清单](../../docs/thesis/cleanup-inventory-2026-11-13.md)

六集、五 seed 的 PIv0、PIv1 正式实验及实时日志命令，见[硕士实验说明](../../README.md)。
