# ICOMP 2026：英文作者回复草稿

## 0. 使用说明：尚未提交

- 用户已确认这是正式评审，且本项目 [当前最终论文 PDF](conference-paper/paper/icomp2026_conference.pdf) 就是送审版本。本草稿已按该 PDF 的摘要、方法、表格、图注与结论核对，不再把 TeX/PDF 版本差异当作未解问题。
- PDF：11页，正文至第9页，参考文献从第9页开始，附录第11页；SHA256 `6c96629857271ae4bb572fc7add0f220412a68b51a80c54ede9f3480f4765ec3`。
- 当前状态：只有回复草稿和大修设计，尚未更改送审论文、算法或产生新实验。下面统一用将来时描述拟修订内容，不能改成“已经完成”后提交。
- 用户提供回复字段 `max length: 2500`，尚未注明是字符还是单词。保守按2500字符（含空格和换行）制作推荐粘贴版；是否可替换稿件/添加附件/补充新实验仍未确认。各版本互为替代，不应合并提交。
- 当前英文正文按空白分词统计：完整稿约726词（不含分项标题），短版约200词。平台计数可能不同，不能据此假定满足未知上限。
- 官网列本届作者回复窗口为9月7–14日；本文件不推断具体截止时区或权限，不自动提交。[ICOMP 官网](https://www.icomp.cc/)

## 推荐粘贴版本：按 2500 字符上限

只复制下面代码框内的英文，不复制标题、计数说明或其他版本。正文为2313字符（LF换行）；即使按Windows CRLF换行计数也仅2325字符，包含空格与段落换行，低于2500。此版本沿用现有证据，不声称新实验或修订已完成。

```text
Thank you for the review. We will distinguish witness validity, discovery completeness, and deletion value.

1. Practical value. MinPPI identifies an evaluated context where adding a feature changes utility from meeting to failing a fixed quota. This does not guarantee beneficial joint deletion. Nonzero deletion occurred in 14/30 dataset-seed units, and the pooled HV gain was not significant after correction. We will narrow the claims to a contextual audit with a separately validated removal decision, not a generally superior selector.

2. Algorithm B. We agree that B has lower five-seed mean F1 than A at every reported budget on Abalone (K=25: 0.313 vs 0.443). Section 2.3 and Figure 2 identify the oracle-conditioned example, but its prominence overstates its empirical role. We will separate it from real-data evidence and present B as a heuristic ablation. The detection bound is conditional on witness probability; it does not establish an advantage over A.

3. Ranking and abstract. Rank 4.08 summarizes ordering across datasets; the Madelon result is a local effect size. The abstract names Madelon, but should connect this result to the overall comparison. Table 3 reports utility 0.7923 for MinPPI, 0.7144 for random removal, and 0.7985 for SHAP on Madelon. We will give the stronger SHAP result equal prominence and avoid interpreting the random comparison as general superiority.

4. Exact validation. One real-data reference is insufficient to characterize estimator reliability. The hybrid's perfect Abalone recovery comes from exact enumeration. We propose additional prespecified small-dimensional real-data games, with training-only panel construction, common total query budgets, and empty-harmful-set cases retained. Their truth applies only to each defined game, not the original high-dimensional problem. These experiments are proposed, not completed, and do not yet close the evidence gap.

5. Reproducibility. We will add implementation-checked numbered pseudocode for the audit, A and B, plus the gate rule, specifying proposals, stopping, budgets, and witnesses. Logical queries, model fits, and cache-dependent runtime will be distinguished.

Minimum aggregation builds on prior work; observed negative crossings certify a fixed-game witness, not population harm or safe deletion.
```

## 1. Main response — 完整英文草稿

We thank the reviewer for identifying the gap between the formal meaning of MinPPI and the evidence for practical selection benefits. We will separate witness validity, discovery completeness, and deletion value throughout. The proposed revisions below are not completed experiments; the empirical evidence discussed here is from the submitted paper.

### 1. Practical value and dependence on Madelon

We agree that the current results do not establish a generally superior selector. Nonzero deletion occurred in only 14 of 30 dataset–seed units, and the average hypervolume change was not significant after correction. We will treat multi-objective prefiltering as a secondary application rather than evidence of broad optimization gains.

The narrower contribution is a checkable statement about a specified empirical game: adding a feature to an evaluated coalition changes its utility from meeting to failing a fixed quota. Minimum aggregation is inherited from prior work; our specialization concerns the semantics and discovery of these threshold crossings. A witness establishes neither population-level harm nor beneficial joint deletion. The practitioner-facing output is therefore a coalition and its two evaluated utilities, not an unconditional instruction to discard a feature. Deletion requires a separate validation decision and held-out assessment. The present results do not establish a dataset characteristic that predicts when the resulting compression will be worthwhile; we will remove wording suggesting otherwise.

### 2. Algorithm B and the oracle illustration

The reviewer is correct about the empirical comparison. In Section 4.4, B has lower five-seed mean harmful-label F1 than A at every reported budget on Abalone; at K=25 the values are 0.313 and 0.443. This concerns means, not every individual seed. We will present B as a heuristic ablation, not a demonstrated algorithmic advantage.

Section 2.3 and the Figure 2 caption identify the oracle-conditioned seed and distinguish this illustration from learned guidance. Nevertheless, placing that curve in the main overview gives it excessive evidential weight. We will separate it from the empirical comparison and move it to theoretical supplementary material, if permitted. Its detection bound is conditional on positive witness probability; it does not show that B assigns more probability than A. The main empirical message will instead be the observed A/B comparison and the distinction between certified discoveries and potentially missed witnesses.

### 3. Overall ranking versus the highlighted removal result

The submitted abstract singles out Madelon, but we accept that its emphasis is insufficiently connected to the overall ranking. Mean rank 4.08 summarizes ordering across the six datasets; the Madelon comparison measures a local effect size. A large local gain over random removal does not establish a good overall rank or superiority over established ranking methods.

Table 3 makes the stronger comparison available: on Madelon, matched-count held-out utility is 0.7923 for MinPPI, 0.7144 for random removal, and 0.7985 for SHAP. We will give this SHAP result equal prominence and align the abstract with the complete comparison. We will also distinguish zero deletion from successful compression and avoid interpreting an average across AUC and R² as a common non-inferiority guarantee. Adapted minimum- and mean-marginal controls will be identified explicitly rather than presented as complete reproductions of the cited methods.

### 4. Insufficient exact-reference validation

We agree that one real-data reference is insufficient to establish estimator reliability. Higher-dimensional harmful-label counts are discoveries, not recall estimates. The hybrid dispatcher's perfect recovery on Abalone comes from its exact branch and is not evidence for approximation quality.

The necessary extension is evaluation on prespecified small-dimensional real-data panels and controlled games. Panel construction must use training data without inspecting exact harmful labels; each panel must define its own game, without implying exactness for the original high-dimensional problem. We propose comparing recall and discovery cost under a common total marginal-query budget, retaining empty-harmful-set games, and separating sampling repeats within a frozen game from data-split variation. These evaluations remain proposed, subject to the revision policy. We acknowledge that a future experiment plan does not close the current evidence gap; until results are available, estimator conclusions must remain limited to the existing reference.

### 5. Missing numbered pseudocode

We will add numbered pseudocode for the audit procedure, A, and B, with an explicit validation-gate rule, and check it against the implementation. It will specify utility and quota, proposal distributions, initialization, stopping, budgets, and returned labels and witness records. Exact enumeration and approximate dispatcher branches will be distinguished, including combined budgets for multiple sampling routes.

The gate will be described as a separate decision about joint removal, not a consequence of individual witnesses. We will also distinguish logical queries, actual model fits, and cache-dependent runtime. This addresses reproducibility without implying an unsupported speed advantage.

## 2. Short response — 篇幅很紧时的替代稿

We thank the reviewer for distinguishing a meaningful index from demonstrated selection value. We will separate witness validity, discovery completeness, and deletion utility. Minimum aggregation is inherited from prior work; our specialization concerns negative threshold crossings in a fixed empirical game, not population harm or guaranteed safe deletion.

We agree that B is not supported as superior: its five-seed mean F1 is lower at every reported budget on the sole exact real-data reference. The oracle curve assumes a known witness. Although identified as illustrative, its prominence is misleading; we will separate it from empirical evidence and treat B as a heuristic ablation.

The abstract singles out Madelon, but the overall rank and stronger SHAP result must receive equal prominence. A large advantage over random removal does not establish general selection superiority.

Additional exact small-game evaluations are needed, with prespecified training-only panels, common query budgets, and empty-harmful-set cases retained. These are proposed evaluations, not completed evidence. We will add implementation-checked numbered pseudocode and clarify the gate, dispatcher, and cost accounting.

Practically, MinPPI supplies an evaluated context that fails a quota when a feature is added; whether acting on that witness is worthwhile remains a separate decision requiring validation and held-out assessment.

## 3. 给作者的逐项证据核对：不要粘贴为正式回复

| 回复点 | 已在送审 PDF 核对的位置 | 写作边界 |
| --- | --- | --- |
| 14/30 发生非零删除 | 第4页 §4.1；第5页 Table 2 | 27/30 raw pass 含13个空候选，不能说27次有效压缩 |
| B 在所有 K 的均值低于 A | 第8页 §4.4 | K25：B .313，A .443；不写每个 seed 均如此 |
| oracle seed 与条件界 | 第3页 §2.3、式(5)(6)；第7页 Figure 2(b)及图注 | 原稿已经披露条件；需要调整视觉叙事，不错误承认完全隐瞒 |
| 摘要与总体名次 | 第1页 abstract；第6页 §4.1 | 摘要确实只突出 Madelon；review 的转述非逐字原句，不据此否定其主问题 |
| Madelon 的三个 utility | 第6页 Table 3 | .7923/.7144/.7985 分别为 MinPPI/random/SHAP，五种子均值 |
| rank 与显著性 | 第6页 Table 3 下文字 | rank4.08；总体排名无优势，不以不显著解释为等效 |
| HV 的证据边界 | 第7页 §4.3、第8页 Figure 3 | 平均held-out ΔHV .01584；校正p .4375；不作加速主张 |
| exact 只在 Abalone，hybrid走exact | 第4页 §3.1、第8页 §4.4 | 准确恢复该游戏不等于联合删除更好 |
| A/B 缺少编号伪代码 | 第2–3页 §2.3；第11页 Appendix A/C | 现有公式及 prose 不是编号算法，建议接受这条意见 |
| 引用错误 | 第1页引用与第10页 references | Banzhaf论文作者、ShapG第二作者须修正；Lippl归因和 adapted baseline 语义须核实后改正 |

## 4. 回复策略与尚未完成的动作

这不是一份靠“承诺以后补实验”就能消除疑虑的回复。R1、R2、R4 的证据不足仍然存在；当前能做的是准确解释已有证据、接受合理批评、收缩过强结论，并给出具体修订边界。

建议顺序：

1. 确认 OpenReview 回复长度、附件和修改稿权限，先保留一份符合限制的文字回复。
2. 进入 MR2 后完成可以直接核验的引文修正、编号算法和定义澄清；真正完成并验收后，再将回复里相应将来时改为完成时，并给修改位置。
3. 只有在规则允许且实验已完成、核验通过时，才补写新证据；不要把35游戏计划写成已有结果，也不承诺 B 将超过 A。
4. 最终提交由作者操作。核对回复没有作者身份泄露、内部路径、未实现承诺被写成事实、以及超出实际 invitation 的内容。

大修的完整设计见 [MAJOR_REVISION_PROTOCOL.md](MAJOR_REVISION_PROTOCOL.md)。本文件只负责本轮回复，不替代长期修订协议。
