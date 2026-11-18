ASCOMP 2026 - MinPI poster
=========================

CURRENT REVISION: v7, audited three-column poster with XeLaTeX mathematics
and two upper-right QR codes. Conference and paper titles remain centered;
author names/institution are on the left, address/contacts on the right.
Columns: principle and toolkit; algorithm flow; experimental evidence.
The body adds the exact MinSS/MinBz identity and labels, core tool roles,
complete A/B sampling and update rules, hybrid/gate decisions, NSGA-II
objectives/settings, matched controls and exact-reference estimator results.
The supplied reference poster informed organization only, not paper content.
Previous poster revisions are retained under ../archive/poster-versions/.
The original article, figures and poster/PPTX before consolidation are under
../archive/pre-final-20261025/. Current path index: ../FINAL_FILES.txt.

PRINT MASTER
ASCOMP_MinPPI_Poster_Wenjie_SHI.pdf
The old filename is retained; the content is the final MinPI XeLaTeX poster.

Single landscape page: 1200 mm wide x 900 mm high (120 x 90 cm).
Print at 100% / Actual size. Do not select Fit to page.
The PDF is the print master; the PNG is for screen preview only.
Ask the print shop to preserve the page dimensions and embedded fonts.
The PDF uses RGB colours; no specific printer ICC profile or PDF/X standard
was supplied. If the shop requires CMYK, PDF/X or bleed, request conversion
to its production specification without scaling the artwork.

QUALITY
All text and mathematics are typeset by XeLaTeX, with embedded Arial and
Latin Modern Math fonts. Equations are vector text, not bitmap screenshots.
True LaTeX subscripts, superscripts, minimum domains, hats, set notation,
floor brackets and fractions replace the previous manually assembled math.
The algorithm flow, chart, rules, shapes and Telegram QR are vector objects.
ASCOMP 2026 POSTER: 96 pt, centered (previously 42 pt).
Paper title: 124 pt, centered. Authors: 76 pt, left-aligned.
Affiliation: 52 pt, left-aligned. Address: 42 pt, right-aligned.
Contact lines: 40 pt, right-aligned.
Section headings: 46 pt. Main body: approximately 32-39 pt.
Chart captions and data labels: 30-34 pt. Formula text: 38-42 pt.
Small QR labels and algorithm routing notes: 30-32 pt.
This large-type layout is designed for viewing at approximately 1.5 m.

The original paper QR is 90 x 108 mm, using the original 1100 x 1320 px
image (310.44 effective dpi), with preserved aspect ratio and quiet zone.
Decoded payload: https://hlnks.co/63671101
The embedded SCAN ME label is retained; the separate reading invitation and
visible URL are removed. There are no PDF hyperlink annotations.
The Telegram QR is 90 x 90 mm, regenerated as black vector modules from the
exact payload decoded from assets/tele_code.png:
https://t.me/+79516564253
It has a four-module quiet zone, error correction M, and labels Contact author
and Telegram. The supplied Telegram screenshot is unchanged.
Both codes are in the upper-right, aligned at the top, 12.7 mm apart.
Both originals, the complete 3600 x 2701 preview, and PDF header crops at
150 and 300 dpi decode successfully. There is no title or label overlap.
Destinations were preserved; remote availability was not changed or tested.

SOURCE AND EVIDENCE
Article: ../paper/icomp2026_conference.pdf (the user's final arXiv version).
Identity: title page; two authors share the full affiliation shown.
Definitions and gate: Section 2; algorithms: Appendix A.
Design: Section 3.
Algorithm: prepare the threshold game; choose exact/A/B/hybrid estimation;
collect harmful feature witnesses; validate joint removal; evaluate the retained
subset, with optional MinPI-prefiltered NSGA-II. Coalition models fit on train,
selection uses validation, and final utility uses the held-out test split.
A samples a coalition size uniformly then a subset uniformly. B perturbs a
training-only MI-correlation seed, dropping each member independently with
probability 0.3 and adding up to five features. Exact/B/hybrid routing and the
B-to-A fallback remain explicit. B's seed uses up to ten tree neighbors ranked
by absolute correlation and padding from the top-MI half of N excluding i;
the size target is min(20, floor(n/3)). Each draw restarts from that seed.
Both estimators track the minimum observed marginal and exit at the first -1.
Budgets are 2/5/10/25 per feature per estimator, written with set membership;
ISOLET uses up to 10. Removal uses the largest. B restarts S from its seed on
each draw and caps additions at the number of available absent features.
The joint-removal gate is all-or-none.
Core tools: training-only variance filtering, MI selection, standardization,
XGBoost CUDA (100 rounds, depth 4, learning rate 0.1), maximum-weight absolute-
correlation tree, TreeSHAP and the non-dominated sorting genetic algorithm II.
NSGA-II objectives are feature fraction and clipped validation utility loss;
the original analyzed dimension is shared. Population 40, 800 evaluations per
variant, HV reference (1.1, 1.1). The selected front is evaluated on held-out data.
Compression chart and held-out utility values: Table 2, five-seed mean and SD.
Aggregate values: Section 4.1, all 30 dataset-seed units.
NSGA-II held-out hypervolume: Section 4.3; absolute change, not a percentage.
Its delta is explicitly MinPI-prefiltered search minus full-space search.
Matched-count removal comparisons: Section 4.1, mean held-out utility changes:
Madelon +0.0047 for MinPI versus -0.0732 for random; Abalone -0.0056 versus
-0.0370, with equal deletion counts.
Random uses 30 trials averaged within each of the five seeds.
Estimator accuracy: Section 4.4, Abalone (8 features), five-seed mean harmful-set
F1 against exact MinPI. At K=2/5/10/25, A = .100/.150/.230/.443 and
B = .000/.080/.133/.313; hybrid selects exact enumeration and F1=1 at each K.
The dispatcher exact branch is n<=20; the paper's formal exact-reference
experiment uses the small-dimensional Abalone dataset. These are distinct roles.
All six datasets are shown. Compression refers to analyzed features, not
native dimensions. The bars share a 0-80% linear scale; whiskers show one SD
and are truncated at zero where necessary. Numeric SD values are also shown.
The algorithm flow summarizes the paper, not additional experimental data.
The results chart has been redrawn as vector graphics from the article's
reported values so that labels remain readable at poster scale.

EDITORIAL EMPHASIS
From average importance to worst-case evidence.
Each detected harmful feature has an evaluated coalition witness.
Candidate joint removals pass a validation gate.
All six dataset means and standard deviations remain visible.
NSGA-II includes its observed aggregate and Madelon changes, matched budget
and overall Holm-adjusted p = 0.4375 as neutral numerical context.
The evidence is presented as measured outcomes and concrete comparisons.
The F1 comparison explicitly means harmful-set F1 against exact labels;
hybrid uses exact enumeration. Holm p is attached to the aggregate HV result.
The quota sweep is explicitly identified as seed 42. Exact neutral labels
require all marginals nonnegative and at least one zero. The validation
utility, accepted removal count and retained subset are explicitly defined.

FILES AND REGENERATION
build_poster.py: editable content, exact result values and vector layout.
check_poster.py: 18 automated read-only preflight checks, including three columns,
centered titles, left/right identity, core tools/algorithm coverage, exact F1
arrays, dual QR, source integrity, embedded math fonts, actual LaTeX box
dimensions and non-overlap, compilation diagnostics and formula structure.
layout_manifest.json: layout, result values, glyph audit, source hashes and
LaTeX-measured text box heights (not estimated line-height measurements).
ASCOMP2026_MinPI_Poster_preview.png: 3600 x 2701 screen preview.
latex/main.tex: editable XeLaTeX project entry point and typography.
latex/poster-content.tex: generated editable TeX text and formula placement.
latex/poster-background.pdf: generated vector shapes, chart and dual QR.
latex/fonts/: Latin Modern Math font with its original license and README.

From Master/, use Python with reportlab, Pillow and pypdf, plus
XeLaTeX (TeX Live) and Poppler on PATH:
  python conference-paper/poster/build_poster.py
  pdftoppm -scale-to 3600 -singlefile -png conference-paper/poster/ASCOMP_MinPPI_Poster_Wenjie_SHI.pdf conference-paper/poster/ASCOMP2026_MinPI_Poster_preview
  python conference-paper/poster/check_poster.py

The builder exports the graphic background, generates the LaTeX content,
runs XeLaTeX twice, checks every text box and checks for missing characters
and overfull boxes before copying the PDF to the stable delivery path.
Compiler intermediates are kept in Master/tmp/pdfs/ascomp-poster/latex-build/.
Both the installed TeX Live XeLaTeX and the desktop editor's compiler were
verified successfully. Open the latex/ folder to inspect the editable source.
For persistent content/layout changes edit build_poster.py; its next run
regenerates poster-content.tex. main.tex controls the overall font setup.

The build uses Windows fonts under C:/Windows/Fonts and reads the original
assets/QRcode.png. Telegram vector QR uses its verified unchanged payload.
The QR check uses zxing-cpp installed only in the task-local
tmp/pdfs/ascomp-poster/deps directory. No global environment was changed.

PROTECTED INPUTS (SHA256, unchanged)
Master/conference-paper/paper/icomp2026_conference.pdf
2c7cc05471252f1e464230d3a01cd5d1e8d4fb8852fb46f37162a6ca5fc26424

Master/conference-paper/poster/assets/QRcode.png
3b007243b94c94bcc09a92cc86a5c0900ace9d4a4c103e565a84a07e2245435d

MinPI_arXiv_source(1).zip
5e09469bd8731d4158b16921b737e5c99b56a46e0e95163c2ea93e03d09e4cf6

Master/conference-paper/poster/assets/tele_code.png
f9ac461d4908cecd45fc29d684eb90c68798e8ab771661c5b0a60bcf70538d63

中文印刷提示
请使用 PDF 印刷原稿，纸张宽 120 cm、高 90 cm，横向，按 100% 实际大小打印。
PNG 仅供屏幕预览，请勿将其替代矢量 PDF 作为印刷原稿。
原论文、最终源文件 ZIP、论文二维码、Telegram 二维码原图均保持不变。
当前为 v7 三栏 LaTeX 终检版，会议名称和文章标题居中；作者及机构在左侧，地址及联系方式在右侧。
正文与公式均经 XeLaTeX 排版，数学字体已嵌入；分式、上下标、最小值范围和帽号为标准 LaTeX。
三栏依次为：原理与工具、算法流程、实验证据。正文适度增加密度，补齐核心工具和算法细节。
新增 Abalone 精确基准上的估计器 F1 对照，保留六数据集、随机删除对照及 NSGA-II 结果。
两个二维码均位于右上角：SCAN ME 与 Contact author / Telegram。
已移除原来的 scan-to-read 提示、可见链接和 PDF 超链接。
历史海报位于 ../archive/poster-versions/；原文章、原海报及 PPTX 位于 ../archive/pre-final-20261025/。
最终 PDF 已覆盖原海报路径；保留旧文件名以兼容原有引用，内容为最终 MinPI 三栏版。
