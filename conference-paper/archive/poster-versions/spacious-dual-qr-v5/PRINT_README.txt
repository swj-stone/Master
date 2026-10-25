ASCOMP 2026 - MinPI poster
=========================

CURRENT REVISION: spacious two-column poster with a four-step algorithm flow
and two upper-right QR codes. Conference and paper titles remain centered;
author names/institution are on the left, address/contacts on the right.
The former objectives ribbon and lower-left QR area now give the algorithm
more room. Core idea and algorithm are on the left; results on the right.
The supplied reference poster informed organization only, not paper content.
Previous versions are retained under versions/three-column-v1,
versions/two-column-v2, versions/contribution-focused-v3 and
versions/split-identity-v4.

PRINT MASTER
../pdf/ASCOMP2026_MinPI_Poster_120x90cm.pdf

Single landscape page: 1200 mm wide x 900 mm high (120 x 90 cm).
Print at 100% / Actual size. Do not select Fit to page.
The PDF is the print master; the PNG is for screen preview only.
Ask the print shop to preserve the page dimensions and embedded fonts.
The PDF uses RGB colours; no specific printer ICC profile or PDF/X standard
was supplied. If the shop requires CMYK, PDF/X or bleed, request conversion
to its production specification without scaling the artwork.

QUALITY
All text is embedded; Arial and Cambria Math are subset-embedded TrueType.
The algorithm flow, chart, rules, shapes and Telegram QR are vector objects.
ASCOMP 2026 POSTER: 96 pt, centered (previously 42 pt).
Paper title: 124 pt, centered. Authors: 76 pt, left-aligned.
Affiliation: 52 pt, left-aligned. Address: 42 pt, right-aligned.
Contact lines: 40 pt, right-aligned.
Section headings: 54 pt. Main body: approximately 35-42 pt.
Chart captions and data labels: 33-41 pt. Formula text: 50 pt.
Small QR labels and algorithm routing notes: 30-32 pt.
This large-type layout is designed for viewing at approximately 1.5 m.

The original paper QR is 90 x 108 mm, using the original 1100 x 1320 px
image (310.44 effective dpi), with preserved aspect ratio and quiet zone.
Decoded payload: https://hlnks.co/63671101
The embedded SCAN ME label is retained; the separate reading invitation and
visible URL are removed. There are no PDF hyperlink annotations.
The Telegram QR is 90 x 90 mm, regenerated as black vector modules from the
exact payload decoded from tmp/pdfs/tele_code.png:
https://t.me/+79516564253
It has a four-module quiet zone, error correction M, and labels Contact author
and Telegram. The supplied Telegram screenshot is unchanged.
Both codes are in the upper-right, aligned at the top, 12.7 mm apart.
Both originals, the complete 3200 x 2400 preview, and PDF header crops at
150 and 300 dpi decode successfully. There is no title or label overlap.
Destinations were preserved; remote availability was not changed or tested.

SOURCE AND EVIDENCE
Article: ../pdf/MinPI_arXiv_final.pdf (the user's final version).
Identity: title page; two authors share the full affiliation shown.
Definitions and gate: Section 2; algorithms: Appendix A.
Design: Section 3.
Four-step flow: prepare the threshold game; estimate MinPI and collect harmful
feature witnesses; validate joint removal; evaluate the selected subset, with
optional MinPI-prefiltered NSGA-II. Coalition models fit on train, selection
uses validation, and final utility uses the held-out test split.
A samples a coalition size uniformly then a subset uniformly. B perturbs a
training-only MI-correlation seed, dropping each member independently with
probability 0.3 and adding up to five features. Exact/B/hybrid routing and the
B-to-A fallback remain explicit. The joint-removal gate is all-or-none.
Compression chart and held-out utility values: Table 2, five-seed mean and SD.
Aggregate values: Section 4.1, all 30 dataset-seed units.
NSGA-II held-out hypervolume: Section 4.3; absolute change, not a percentage.
Matched-count removal comparisons: Section 4.1, mean held-out utility changes:
Madelon +0.0047 for MinPI versus -0.0732 for random; Abalone -0.0056 versus
-0.0370, with equal deletion counts.
Random uses 30 trials averaged within each of the five seeds.
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

FILES AND REGENERATION
build_poster.py: editable content, exact result values and vector layout.
check_poster.py: 13 automated read-only preflight checks, including centered
titles, left/right identity groups, algorithm copy, dual QR and source integrity.
layout_manifest.json: layout, result values, glyph audit and source hashes.
ASCOMP2026_MinPI_Poster_preview.png: 3200 x 2400 screen preview.

From the workspace root, use Python with reportlab, Pillow and pypdf:
  python output/poster/build_poster.py
  pdftoppm -scale-to 3200 -singlefile -png output/pdf/ASCOMP2026_MinPI_Poster_120x90cm.pdf output/poster/ASCOMP2026_MinPI_Poster_preview
  python output/poster/check_poster.py

The build uses Windows fonts under C:/Windows/Fonts and reads the original
output/QRcode.png. Telegram vector QR uses its verified unchanged payload.
The QR check uses zxing-cpp installed only in the task-local
tmp/pdfs/ascomp-poster/deps directory. No global environment was changed.

PROTECTED INPUTS (SHA256, unchanged)
output/pdf/MinPI_arXiv_final.pdf
2c7cc05471252f1e464230d3a01cd5d1e8d4fb8852fb46f37162a6ca5fc26424

output/QRcode.png
3b007243b94c94bcc09a92cc86a5c0900ace9d4a4c103e565a84a07e2245435d

MinPI_arXiv_source(1).zip
5e09469bd8731d4158b16921b737e5c99b56a46e0e95163c2ea93e03d09e4cf6

tmp/pdfs/tele_code.png
f9ac461d4908cecd45fc29d684eb90c68798e8ab771661c5b0a60bcf70538d63

中文印刷提示
请使用 PDF 印刷原稿，纸张宽 120 cm、高 90 cm，横向，按 100% 实际大小打印。
PNG 仅供屏幕预览，请勿将其替代矢量 PDF 作为印刷原稿。
原论文、最终源文件 ZIP、论文二维码、Telegram 二维码原图均保持不变。
当前为疏朗两栏版，会议名称和文章标题居中；作者及机构在左侧，地址及联系方式在右侧。
左栏展示数学定义及四步算法流程；右栏展示六个数据集、总体结果、随机删除对照和 NSGA-II 结果。
两个二维码均位于右上角：SCAN ME 与 Contact author / Telegram。
已移除原来的 scan-to-read 提示、可见链接和 PDF 超链接。
历史版本备份于 versions/，本次修改前的版本位于 split-identity-v4。
