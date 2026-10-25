ASCOMP 2026 - MinPI poster
=========================

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
The diagram, chart, rules and shapes are vector objects.
Title: 112 pt. Section headings: 50 pt. Main body: approximately 35-47 pt.
Captions/data labels/contact information: 30-36 pt.
This large-type layout is designed for viewing at approximately 1.5 m.

The original lower-left QR is 90 x 108 mm, using the original 1100 x 1320 px
image (310.44 effective dpi), with preserved aspect ratio and quiet zone.
Decoded payload: https://hlnks.co/63671101
Both the original and the complete 3200 x 2400 preview decode successfully.
The supplied QR destination was retained; its remote landing page was not
changed or independently validated for continued availability.

SOURCE AND EVIDENCE
Article: ../pdf/MinPI_arXiv_final.pdf (the user's final version).
Identity: title page; two authors share the full affiliation shown.
Definitions and gate: Section 2.
Design: Section 3.
Compression chart and held-out utility values: Table 2, five-seed mean and SD.
Headline values and gate counts: Section 4.1, all 30 dataset-seed units.
NSGA-II held-out hypervolume: Section 4.3; absolute change, not a percentage.
All six datasets are shown. Compression refers to analyzed features, not
native dimensions. The bars share a 0-80% linear scale; whiskers show one SD
and are truncated at zero where necessary. Numeric SD values are also shown.
The threshold-crossing diagram is conceptual, not additional experimental data.
The source figures have been redrawn as vector graphics from the article's
reported values so that labels remain readable at poster scale.

LIMITATIONS KEPT VISIBLE
Finite sampling may miss harmful witnesses.
Validation gating is not a guarantee of held-out non-inferiority.
NSGA-II: Holm-adjusted p = 0.4375; the gain is driven mainly by Madelon.
No overall superiority or runtime speedup is established.

FILES AND REGENERATION
build_poster.py: editable content, exact result values and vector layout.
check_poster.py: 8 automated read-only preflight checks.
layout_manifest.json: layout, result values, glyph audit and source hashes.
ASCOMP2026_MinPI_Poster_preview.png: 3200 x 2400 screen preview.

From the workspace root, use Python with reportlab, Pillow and pypdf:
  python output/poster/build_poster.py
  pdftoppm -scale-to 3200 -singlefile -png output/pdf/ASCOMP2026_MinPI_Poster_120x90cm.pdf output/poster/ASCOMP2026_MinPI_Poster_preview
  python output/poster/check_poster.py

The build uses Windows fonts under C:/Windows/Fonts and reads the original
output/QRcode.png. The QR check uses zxing-cpp installed only in the task-local
tmp/pdfs/ascomp-poster/deps directory. No global environment was changed.

PROTECTED INPUTS (SHA256, unchanged)
output/pdf/MinPI_arXiv_final.pdf
2c7cc05471252f1e464230d3a01cd5d1e8d4fb8852fb46f37162a6ca5fc26424

output/QRcode.png
3b007243b94c94bcc09a92cc86a5c0900ace9d4a4c103e565a84a07e2245435d

MinPI_arXiv_source(1).zip
5e09469bd8731d4158b16921b737e5c99b56a46e0e95163c2ea93e03d09e4cf6

中文印刷提示
请使用 PDF 印刷原稿，纸张宽 120 cm、高 90 cm，横向，按 100% 实际大小打印。
PNG 仅供屏幕预览，请勿将其替代矢量 PDF 作为印刷原稿。
原论文、最终源文件 ZIP、专属二维码均保持不变。
