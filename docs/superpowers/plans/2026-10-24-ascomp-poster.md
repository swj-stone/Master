# ASCOMP 2026 Poster Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the approved artifact design. Checklist steps track generation and verification.

**Current goal:** Produce a three-column English poster at exactly 120 x 90 cm, with complete core algorithm/tool coverage and two upper-right QR codes. Earlier completed revisions are recorded below for history.

**Architecture:** A self-contained ReportLab generator draws embedded text and vector graphics from verified paper values. The original QR is embedded unchanged. All new assets live separately from the protected article and ZIP.

**Tech Stack:** Python, ReportLab, pypdf, Pillow, Poppler, task-local zxing-cpp.

**Spec:** ../specs/2026-10-24-ascomp-poster.md

## Global Constraints

- One landscape page, 1200 x 900 mm; English; ASCOMP 2026 POSTER.
- All authors and full shared affiliation; paper and Telegram QR in upper-right at 90 mm width.
- Preserve source PDF, final ZIP and both original QR hashes.
- Main text approximately 38-42 pt, smallest reference/caption text 30 pt.
- All six datasets; explicitly label means, SD and held-out endpoints; no defensive warning prose.
- Artifact-only work; no repository changes, commits or worktree operations in the unrelated Master repositories.

## Review Focus

- Clipped formula subscripts or narrow chart labels: visual crop review and text bounds.
- Misleading evidence: independent content audit against the final article.
- QR size or quiet-zone damage: decode both source and rendered output.
- Print scaling and font substitution: MediaBox and embedded-font inspection.
- Accidental original-file changes: before/after SHA256 checks.

## Current task: three-column algorithm-complete revision

Files: output/poster/build_poster.py owns vector layout and verified copy; check_poster.py validates the rendered artifact and original inputs; PRINT_README.txt documents print and evidence details. Output paths remain stable. This is an artifact revision in the user's requested directory, not a software branch change; no Git operations are needed.

- [x] Back up the predecessor as versions/spacious-dual-qr-v5 and audit methods/results independently against the final TeX.
- [x] Add failing checks for three columns, core tool names/roles, complete A/B update and seed rules, NSGA-II objectives/settings and estimator results.
- [x] Implement three body columns while retaining the exact established header and dual QR; use >=30 pt captions, ~33-36 pt body and large vector formulas/figures.
- [x] Render and inspect the overview and dense sections; run all preflight checks and obtain a fresh independent visual/content review.
- [x] Update notes and deliver the verified PDF and preview with previous versions and source inputs preserved.

Verification: all 14 checks pass. Both QR codes decode from the 3600 x 2700 overview and 150/300 dpi PDF crops. Fresh review found no blockers. Follow-up precision edits identify B's MI pool as N excluding i, seed restart per trial, the positive full-utility condition, and the exact NSGA-II difference comparison. Fonts are embedded and the original article/ZIP/QR hashes are unchanged.

## Task 1: Author and verify the poster

**Files:** output/poster/build_poster.py; output/poster/check_poster.py; output/pdf/ASCOMP2026_MinPI_Poster_120x90cm.pdf; output/poster/ASCOMP2026_MinPI_Poster_preview.png; output/poster/PRINT_README.txt.

**Interfaces:** build_poster.py consumes the unchanged QR and verified constants; emits a one-page PDF and layout manifest. check_poster.py reads the PDF, manifest and original hashes.

- [x] Write and run the artifact checks before generation; confirm missing-poster failure.
- [x] Generate the PDF with large-label vector diagrams/charts and the original QR.
- [x] Render with Poppler and inspect overview and detail crops.
- [x] Run dimensions, fonts, content, layout, source-integrity and rendered-QR checks.
- [x] Obtain independent content/layout review and address material findings.
- [x] Save print notes, preview and source; open the PDF and deliver verified outputs.

Verification: 8 automated checks passed. Independent reviewer reported no blocking issues. The first visual render exposed absent Arial set-operator glyphs; embedded Cambria Math and a character-outline preflight fixed and guard against recurrence. All source SHA256 values remained unchanged.

## User-requested revision: two columns and centered enlarged identity

- [x] Preserve the three-column PDF, preview, source, manifest, tests and notes under output/poster/versions/three-column-v1.
- [x] Add failing checks for two columns, enlarged centered identity and at least 15% less text.
- [x] Revise the generator: 96 pt event, 124 pt title, 76 pt authors, 52 pt affiliation, 42 pt address and 40 pt contacts, all centered.
- [x] Condense the body into Problem & Method and Experiments & Results, preserving all six datasets and scientific caveats.
- [x] Re-render and inspect; run all 10 automated checks, including original and full-preview QR decoding.
- [x] Update print notes and preserve the original article, source ZIP and QR hashes.

Text extraction counts: 407 words in the previous poster, 280 in this revision (about 31% fewer). Exact page dimensions remain 1200 x 900 mm. The QR remains 90 x 108 mm at approximately 310 dpi in the lower-left corner.

## Editorial revision: 2026-10-25

- [x] Back up the preceding two-column PDF, preview, source and support files under versions/two-column-v2.
- [x] Add and run a failing test for contribution-focused copy and removal of warning-style descriptions.
- [x] Emphasize worst-case evidence, coalition witnesses and validation-gated removal; add Madelon matched-count random-removal comparison.
- [x] Retain all six datasets, numerical SD and overall Holm p as neutral empirical context.
- [x] Render and inspect the new poster; 11 automated checks pass, including QR decoding and protected input hashes.
- [x] Independent visual review reports no overflow, clipping, overlap or remaining defensive warning copy.

## Layout refinement: author/contact groups

- [x] Preserve the prior version under versions/contribution-focused-v3.
- [x] Replace the centered-identity test with centered-title and split-identity assertions; observe the new placement check fail before editing.
- [x] Move names/institution left and address/contacts right without reducing their type sizes.
- [x] Add the optimization-objectives ribbon in the released header space.
- [x] Render and visually inspect; all 12 preflight checks pass. Independent review confirms balanced placement and no overlap or clipping.

The user's selection of option 1 approves the design. Execution continues directly in the current session without another design-selection round.

## Latest refinement: reference-led algorithm flow and dual QR

- [x] Preserve the preceding deliverables and source under versions/split-identity-v4.
- [x] Inspect the reference poster and both QR sources. Use its flow organization, not its unrelated paper content.
- [x] Add and observe failing checks for upper-right dual QR and algorithm flow before implementation.
- [x] Reclaim the objectives ribbon/lower-left QR area for numbered four-step flow, with A/B/hybrid details and the all-or-none gate.
- [x] Keep six-dataset results and aggregate values; add Abalone to the matched random comparison; retain NSGA-II budget and measured HV changes.
- [x] Keep original SCAN ME artwork; regenerate Telegram from its decoded exact payload as vector QR with four-module quiet zone.
- [x] Remove visible paper URL, scan-to-read sentence and hyperlink annotations.
- [x] Render and inspect overview/header; correct B wording to independent per-member drop probability and clarify validation/test order following independent content review.
- [x] Fix the QR renderer's default Times-Roman resource at its source by setting the drawing's initial font to embedded Arial; font check then passes.
- [x] Run all 13 preflight checks successfully, inspect all three embedded font resources and exact single-page dimensions.
- [x] Decode both PDF-rendered codes at 150 and 300 dpi; independent visual review reports no overlap/clipping.
- [x] Update print notes; preserve original source/QR hashes.

## Final wording/math audit and XeLaTeX revision (v7)

- [x] Preserve the previous complete three-column PDF, preview, generator, checks, manifest and notes under versions/three-column-complete-v6.
- [x] Audit all equations, sampler boundaries, empirical quantities and wording against the final article and the B implementation.
- [x] Correct K membership/per-estimator budget, available-feature cap, seed restart, exact neutral label, accepted removal count, seed-42 sweep, harmful-set F1 and Holm p association.
- [x] Move all text and mathematics into an editable XeLaTeX project; retain the vector background/chart/Telegram QR and original paper QR.
- [x] Fix TikZ's text-font context by constructing measured LaTeX boxes outside the picture environment; reposition the display fraction and following copy to eliminate collision.
- [x] Embed Arial and Latin Modern Math, include the math font and original license, and verify both local XeLaTeX and the desktop editor compiler.
- [x] Render and inspect the overview and enlarged equation crops; independent review confirms the minimum domain, hats, membership, seed restart and fraction are correctly rendered.
- [x] Run 18 automated preflight checks, covering all 101 measured text boxes, nonoverlap, formula structure, fonts, content, dimensions, QR decoding and protected source hashes.
- [x] Decode both PDF-rendered QR codes at 150 and 300 dpi; update regeneration and print notes.

## Final file consolidation (2026-10-25)

- Final article and poster restored to their original `Master/conference-paper/paper` and `poster` paths; original filenames retained.
- Previous article, figures, PDF/PPTX poster preserved under `conference-paper/archive/pre-final-20261025/`; poster revision history under `archive/poster-versions/`.
- Historical packages moved into `Master/packages/`; build/debug scratch and PDF working files into `Master/tmp/`; design history into `Master/docs/`.
- Updated only article figure paths and poster script paths. The protected final source ZIP remains untouched at the workspace root.
- Current paths and rebuild instructions: `Master/conference-paper/FINAL_FILES.txt`.
