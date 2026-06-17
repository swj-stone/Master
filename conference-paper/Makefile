# ICOMP 2026 paper build pipeline.
#
# Deliverables (both named minppi_mofs_<DD-Month>):
#   * minppi_mofs_<DD-Month>.pdf   the compiled paper
#   * minppi_mofs_<DD-Month>.zip   sources only (figures + *.tex/.sty/.bst + .bib)
#
#   make assets    Collect figures/ from results/figures/ (sync_figures.py).
#   make pdf       Compile -> minppi_mofs_<DD-Month>.pdf.
#   make zip       Package sources -> minppi_mofs_<DD-Month>.zip.
#   make all       assets + pdf + zip.
#   make clean     Remove build/ and today's deliverables only;
#                  historical pdf/zip (other dates) are preserved.
#   make help      Show this list.
#
# Layout note: the .tex/.sty/.bst/.bib live in paper/; figures live in the
# local figures/ dir and are referenced by the paper as ../figures/. LaTeX
# runs inside build/ (a sibling of paper/ and figures/, same depth), so the
# relative figure paths resolve identically during the build.

# ---- knobs ----------------------------------------------------------------
PYTHON   ?= python
LATEX    ?= pdflatex
BIBTEX   ?= bibtex

SRCDIR   := paper
FIGDIR   := figures
MAIN     := icomp2026_conference
BUILDDIR := build
STEM     := minppi_mofs

# Date stamp: 17-June, 06-April, ...
DATE     := $(shell date +%d-%B)
PDF_NAME := $(STEM)_$(DATE).pdf
ZIP_NAME := $(STEM)_$(DATE).zip

# Source files that end up in the zip (no pdf, no build artifacts)
SRC_FILES := $(wildcard $(SRCDIR)/*.tex) $(wildcard $(SRCDIR)/*.sty) \
             $(wildcard $(SRCDIR)/*.bst) $(wildcard $(SRCDIR)/*.bib)

# Point LaTeX/BibTeX at the source dir while running from build/.
TEXENV := TEXINPUTS=../$(SRCDIR): BIBINPUTS=../$(SRCDIR): BSTINPUTS=../$(SRCDIR):

# ---- targets --------------------------------------------------------------
.PHONY: help all assets pdf zip clean
.DEFAULT_GOAL := help

help:
	@echo "make assets   collect $(FIGDIR)/ from results/figures/ (sync_figures.py)"
	@echo "make pdf      compile -> $(PDF_NAME)"
	@echo "make zip      package sources -> $(ZIP_NAME) (no pdf inside)"
	@echo "make all      assets + pdf + zip"
	@echo "make clean    remove $(BUILDDIR)/, $(PDF_NAME), $(ZIP_NAME)"

all: assets pdf zip

assets:
	$(PYTHON) sync_figures.py

# pdflatex + bibtex + pdflatex + pdflatex resolves \cite and \ref reliably.
# LaTeX runs in build/ to keep the source tree tidy; the final pdf is
# copied to the top level under a date-stamped name.
pdf: $(PDF_NAME)

$(PDF_NAME): $(BUILDDIR)/$(MAIN).pdf
	cp $< $@

$(BUILDDIR)/$(MAIN).pdf: $(SRC_FILES) | $(BUILDDIR)
	cd $(BUILDDIR) && $(TEXENV) $(LATEX) -interaction=nonstopmode $(MAIN).tex
	cd $(BUILDDIR) && $(TEXENV) $(BIBTEX) $(MAIN)
	cd $(BUILDDIR) && $(TEXENV) $(LATEX) -interaction=nonstopmode $(MAIN).tex
	cd $(BUILDDIR) && $(TEXENV) $(LATEX) -interaction=nonstopmode $(MAIN).tex

$(BUILDDIR):
	mkdir -p $@

# Package sources only: paper/ sources + figures/. The paper/ and figures/
# dirs are kept as siblings so ../figures/ in the .tex still resolves when a
# reviewer compiles from the extracted tree. The pre-built pdf is excluded.
zip: $(ZIP_NAME)

$(ZIP_NAME): $(SRC_FILES)
	@echo "Packaging $(ZIP_NAME)..."
	@rm -f $@
	@test -d $(FIGDIR) || echo "  warning: $(FIGDIR)/ missing — run 'make assets' first"
	zip -r $@ $(SRC_FILES) $(wildcard $(FIGDIR)) -x '*.DS_Store' '*/__pycache__/*'
	@echo "Done: $@ ($$(du -h $@ | cut -f1))"

clean:
	rm -rf $(BUILDDIR)
	rm -f $(PDF_NAME) $(ZIP_NAME)
