# Minimal Makefile for Sphinx documentation
#

SPHINXOPTS    =

# Prefer the repository venv when available (repo root: ../.venv).
ifeq ($(wildcard ../.venv/bin/python),../.venv/bin/python)
SPHINXBUILD   = ../.venv/bin/python -m sphinx
else
SPHINXBUILD   = sphinx-build
endif
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean pdf

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"

# 'pdf' is not a built-in Sphinx builder; route to latexpdf (requires xelatex).
pdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
