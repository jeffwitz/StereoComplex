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

.PHONY: help clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"

.PHONY: Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
