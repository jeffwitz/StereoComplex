# ======================================================================
# Reproduce the CMO paper (paper/cmo/manuscript.tex)
# ======================================================================
# Quick start (clean checkout, offline, minutes — uses the versioned figures):
#   make repro       # compile the PDF + audit every manuscript number
#   make pdf         # just compile the manuscript
#   make audit       # cross-check every manuscript number against the assets
#   make clean       # remove build/
#
# Regenerate the figures (the specimen ones need the heavy .npz present):
#   make figures     # regenerate every figure from the current data
#   make specimen    # rebuild the full-image specimen .npz (needs Pycaso images)
#   make fetch       # specimen .npz from Zenodo (PREDATES the full-image
#                    #   regeneration — stale for the current numbers)
#   make schur       # re-run the corner Schur-BA (diagnostics + thetas, ~2 h)
#   make repro-full  # schur + specimen + figures + pdf + audit (from raw)
#
# See REPRODUCE.md for the data-provenance and one-command recipe.
# ======================================================================

PY    ?= rtk .venv/bin/python
ROOT  := $(abspath $(CURDIR)/../..)
NB    := examples/notebooks
SCHUR_OUT := docs/assets/pycaso_real_data/schur_ba
STATE     := docs/assets/pycaso_real_data/intermediate_state.npz

# one generate_fig_<name>.py per figure referenced by the manuscript
FIGS := bic_bars bic_vs_order cmo_physical dy_profile pipeline \
        profilo_relief_comparison residual_evolution schur_svd se3_arm_alignment \
        specimen_reconstruction zernike_cmo_rigid_removed

.PHONY: repro repro-full pdf figures fetch specimen schur audit clean help
.DEFAULT_GOAL := help

help:
	@sed -n '2,18p' $(firstword $(MAKEFILE_LIST))

# --- fast reproduction: build the versioned figures into the PDF + audit ------
repro: pdf audit
	@echo "==> CMO paper built + audited (paper/cmo/build/manuscript.pdf)."

# --- full reproduction from raw inputs --------------------------------
repro-full: schur specimen figures pdf audit
	@echo "==> CMO paper reproduced from raw inputs."

pdf:
	cd $(ROOT)/paper/cmo && bash build_pdflatex.sh

figures:
	@cd $(ROOT) && $(foreach f,$(FIGS), \
	    echo "--- generate_fig_$(f).py" && $(PY) $(NB)/generate_fig_$(f).py &&) true
	@cd $(ROOT) && echo "--- generate_tab_dis_robustness.py" && \
	    $(PY) $(NB)/generate_tab_dis_robustness.py

fetch:
	cd $(ROOT) && $(PY) examples/zenodo_fetch.py

specimen:
	cd $(ROOT) && $(PY) $(NB)/regenerate_specimen_reconstructions.py --dis improved

schur:
	cd $(ROOT) && $(PY) examples/pycaso_schur_regularized_ba.py \
	    --mode diagnostic --input $(STATE) --out $(SCHUR_OUT)
	cd $(ROOT) && $(PY) examples/pycaso_schur_regularized_ba.py \
	    --mode ba --input $(STATE) --out $(SCHUR_OUT)

audit:
	cd $(ROOT) && $(PY) $(NB)/audit_paper_numbers.py

clean:
	rm -rf $(ROOT)/paper/cmo/build
