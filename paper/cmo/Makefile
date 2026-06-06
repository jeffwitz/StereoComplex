# ======================================================================
# Reproduce the CMO paper (paper/cmo/manuscript.tex)
# ======================================================================
# Quick start:
#   make repro       # fast path: build all figures + PDF + audit from the
#                    # versioned data (minutes; no raw images, no 2 h BA)
#   make pdf         # just compile the manuscript
#   make figures     # regenerate every figure from the current data
#   make audit       # cross-check every manuscript number against the assets
#
# Heavy / from-scratch:
#   make fetch       # download the heavy specimen .npz from Zenodo
#   make specimen    # regenerate the full-image specimen reconstructions
#                    #   (needs the Pycaso raw images, see REPRODUCE.md)
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
FIGS := bic_bars bic_vs_order cmo_physical dy_profile pareto pipeline \
        profilo_relief_comparison residual_evolution schur_svd se3_arm_alignment \
        specimen_reconstruction specimen_schur_regularized subpupil_3d \
        zernike_cmo_rigid_removed

.PHONY: repro repro-full pdf figures fetch specimen schur audit clean help
.DEFAULT_GOAL := help

help:
	@sed -n '2,18p' $(firstword $(MAKEFILE_LIST))

# --- fast reproduction (versioned data only) --------------------------
repro: figures pdf audit
	@echo "==> CMO paper reproduced from versioned data (paper/cmo/build/manuscript.pdf)."

# --- full reproduction from raw inputs --------------------------------
repro-full: schur specimen figures pdf audit
	@echo "==> CMO paper reproduced from raw inputs."

pdf:
	cd $(ROOT)/paper/cmo && bash build_pdflatex.sh

figures:
	@cd $(ROOT) && $(foreach f,$(FIGS), \
	    echo "--- generate_fig_$(f).py" && $(PY) $(NB)/generate_fig_$(f).py &&) true

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
