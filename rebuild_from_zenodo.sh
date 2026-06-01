#!/usr/bin/env bash
# Rebuild the CMO paper from the Zenodo archive.
#
# Two usage modes:
#
# A) From an extracted Zenodo archive (no git clone needed):
#      unzip sterocomplex_cmo_paper.zip
#      cd sterocomplex_cmo_paper/
#      bash rebuild_from_zenodo.sh
#
# B) From a git clone (fetches Zenodo automatically):
#      git clone https://github.com/jeffwitz/StereoComplex
#      cd StereoComplex
#      bash rebuild_from_zenodo.sh
#
# After running, you get the full directory structure and can:
#   cd paper/cmo && bash build_pdflatex.sh      # rebuild the PDF
#   python examples/notebooks/generate_fig_*.py  # regenerate figures

set -euo pipefail

# ── Detect context ────────────────────────────────────────────────────
# If manuscript.tex is in the current directory, we're inside an already-
# extracted Zenodo archive (mode A). Otherwise, download from Zenodo.
if [ -f manuscript.tex ]; then
    EXTRACTED_DIR="$(pwd)"
    echo "=== Mode A: using already-extracted Zenodo at $EXTRACTED_DIR ==="
else
    ZENODO_URL="https://zenodo.org/api/records/20444786/files-archive"
    TMPDIR="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR"' EXIT
    echo "=== Mode B: downloading Zenodo archive ... ==="
    curl -sL "$ZENODO_URL" -o "$TMPDIR/zenodo.zip"
    unzip -qo "$TMPDIR/zenodo.zip" -d "$TMPDIR/extracted"
    EXTRACTED_DIR="$TMPDIR/extracted"
fi

# ── Restore paper/cmo/ structure ─────────────────────────────────────
echo "=== Restoring paper/cmo/ structure ..."
mkdir -p paper/cmo/figures paper/cmo/tables paper/cmo/build

for f in manuscript.tex references.bib build_pdflatex.sh cover_letter.txt \
         generate_tables.py check_manuscript_numbers.py number_audit_report.md \
         SUBMISSION_CHECKLIST.md; do
  [ -f "$EXTRACTED_DIR/$f" ] && cp "$EXTRACTED_DIR/$f" paper/cmo/
done

[ -f "$EXTRACTED_DIR/manuscript.pdf" ] && cp "$EXTRACTED_DIR/manuscript.pdf" paper/cmo/
[ -f "$EXTRACTED_DIR/manuscript.log" ] && cp "$EXTRACTED_DIR/manuscript.log" paper/cmo/build/

# ── Figures ──────────────────────────────────────────────────────────
for f in "$EXTRACTED_DIR"/*.pdf "$EXTRACTED_DIR"/*.png; do
  fname=$(basename "$f")
  [ "$fname" = "manuscript.pdf" ] && continue
  cp "$f" "paper/cmo/figures/"
done

# ── Tables ────────────────────────────────────────────────────────────
for f in "$EXTRACTED_DIR"/*.tex; do
  fname=$(basename "$f")
  [ "$fname" = "manuscript.tex" ] && continue
  cp "$f" "paper/cmo/tables/"
done

# ── Figure asset folders ─────────────────────────────────────────────
echo "=== Restoring figure assets ..."
FIG_DIRS=(
  figure1_cmo_physical
  figure2_pipeline
  figure3_BIC_vs_order
  figure4_subpupil_3d
  figure5_dy_profile_comparison
  figure6_residual_evolution
  figure7_pareto_gauge_regularization
  figure8_schur_singular_values
  figure9_specimen_reconstruction
  figure10_zernike_cmo_rigid_removed
  figure11_specimen_schur_regularized
  figure12_bic_bars
)

for fig_dir in "${FIG_DIRS[@]}"; do
  prefix="${fig_dir}_"
  mkdir -p "docs/assets/cmo_paper/$fig_dir"
  for f in "$EXTRACTED_DIR/${prefix}"*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    orig_name="${fname#$prefix}"
    cp "$f" "docs/assets/cmo_paper/$fig_dir/$orig_name"
  done
done

# ── Reproducibility scripts ──────────────────────────────────────────
mkdir -p examples/notebooks
for f in "$EXTRACTED_DIR"/generate_fig_*.py \
         "$EXTRACTED_DIR"/audit_paper_numbers.py \
         "$EXTRACTED_DIR"/sensitivity_coupling_norm.py \
         "$EXTRACTED_DIR"/parameter_identifiability.py \
         "$EXTRACTED_DIR"/zenodo_fetch.py \
         "$EXTRACTED_DIR"/pycaso_schur_regularized_ba.py; do
  [ -f "$f" ] && cp "$f" examples/notebooks/
done

# ── Key data files ────────────────────────────────────────────────────
mkdir -p docs/assets/pycaso_real_data/schur_ba
for f in intermediate_state.npz summary.json \
         schur_ba_diagnostic.json \
         multi_oracle_coupling.json corner_ba_refinement.json \
         bic_model_selection.json validation_experiments.json \
         zernike_order_sweep.json \
         specimen_correspondences.npz \
         specimen_reconstruction_cmo26.npz \
         specimen_reconstruction_zernike.npz \
         specimen_reconstruction_metrics.json \
         specimen_dataset_inventory.json; do
  if [ -f "$EXTRACTED_DIR/$f" ]; then
    fname=$(basename "$f")
    cp "$EXTRACTED_DIR/$f" "docs/assets/pycaso_real_data/$fname"
  fi
done

[ -f "$EXTRACTED_DIR/schur_ba_diagnostic.json" ] && \
  cp "$EXTRACTED_DIR/schur_ba_diagnostic.json" docs/assets/pycaso_real_data/schur_ba/

# ── Audit ─────────────────────────────────────────────────────────────
mkdir -p docs/assets/cmo_paper
[ -f "$EXTRACTED_DIR/AUDIT.md" ] && cp "$EXTRACTED_DIR/AUDIT.md" docs/assets/cmo_paper/

# ── Note about heavy Schur BA data (separate Zenodo) ─────────────────
if [ ! -f docs/assets/pycaso_real_data/schur_ba/optical_ba_unregularized.json ]; then
  echo ""
  echo "NOTE: Heavy Schur BA snapshots (~140 MB) are archived separately at"
  echo "  https://doi.org/10.5281/zenodo.20369312"
  echo "Run 'python examples/notebooks/zenodo_fetch.py' to download them."
  echo "They are only needed for figures 8 and 12."
fi

echo ""
echo "=== Done. You can now: ==="
echo "  # Rebuild the PDF"
echo "  cd paper/cmo && bash build_pdflatex.sh"
echo ""
echo "  # Run the numerical audit"
echo "  python examples/notebooks/audit_paper_numbers.py"
echo ""
echo "  # Regenerate a figure (example: Figure 1)"
echo "  python examples/notebooks/generate_fig_cmo_physical.py"
