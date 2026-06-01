#!/usr/bin/env bash
# Rebuild the CMO paper from the Zenodo archive + StereoComplex repo.
#
# Usage:
#   1. git clone https://github.com/jeffwitz/StereoComplex
#   2. cd StereoComplex
#   3. bash rebuild_from_zenodo.sh
#
# Downloads the Zenodo archive (DOI 10.5281/zenodo.20444216) and restores
# the expected directory structure so that all figure generators, the
# numerical audit, and the LaTeX build can run.

set -euo pipefail

ZENODO_URL="https://zenodo.org/api/records/20444216/files-archive"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== Downloading Zenodo archive ..."
curl -sL "$ZENODO_URL" -o "$TMPDIR/zenodo.zip"
unzip -qo "$TMPDIR/zenodo.zip" -d "$TMPDIR/extracted"

echo "=== Restoring paper/cmo/ structure ..."
mkdir -p paper/cmo/figures paper/cmo/tables paper/cmo/build

# --- Core manuscript files ---
for f in manuscript.tex references.bib build_pdflatex.sh cover_letter.txt \
         generate_tables.py check_manuscript_numbers.py number_audit_report.md \
         SUBMISSION_CHECKLIST.md; do
  [ -f "$TMPDIR/extracted/$f" ] && cp "$TMPDIR/extracted/$f" paper/cmo/
done

# --- Build artifacts ---
[ -f "$TMPDIR/extracted/manuscript.pdf" ] && cp "$TMPDIR/extracted/manuscript.pdf" paper/cmo/
[ -f "$TMPDIR/extracted/manuscript.log" ] && cp "$TMPDIR/extracted/manuscript.log" paper/cmo/build/

# --- Figures (PDF + PNG, skip manuscript.pdf) ---
for f in "$TMPDIR/extracted"/*.pdf "$TMPDIR/extracted"/*.png; do
  fname=$(basename "$f")
  [ "$fname" = "manuscript.pdf" ] && continue
  cp "$f" "paper/cmo/figures/"
done

# --- Tables (tex files, skip manuscript.tex) ---
for f in "$TMPDIR/extracted"/*.tex; do
  fname=$(basename "$f")
  [ "$fname" = "manuscript.tex" ] && continue
  cp "$f" "paper/cmo/tables/"
done

# --- Figure asset folders ---
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
  for f in "$TMPDIR/extracted/${prefix}"*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    orig_name="${fname#$prefix}"
    cp "$f" "docs/assets/cmo_paper/$fig_dir/$orig_name"
  done
done

# --- Reproducibility scripts ---
mkdir -p examples/notebooks
for f in "$TMPDIR/extracted"/generate_fig_*.py \
         "$TMPDIR/extracted"/audit_paper_numbers.py \
         "$TMPDIR/extracted"/sensitivity_coupling_norm.py \
         "$TMPDIR/extracted"/parameter_identifiability.py \
         "$TMPDIR/extracted"/zenodo_fetch.py \
         "$TMPDIR/extracted"/pycaso_schur_regularized_ba.py; do
  [ -f "$f" ] && cp "$f" examples/notebooks/
done

# --- Key data files ---
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
  if [ -f "$TMPDIR/extracted/$f" ]; then
    fname=$(basename "$f")
    cp "$TMPDIR/extracted/$f" "docs/assets/pycaso_real_data/$fname"
  fi
done

# schur_ba_diagnostic also goes to schur_ba/ subdir
[ -f "$TMPDIR/extracted/schur_ba_diagnostic.json" ] && \
  cp "$TMPDIR/extracted/schur_ba_diagnostic.json" docs/assets/pycaso_real_data/schur_ba/

# --- Audit ---
mkdir -p docs/assets/cmo_paper
[ -f "$TMPDIR/extracted/AUDIT.md" ] && cp "$TMPDIR/extracted/AUDIT.md" docs/assets/cmo_paper/

# --- Note about heavy Schur BA data ---
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
echo ""
echo "  # Regenerate all figures (requires heavy Schur BA data)"
echo "  for f in examples/notebooks/generate_fig_*.py; do python \$f; done"
