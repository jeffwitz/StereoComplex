# Submission notes — Optics and Lasers in Engineering (OLE)

Working notes documenting the editorial conformance pass for the OLE submission
of *Rayfield-Based Calibration and Effective Physical Model Identification of a
Common Main Objective Stereo Microscope*. Not part of the submitted manuscript.

## Abstract

- Length: **210 words** (units counted; macro-stripped count 205). Target < 250: **met**.
- "reveals which physical degree of freedom is missing" softened to
  "guides compact physical-model construction".
- All key numbers retained: 1.06 px, > 300 px (OpenCV), 0.28 px, 0.18 px
  (Soloff floor), ~10 % relief agreement.
- Schur coupling norm c = 0.98 dropped from the abstract for brevity (retained
  in the body).

## Figures — resolution report

All figures referenced by `manuscript.tex` are now **vector PDF**, so there is
no raster-resolution risk for Elsevier.

| # | File | Type | Status |
|---|---|---|---|
| 1 | cmo_physical.pdf | vector PDF | OK |
| 2 | pipeline.pdf | vector PDF | OK |
| 3 | BIC_vs_order.pdf | vector PDF | OK |
| 4 | dy_profile_comparison.pdf | vector PDF | OK |
| 5 | residual_evolution.pdf | vector PDF | OK |
| 6 | se3_arm_alignment.pdf | vector PDF | OK |
| 7 | schur_singular_values.pdf | vector PDF | OK |
| 8 | specimen_reconstruction.pdf | vector PDF | OK |
| 9 | zernike_cmo_rigid_removed.pdf | vector PDF | OK |
| 10 | zernike_stage_prior.pdf | vector PDF | OK |
| 11 | profilo_relief_comparison.pdf | vector PDF (relief panels embedded at native res, 300 dpi) | OK — converted from a 150-dpi PNG in this pass |
| 12 | bic_bars.pdf | vector PDF | OK |

Note: `profilo_relief_comparison` was previously the only raster figure
(1489×2088 PNG, ~250 dpi effective at the printed size, below Elsevier's
300-dpi halftone floor). It is now rendered to PDF (vector axes/text + relief
panels embedded at native resolution). If the editor later requests even higher
panel resolution, re-run `examples/notebooks/generate_fig_profilo_relief_comparison.py`
with a higher `dpi` in the `savefig` calls.

## Highlights

`highlights_ole.txt`, 5 bullets, each ≤ 85 characters (max observed 63).

## End-matter section order (Elsevier-compatible)

Reproducibility Statement → Appendices (A, B) → Author Contributions → Funding →
Acknowledgments → Declaration of generative AI → Code and Data Availability →
References.

- The generative-AI declaration was moved out of Acknowledgments into its own
  dedicated section before the references.
- Acknowledgments reduced to a sober statement (Pycaso dataset + Python ecosystem).
- Funding statement added (CNRS / LaMcube institutional context, no specific grant).

## Reproducibility / DOI consistency

- GitHub: https://github.com/jeffwitz/StereoComplex
- Pycaso dataset: https://github.com/LaboratoireMecaniqueLille/Pycaso
- Zenodo concept DOI: 10.5281/zenodo.20444215 (consistent across manuscript + checklist)
- Zenodo version DOI: 10.5281/zenodo.21025322 (v8, consistent across manuscript + checklist)
- Added a Zenodo bullet to the Code and Data Availability section.

## Sweep results

- No real "Optics Express" / "Optica" mention in the submission files
  (manuscript, cover letter, highlights, checklist). The only match is the
  checklist meta-item asserting their absence.
- No `TODO` / `placeholder` / `XXXX` / `FIXME` in submission files.
- LaTeX build: 46 pages, 0 undefined references, 0 undefined citations,
  0 overfull hbox > 40 pt.

## Human checks still pending

- Confirm corresponding-author email is active.
- Confirm the final Zenodo version corresponds to the submitted PDF.
- Confirm the Funding wording is acceptable to the author.
- Confirm the generative-AI declaration wording is acceptable to the author.
- Confirm no concurrent submission elsewhere.
