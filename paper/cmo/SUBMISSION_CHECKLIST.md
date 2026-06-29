# Submission checklist — Optics and Lasers in Engineering

Target journal: **Optics and Lasers in Engineering** (Elsevier).
Manuscript: `paper/cmo/manuscript.tex`. Title: *Rayfield-Based Calibration and
Effective Physical Model Identification of a Common Main Objective Stereo
Microscope*.

## Manuscript

- [x] `manuscript.tex` compiles cleanly (0 undefined refs, 0 undefined citations, 0 overfull > 40 pt)
- [x] abstract < 250 words (~210 words; see `submission_notes_ole.md`)
- [x] keywords present
- [x] all `\cite{}` resolve
- [x] all `\ref{}` / `\label{}` resolve
- [x] all figures present and embedded
- [x] all tables present
- [x] author affiliation correct (CNRS, Univ. Lille, Centrale Lille, UMR 9013 -- LaMcube)
- [x] ORCID present (0000-0002-7240-9476)
- [ ] corresponding author email active (human check: jean-francois.witz@centralelille.fr)
- [x] author contributions present
- [x] funding statement present
- [x] acknowledgments present (minimal, no AI text)
- [x] generative AI declaration present (dedicated section, before references)
- [x] code/data availability present (with Zenodo concept + version DOI)
- [x] reproducibility statement present
- [x] no remaining mention of Optics Express in submission files

## Submission files

- [x] manuscript source `manuscript.tex`
- [x] compiled PDF `manuscript.pdf`
- [x] bibliography `references.bib`
- [x] all figure files (`figures/`)
- [x] all table inputs (`tables/`)
- [x] cover letter for OLE (`cover_letter.txt`)
- [x] highlights file (`highlights_ole.txt`)
- [x] reproducibility archive available (Zenodo v8)

## Highlights

- [x] 5 bullet points
- [x] each bullet <= 85 characters (max 63; verified)
- [x] file name contains `highlights` (`highlights_ole.txt`)

## Figures

- [x] vector PDFs for all line drawings and plots (11 of 12 figures)
- [x] remaining figure (`profilo_relief_comparison.pdf`) is now vector PDF (axes/text vector + relief panels embedded at native resolution, 300 dpi)
- [x] no missing figure files
- [x] captions present in manuscript

## Reproducibility

- [x] GitHub repository URL present (https://github.com/jeffwitz/StereoComplex)
- [x] Pycaso dataset URL present (https://github.com/LaboratoireMecaniqueLille/Pycaso)
- [x] Zenodo concept DOI present (10.5281/zenodo.20444215)
- [x] Zenodo version DOI present (10.5281/zenodo.21025322, v8)
- [x] intermediate state documented (`intermediate_state.npz`)
- [x] reproduction scripts documented (`paper/cmo/REPRODUCE.md`, figure generators)

## Final human checks

- [ ] confirm corresponding-author email is active
- [ ] confirm the final Zenodo version corresponds to the submitted PDF
- [ ] confirm the Funding statement wording is acceptable to the author
- [ ] confirm the generative-AI declaration wording is acceptable to the author
- [ ] confirm no confidential data in the submission
- [ ] confirm no concurrent submission elsewhere
- [ ] decide whether to raise `profilo_relief_comparison` raster panels above 300 dpi if the editor requests it
