# Numeric verification

The current checker is `check_manuscript_numbers.py`. It exits unsuccessfully
if generated tables or macros differ from the recorded JSON, an input hash
changes, inversion fails to close, exact gauge/centrality checks fail, a fit
fails, the synthetic experiment is incomplete, or a required asset is absent.

The prior report's unconditional `OK` labels are superseded. For the scientific
interpretation and the status of unreproduced historical aggregates, see
`REORIENTATION.md` and `supplementary.pdf`.

Verified in this revision: input hashes; generated numbers; all four inverse
mappings with closure below 1e-10 mm; two analytic gauge transformations;
centrality in both reference channels; all model starts and forty synthetic
cases; all referenced figure/table assets. This is a computational consistency
check, not independent validation of microscope accuracy.
