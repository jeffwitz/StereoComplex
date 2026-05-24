"""Observability-aware bundle adjustment helpers for the CMO Pycaso work.

This sub-package implements the diagnostic and regularization tools described
in ``CdC_BA_optique_Schur_CMO_Pycaso.md``:

- :mod:`stereocomplex.optical_ba.fisher` builds a scaled, block-partitioned
  Fisher (Gauss-Newton) approximation of the bundle-adjustment Hessian;
- :mod:`stereocomplex.optical_ba.schur` computes the Schur complement of that
  Fisher with respect to the pose block and exposes its weak observability
  modes.

The naming convention follows the CDC: ``theta`` denotes the optical
parameters, ``eta`` denotes the per-frame pose parameters.
"""

from stereocomplex.optical_ba.fisher import (
    FisherBlocks,
    build_fisher_blocks,
    finite_difference_jacobian_scaled,
)
from stereocomplex.optical_ba.priors import (
    SchurPrior,
    isotropic_prior_residuals,
    schur_prior_residuals,
)
from stereocomplex.optical_ba.regularized_ba import (
    OpticalBAResult,
    default_bounds,
    run_optical_ba,
    run_schur_regularized_optical_ba,
)
from stereocomplex.optical_ba.residuals import (
    PycasoCMOObservations,
    default_parameter_scales,
    point_to_ray_residuals_cmo_se3,
    reprojection_residuals_cmo_se3,
)
from stereocomplex.optical_ba.schur import (
    SchurDiagnostic,
    coupling_norm_schur,
    diagnose_schur_modes,
    schur_complement_theta,
)
from stereocomplex.optical_ba.specimen import (
    EUR_2_CENT_DIAMETER_MM,
    SpecimenReconstruction,
    load_specimen_npz,
    load_zernike_baseline,
    magnification_ratio,
    plot_specimen_grid,
    reconstruct_with_cmo_se3,
)

__all__ = [
    "EUR_2_CENT_DIAMETER_MM",
    "FisherBlocks",
    "OpticalBAResult",
    "PycasoCMOObservations",
    "SchurDiagnostic",
    "SpecimenReconstruction",
    "build_fisher_blocks",
    "coupling_norm_schur",
    "default_bounds",
    "default_parameter_scales",
    "diagnose_schur_modes",
    "finite_difference_jacobian_scaled",
    "load_specimen_npz",
    "load_zernike_baseline",
    "magnification_ratio",
    "plot_specimen_grid",
    "point_to_ray_residuals_cmo_se3",
    "reconstruct_with_cmo_se3",
    "reprojection_residuals_cmo_se3",
    "run_optical_ba",
    "run_schur_regularized_optical_ba",
    "schur_complement_theta",
    "SchurPrior",
    "schur_prior_residuals",
    "isotropic_prior_residuals",
]
