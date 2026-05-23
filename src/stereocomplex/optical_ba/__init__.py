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
from stereocomplex.optical_ba.regularized_ba import (
    OpticalBAResult,
    default_bounds,
    run_optical_ba,
)
from stereocomplex.optical_ba.residuals import (
    PycasoCMOObservations,
    default_parameter_scales,
    point_to_ray_residuals_cmo_se3,
)
from stereocomplex.optical_ba.schur import (
    SchurDiagnostic,
    coupling_norm_schur,
    diagnose_schur_modes,
    schur_complement_theta,
)

__all__ = [
    "FisherBlocks",
    "OpticalBAResult",
    "PycasoCMOObservations",
    "SchurDiagnostic",
    "build_fisher_blocks",
    "coupling_norm_schur",
    "default_bounds",
    "default_parameter_scales",
    "diagnose_schur_modes",
    "finite_difference_jacobian_scaled",
    "point_to_ray_residuals_cmo_se3",
    "run_optical_ba",
    "schur_complement_theta",
]
