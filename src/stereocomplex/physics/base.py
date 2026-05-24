from __future__ import annotations

from typing import Protocol, Self

import numpy as np


class PhysicalRayFieldModel(Protocol):
    """Protocol for physical optical candidates compared in ray space.

    A model maps image coordinates to 3D lines in one camera/channel frame. The
    returned origin is only one representative point on each line; candidates
    are compared through line geometry, not through raw origin equality.
    """

    name: str
    is_stereo_shared: bool

    @property
    def n_parameters(self) -> int:
        """Number of free parameters for model selection."""
        ...

    def parameter_vector(self) -> np.ndarray:
        """Pack model parameters into a flat vector for optimisation."""
        ...

    @classmethod
    def from_parameter_vector(cls, x: np.ndarray, **kwargs) -> Self:
        """Reconstruct model from a parameter vector."""
        ...

    def parameter_dict(self) -> dict[str, float]:
        """Model parameters as a dict keyed by coefficient name."""
        ...

    def ray(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute ray (origin, direction) for a pixel."""
        ...


__all__ = ["PhysicalRayFieldModel"]
