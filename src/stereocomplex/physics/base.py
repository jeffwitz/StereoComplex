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
        ...

    def parameter_vector(self) -> np.ndarray:
        ...

    @classmethod
    def from_parameter_vector(cls, x: np.ndarray, **kwargs) -> Self:
        ...

    def parameter_dict(self) -> dict[str, float]:
        ...

    def ray(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


__all__ = ["PhysicalRayFieldModel"]
