from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _monomial_powers_4vars(max_degree: int) -> np.ndarray:
    """
    Enumerate monomial powers (a,b,c,d) for 4 variables with total degree <= max_degree.
    Returned shape: (M,4), deterministic order.
    """
    p = int(max_degree)
    if p < 0:
        raise ValueError("max_degree must be >= 0")
    powers = []
    for total in range(p + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                for c in range(total - a - b + 1):
                    d = total - a - b - c
                    powers.append((a, b, c, d))
    return np.asarray(powers, dtype=np.int32)


def _poly_design_matrix_4vars(X: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """
    Build a polynomial design matrix for 4-variable inputs.

    X: (N,4)
    powers: (M,4)
    Returns: A (N,M) with A[n,m] = prod_j X[n,j] ** powers[m,j]
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 4)
    powers = np.asarray(powers, dtype=np.int32).reshape(-1, 4)
    # Compute as product of per-variable powers, vectorized.
    A = np.ones((X.shape[0], powers.shape[0]), dtype=np.float64)
    for j in range(4):
        pj = powers[:, j].reshape(1, -1)
        if np.max(pj) == 0:
            continue
        A *= X[:, j : j + 1] ** pj
    return A


@dataclass(frozen=True)
class SoloffPolynomialModel:
    """
    A Pycaso/Soloff-style direct polynomial mapping from stereo pixels to 3D.

    We fit three polynomial regressors:
      X = f_x(uL,vL,uR,vR)
      Y = f_y(uL,vL,uR,vR)
      Z = f_z(uL,vL,uR,vR)
    """

    degree: int
    powers: np.ndarray  # (M,4)
    x_mean: np.ndarray  # (4,)
    x_scale: np.ndarray  # (4,)
    y_mean: np.ndarray  # (3,)
    y_scale: np.ndarray  # (3,)
    coeffs_X: np.ndarray  # (M,)
    coeffs_Y: np.ndarray  # (M,)
    coeffs_Z: np.ndarray  # (M,)

    @classmethod
    def fit(
        cls,
        uv_left_px: np.ndarray,
        uv_right_px: np.ndarray,
        XYZ_mm: np.ndarray,
        *,
        degree: int = 3,
        ridge: float = 0.0,
    ) -> "SoloffPolynomialModel":
        uv_left_px = np.asarray(uv_left_px, dtype=np.float64).reshape(-1, 2)
        uv_right_px = np.asarray(uv_right_px, dtype=np.float64).reshape(-1, 2)
        XYZ_mm = np.asarray(XYZ_mm, dtype=np.float64).reshape(-1, 3)
        if uv_left_px.shape[0] != uv_right_px.shape[0] or uv_left_px.shape[0] != XYZ_mm.shape[0]:
            raise ValueError("Input sizes must match")

        powers = _monomial_powers_4vars(int(degree))
        X4 = np.concatenate([uv_left_px, uv_right_px], axis=1)

        # Normalize inputs/outputs for numerical stability (important for large pixel/mm ranges).
        x_mean = np.mean(X4, axis=0)
        x_scale = np.std(X4, axis=0)
        x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
        X4n = (X4 - x_mean) / x_scale

        y_mean = np.mean(XYZ_mm, axis=0)
        y_scale = np.std(XYZ_mm, axis=0)
        y_scale = np.where(y_scale > 1e-12, y_scale, 1.0)
        Yn = (XYZ_mm - y_mean) / y_scale

        A = _poly_design_matrix_4vars(X4n, powers)

        if ridge < 0:
            raise ValueError("ridge must be >= 0")
        if ridge > 0:
            ATA = A.T @ A + float(ridge) * np.eye(A.shape[1], dtype=np.float64)
            AT = A.T
            coeffs_X = np.linalg.solve(ATA, AT @ Yn[:, 0])
            coeffs_Y = np.linalg.solve(ATA, AT @ Yn[:, 1])
            coeffs_Z = np.linalg.solve(ATA, AT @ Yn[:, 2])
        else:
            coeffs_X, *_ = np.linalg.lstsq(A, Yn[:, 0], rcond=None)
            coeffs_Y, *_ = np.linalg.lstsq(A, Yn[:, 1], rcond=None)
            coeffs_Z, *_ = np.linalg.lstsq(A, Yn[:, 2], rcond=None)

        return cls(
            degree=int(degree),
            powers=powers,
            x_mean=np.asarray(x_mean, dtype=np.float64).reshape(4),
            x_scale=np.asarray(x_scale, dtype=np.float64).reshape(4),
            y_mean=np.asarray(y_mean, dtype=np.float64).reshape(3),
            y_scale=np.asarray(y_scale, dtype=np.float64).reshape(3),
            coeffs_X=np.asarray(coeffs_X, dtype=np.float64).reshape(-1),
            coeffs_Y=np.asarray(coeffs_Y, dtype=np.float64).reshape(-1),
            coeffs_Z=np.asarray(coeffs_Z, dtype=np.float64).reshape(-1),
        )

    def predict(self, uv_left_px: np.ndarray, uv_right_px: np.ndarray) -> np.ndarray:
        uv_left_px = np.asarray(uv_left_px, dtype=np.float64).reshape(-1, 2)
        uv_right_px = np.asarray(uv_right_px, dtype=np.float64).reshape(-1, 2)
        if uv_left_px.shape[0] != uv_right_px.shape[0]:
            raise ValueError("uv_left_px and uv_right_px must have same length")
        X4 = np.concatenate([uv_left_px, uv_right_px], axis=1)
        X4n = (X4 - self.x_mean.reshape(1, 4)) / self.x_scale.reshape(1, 4)
        A = _poly_design_matrix_4vars(X4n, self.powers)
        Xn = A @ self.coeffs_X
        Yn = A @ self.coeffs_Y
        Zn = A @ self.coeffs_Z
        X = Xn * float(self.y_scale[0]) + float(self.y_mean[0])
        Y = Yn * float(self.y_scale[1]) + float(self.y_mean[1])
        Z = Zn * float(self.y_scale[2]) + float(self.y_mean[2])
        return np.stack([X, Y, Z], axis=1)
