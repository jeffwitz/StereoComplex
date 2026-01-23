from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _monomial_powers_3vars(max_degree: int) -> np.ndarray:
    """
    Enumerate monomial powers (a,b,c) for 3 variables with total degree <= max_degree.
    Returned shape: (M,3), deterministic order.
    """
    p = int(max_degree)
    if p < 0:
        raise ValueError("max_degree must be >= 0")
    powers: list[tuple[int, int, int]] = []
    for total in range(p + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                c = total - a - b
                powers.append((a, b, c))
    return np.asarray(powers, dtype=np.int32)


def _poly_design_matrix_3vars(X: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """
    X: (N,3)
    powers: (M,3)
    Returns: A (N,M) with A[n,m] = prod_j X[n,j] ** powers[m,j]
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    powers = np.asarray(powers, dtype=np.int32).reshape(-1, 3)
    A = np.ones((X.shape[0], powers.shape[0]), dtype=np.float64)
    for j in range(3):
        pj = powers[:, j].reshape(1, -1)
        if int(np.max(pj)) == 0:
            continue
        A *= X[:, j : j + 1] ** pj
    return A


def _poly_features_and_jacobian_3vars(xyz: np.ndarray, powers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For a single xyz point, return (phi, Jphi) where:
      - phi: (M,)
      - Jphi: (M,3) with columns dphi/dx, dphi/dy, dphi/dz
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    powers = np.asarray(powers, dtype=np.int32).reshape(-1, 3)
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    ax = powers[:, 0].astype(np.int32)
    by = powers[:, 1].astype(np.int32)
    cz = powers[:, 2].astype(np.int32)

    # Feature values.
    x_a = np.power(x, ax, dtype=np.float64)
    y_b = np.power(y, by, dtype=np.float64)
    z_c = np.power(z, cz, dtype=np.float64)
    phi = x_a * y_b * z_c

    # Derivatives.
    dphi_dx = np.zeros_like(phi)
    dphi_dy = np.zeros_like(phi)
    dphi_dz = np.zeros_like(phi)

    mask = ax > 0
    if np.any(mask):
        dphi_dx[mask] = ax[mask] * np.power(x, ax[mask] - 1, dtype=np.float64) * y_b[mask] * z_c[mask]
    mask = by > 0
    if np.any(mask):
        dphi_dy[mask] = by[mask] * x_a[mask] * np.power(y, by[mask] - 1, dtype=np.float64) * z_c[mask]
    mask = cz > 0
    if np.any(mask):
        dphi_dz[mask] = cz[mask] * x_a[mask] * y_b[mask] * np.power(z, cz[mask] - 1, dtype=np.float64)

    Jphi = np.stack([dphi_dx, dphi_dy, dphi_dz], axis=1)
    return phi, Jphi


@dataclass(frozen=True)
class PycasoSoloffStereoModel:
    """
    Soloff-style calibration:

      C_L = S_C_L(X,Y,Z)
      R_L = S_R_L(X,Y,Z)
      C_R = S_C_R(X,Y,Z)
      R_R = S_R_R(X,Y,Z)

    where each S(.) is a polynomial in (X,Y,Z) with crossed terms up to a chosen degree.

    Identification (reconstruction) solves for (X,Y,Z) given (C_L,R_L,C_R,R_R) by non-linear least squares.
    """

    degree: int
    powers: np.ndarray  # (M,3)
    xyz_mean: np.ndarray  # (3,)
    xyz_scale: np.ndarray  # (3,)
    coeffs_Cl: np.ndarray  # (M,)
    coeffs_Rl: np.ndarray  # (M,)
    coeffs_Cr: np.ndarray  # (M,)
    coeffs_Rr: np.ndarray  # (M,)
    # Degree-1 coefficients used for initialization.
    coeffs1_Cl: np.ndarray  # (4,) for [1,X,Y,Z]
    coeffs1_Rl: np.ndarray
    coeffs1_Cr: np.ndarray
    coeffs1_Rr: np.ndarray

    @classmethod
    def fit(
        cls,
        *,
        XYZ_mm: np.ndarray,
        uv_left_px: np.ndarray,
        uv_right_px: np.ndarray,
        degree: int = 3,
        ridge: float = 0.0,
    ) -> "PycasoSoloffStereoModel":
        XYZ_mm = np.asarray(XYZ_mm, dtype=np.float64).reshape(-1, 3)
        uv_left_px = np.asarray(uv_left_px, dtype=np.float64).reshape(-1, 2)
        uv_right_px = np.asarray(uv_right_px, dtype=np.float64).reshape(-1, 2)
        if XYZ_mm.shape[0] != uv_left_px.shape[0] or XYZ_mm.shape[0] != uv_right_px.shape[0]:
            raise ValueError("Input sizes must match")

        # Normalize XYZ for numerical stability (critical when coordinates span large ranges).
        xyz_mean = np.mean(XYZ_mm, axis=0)
        xyz_scale = np.std(XYZ_mm, axis=0)
        xyz_scale = np.where(xyz_scale > 1e-12, xyz_scale, 1.0)
        XYZ_n = (XYZ_mm - xyz_mean) / xyz_scale

        p = int(degree)
        powers = _monomial_powers_3vars(p)
        A = _poly_design_matrix_3vars(XYZ_n, powers)

        if ridge < 0:
            raise ValueError("ridge must be >= 0")
        if ridge > 0:
            ATA = A.T @ A + float(ridge) * np.eye(A.shape[1], dtype=np.float64)
            AT = A.T
            coeffs_Cl = np.linalg.solve(ATA, AT @ uv_left_px[:, 0])
            coeffs_Rl = np.linalg.solve(ATA, AT @ uv_left_px[:, 1])
            coeffs_Cr = np.linalg.solve(ATA, AT @ uv_right_px[:, 0])
            coeffs_Rr = np.linalg.solve(ATA, AT @ uv_right_px[:, 1])
        else:
            coeffs_Cl, *_ = np.linalg.lstsq(A, uv_left_px[:, 0], rcond=None)
            coeffs_Rl, *_ = np.linalg.lstsq(A, uv_left_px[:, 1], rcond=None)
            coeffs_Cr, *_ = np.linalg.lstsq(A, uv_right_px[:, 0], rcond=None)
            coeffs_Rr, *_ = np.linalg.lstsq(A, uv_right_px[:, 1], rcond=None)

        # Degree-1 fit for initialization: features [1, Xn, Yn, Zn].
        A1 = np.concatenate([np.ones((XYZ_n.shape[0], 1), dtype=np.float64), XYZ_n], axis=1)
        if ridge > 0:
            ATA1 = A1.T @ A1 + float(ridge) * np.eye(4, dtype=np.float64)
            AT1 = A1.T
            coeffs1_Cl = np.linalg.solve(ATA1, AT1 @ uv_left_px[:, 0])
            coeffs1_Rl = np.linalg.solve(ATA1, AT1 @ uv_left_px[:, 1])
            coeffs1_Cr = np.linalg.solve(ATA1, AT1 @ uv_right_px[:, 0])
            coeffs1_Rr = np.linalg.solve(ATA1, AT1 @ uv_right_px[:, 1])
        else:
            coeffs1_Cl, *_ = np.linalg.lstsq(A1, uv_left_px[:, 0], rcond=None)
            coeffs1_Rl, *_ = np.linalg.lstsq(A1, uv_left_px[:, 1], rcond=None)
            coeffs1_Cr, *_ = np.linalg.lstsq(A1, uv_right_px[:, 0], rcond=None)
            coeffs1_Rr, *_ = np.linalg.lstsq(A1, uv_right_px[:, 1], rcond=None)

        return cls(
            degree=p,
            powers=powers,
            xyz_mean=np.asarray(xyz_mean, dtype=np.float64).reshape(3),
            xyz_scale=np.asarray(xyz_scale, dtype=np.float64).reshape(3),
            coeffs_Cl=np.asarray(coeffs_Cl, dtype=np.float64).reshape(-1),
            coeffs_Rl=np.asarray(coeffs_Rl, dtype=np.float64).reshape(-1),
            coeffs_Cr=np.asarray(coeffs_Cr, dtype=np.float64).reshape(-1),
            coeffs_Rr=np.asarray(coeffs_Rr, dtype=np.float64).reshape(-1),
            coeffs1_Cl=np.asarray(coeffs1_Cl, dtype=np.float64).reshape(-1),
            coeffs1_Rl=np.asarray(coeffs1_Rl, dtype=np.float64).reshape(-1),
            coeffs1_Cr=np.asarray(coeffs1_Cr, dtype=np.float64).reshape(-1),
            coeffs1_Rr=np.asarray(coeffs1_Rr, dtype=np.float64).reshape(-1),
        )

    def _xyz_norm(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        return (xyz - self.xyz_mean) / self.xyz_scale

    def _predict_all(self, xyz: np.ndarray) -> np.ndarray:
        phi, _Jphi = _poly_features_and_jacobian_3vars(self._xyz_norm(xyz), self.powers)
        return np.array(
            [
                float(phi @ self.coeffs_Cl),
                float(phi @ self.coeffs_Rl),
                float(phi @ self.coeffs_Cr),
                float(phi @ self.coeffs_Rr),
            ],
            dtype=np.float64,
        )

    def _predict_all_with_jac(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyz_n = self._xyz_norm(xyz)
        phi, Jphi = _poly_features_and_jacobian_3vars(xyz_n, self.powers)
        pred = np.array(
            [
                float(phi @ self.coeffs_Cl),
                float(phi @ self.coeffs_Rl),
                float(phi @ self.coeffs_Cr),
                float(phi @ self.coeffs_Rr),
            ],
            dtype=np.float64,
        )
        # Jacobian in normalized xyz: (4,3)
        J = np.stack(
            [
                Jphi.T @ self.coeffs_Cl,
                Jphi.T @ self.coeffs_Rl,
                Jphi.T @ self.coeffs_Cr,
                Jphi.T @ self.coeffs_Rr,
            ],
            axis=0,
        )
        # Chain rule to physical xyz (xyz_n = (xyz - mean)/scale).
        J = J / self.xyz_scale.reshape(1, 3)
        return pred, J

    def initial_guess(self, uv_left_px: np.ndarray, uv_right_px: np.ndarray) -> np.ndarray:
        """
        Linear initialization from the degree-1 model.

        We solve a 4x3 least-squares problem:
            [C_L, R_L, C_R, R_R]^T - c0  ≈  B [X,Y,Z]^T
        """
        uv_left_px = np.asarray(uv_left_px, dtype=np.float64).reshape(2)
        uv_right_px = np.asarray(uv_right_px, dtype=np.float64).reshape(2)
        obs = np.array([uv_left_px[0], uv_left_px[1], uv_right_px[0], uv_right_px[1]], dtype=np.float64)

        c0 = np.array([self.coeffs1_Cl[0], self.coeffs1_Rl[0], self.coeffs1_Cr[0], self.coeffs1_Rr[0]], dtype=np.float64)
        B = np.array(
            [
                self.coeffs1_Cl[1:4],
                self.coeffs1_Rl[1:4],
                self.coeffs1_Cr[1:4],
                self.coeffs1_Rr[1:4],
            ],
            dtype=np.float64,
        )  # (4,3)
        x0_n, *_ = np.linalg.lstsq(B, obs - c0, rcond=None)
        x0 = x0_n * self.xyz_scale + self.xyz_mean
        return np.asarray(x0, dtype=np.float64).reshape(3)

    def solve(
        self,
        uv_left_px: np.ndarray,
        uv_right_px: np.ndarray,
        *,
        max_nfev: int = 50,
    ) -> np.ndarray:
        """
        Solve for XYZ from measured stereo pixel coordinates.
        """
        from scipy.optimize import least_squares  # type: ignore

        uv_left_px = np.asarray(uv_left_px, dtype=np.float64).reshape(-1, 2)
        uv_right_px = np.asarray(uv_right_px, dtype=np.float64).reshape(-1, 2)
        if uv_left_px.shape[0] != uv_right_px.shape[0]:
            raise ValueError("uv_left_px and uv_right_px must have same length")

        out = np.zeros((uv_left_px.shape[0], 3), dtype=np.float64)
        for i in range(uv_left_px.shape[0]):
            obs = np.array(
                [uv_left_px[i, 0], uv_left_px[i, 1], uv_right_px[i, 0], uv_right_px[i, 1]],
                dtype=np.float64,
            )

            x0 = self.initial_guess(uv_left_px[i], uv_right_px[i])

            def fun(x):
                return self._predict_all(x) - obs

            def jac(x):
                _pred, J = self._predict_all_with_jac(x)
                return J

            res = least_squares(fun, x0=x0, jac=jac, method="lm", max_nfev=int(max_nfev))
            out[i] = np.asarray(res.x, dtype=np.float64).reshape(3)

        return out
