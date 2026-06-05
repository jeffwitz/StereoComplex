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
    ) -> SoloffPolynomialModel:
        """Fit a Soloff polynomial to stereo calibration data.

        Parameters
        ----------
        uv_left_px : ndarray, shape (N, 2)
            Left-channel pixel coordinates.
        uv_right_px : ndarray, shape (N, 2)
            Right-channel pixel coordinates.
        XYZ_mm : ndarray, shape (N, 3)
            3-D calibration points in millimetres.
        degree : int
            Polynomial degree (2 or 3).
        ridge : float
            Ridge (Tikhonov) regularisation strength.

        Returns
        -------
        SoloffPolynomialModel
            Fitted polynomial model with per-channel coefficients.
        """
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
        """Evaluate the fitted polynomial at query points."""
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


def _soloff_monomial_powers(form: int) -> np.ndarray:
    """Enumerate the 3-variable monomial powers of a Soloff polynomial form.

    The Soloff method (Soloff et al., *Meas. Sci. Technol.* 8 (1997) 1441,
    doi:10.1088/0957-0233/8/12/008) models each detector coordinate as a
    polynomial of the 3-D object point. Pycaso encodes the polynomial by a
    three-digit ``form`` code ``d1 d2 d3`` giving the per-axis degree caps for
    ``(x, y, z)``; the retained monomials ``x^i y^j z^k`` are those with
    ``i <= d1``, ``j <= d2``, ``k <= d3`` and total degree ``i+j+k <=
    max(d1, d2, d3)``. Form ``332`` (the Pycaso default) yields 19 monomials —
    cubic laterally, quadratic axially.

    Parameters
    ----------
    form : int
        Three-digit Soloff form code, e.g. ``111`` (4 monomials, linear),
        ``222`` (10), ``332`` (19), ``333`` (20).

    Returns
    -------
    ndarray, shape (M, 3)
        Integer powers ``(i, j, k)`` of ``x, y, z`` for the ``M`` monomials,
        in a deterministic order (ascending total degree).
    """
    digits = [int(c) for c in f"{int(form):03d}"]
    d1, d2, d3 = digits[-3], digits[-2], digits[-1]
    total_cap = max(d1, d2, d3)
    powers = []
    for total in range(total_cap + 1):
        for i in range(min(d1, total) + 1):
            for j in range(min(d2, total - i) + 1):
                k = total - i - j
                if 0 <= k <= d3:
                    powers.append((i, j, k))
    return np.asarray(powers, dtype=np.int32)


def _soloff_design(xyz: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Monomial design matrix ``M`` such that detector coord ``= a . M``.

    Parameters
    ----------
    xyz : ndarray, shape (N, 3)
        Object points in millimetres.
    powers : ndarray, shape (P, 3)
        Monomial powers from :func:`_soloff_monomial_powers`.

    Returns
    -------
    ndarray, shape (N, P)
        ``M[n, p] = x^i y^j z^k`` for point ``n`` and monomial ``p``.
    """
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]
    return (x ** powers[:, 0]) * (y ** powers[:, 1]) * (z ** powers[:, 2])


def _soloff_design_jac(xyz: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Analytic Jacobian ``dM/d(x,y,z)`` of the Soloff design matrix.

    Parameters
    ----------
    xyz : ndarray, shape (N, 3)
        Object points in millimetres.
    powers : ndarray, shape (P, 3)
        Monomial powers.

    Returns
    -------
    ndarray, shape (N, P, 3)
        Partial derivatives of each monomial with respect to ``x, y, z``;
        used by the batched Levenberg--Marquardt inversion.
    """
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]
    i, j, k = powers[:, 0], powers[:, 1], powers[:, 2]
    xi = x ** np.clip(i - 1, 0, None)
    yj = y ** np.clip(j - 1, 0, None)
    zk = z ** np.clip(k - 1, 0, None)
    xfull = x ** i
    yfull = y ** j
    zfull = z ** k
    dx = i * xi * yfull * zfull
    dy = j * xfull * yj * zfull
    dz = k * xfull * yfull * zk
    return np.stack([dx, dy, dz], axis=2)


@dataclass
class SoloffForwardModel:
    """True Soloff stereo calibration: forward polynomials + LM inversion.

    Unlike :class:`SoloffPolynomialModel` (which directly regresses the inverse
    map ``pixels -> XYZ``), this is the *bona fide* Soloff method
    (Soloff et al. 1997, doi:10.1088/0957-0233/8/12/008): it fits the **forward**
    projection of each detector coordinate as a polynomial of the object point,

    ``[u_L, v_L, u_R, v_R]^T = A . M(x, y, z)``,

    and recovers ``(x, y, z)`` from a stereo pixel pair by **nonlinear
    least-squares inversion** (Levenberg--Marquardt). The forward model is
    well-conditioned (a smooth 3-D -> 2-D projection), so the inverse is more
    faithful than a direct inverse-polynomial regression — at microscope
    magnification the difference is measurable on dense specimen relief.

    This is a dependency-free reimplementation of Pycaso's ``Soloff_*`` routines;
    with identical calibration data and form it reproduces Pycaso's identified
    points to machine precision.

    Attributes
    ----------
    form : int
        Soloff polynomial form code (e.g. ``332``).
    powers : ndarray, shape (P, 3)
        Monomial powers of the chosen form.
    coeffs : ndarray, shape (4, P)
        Forward coefficients ``A``; rows are ``u_L, v_L, u_R, v_R``. They act on
        the **normalised** object point ``(xyz - x_mean) / x_scale`` so the
        high-degree monomial design matrix stays well-conditioned.
    coeffs_linear : ndarray, shape (4, 4)
        Linear (form ``111``) coefficients, used only to seed the inversion.
    x_mean, x_scale : ndarray, shape (3,)
        Object-space normalisation (mean and standard deviation, mm) applied
        before evaluating the monomials.
    """

    form: int
    powers: np.ndarray
    coeffs: np.ndarray
    coeffs_linear: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray

    @classmethod
    def fit(
        cls,
        uv_left_px: np.ndarray,
        uv_right_px: np.ndarray,
        xyz_mm: np.ndarray,
        *,
        form: int = 332,
        ridge: float = 0.0,
    ) -> SoloffForwardModel:
        """Fit the forward Soloff polynomials from stereo calibration data.

        Parameters
        ----------
        uv_left_px, uv_right_px : ndarray, shape (N, 2)
            Detected left/right pixel coordinates of the calibration points.
        xyz_mm : ndarray, shape (N, 3)
            Known 3-D object points in millimetres (board ``x, y`` at the
            controlled stage ``z``).
        form : int
            Soloff polynomial form (default ``332``, the Pycaso default).
        ridge : float
            Optional Tikhonov regularisation on the linear least-squares fit.

        Returns
        -------
        SoloffForwardModel
            Calibrated model ready for :meth:`identify`.
        """
        uvl = np.asarray(uv_left_px, dtype=np.float64).reshape(-1, 2)
        uvr = np.asarray(uv_right_px, dtype=np.float64).reshape(-1, 2)
        xyz = np.asarray(xyz_mm, dtype=np.float64).reshape(-1, 3)
        obs = np.column_stack([uvl, uvr])  # (N, 4): u_L, v_L, u_R, v_R

        x_mean = xyz.mean(axis=0)
        x_scale = xyz.std(axis=0)
        x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
        xyz_n = (xyz - x_mean) / x_scale

        def _solve(powers: np.ndarray) -> np.ndarray:
            design = _soloff_design(xyz_n, powers)  # (N, P)
            if ridge > 0:
                gram = design.T @ design + ridge * np.eye(design.shape[1])
                return (np.linalg.solve(gram, design.T @ obs)).T  # (4, P)
            return np.linalg.lstsq(design, obs, rcond=None)[0].T

        powers = _soloff_monomial_powers(form)
        coeffs = _solve(powers)
        coeffs_lin = _solve(_soloff_monomial_powers(111))
        return cls(
            form=int(form), powers=powers, coeffs=coeffs, coeffs_linear=coeffs_lin,
            x_mean=x_mean, x_scale=x_scale,
        )

    def project(self, xyz_mm: np.ndarray) -> np.ndarray:
        """Forward-project object points to the 4 detector coordinates.

        Parameters
        ----------
        xyz_mm : ndarray, shape (N, 3)
            Object points in millimetres.

        Returns
        -------
        ndarray, shape (N, 4)
            Predicted ``u_L, v_L, u_R, v_R`` pixel coordinates.
        """
        xyz = np.asarray(xyz_mm, dtype=np.float64).reshape(-1, 3)
        xyz_n = (xyz - self.x_mean) / self.x_scale
        return _soloff_design(xyz_n, self.powers) @ self.coeffs.T

    def identify(
        self,
        uv_left_px: np.ndarray,
        uv_right_px: np.ndarray,
        *,
        max_iter: int = 80,
        tol_px: float = 1e-4,
    ) -> np.ndarray:
        """Recover object points from stereo pixel pairs by LM inversion.

        Solves, for every correspondence simultaneously, the 4-equation /
        3-unknown nonlinear least-squares problem ``A . M(x,y,z) = [u_L, v_L,
        u_R, v_R]`` with a **batched, analytically-differentiated**
        Levenberg--Marquardt scheme (no per-point Python loop). The seed comes
        from the linear (form ``111``) coefficients.

        Parameters
        ----------
        uv_left_px, uv_right_px : ndarray, shape (N, 2)
            Observed left/right pixel coordinates.
        max_iter : int
            Maximum LM iterations.
        tol_px : float
            Convergence threshold on the per-point reprojection RMS update.

        Returns
        -------
        ndarray, shape (N, 3)
            Recovered object points ``(x, y, z)`` in millimetres.
        """
        uvl = np.asarray(uv_left_px, dtype=np.float64).reshape(-1, 2)
        uvr = np.asarray(uv_right_px, dtype=np.float64).reshape(-1, 2)
        obs = np.column_stack([uvl, uvr])  # (N, 4)

        # The inversion runs in the normalised object space the coeffs act on.
        # Linear seed: obs - a0 = A_lin[:, 1:] @ (xn, yn, zn)
        a_lin = self.coeffs_linear  # (4, 4): columns [1, xn, yn, zn]
        rhs = (obs - a_lin[:, 0][None, :]).T  # (4, N)
        xyz_n = np.linalg.lstsq(a_lin[:, 1:4], rhs, rcond=None)[0].T  # (N, 3)

        lam = np.full(xyz_n.shape[0], 1e-3)
        eye3 = np.eye(3)[None]
        for _ in range(max_iter):
            resid = _soloff_design(xyz_n, self.powers) @ self.coeffs.T - obs  # (N, 4)
            jac = np.einsum("rp,npk->nrk", self.coeffs, _soloff_design_jac(xyz_n, self.powers))
            gram = np.einsum("nrk,nrl->nkl", jac, jac) + lam[:, None, None] * eye3
            grad = np.einsum("nrk,nr->nk", jac, resid)
            step = np.linalg.solve(gram, -grad[..., None])[..., 0]
            cand = xyz_n + step
            resid_c = _soloff_design(cand, self.powers) @ self.coeffs.T - obs
            improved = np.sum(resid_c ** 2, axis=1) < np.sum(resid ** 2, axis=1)
            xyz_n = np.where(improved[:, None], cand, xyz_n)
            lam = np.where(improved, lam * 0.5, lam * 2.0)
            if np.max(np.abs(step)) < tol_px:
                break
        return xyz_n * self.x_scale + self.x_mean
