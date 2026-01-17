from __future__ import annotations

import numpy as np


def _fit_affine(obj_xy: np.ndarray, img_uv: np.ndarray) -> np.ndarray | None:
    """
    Fit affine transform (2x3) mapping obj_xy -> img_uv in least squares.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64).reshape(-1, 2)
    img_uv = np.asarray(img_uv, dtype=np.float64).reshape(-1, 2)
    if obj_xy.shape[0] < 3:
        return None
    A = np.concatenate([obj_xy, np.ones((obj_xy.shape[0], 1), dtype=np.float64)], axis=1)  # (N,3)
    try:
        M, *_ = np.linalg.lstsq(A, img_uv, rcond=None)  # (3,2)
    except np.linalg.LinAlgError:
        return None
    return M.T.astype(np.float64)  # (2,3)


def _apply_affine(M: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    query_xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    qh = np.concatenate([query_xy, np.ones((query_xy.shape[0], 1), dtype=np.float64)], axis=1)  # (M,3)
    return (qh @ M.T).astype(np.float64)


def _predict_points_tps_irls(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    *,
    lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> np.ndarray:
    """
    Thin-plate spline warp (2D->2D) with robust IRLS weights.

    This is a weighted TPS smoothing spline where the diagonal regularization term becomes
    `lam * W^{-1}`. With IRLS, weights are updated from the pointwise residuals using a
    Huber rule, down-weighting outliers.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64).reshape(-1, 2)
    img_uv = np.asarray(img_uv, dtype=np.float64).reshape(-1, 2)
    query_xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have the same length")

    N = int(obj_xy.shape[0])
    if N < 6:
        M = _fit_affine(obj_xy, img_uv)
        if M is None:
            return np.full((query_xy.shape[0], 2), np.nan, dtype=np.float64)
        return _apply_affine(M, query_xy)

    m = np.mean(obj_xy, axis=0)
    d = np.sqrt(np.sum((obj_xy - m[None, :]) ** 2, axis=1))
    s = float(np.median(d) + 1e-12)
    X = (obj_xy - m[None, :]) / s
    Q = (query_xy - m[None, :]) / s

    def U(r2: np.ndarray) -> np.ndarray:
        # U(r) = r^2 log(r^2), with U(0)=0.
        r2 = np.asarray(r2, dtype=np.float64)
        out = np.zeros_like(r2)
        mask = r2 > 1e-18
        out[mask] = r2[mask] * np.log(r2[mask])
        return out

    dx = X[:, 0:1] - X[:, 0:1].T
    dy = X[:, 1:2] - X[:, 1:2].T
    K = U(dx * dx + dy * dy)
    P = np.concatenate([np.ones((N, 1), dtype=np.float64), X], axis=1)  # (N,3)

    w_data = np.ones((N,), dtype=np.float64)
    huber_c = float(max(0.25, huber_c))
    lam = float(max(0.0, lam))

    coeff = None
    eps = 1e-12
    for _ in range(int(max(1, iters))):
        # Weighted smoothing spline: (K + lam * W^{-1}) w + P a = y;  P^T w = 0.
        D = lam / (w_data + eps)
        A = np.zeros((N + 3, N + 3), dtype=np.float64)
        A[:N, :N] = K + np.diag(D)
        A[:N, N:] = P
        A[N:, :N] = P.T

        Y = np.zeros((N + 3, 2), dtype=np.float64)
        Y[:N, :] = img_uv

        try:
            coeff = np.linalg.solve(A, Y)  # (N+3,2): [W; a0,a1,a2]
        except np.linalg.LinAlgError:
            M = _fit_affine(obj_xy, img_uv)
            if M is None:
                return np.full((query_xy.shape[0], 2), np.nan, dtype=np.float64)
            return _apply_affine(M, query_xy)

        Wc = coeff[:N, :]
        ac = coeff[N:, :]  # (3,2)

        pred_i = K @ Wc + P @ ac
        r = np.sqrt(np.sum((pred_i - img_uv) ** 2, axis=1))
        w_data = np.where(r <= huber_c, 1.0, huber_c / (r + eps))

    if coeff is None:
        M = _fit_affine(obj_xy, img_uv)
        if M is None:
            return np.full((query_xy.shape[0], 2), np.nan, dtype=np.float64)
        return _apply_affine(M, query_xy)

    Wc = coeff[:N, :]
    ac = coeff[N:, :]
    dxq = Q[:, 0:1] - X[:, 0:1].T  # (M,N)
    dyq = Q[:, 1:2] - X[:, 1:2].T
    Kq = U(dxq * dxq + dyq * dyq)  # (M,N)
    Pq = np.concatenate([np.ones((Q.shape[0], 1), dtype=np.float64), Q], axis=1)  # (M,3)
    return (Kq @ Wc + Pq @ ac).astype(np.float64)


def predict_points_rayfield_tps_robust(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    *,
    lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    ransac_reproj_px: float = 3.0,
) -> np.ndarray:
    """
    Board→image mapping: global projective base + robust TPS residual field.

    - Base mapping is a homography (estimated robustly with RANSAC).
    - Residuals are smoothed with a TPS spline fitted by IRLS (Huber).
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64).reshape(-1, 2)
    img_uv = np.asarray(img_uv, dtype=np.float64).reshape(-1, 2)
    query_xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have the same length")

    N = int(obj_xy.shape[0])
    if N < 4:
        M = _fit_affine(obj_xy, img_uv)
        if M is None:
            return np.full((query_xy.shape[0], 2), np.nan, dtype=np.float64)
        return _apply_affine(M, query_xy)

    import cv2  # type: ignore

    H, _mask = cv2.findHomography(obj_xy, img_uv, method=cv2.RANSAC, ransacReprojThreshold=float(ransac_reproj_px))
    if H is None:
        H, _mask = cv2.findHomography(obj_xy, img_uv, method=0)
    if H is None:
        M = _fit_affine(obj_xy, img_uv)
        if M is None:
            return np.full((query_xy.shape[0], 2), np.nan, dtype=np.float64)
        return _apply_affine(M, query_xy)

    def proj(Hh: np.ndarray, pts: np.ndarray) -> np.ndarray:
        ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        uvw = (Hh @ ph.T).T
        return uvw[:, :2] / (uvw[:, 2:3] + 1e-12)

    base_obs = proj(H, obj_xy)
    res_obs = img_uv - base_obs
    res_q = _predict_points_tps_irls(obj_xy, res_obs, query_xy, lam=float(lam), huber_c=float(huber_c), iters=int(iters))
    base_q = proj(H, query_xy)
    return (base_q + res_q).astype(np.float64)

