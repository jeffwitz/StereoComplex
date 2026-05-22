"""ChArUco corner predictors: smooth board-plane warps from marker correspondences.

Each ``predict_points_*`` function maps board coordinates (mm) to image
coordinates (px) and shares the signature ``(obj_xy, img_uv, query_xy) -> xy``,
where ``xy`` is an ``(M, 2)`` array (or ``None`` when the fit fails). They are
dispatched by name through :data:`stereocomplex.eval.predictors.PREDICTORS`.
"""

from __future__ import annotations

import numpy as np


def build_marker_correspondences(
    charuco_board,
    marker_ids,
    marker_corners,
    *,
    ndim: int = 2,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Pair detected ArUco marker corners with their board object points.

    Parameters
    ----------
    charuco_board:
        OpenCV ``CharucoBoard`` exposing ``getIds`` / ``getObjPoints``.
    marker_ids, marker_corners:
        Raw detector output (ids array and per-marker corner arrays).
    ndim:
        ``2`` to keep only the board-plane object coordinates, ``3`` to keep the
        full 3D object points (needed for PnP).

    Returns
    -------
    tuple of (obj, img) arrays, or ``None`` when no marker matches the board.
    """
    marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
    board_ids = np.asarray(charuco_board.getIds(), dtype=np.int32).reshape(-1)
    board_obj = charuco_board.getObjPoints()
    sl = slice(None) if ndim == 3 else slice(0, 2)
    id_to_obj = {
        int(i): np.asarray(p, dtype=np.float64)[:, sl]
        for i, p in zip(board_ids.tolist(), board_obj, strict=True)
    }

    obj_pts = []
    img_pts = []
    for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
        o = id_to_obj.get(int(mid))
        if o is None:
            continue
        mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)  # noqa: PLW2901
        if mc.shape[0] != 4 or o.shape[0] != 4:
            continue
        obj_pts.append(o)
        img_pts.append(mc)
    if not obj_pts:
        return None
    return np.concatenate(obj_pts, axis=0), np.concatenate(img_pts, axis=0)


def predict_points_mls_affine(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    k: int = 40,
    sigma_obj: float | None = None,
) -> np.ndarray:
    """
    Moving least squares (affine) mapping from board coords -> image coords.

    For each query point, fits an affine map using the k strongest Gaussian-weighted
    correspondences in object space.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    N = int(obj_xy.shape[0])
    k = int(max(6, min(k, N)))
    if sigma_obj is None:
        # Heuristic: use a few square sizes worth of influence, derived from marker spacing.
        # If correspondences are very sparse, inflate sigma.
        span = float(
            np.median(
                np.sqrt(np.sum((obj_xy - np.median(obj_xy, axis=0)) ** 2, axis=1))
            )
            + 1e-9
        )
        sigma_obj = max(10.0, 0.25 * span)
    sigma2 = float(sigma_obj) ** 2

    out = np.empty((query_xy.shape[0], 2), dtype=np.float64)

    for i in range(query_xy.shape[0]):
        q = query_xy[i]
        d2 = np.sum((obj_xy - q[None, :]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / sigma2)

        # Take k best weights.
        idx = np.argpartition(w, -k)[-k:] if k < N else np.arange(N)
        ww = w[idx]
        if float(np.max(ww)) <= 1e-12:
            out[i] = np.array([np.nan, np.nan], dtype=np.float64)
            continue

        X = np.concatenate(
            [obj_xy[idx], np.ones((idx.shape[0], 1), dtype=np.float64)], axis=1
        )  # (k,3)
        u = img_uv[idx, 0]
        v = img_uv[idx, 1]

        # Weighted normal equations: (X^T W X) a = X^T W u
        WX = X * ww[:, None]
        A = X.T @ WX  # (3,3)
        bu = X.T @ (ww * u)
        bv = X.T @ (ww * v)
        detA = float(np.linalg.det(A))
        if abs(detA) < 1e-12:
            out[i] = np.array([np.nan, np.nan], dtype=np.float64)
            continue
        au = np.linalg.solve(A, bu)
        av = np.linalg.solve(A, bv)
        qh = np.array([q[0], q[1], 1.0], dtype=np.float64)
        out[i, 0] = float(qh @ au)
        out[i, 1] = float(qh @ av)

    return out


def predict_points_affine_field(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    grid_size: tuple[int, int] = (9, 6),
    k: int = 80,
    sigma_obj: float | None = None,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Smooth "field of local intrinsics" surrogate.

    Instead of a single global K, estimate a low-frequency field of local affine maps
    from board (x,y) to image (u,v), by fitting affines at anchor locations and
    smoothing/interpolating their parameters.
    """
    import cv2  # type: ignore

    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    N = int(obj_xy.shape[0])
    if N < 6:
        return predict_points_mls_affine(obj_xy, img_uv, query_xy)

    nx, ny = (int(grid_size[0]), int(grid_size[1]))
    nx = max(2, nx)
    ny = max(2, ny)

    xmin = float(np.min(obj_xy[:, 0]))
    xmax = float(np.max(obj_xy[:, 0]))
    ymin = float(np.min(obj_xy[:, 1]))
    ymax = float(np.max(obj_xy[:, 1]))
    if (
        not np.isfinite([xmin, xmax, ymin, ymax]).all()
        or (xmax - xmin) < 1e-9
        or (ymax - ymin) < 1e-9
    ):
        return predict_points_mls_affine(obj_xy, img_uv, query_xy)

    if sigma_obj is None:
        span = float(
            np.median(
                np.sqrt(np.sum((obj_xy - np.median(obj_xy, axis=0)) ** 2, axis=1))
            )
            + 1e-9
        )
        sigma_obj = max(10.0, 0.35 * span)
    sigma2 = float(sigma_obj) ** 2

    # Global affine fallback.
    Xg = np.concatenate([obj_xy, np.ones((N, 1), dtype=np.float64)], axis=1)
    Ag = Xg.T @ Xg
    bg_u = Xg.T @ img_uv[:, 0]
    bg_v = Xg.T @ img_uv[:, 1]
    try:
        au_g = np.linalg.solve(Ag, bg_u)
        av_g = np.linalg.solve(Ag, bg_v)
    except np.linalg.LinAlgError:
        return predict_points_mls_affine(obj_xy, img_uv, query_xy)

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)

    Pu = np.empty((ny, nx, 3), dtype=np.float64)
    Pv = np.empty((ny, nx, 3), dtype=np.float64)

    k = int(max(6, min(int(k), N)))
    for j in range(ny):
        for i in range(nx):
            q = np.array([xs[i], ys[j]], dtype=np.float64)
            d2 = np.sum((obj_xy - q[None, :]) ** 2, axis=1)
            w = np.exp(-0.5 * d2 / sigma2)
            idx = np.argpartition(w, -k)[-k:] if k < N else np.arange(N)
            ww = w[idx]
            if float(np.max(ww)) <= 1e-12:
                Pu[j, i] = au_g
                Pv[j, i] = av_g
                continue

            X = np.concatenate([obj_xy[idx], np.ones((idx.shape[0], 1), dtype=np.float64)], axis=1)
            u = img_uv[idx, 0]
            v = img_uv[idx, 1]
            WX = X * ww[:, None]
            A = X.T @ WX
            bu = X.T @ (ww * u)
            bv = X.T @ (ww * v)
            try:
                Pu[j, i] = np.linalg.solve(A, bu)
                Pv[j, i] = np.linalg.solve(A, bv)
            except np.linalg.LinAlgError:
                Pu[j, i] = au_g
                Pv[j, i] = av_g

    for c in range(3):
        Pu[:, :, c] = (
            cv2.GaussianBlur(
                Pu[:, :, c].astype(np.float32), ksize=(0, 0), sigmaX=float(smooth_sigma)
            ).astype(np.float64)
        )
        Pv[:, :, c] = (
            cv2.GaussianBlur(
                Pv[:, :, c].astype(np.float32), ksize=(0, 0), sigmaX=float(smooth_sigma)
            ).astype(np.float64)
        )

    def lerp(a0: np.ndarray, a1: np.ndarray, t: float) -> np.ndarray:
        return (1.0 - t) * a0 + t * a1

    out = np.empty((query_xy.shape[0], 2), dtype=np.float64)
    for n in range(query_xy.shape[0]):
        x, y = float(query_xy[n, 0]), float(query_xy[n, 1])
        tx = (x - xmin) / (xmax - xmin)
        ty = (y - ymin) / (ymax - ymin)
        tx = float(np.clip(tx, 0.0, 1.0))
        ty = float(np.clip(ty, 0.0, 1.0))
        fx = tx * (nx - 1)
        fy = ty * (ny - 1)
        i0 = int(np.floor(fx))
        j0 = int(np.floor(fy))
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        ax = fx - i0
        ay = fy - j0

        Pu0 = lerp(Pu[j0, i0], Pu[j0, i1], ax)
        Pu1 = lerp(Pu[j1, i0], Pu[j1, i1], ax)
        Pv0 = lerp(Pv[j0, i0], Pv[j0, i1], ax)
        Pv1 = lerp(Pv[j1, i0], Pv[j1, i1], ax)
        au = lerp(Pu0, Pu1, ay)
        av = lerp(Pv0, Pv1, ay)

        qh = np.array([x, y, 1.0], dtype=np.float64)
        out[n, 0] = float(qh @ au)
        out[n, 1] = float(qh @ av)

    return out


def predict_points_rayfield(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    grid_size: tuple[int, int] = (16, 10),
    smooth_lambda: float = 3.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> np.ndarray:
    """
    Smooth ray-field (2D warp) on the board plane.

    Fits a regularized grid warp u(x,y), v(x,y) defined at grid nodes in object space.
    This is equivalent to a low-frequency non-parametric camera model restricted to
    the calibration plane.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    import cv2  # type: ignore

    N = int(obj_xy.shape[0])
    if N < 8:
        return predict_points_mls_homography(obj_xy, img_uv, query_xy)

    nx, ny = (int(grid_size[0]), int(grid_size[1]))
    nx = max(3, nx)
    ny = max(3, ny)
    M = nx * ny

    # Domain covers both constraints and all queries to avoid unstable extrapolation.
    all_xy = np.concatenate([obj_xy, query_xy], axis=0)
    xmin = float(np.min(all_xy[:, 0]))
    xmax = float(np.max(all_xy[:, 0]))
    ymin = float(np.min(all_xy[:, 1]))
    ymax = float(np.max(all_xy[:, 1]))
    if (xmax - xmin) < 1e-9 or (ymax - ymin) < 1e-9:
        return predict_points_mls_homography(obj_xy, img_uv, query_xy)

    # Pad to avoid boundary artifacts.
    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    def node_index(ix: int, iy: int) -> int:
        return iy * nx + ix

    def weights_for_points(
        pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = pts[:, 0]
        y = pts[:, 1]
        tx = (x - xmin) / (xmax - xmin)
        ty = (y - ymin) / (ymax - ymin)
        tx = np.clip(tx, 0.0, 1.0)
        ty = np.clip(ty, 0.0, 1.0)
        fx = tx * (nx - 1)
        fy = ty * (ny - 1)
        i0 = np.floor(fx).astype(np.int32)
        j0 = np.floor(fy).astype(np.int32)
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        ax = fx - i0
        ay = fy - j0
        w00 = (1.0 - ax) * (1.0 - ay)
        w10 = ax * (1.0 - ay)
        w01 = (1.0 - ax) * ay
        w11 = ax * ay
        idx00 = (j0 * nx + i0).astype(np.int32)
        idx10 = (j0 * nx + i1).astype(np.int32)
        idx01 = (j1 * nx + i0).astype(np.int32)
        idx11 = (j1 * nx + i1).astype(np.int32)
        return idx00, idx10, idx01, idx11, np.stack([w00, w10, w01, w11], axis=1)

    # Smoothness: graph Laplacian on the grid.
    L = np.zeros((M, M), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            p = node_index(ix, iy)
            neigh = []
            if ix > 0:
                neigh.append(node_index(ix - 1, iy))
            if ix + 1 < nx:
                neigh.append(node_index(ix + 1, iy))
            if iy > 0:
                neigh.append(node_index(ix, iy - 1))
            if iy + 1 < ny:
                neigh.append(node_index(ix, iy + 1))
            deg = len(neigh)
            L[p, p] += deg
            for q in neigh:
                L[p, q] -= 1.0

    # Base mapping: single homography from all correspondences (gives sane extrapolation).
    Hb, _mask = cv2.findHomography(obj_xy, img_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if Hb is None:
        return predict_points_mls_homography(obj_xy, img_uv, query_xy)

    def proj(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        uvw = (H @ ph.T).T
        return uvw[:, :2] / (uvw[:, 2:3] + 1e-12)

    base_obs = proj(Hb, obj_xy)
    res_obs = img_uv - base_obs

    idx00, idx10, idx01, idx11, ww = weights_for_points(obj_xy)
    nodes_u = np.zeros((M,), dtype=np.float64)
    nodes_v = np.zeros((M,), dtype=np.float64)

    w_data = np.ones((N,), dtype=np.float64)
    huber_c = float(max(0.25, huber_c))

    for _ in range(int(max(1, iters))):
        AtA = np.zeros((M, M), dtype=np.float64)
        Atu = np.zeros((M,), dtype=np.float64)
        Atv = np.zeros((M,), dtype=np.float64)

        for n in range(N):
            wrow = float(w_data[n])
            if wrow <= 0.0:
                continue
            ids = (int(idx00[n]), int(idx10[n]), int(idx01[n]), int(idx11[n]))
            ws = ww[n]
            u_obs = float(res_obs[n, 0])
            v_obs = float(res_obs[n, 1])

            for a in range(4):
                ia = ids[a]
                wa = float(ws[a])
                Atu[ia] += wrow * wa * u_obs
                Atv[ia] += wrow * wa * v_obs
                for b in range(4):
                    ib = ids[b]
                    wb = float(ws[b])
                    AtA[ia, ib] += wrow * wa * wb

        lam = float(smooth_lambda) * (float(N) / float(M))
        AtA = AtA + lam * (L.T @ L) + (0.1 * lam + 1e-6) * np.eye(M, dtype=np.float64)

        nodes_u = np.linalg.solve(AtA, Atu)
        nodes_v = np.linalg.solve(AtA, Atv)

        # Update robust weights.
        du_pred = (
            nodes_u[idx00] * ww[:, 0]
            + nodes_u[idx10] * ww[:, 1]
            + nodes_u[idx01] * ww[:, 2]
            + nodes_u[idx11] * ww[:, 3]
        )
        dv_pred = (
            nodes_v[idx00] * ww[:, 0]
            + nodes_v[idx10] * ww[:, 1]
            + nodes_v[idx01] * ww[:, 2]
            + nodes_v[idx11] * ww[:, 3]
        )
        r = np.sqrt((du_pred - res_obs[:, 0]) ** 2 + (dv_pred - res_obs[:, 1]) ** 2)
        w_data = np.where(r <= huber_c, 1.0, huber_c / (r + 1e-12))

    # Predict queries: base homography + smoothed residual field.
    base_q = proj(Hb, query_xy)
    q00, q10, q01, q11, qw = weights_for_points(query_xy)
    du = (
        nodes_u[q00] * qw[:, 0]
        + nodes_u[q10] * qw[:, 1]
        + nodes_u[q01] * qw[:, 2]
        + nodes_u[q11] * qw[:, 3]
    )
    dv = (
        nodes_v[q00] * qw[:, 0]
        + nodes_v[q10] * qw[:, 1]
        + nodes_v[q01] * qw[:, 2]
        + nodes_v[q11] * qw[:, 3]
    )
    return (base_q + np.stack([du, dv], axis=1)).astype(np.float64)


def predict_points_rayfield_tps(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    lam: float = 10.0,
) -> np.ndarray:
    """
    Ray-field variant: global homography + TPS-smoothed residuals.

    This keeps the "projective base" (good extrapolation) while using a thin-plate
    spline (with smoothing `lam`) to reconstruct a smooth residual field from
    sparse ArUco samples.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    import cv2  # type: ignore

    N = int(obj_xy.shape[0])
    if N < 8:
        return predict_points_mls_homography(obj_xy, img_uv, query_xy)

    Hb, _mask = cv2.findHomography(obj_xy, img_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if Hb is None:
        return predict_points_mls_homography(obj_xy, img_uv, query_xy)

    def proj(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        uvw = (H @ ph.T).T
        return uvw[:, :2] / (uvw[:, 2:3] + 1e-12)

    base_obs = proj(Hb, obj_xy)
    res_obs = img_uv - base_obs

    # TPS reconstructs residual field from sparse samples.
    res_q = predict_points_tps(obj_xy, res_obs, query_xy, lam=float(lam))
    base_q = proj(Hb, query_xy)
    return (base_q + res_q).astype(np.float64)


def predict_points_rayfield_tps_robust(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> np.ndarray:
    """
    Ray-field variant: global homography + robust TPS-smoothed residuals (IRLS).
    """
    from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust as _impl

    return _impl(
        obj_xy,
        img_uv,
        query_xy,
        lam=float(lam),
        huber_c=float(huber_c),
        iters=int(iters),
    )


def predict_points_mls_homography(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    k: int = 60,
    sigma_obj: float | None = None,
) -> np.ndarray:
    """
    Moving least squares (projective) mapping from board coords -> image coords.

    Fits a local homography per query point using weighted DLT on the k strongest
    correspondences in object space.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    N = int(obj_xy.shape[0])
    k = int(max(8, min(k, N)))
    if sigma_obj is None:
        span = float(
            np.median(
                np.sqrt(np.sum((obj_xy - np.median(obj_xy, axis=0)) ** 2, axis=1))
            )
            + 1e-9
        )
        sigma_obj = max(10.0, 0.25 * span)
    sigma2 = float(sigma_obj) ** 2

    def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.mean(pts, axis=0)
        d = np.sqrt(np.sum((pts - m[None, :]) ** 2, axis=1))
        s = float(np.sqrt(2.0) / (np.mean(d) + 1e-12))
        T = np.array([[s, 0.0, -s * m[0]], [0.0, s, -s * m[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
        ph = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        pn = (T @ ph.T).T[:, :2]
        return pn, T

    # Global fallback homography (helps for queries outside convex hull).
    H_global = None
    if N >= 4:
        A = []
        for (x, y), (u, v) in zip(obj_xy.tolist(), img_uv.tolist(), strict=True):
            A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
            A.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
        A = np.asarray(A, dtype=np.float64)
        _u, _s, vt = np.linalg.svd(A, full_matrices=False)
        H_global = vt[-1].reshape(3, 3)

    out = np.empty((query_xy.shape[0], 2), dtype=np.float64)
    for i in range(query_xy.shape[0]):
        q = query_xy[i]
        d2 = np.sum((obj_xy - q[None, :]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / sigma2)
        idx = np.argpartition(w, -k)[-k:] if k < N else np.arange(N)
        ww = w[idx]
        if float(np.max(ww)) <= 1e-12:
            if H_global is None:
                out[i] = np.array([np.nan, np.nan], dtype=np.float64)
            else:
                uvw = H_global @ np.array([q[0], q[1], 1.0], dtype=np.float64)
                out[i] = uvw[:2] / (uvw[2] + 1e-12)
            continue

        X = obj_xy[idx]
        U = img_uv[idx]
        Xn, T1 = _normalize(X)
        Un, T2 = _normalize(U)

        A = np.zeros((2 * idx.shape[0], 9), dtype=np.float64)
        for j, ((x, y), (u, v)) in enumerate(zip(Xn.tolist(), Un.tolist(), strict=True)):
            A[2 * j + 0] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
            A[2 * j + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
        sw = np.sqrt(np.repeat(ww, 2))
        A *= sw[:, None]

        _u, _s, vt = np.linalg.svd(A, full_matrices=False)
        Hn = vt[-1].reshape(3, 3)
        H = np.linalg.inv(T2) @ Hn @ T1
        uvw = H @ np.array([q[0], q[1], 1.0], dtype=np.float64)
        out[i, 0] = float(uvw[0] / (uvw[2] + 1e-12))
        out[i, 1] = float(uvw[1] / (uvw[2] + 1e-12))

    return out


def predict_points_piecewise_affine(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
) -> np.ndarray:
    """
    Piecewise-affine mapping using Delaunay triangulation in object space.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    import cv2  # type: ignore

    xmin = float(np.min(obj_xy[:, 0]))
    ymin = float(np.min(obj_xy[:, 1]))
    xmax = float(np.max(obj_xy[:, 0]))
    ymax = float(np.max(obj_xy[:, 1]))
    pad = 1.0
    rect = (
        int(np.floor(xmin - pad)),
        int(np.floor(ymin - pad)),
        int(np.ceil((xmax - xmin) + 2 * pad)),
        int(np.ceil((ymax - ymin) + 2 * pad)),
    )
    subdiv = cv2.Subdiv2D(rect)
    for p in obj_xy.tolist():
        subdiv.insert((float(p[0]), float(p[1])))
    tris = subdiv.getTriangleList()
    tris = np.asarray(tris, dtype=np.float64).reshape(-1, 6)

    def _closest_index(pt: np.ndarray) -> int | None:
        d2 = np.sum((obj_xy - pt[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        if float(d2[j]) > 1e-8:
            return None
        return j

    tri_idx: list[tuple[int, int, int]] = []
    for t in tris:
        p1 = np.array([t[0], t[1]], dtype=np.float64)
        p2 = np.array([t[2], t[3]], dtype=np.float64)
        p3 = np.array([t[4], t[5]], dtype=np.float64)
        i1 = _closest_index(p1)
        i2 = _closest_index(p2)
        i3 = _closest_index(p3)
        if i1 is None or i2 is None or i3 is None:
            continue
        if len({i1, i2, i3}) < 3:
            continue
        tri_idx.append((i1, i2, i3))
    if not tri_idx:
        return predict_points_mls_affine(obj_xy, img_uv, query_xy)

    # Precompute triangles in both spaces.
    obj_tris = []
    img_tris = []
    for a, b, c in tri_idx:
        obj_tris.append(obj_xy[[a, b, c], :])
        img_tris.append(img_uv[[a, b, c], :])

    out = np.empty((query_xy.shape[0], 2), dtype=np.float64)
    eps = -1e-8
    for i, q in enumerate(query_xy):
        found = False
        for P, Q in zip(obj_tris, img_tris, strict=True):
            p0, p1, p2 = P
            v0 = p1 - p0
            v1 = p2 - p0
            v2 = q - p0
            den = float(v0[0] * v1[1] - v0[1] * v1[0])
            if abs(den) < 1e-12:
                continue
            a = float((v2[0] * v1[1] - v2[1] * v1[0]) / den)
            b = float((v0[0] * v2[1] - v0[1] * v2[0]) / den)
            c = 1.0 - a - b
            if a >= eps and b >= eps and c >= eps:
                out[i] = a * Q[1] + b * Q[2] + c * Q[0]
                found = True
                break
        if not found:
            out[i] = predict_points_mls_affine(obj_xy, img_uv, q.reshape(1, 2))[0]
    return out


def predict_points_tps(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
    lam: float = 1e-1,
) -> np.ndarray:
    """
    Thin-plate spline warp (2D->2D) from board coords -> image coords.

    Uses only 2D correspondences (e.g., ArUco marker corners), so it can model
    non-pinhole / non-Brown camera mappings as a smooth deformation.
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if obj_xy.ndim != 2 or obj_xy.shape[1] != 2:
        raise ValueError("obj_xy must be (N,2)")
    if img_uv.ndim != 2 or img_uv.shape[1] != 2:
        raise ValueError("img_uv must be (N,2)")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("query_xy must be (M,2)")
    if obj_xy.shape[0] != img_uv.shape[0]:
        raise ValueError("obj_xy and img_uv must have same length")

    N = int(obj_xy.shape[0])
    if N < 6:
        # Not enough constraints; fall back to affine MLS.
        return predict_points_mls_affine(obj_xy, img_uv, query_xy)

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
    if lam > 0:
        K = K + float(lam) * np.eye(N, dtype=np.float64)

    P = np.concatenate([np.ones((N, 1), dtype=np.float64), X], axis=1)  # (N,3)
    A = np.zeros((N + 3, N + 3), dtype=np.float64)
    A[:N, :N] = K
    A[:N, N:] = P
    A[N:, :N] = P.T

    Y = np.zeros((N + 3, 2), dtype=np.float64)
    Y[:N, :] = img_uv

    try:
        coeff = np.linalg.solve(A, Y)  # (N+3,2): [W; a0,a1,a2]
    except np.linalg.LinAlgError:
        return predict_points_mls_affine(obj_xy, img_uv, query_xy)

    W = coeff[:N, :]
    a = coeff[N:, :]  # (3,2)

    # Compute warp for queries.
    dxq = Q[:, 0:1] - X[:, 0:1].T  # (M,N)
    dyq = Q[:, 1:2] - X[:, 1:2].T
    Kq = U(dxq * dxq + dyq * dyq)  # (M,N)
    Pq = np.concatenate([np.ones((Q.shape[0], 1), dtype=np.float64), Q], axis=1)  # (M,3)
    return Kq @ W + Pq @ a


def predict_points_homography(
    obj_xy: np.ndarray,
    img_uv: np.ndarray,
    query_xy: np.ndarray,
) -> np.ndarray | None:
    """
    Global RANSAC homography from board coords -> image coords.

    Returns ``None`` when ``cv2.findHomography`` fails to estimate a model.
    """
    import cv2  # type: ignore

    obj_xy = np.asarray(obj_xy, dtype=np.float64)
    img_uv = np.asarray(img_uv, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)

    H, _mask = cv2.findHomography(obj_xy, img_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        return None
    return (
        cv2.perspectiveTransform(query_xy.reshape(-1, 1, 2).astype(np.float32), H)
        .reshape(-1, 2)
        .astype(np.float64)
    )


def predict_hybrid(
    charuco_board,
    charuco_ids,
    charuco_corners,
    marker_ids,
    marker_corners,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Hybrid predictor: marker-based projective base + interpolated residual field.

    Builds a base prediction from ArUco markers (local projective MLS), then, when
    enough ChArUco corners are detected, corrects it with a locally interpolated
    residual field learned from those detections.

    Returns ``(ids, xy)`` for all chessboard corners, or ``None`` on missing data.
    """
    if charuco_ids is None or charuco_corners is None:
        return None
    if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
        return None

    det_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    # Match dataset pixel-center convention.
    det_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5

    corr = build_marker_correspondences(charuco_board, marker_ids, marker_corners, ndim=2)
    if corr is None:
        return None
    obj_pts, img_pts = corr

    chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
    base_xy = predict_points_mls_homography(obj_pts, img_pts, chess)
    ids = np.arange(chess.shape[0], dtype=np.int32)

    if det_ids.size < 12:
        return ids, base_xy

    res = det_xy - base_xy[det_ids]
    # Residual field is fit in board coordinates using detected ChArUco corners.
    res_pred = predict_points_mls_affine(chess[det_ids], res, chess, k=min(80, det_ids.size))
    return ids, base_xy + res_pred
