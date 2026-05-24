"""Structure-tensor ChArUco corner refiners."""

from __future__ import annotations

import numpy as np


def refine_points_tensor_symmetry(
    cv2,
    img_u8: np.ndarray,
    pts_xy: np.ndarray,
    search_radius: float,
    tensor_sigma: float,
) -> np.ndarray:
    """Pycaso-like second pass: refine corners by maximising quadrant symmetry.

    Estimates local axes from the structure tensor, then maximises a
    quadrant-symmetry score evaluated along those axes.

    Parameters
    ----------
    cv2 : module
        OpenCV module.
    img_u8 : ndarray, uint8
        Grayscale input image.
    pts_xy : ndarray, shape (N, 2)
        Initial corner coordinates in pixels.
    search_radius : float
        Search radius in pixels.
    tensor_sigma : float
        Gaussian sigma for the structure tensor computation.

    Returns
    -------
    ndarray, shape (N, 2)
        Refined corner coordinates in pixels.
    """
    search_radius = float(max(0.25, search_radius))
    tensor_sigma = float(max(0.8, tensor_sigma))

    img = img_u8.astype(np.float32) / 255.0
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    H, W = img_u8.shape[:2]
    out = pts_xy.astype(np.float64, copy=True)

    # Window for tensor estimation.
    win_r = int(max(3, round(2.5 * tensor_sigma)))
    ys, xs = np.mgrid[-win_r : win_r + 1, -win_r : win_r + 1]
    w = np.exp(-(xs * xs + ys * ys) / (2.0 * tensor_sigma * tensor_sigma)).astype(np.float32)

    # Candidate offsets (subpixel grid).
    step = 0.25
    offs = np.arange(-search_radius, search_radius + 1e-9, step, dtype=np.float64)

    # Radii for quadrant sampling along axes.
    radii = np.array([1.5, 2.5, 3.5], dtype=np.float64)

    for i in range(out.shape[0]):
        x0, y0 = out[i]
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if xi < win_r + 4 or xi >= (W - win_r - 4) or yi < win_r + 4 or yi >= (H - win_r - 4):
            continue

        gx = Ix[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]
        gy = Iy[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]

        Sxx = float(np.sum(w * gx * gx))
        Sxy = float(np.sum(w * gx * gy))
        Syy = float(np.sum(w * gy * gy))
        J = np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float64)
        vals, vecs = np.linalg.eigh(J)
        if float(vals[1]) < 1e-10:
            continue
        # vecs columns are eigenvectors; use them as orthonormal axes.
        e1 = vecs[:, 1]
        e2 = vecs[:, 0]

        best_score = -1.0
        best_xy = (x0, y0)

        for dy in offs:
            for dx in offs:
                x = x0 + dx
                y = y0 + dy
                s = 0.0
                for rr in radii:
                    ppp = _bilinear(img, x + rr * (e1[0] + e2[0]), y + rr * (e1[1] + e2[1]))
                    pmm = _bilinear(img, x - rr * (e1[0] + e2[0]), y - rr * (e1[1] + e2[1]))
                    ppm = _bilinear(img, x + rr * (e1[0] - e2[0]), y + rr * (e1[1] - e2[1]))
                    pmp = _bilinear(img, x - rr * (e1[0] - e2[0]), y - rr * (e1[1] - e2[1]))
                    s += abs((ppp + pmm) - (ppm + pmp))
                if s > best_score:
                    best_score = s
                    best_xy = (x, y)

        out[i, 0] = best_xy[0]
        out[i, 1] = best_xy[1]

    return out


def refine_points_tensor_noble(
    cv2,
    img_u8: np.ndarray,
    pts_xy: np.ndarray,
    search_radius: float,
    tensor_sigma: float,
) -> np.ndarray:
    """
    Second pass based on a rotation-invariant structure tensor cornerness measure.

    Uses the "Noble" measure: det(J) / trace(J), where J is the (Gaussian-smoothed)
    structure tensor built from image gradients. For each point, searches a small
    window around the initial estimate and refines to subpixel precision via a
    1D parabolic fit around the maximum.
    """
    search_radius = float(max(1.0, search_radius))
    tensor_sigma = float(max(0.8, tensor_sigma))

    img = img_u8.astype(np.float32) / 255.0
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    # Smoothed gradient products (structure tensor components).
    A = cv2.GaussianBlur(
        Ix * Ix,
        ksize=(0, 0),
        sigmaX=tensor_sigma,
        sigmaY=tensor_sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    B = cv2.GaussianBlur(
        Ix * Iy,
        ksize=(0, 0),
        sigmaX=tensor_sigma,
        sigmaY=tensor_sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    C = cv2.GaussianBlur(
        Iy * Iy,
        ksize=(0, 0),
        sigmaX=tensor_sigma,
        sigmaY=tensor_sigma,
        borderType=cv2.BORDER_REFLECT,
    )

    trace = A + C
    det = A * C - B * B
    R = det / (trace + 1e-12)

    H, W = img_u8.shape[:2]
    out = pts_xy.astype(np.float64, copy=True)
    r = round(search_radius)
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1]
    # Favor the peak closest to the initial estimate to avoid snapping to nearby
    # high-contrast features (e.g., marker micro-corners).
    w = np.exp(-(xs * xs + ys * ys) / (2.0 * max(1.0, 0.5 * r) ** 2)).astype(np.float32)

    def _parabolic_delta(v_m1: float, v_0: float, v_p1: float) -> float:
        denom = v_m1 - 2.0 * v_0 + v_p1
        if abs(denom) < 1e-12:
            return 0.0
        d = 0.5 * (v_m1 - v_p1) / denom
        # Keep refinement local and stable.
        return float(np.clip(d, -1.0, 1.0))

    for i in range(out.shape[0]):
        x0, y0 = out[i]
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if xi < r + 1 or xi >= (W - r - 1) or yi < r + 1 or yi >= (H - r - 1):
            continue

        patch = R[yi - r : yi + r + 1, xi - r : xi + r + 1]
        if patch.size == 0:
            continue

        flat_idx = int(np.argmax(patch * w))
        my, mx = np.unravel_index(flat_idx, patch.shape)
        px = (xi - r) + int(mx)
        py = (yi - r) + int(my)

        if px < 1 or px >= (W - 1) or py < 1 or py >= (H - 1):
            continue

        v0 = float(R[py, px])
        if not np.isfinite(v0) or v0 <= 0.0:
            continue

        dx = _parabolic_delta(float(R[py, px - 1]), v0, float(R[py, px + 1]))
        dy = _parabolic_delta(float(R[py - 1, px]), v0, float(R[py + 1, px]))

        out[i, 0] = float(px) + dx
        out[i, 1] = float(py) + dy

    return out


def refine_points_tensor_lines(
    cv2,
    img_u8: np.ndarray,
    pts_xy: np.ndarray,
    search_radius: float,
    tensor_sigma: float,
) -> np.ndarray:
    """
    Pycaso-like second pass using the structure tensor to estimate two edge normals,
    then re-localizing each edge by 1D search along its normal and intersecting the
    two edge lines.

    This targets a more "geometric" corner center than simply maximizing a cornerness map.
    """
    search_radius = float(max(0.5, search_radius))
    tensor_sigma = float(max(0.8, tensor_sigma))

    img = img_u8.astype(np.float32) / 255.0
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    H, W = img_u8.shape[:2]
    out = pts_xy.astype(np.float64, copy=True)

    win_r = int(max(3, round(2.5 * tensor_sigma)))
    ys, xs = np.mgrid[-win_r : win_r + 1, -win_r : win_r + 1]
    w = np.exp(-(xs * xs + ys * ys) / (2.0 * tensor_sigma * tensor_sigma)).astype(np.float32)

    # 1D search settings.
    step = 0.25
    offs = np.arange(-search_radius, search_radius + 1e-9, step, dtype=np.float64)
    line_half_len = int(max(2, round(2.0 * tensor_sigma)))
    line_s = np.arange(-line_half_len, line_half_len + 1, 1.0, dtype=np.float64)

    def _edge_score_at(p: np.ndarray, n: np.ndarray, t: np.ndarray) -> float:
        s = 0.0
        for ss in line_s:
            x = float(p[0] + ss * t[0])
            y = float(p[1] + ss * t[1])
            gx = _bilinear(Ix, x, y)
            gy = _bilinear(Iy, x, y)
            s += abs(gx * float(n[0]) + gy * float(n[1]))
        return s

    def _refine_edge_offset(p0: np.ndarray, n: np.ndarray) -> float:
        # Edge tangent is perpendicular to its normal.
        t = np.array([-n[1], n[0]], dtype=np.float64)

        scores = np.empty((offs.size,), dtype=np.float64)
        for k, a in enumerate(offs.tolist()):
            scores[k] = _edge_score_at(p0 + a * n, n, t)

        k0 = int(np.argmax(scores))
        if 0 < k0 < (scores.size - 1):
            v_m1 = scores[k0 - 1]
            v_0 = scores[k0]
            v_p1 = scores[k0 + 1]
            denom = v_m1 - 2.0 * v_0 + v_p1
            if abs(denom) > 1e-12:
                d = 0.5 * (v_m1 - v_p1) / denom
                d = float(np.clip(d, -1.0, 1.0))
                return float(offs[k0] + d * step)
        return float(offs[k0])

    for i in range(out.shape[0]):
        x0, y0 = out[i]
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if xi < win_r + 3 or xi >= (W - win_r - 3) or yi < win_r + 3 or yi >= (H - win_r - 3):
            continue

        gx = Ix[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]
        gy = Iy[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]

        Sxx = float(np.sum(w * gx * gx))
        Sxy = float(np.sum(w * gx * gy))
        Syy = float(np.sum(w * gy * gy))
        J = np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float64)
        vals, vecs = np.linalg.eigh(J)
        if float(vals[0]) < 1e-10 or float(vals[1]) < 1e-10:
            continue

        # Eigenvectors of J are the principal directions of gradient energy (edge normals).
        n1 = vecs[:, 1].astype(np.float64, copy=False)
        n2 = vecs[:, 0].astype(np.float64, copy=False)

        p0 = np.array([x0, y0], dtype=np.float64)
        a1 = _refine_edge_offset(p0, n1)
        a2 = _refine_edge_offset(p0, n2)

        p1 = p0 + a1 * n1
        p2 = p0 + a2 * n2

        c1 = float(n1[0] * p1[0] + n1[1] * p1[1])
        c2 = float(n2[0] * p2[0] + n2[1] * p2[1])

        M = np.array([[n1[0], n1[1]], [n2[0], n2[1]]], dtype=np.float64)
        detM = float(np.linalg.det(M))
        if abs(detM) < 1e-8:
            continue
        xy = np.linalg.solve(M, np.array([c1, c2], dtype=np.float64))

        if float(np.linalg.norm(xy - p0)) > (search_radius + 1.0):
            continue

        out[i, 0] = float(xy[0])
        out[i, 1] = float(xy[1])

    return out


def refine_points_tensor_lsq(
    cv2,
    img_u8: np.ndarray,
    pts_xy: np.ndarray,
    search_radius: float,
    tensor_sigma: float,
) -> np.ndarray:
    """
    Structure-tensor based corner refinement via least-squares intersection of local
    gradient-normals (line constraints).

    For pixels p in a window around the initial estimate, treat the local edge normal
    as n = grad / ||grad|| and add a constraint nᵀ x = nᵀ p. Solve the weighted normal
    equations for x (the corner).
    """
    search_radius = float(max(0.5, search_radius))
    tensor_sigma = float(max(0.8, tensor_sigma))

    img = img_u8.astype(np.float32) / 255.0
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    H, W = img_u8.shape[:2]
    out = pts_xy.astype(np.float64, copy=True)

    win_r = int(max(3, round(2.5 * tensor_sigma)))
    ys, xs = np.mgrid[-win_r : win_r + 1, -win_r : win_r + 1]
    w = np.exp(-(xs * xs + ys * ys) / (2.0 * tensor_sigma * tensor_sigma)).astype(np.float32)
    mag_thresh = 1e-3

    for i in range(out.shape[0]):
        x0, y0 = out[i]
        xi = int(np.rint(x0))
        yi = int(np.rint(y0))
        if xi < win_r + 2 or xi >= (W - win_r - 2) or yi < win_r + 2 or yi >= (H - win_r - 2):
            continue

        gx = (
            Ix[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]
            .astype(np.float64, copy=False)
        )
        gy = (
            Iy[yi - win_r : yi + win_r + 1, xi - win_r : xi + win_r + 1]
            .astype(np.float64, copy=False)
        )

        mag = np.sqrt(gx * gx + gy * gy)
        m = mag > mag_thresh
        if int(np.count_nonzero(m)) < 10:
            continue

        nx = np.zeros_like(gx)
        ny = np.zeros_like(gy)
        nx[m] = gx[m] / mag[m]
        ny[m] = gy[m] / mag[m]

        # Pixel-center coordinates in the global image frame.
        px = (xi + xs).astype(np.float64)
        py = (yi + ys).astype(np.float64)

        # Weights: Gaussian window * gradient magnitude (favor strong edges).
        ww = (w.astype(np.float64) * mag)[m]
        nxv = nx[m]
        nyv = ny[m]
        dot = nxv * px[m] + nyv * py[m]

        A11 = float(np.sum(ww * nxv * nxv))
        A12 = float(np.sum(ww * nxv * nyv))
        A22 = float(np.sum(ww * nyv * nyv))
        b1 = float(np.sum(ww * nxv * dot))
        b2 = float(np.sum(ww * nyv * dot))

        detA = A11 * A22 - A12 * A12
        if abs(detA) < 1e-10:
            continue

        x = (A22 * b1 - A12 * b2) / detA
        y = (-A12 * b1 + A11 * b2) / detA
        xy = np.array([x, y], dtype=np.float64)

        if float(np.linalg.norm(xy - np.array([x0, y0], dtype=np.float64))) > (search_radius + 1.0):
            continue

        out[i, 0] = float(xy[0])
        out[i, 1] = float(xy[1])

    return out


def _bilinear(img: np.ndarray, x: float, y: float) -> float:
    H, W = img.shape[:2]
    if x < 0.0 or y < 0.0 or x > (W - 1) or y > (H - 1):
        return 0.0
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    wx = x - x0
    wy = y - y0
    Ia = float(img[y0, x0])
    Ib = float(img[y0, x1])
    Ic = float(img[y1, x0])
    Id = float(img[y1, x1])
    return (1.0 - wx) * (1.0 - wy) * Ia + wx * (1.0 - wy) * Ib + (1.0 - wx) * wy * Ic + wx * wy * Id
