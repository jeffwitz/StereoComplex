from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BrownDistortion:
    """
    Brown-Conrady distortion on normalized camera coordinates (x=X/Z, y=Y/Z).

    Parameters follow common OpenCV naming:
      radial: k1, k2, k3
      tangential: p1, p2
    """

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        x2 = x * x
        y2 = y * y
        xy = x * y
        x_tan = 2.0 * self.p1 * xy + self.p2 * (r2 + 2.0 * x2)
        y_tan = self.p1 * (r2 + 2.0 * y2) + 2.0 * self.p2 * xy
        xd = x * radial + x_tan
        yd = y * radial + y_tan
        return xd, yd

    def undistort(self, xd: np.ndarray, yd: np.ndarray, iterations: int = 7) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterative inverse of distort() for small/moderate distortion.
        """
        xd = np.asarray(xd, dtype=np.float64)
        yd = np.asarray(yd, dtype=np.float64)
        x = xd.copy()
        y = yd.copy()
        for _ in range(int(iterations)):
            x_est, y_est = self.distort(x, y)
            x += xd - x_est
            y += yd - y_est
        return x, y


def brown_from_dict(d: dict) -> BrownDistortion:
    return BrownDistortion(
        k1=float(d.get("k1", 0.0)),
        k2=float(d.get("k2", 0.0)),
        p1=float(d.get("p1", 0.0)),
        p2=float(d.get("p2", 0.0)),
        k3=float(d.get("k3", 0.0)),
    )


def brown_to_dict(m: BrownDistortion) -> dict:
    return {"k1": m.k1, "k2": m.k2, "p1": m.p1, "p2": m.p2, "k3": m.k3}

