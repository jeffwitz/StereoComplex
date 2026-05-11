# %% [markdown]
# # 09 — StereoComplex on a real CMO microscope (Pycaso data)
#
# This notebook runs the full StereoComplex pipeline on the public
# Pycaso calibration dataset.  Pycaso is an open-source CMO stereo
# microscope calibration tool by the Lille Mechanics Laboratory.
#
# **First run:** ~20 MB of calibration images are downloaded automatically
# to `examples/pycaso_calib_data/` (git-ignored, persistent).
#
# **Subsequent runs:** instant — the data is already cached locally.

# %%
from __future__ import annotations

import io
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import stereocomplex as sc

# ── Pycaso data cache ────────────────────────────────────────────────
# Stored alongside this notebook so it survives reboots but is never
# committed (the directory is in .gitignore).
CACHE = Path(__file__).resolve().parent / "pycaso_calib_data"
PYCASO_REPO = "LaboratoireMecaniqueLille/Pycaso"
PYCASO_PATH = "data/calibration/CMO"  # Path inside the Pycaso repo


def _download_pycaso_calibration(target: Path) -> None:
    """Download the Pycaso CMO calibration frames from GitHub.

    Uses the GitHub API to fetch the directory listing, then downloads
    individual files via ``raw.githubusercontent.com``.  No git clone
    is needed.
    """
    import json

    api_url = f"https://api.github.com/repos/{PYCASO_REPO}/contents/{PYCASO_PATH}"
    print(f"  Fetching file list from {api_url} …")
    with urlopen(api_url) as resp:
        entries = json.loads(resp.read().decode())

    for entry in entries:
        if entry["type"] != "file":
            continue
        name = entry["name"]
        dl_url = entry["download_url"]
        if dl_url is None:
            continue
        dest = target / name
        print(f"  Downloading {name} ({entry['size']:,} bytes) …")
        with urlopen(dl_url) as src:
            dest.write_bytes(src.read())


def _ensure_pycaso_data() -> tuple[Path, Path]:
    """Return ``(left_dir, right_dir)``, downloading if needed."""
    if not (CACHE / "left").exists() or not (CACHE / "right").exists():
        print("Pycaso calibration data not found — downloading …")
        CACHE.mkdir(parents=True, exist_ok=True)

        _download_pycaso_calibration(CACHE)

        # Pycaso stores all frames flat; split into left/right by prefix.
        # Typical naming:  "CMO_L_0001.png", "CMO_R_0001.png".
        left_dir = CACHE / "left"
        right_dir = CACHE / "right"
        left_dir.mkdir(exist_ok=True)
        right_dir.mkdir(exist_ok=True)
        for f in sorted(CACHE.glob("*_L_*")):
            shutil.move(str(f), str(left_dir / f.name))
        for f in sorted(CACHE.glob("*_R_*")):
            shutil.move(str(f), str(right_dir / f.name))
        print(f"  Done — {len(list(left_dir.iterdir()))} left, "
              f"{len(list(right_dir.iterdir()))} right frames")
    else:
        print("Pycaso data already cached — skipping download.")

    return CACHE / "left", CACHE / "right"


# %% [markdown]
# ## 1 — Load calibration data

# %%
left_dir, right_dir = _ensure_pycaso_data()

# %% [markdown]
# ## 2 — Detect corners and calibrate
#
# Pycaso uses a dot-grid target.  StereoComplex works with ChArUco.
# For this demo, we define a ChArUco board matching the Pycaso target
# dimensions.  **Adjust these values to match your printed target.**

# %%
board = sc.CharucoBoardSpec(
    squares_x=17, squares_y=12,
    square_size_mm=1.0,    # Pycaso default: 1 mm squares
    marker_size_mm=0.7,
    aruco_dictionary="DICT_4X4_50",
)

# %% [markdown]
# ## 3 — Compare OpenCV raw vs Ray2D-refined

# %%
report = sc.compare_opencv_stereo_calibration(
    left_dir=left_dir,
    right_dir=right_dir,
    board=board,
    max_pairs=30,
)

print(f"Raw stereo RMS:      {report['raw']['stereo_rms_px']:.3f} px")
print(f"Refined stereo RMS:  {report['refined']['stereo_rms_px']:.3f} px")
print(f"Improvement:         {report['improvement_px']:+.3f} px")

# %% [markdown]
# ## 4 — Quality assessment

# %%
assessment = sc.assess_calibration(report["refined_result"])
print(f"Status: {assessment.status}")
for m in assessment.messages:
    print(f"  • {m}")
for r in assessment.recommendations:
    print(f"  → {r}")

# %% [markdown]
# ## 5 — Export to OpenCV format

# %%
K1, d1, K2, d2, R, T = report["refined_result"].to_opencv()
baseline = float(np.linalg.norm(T))
print(f"K_left  = [{K1[0,0]:.1f}, {K1[1,1]:.1f}] px")
print(f"Baseline = {baseline:.2f} mm")

# %% [markdown]
# ## 6 — Next steps
#
# - If the stereo RMS is good (< 0.3 px), the calibration is usable.
# - If residuals show structured patterns, try the non-central pipeline:
#   ``result = sc.calibrate_noncentral(left_dir, right_dir, board)``
# - Then identify the optical model:
#   ``sc.identify_optics(result.left_field, ...)``
#
# See `docs/FROM_OPENCV_TO_STEREOCOMPLEX.md` for the full guide.
