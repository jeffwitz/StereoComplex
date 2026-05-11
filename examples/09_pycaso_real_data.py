# %% [markdown]
# # 09 — StereoComplex on real CMO microscope data
#
# This notebook runs the full StereoComplex pipeline on real calibration
# images from a CMO stereo microscope.  It accepts either:
#
# - a **local Pycaso clone** (point ``PYCASO_CLONE`` below);
# - any **left/right image folders** with calibration frames.
#
# Pycaso (`LaboratoireMecaniqueLille/Pycaso`) is an open-source CMO
# calibration tool whose example data includes paired left/right
# calibration images in ``Exemple/Images_example/``.

# %%
from __future__ import annotations

from pathlib import Path
import stereocomplex as sc
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Set this to your local Pycaso clone, or to any left/right folder pair.
PYCASO_CLONE = Path("../Pycaso")  # ← adjust to your local path
# ══════════════════════════════════════════════════════════════════════

_IMG = PYCASO_CLONE / "Exemple" / "Images_example"
LEFT_DIR = _IMG / "left_calibration"
RIGHT_DIR = _IMG / "right_calibration"

# %% [markdown]
# ## 1 — Load calibration data

# %%
if LEFT_DIR.exists() and RIGHT_DIR.exists():
    left_dir = LEFT_DIR
    right_dir = RIGHT_DIR
    print(f"Using Pycaso data: {left_dir}")
    print(f"  Left frames:  {len(list(left_dir.iterdir()))}")
    print(f"  Right frames: {len(list(right_dir.iterdir()))}")
else:
    print(f"Pycaso data not found at {PYCASO_CLONE}")
    print("Clone it with:  git clone https://github.com/LaboratoireMecaniqueLille/Pycaso ../Pycaso")
    print("Or set PYCASO_CLONE to your own left/right image folders.")
    exit(1)

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
