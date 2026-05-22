# %% [markdown]
# # 00 — Getting started with StereoComplex
#
# **You already know OpenCV.  Here is the 5-minute path to better calibration.**
#
# This notebook assumes you have:
# - a folder of left images and a folder of right images;
# - a ChArUco calibration board (see `docs/CHARUCO_IDENTIFICATION.md`).
#
# ## 1 — Define your board

# %%
from pathlib import Path

import numpy as np

import stereocomplex as sc

# Replace with your board dimensions
board = sc.CharucoBoardSpec(
    squares_x=11, squares_y=7,
    square_size_mm=39.07, marker_size_mm=27.35,
    aruco_dictionary="DICT_4X4_1000",
)

# %% [markdown]
# ## 2 — Compare OpenCV raw vs Ray2D-refined
#
# This is the recommended first step: it runs both methods and shows
# the improvement from rayfield-based corner refinement.

# %%
# Replace with your own image directories:
#   report = sc.compare_opencv_stereo_calibration(
#       left_dir=Path("my_data/left"), right_dir=Path("my_data/right"), board=board,
#   )
# This notebook uses the ChArUco scene bundled in the repository.
scene_root = Path("dataset/v0_png/train/scene_0000")
if not scene_root.exists():
    raise FileNotFoundError(
        f"Sample dataset not found at {scene_root.resolve()}. "
        "Run this notebook from the StereoComplex repository root "
        "(see docs/BRING_YOUR_OWN_DATA.md to use your own images)."
    )

report = sc.compare_opencv_stereo_calibration(
    left_dir=scene_root / "left",
    right_dir=scene_root / "right",
    board=board,
    max_pairs=5,
)
print(f"Raw stereo RMS:     {report['raw']['stereo_rms_px']:.3f} px")
print(f"Refined stereo RMS: {report['refined']['stereo_rms_px']:.3f} px")
print(f"Improvement:        {report['improvement_px']:.3f} px")

# %% [markdown]
# ## 3 — Check calibration quality

# %%
assessment = sc.assess_calibration(report["refined_result"])
print(f"Status: {assessment.status}")
for m in assessment.messages:
    print(f"  • {m}")
for r in assessment.recommendations:
    print(f"  → {r}")

# %% [markdown]
# ## 4 — Export to OpenCV format

# %%
K1, d1, K2, d2, R, T = report["refined_result"].to_opencv()
print(f"K_left   = {K1[0, 0]:.1f} px")
print(f"Baseline = {np.linalg.norm(T):.1f} mm")

# %% [markdown]
# ## 5 — Next steps
#
# - **If stereo RMS < 0.2 px**: your calibration is good.  Export and use.
# - **If stereo RMS > 0.3 px with structured residuals**: try the central
#   rayfield pipeline → `sc.calibrate_central(...)`.
# - **If you suspect non-central optics** (microscope, plate, tilted sensor):
#   try `sc.calibrate_noncentral(...)` → then `sc.identify_optics(...)`.
#
# See `docs/FROM_OPENCV_TO_STEREOCOMPLEX.md` for the full guide.
