StereoComplex
=============

Improve stereo calibration when OpenCV plateaus (blur, distortion, compression)
-------------------------------------------------------------------------------

StereoComplex is a practical toolkit to **refine ChArUco corners before calibration** using a geometric prior on the board plane
(`rayfield_tps_robust`: homography + smooth residual field + robust fitting).

This simple “2D cleanup” step is often enough to make classic OpenCV calibration **much more stable** on challenging data.

.. figure:: assets/rayfield_worked_example/micro_overlays/left_best_frame000000.png
   :alt: Micro overlay showing GT (green), OpenCV raw (red), ray-field (blue)
   :width: 85%

   GT (green) vs OpenCV raw (red) vs ray-field (blue) on a zoomed ChArUco corner.

.. rubric:: Quickstart (what most users want)

1) Refine corners (exports JSON + an OpenCV-ready NPZ):

.. code-block:: bash

   .venv/bin/python -m stereocomplex.cli refine-corners dataset/v0_png \
     --split train --scene scene_0000 \
     --method rayfield_tps_robust \
     --out-json paper/tables/refined_corners_scene0000.json \
     --out-npz paper/tables/refined_corners_scene0000_opencv.npz

2) Run the reproducible OpenCV evaluation (raw vs ray-field) on the same scene:

.. code-block:: bash

   .venv/bin/python paper/experiments/compare_opencv_calibration_rayfield.py dataset/v0_png \
     --split train --scene scene_0000 \
     --out paper/tables/opencv_calibration_rayfield.json

.. rubric:: Documentation map

- I want a practical guide: :doc:`FIX_MY_CALIBRATION`
- I want the full worked example (plots + overlays): :doc:`RAYFIELD_WORKED_EXAMPLE`
- I want stereo/3D metrics and baseline-in-pixels: :doc:`STEREO_RECONSTRUCTION`
- I want to load/export models and reconstruct via an API: :doc:`RECONSTRUCTION_API`
- I want the math and ray-based calibration: :doc:`RAYFIELD3D_RECONSTRUCTION`

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   START_HERE
   FIX_MY_CALIBRATION
   ARCHITECTURE
   DATASET_SPEC
   CONVENTIONS

.. toctree::
   :maxdepth: 2
   :caption: ChArUco / Ray-field
   :hidden:

   CHARUCO_IDENTIFICATION
   RAYFIELD_WORKED_EXAMPLE

.. toctree::
   :maxdepth: 2
   :caption: Calibration / 3D
   :hidden:

   STEREO_RECONSTRUCTION
   ROBUSTNESS_SWEEP
   RECONSTRUCTION_API
   RAYFIELD3D_RECONSTRUCTION
