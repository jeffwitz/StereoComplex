StereoComplex
=============

Robust ChArUco calibration and ray-based stereo reconstruction
----------------------------------------------------------------

StereoComplex is a lightweight Python toolkit for robust stereo calibration,
Ray2D ChArUco refinement, ray-based 3D reconstruction, and non-central optical
model diagnostics.

If you already know OpenCV calibration, start with the onboarding notebook:

.. code-block:: text

   examples/notebooks/00_getting_started.ipynb

It shows the shortest path: define a ChArUco board, compare raw OpenCV against
Ray2D-refined calibration, assess calibration quality, and export back to
OpenCV format.

.. rubric:: Video 1 (overview)

.. raw:: html

   <iframe width="560" height="315"
     src="https://www.youtube.com/embed/8Mbcv5l7ek0"
     title="StereoComplex overview"
     frameborder="0"
     referrerpolicy="strict-origin-when-cross-origin"
     allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
     allowfullscreen></iframe>

.. rubric:: What StereoComplex does

.. list-table::
   :header-rows: 1

   * - Layer
     - Meaning
     - Status
   * - Ray2D / planar ray-field
     - Homography plus smooth residual field on the calibration board plane
     - stable practical preprocessing
   * - Central 3D ray-field
     - Pixel → 3D direction with a shared camera center
     - research prototype
   * - Non-central 3D rayfield
     - Pixel → 3D line ``(O(u,v), d(u,v))``
     - research-grade, validated on synthetic oracles and one real CMO case study

The 2D Ray2D refinement is not itself a 3D non-central camera model. It improves
the image observations fed to calibration. The non-central backend is a separate
3D line-based model.

Engineering footprint: no ROS, no Docker requirement, no C++ toolchain; the core
is a Python package using standard scientific libraries.

.. rubric:: Recommended path

1. **First use / OpenCV migration** → :doc:`FROM_OPENCV_TO_STEREOCOMPLEX` and
   ``examples/notebooks/00_getting_started.ipynb``.
2. **Fix ChArUco / OpenCV calibration** → :doc:`FIX_MY_CALIBRATION`,
   :doc:`BRING_YOUR_OWN_DATA`, and ``examples/notebooks/01_ray2d_vs_opencv.ipynb``.
3. **Ray-based 3D reconstruction** → :doc:`RAYFIELD3D_RECONSTRUCTION` and
   ``examples/notebooks/02_ray3d.ipynb``.
4. **Non-central calibration** → :doc:`NONCENTRAL_FROM_IMAGES` and
   ``examples/notebooks/04_parallel_plate_origin_field.ipynb``.
5. **Optics identification / CMO case study** → :doc:`REAL_CMO_PYCASO_RAYFIELD`,
   :doc:`CMO_PHYSICAL_MODEL`, and ``examples/notebooks/09_pycaso_real_data.ipynb``.

.. rubric:: Key contributions

1. **Robust ChArUco refinement without requiring a camera model.** Ray2D /
   ``rayfield_tps_robust`` uses a homography plus a smooth residual field on the
   board plane.
2. **OpenCV-compatible stereo calibration diagnostics.** StereoComplex compares
   raw and refined ChArUco points and reports stereo/reconstruction metrics.
3. **Central 3D ray-field reconstruction.** The central backend learns a compact
   pixel-to-ray direction model and triangulates from rays.
4. **Non-central Zernike rayfield calibration.** Each pixel can define a 3D line
   instead of sharing one optical center.
5. **Ray-space optical model identification.** A measured Zernike rayfield can be
   used to compare compact physical hypotheses such as Brown-Conrady, inclined
   plate, CMO-like, and generic fallback rayfields.

.. figure:: assets/rayfield_worked_example/zoom_overlays/left_best_ideal_vs_realistic_frame000000.png
   :alt: Ideal (no blur) vs realistic (dataset) corner overlays, raw vs ray-field
   :width: 85%

   Same GT (with geometric distortion) on a strict ideal render (top: no blur/no
   noise, nearest texture sampling) vs realistic (bottom), raw vs ray-field.

.. rubric:: Key result: non-central rendered-image benchmark

On the inclined-plate benchmark, raw OpenCV ChArUco detections impose a high
reconstruction floor. With Ray2D-refined observations, the same non-central
bundle adjustment reaches sub-millimetric reconstruction accuracy:

- OpenCV raw: central RMS ≈ 4.21 mm, oracle-detected RMS ≈ 3.44 mm, non-central BA RMS ≈ 3.36 mm.
- Ray2D refined: central RMS ≈ 2.50 mm, oracle-detected RMS ≈ 0.76 mm, non-central BA RMS ≈ 0.66 mm.

Interpretation: the non-central model works when the 2D observations are good
enough; front-end quality is the limiting factor on rendered or real images.

.. rubric:: Key result: real CMO microscope data

The Pycaso CMO microscope case study is the main real-data validation of the
non-central workflow. It separates a flexible measured rayfield from a compact
physical interpretation:

.. list-table::
   :header-rows: 1

   * - Model / method
     - Role
     - RMS
   * - Standard OpenCV stereo on the tested setup
     - central baseline
     - >300 px
   * - Perspective CMO physical model
     - wrong optical family
     - ~86 px
   * - Telecentric CMO 14p
     - correct family but not usable
     - ~14.6 px
   * - Telecentric CMO + per-arm SE(3), 26p
     - compact usable physical model
     - 1.06 px
   * - Zernike O(0)+d(2), 57p
     - flexible rayfield reference
     - 0.47 px

The key methodological point is that the rayfield is not only a calibration
model; it is also a diagnostic instrument used to build and falsify compact
physical optical models.

See :doc:`REAL_CMO_PYCASO_RAYFIELD`, :doc:`CMO_PHYSICAL_MODEL`, and
``examples/notebooks/09_pycaso_real_data.ipynb``.

.. rubric:: Notebook walkthroughs

If you want a guided, visual introduction before reading the code or the
advanced case studies, start with the notebook series:

- **``00_getting_started.ipynb``**: first OpenCV-to-StereoComplex onboarding.
- ``01_ray2d_vs_opencv.ipynb``: raw OpenCV detections vs Ray2D refinement.
- ``02_ray3d.ipynb``: compact central 3D ray-field and compression sweeps.
- ``03_rayfield_virtual_rectification.ipynb``: virtual rectification maps for dense stereo.
- ``04_parallel_plate_origin_field.ipynb``: inclined-plate non-central oracle.
- ``05_noncentral_calibration_from_images.ipynb``: practical non-central image-folder workflow.
- ``06_cmo_model_selection.ipynb``: CMO-like rayfield measurement and model selection.
- ``09_pycaso_real_data.ipynb``: real Pycaso CMO microscope case study.

Companion ``.py`` exports are stored next to the notebooks for quick inspection
in any text editor. See :doc:`NOTEBOOKS` for the complete sequence, including
scripts 07 and 08.

.. rubric:: Quickstart from OpenCV

.. code-block:: python

   from pathlib import Path
   import stereocomplex as sc

   board = sc.CharucoBoardSpec(
       squares_x=11,
       squares_y=7,
       square_size_mm=39.07,
       marker_size_mm=27.35,
       aruco_dictionary="DICT_4X4_1000",
   )

   report = sc.compare_opencv_stereo_calibration(
       left_dir=Path("my_data/left"),
       right_dir=Path("my_data/right"),
       board=board,
   )

   assessment = sc.assess_calibration(report["refined_result"])
   K1, d1, K2, d2, R, T = report["refined_result"].to_opencv()

.. rubric:: Documentation map

- Getting started:
  :doc:`START_HERE`,
  :doc:`TUTORIAL`,
  :doc:`FROM_OPENCV_TO_STEREOCOMPLEX`,
  :doc:`BRING_YOUR_OWN_DATA`,
  :doc:`FIX_MY_CALIBRATION`,
  :doc:`NOTEBOOKS`.
- ChArUco and 2D refinement:
  :doc:`CHARUCO_IDENTIFICATION`,
  :doc:`RAYFIELD_WORKED_EXAMPLE`.
- Ray-based calibration and reconstruction:
  :doc:`STEREO_RECONSTRUCTION`,
  :doc:`RECONSTRUCTION_API`,
  :doc:`RAYFIELD3D_RECONSTRUCTION`,
  :doc:`RAYFIELD_VIRTUAL_RECTIFY`.
- Non-central and optical model identification:
  :doc:`NONCENTRAL_FROM_IMAGES`,
  :doc:`IDENTIFY_MY_OPTICS`,
  :doc:`CMO_MODEL_SELECTION`,
  :doc:`DIRECT_VS_RAYFIELD_INVERSION`,
  :doc:`PARALLEL_PLATE_ORIGIN_FIELD`.
- Real-data CMO validation:
  :doc:`REAL_CMO_PYCASO_RAYFIELD`,
  :doc:`CMO_PHYSICAL_MODEL`,
  :doc:`PYCASO_Z_SWEEP`.
- Project status:
  :doc:`VALIDATION_STATUS`,
  :doc:`ROADMAP`.
- Reference:
  :doc:`PUBLIC_API`,
  :doc:`ARCHITECTURE`,
  :doc:`DATASET_SPEC`,
  :doc:`CONVENTIONS`,
  :doc:`ALTERNATIVES_POSITIONING`,
  :doc:`LICENSE`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   START_HERE
   TUTORIAL
   FROM_OPENCV_TO_STEREOCOMPLEX
   BRING_YOUR_OWN_DATA
   FIX_MY_CALIBRATION
   NOTEBOOKS

.. toctree::
   :maxdepth: 2
   :caption: ChArUco / Ray-field
   :hidden:

   CHARUCO_IDENTIFICATION
   RAYFIELD_WORKED_EXAMPLE

.. toctree::
   :maxdepth: 2
   :caption: Ray-based calibration / 3D
   :hidden:

   STEREO_RECONSTRUCTION
   RECONSTRUCTION_API
   RAYFIELD3D_RECONSTRUCTION
   RAYFIELD_VIRTUAL_RECTIFY

.. toctree::
   :maxdepth: 2
   :caption: Non-central / optical models
   :hidden:

   NONCENTRAL_FROM_IMAGES
   IDENTIFY_MY_OPTICS
   CMO_MODEL_SELECTION
   DIRECT_VS_RAYFIELD_INVERSION
   PARALLEL_PLATE_ORIGIN_FIELD
   COMPRESSION_RECONSTRUCTION
   ROBUSTNESS_SWEEP

.. toctree::
   :maxdepth: 2
   :caption: Real-data CMO case study
   :hidden:

   REAL_CMO_PYCASO_RAYFIELD
   CMO_PHYSICAL_MODEL
   PYCASO_Z_SWEEP

.. toctree::
   :maxdepth: 2
   :caption: Project status
   :hidden:

   VALIDATION_STATUS
   ROADMAP

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   PUBLIC_API
   ARCHITECTURE
   DATASET_SPEC
   CONVENTIONS
   ALTERNATIVES_POSITIONING
   LICENSE
