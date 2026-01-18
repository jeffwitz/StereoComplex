# Alternatives and positioning

StereoComplex is designed to sit between minimal OpenCV calibration scripts and larger robotics / SfM toolchains.
It keeps an OpenCV-like installation footprint, but emphasizes robust stereo geometry, rectification quality, and explicit diagnostic metrics.

## OpenCV (camera & stereo calibration)

OpenCV is the baseline reference for mono and stereo calibration:

- Simple installation, stable and well-documented APIs, widespread adoption.

In practice, performance can plateau on degraded data (blur, compression, noise), and diagnostics are often limited to reprojection error summaries.
StereoComplex is compatible with OpenCV workflows and adds geometric corner refinement plus explicit metrics and plots on top.

## Kalibr (ETH Zurich)

Kalibr is a robotics-oriented calibration toolbox (camera, IMU) with rich models and global optimization.

- Strengths: mature methodology, multi-sensor focus, strong optimization backbone.
- Trade-offs: heavy dependencies and workflow (ROS/catkin/Docker-style environment), which can be costly to set up and maintain for stereo-only use.

StereoComplex targets lightweight stereo calibration workflows without requiring a robotics stack.

## Basalt (TUM)

Basalt is a VIO/SLAM research framework that includes calibration utilities.

- Strengths: modern optimization methods, strong performance in VIO contexts.
- Trade-offs: C++ research-oriented codebase, non-trivial build/configuration, calibration is not the primary standalone focus.

StereoComplex focuses specifically on stereo geometry, rectification quality, and calibration diagnostics.

## camodocal

camodocal is an academic multi-camera calibration toolbox.

- Strengths: multiple camera models, solid methodological foundations.
- Trade-offs: lower maintenance activity and dated ergonomics compared to newer pipelines; limited modern integration patterns.

StereoComplex focuses on a lightweight Python workflow with reproducible experiments and diagnostics.

## SfM toolchains (COLMAP, OpenMVG)

Structure-from-Motion toolchains are not designed around stereo calibration objectives:

- Intrinsics are typically optimized for reconstruction quality, not stereo rectification stability.
- Stereo constraints and rectification quality are not first-class objectives.

They can be useful for rough initialization or unconstrained environments, but are out of scope for StereoComplex.

## Non-goals (current scope)

- Not a SLAM or VIO framework
- Not a camera–IMU calibration toolbox
- Not a replacement for full robotics stacks
- Not a Structure-from-Motion pipeline

