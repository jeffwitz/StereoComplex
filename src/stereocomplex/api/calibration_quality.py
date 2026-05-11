"""Calibration quality assessment — tells the user whether their result is usable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CalibrationAssessment:
    """Structured quality assessment for a stereo calibration result.

    Attributes
    ----------
    status : str
        ``"ok"``, ``"warning"``, or ``"failed"``.
    messages : list of str
        Human-readable diagnostic messages.
    recommendations : list of str
        Actionable suggestions.
    """

    status: str  # "ok" | "warning" | "failed"
    messages: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def assess_calibration(result: Any) -> CalibrationAssessment:
    """Assess whether a stereo calibration result is usable.

    Works with ``StereoOpenCVCalibrationResult``,
    ``StereoCentralRayFieldFitResult``, and
    ``StereoZernikeOriginFieldFitResult``.

    Parameters
    ----------
    result :
        A calibration result object with a ``.report`` attribute.

    Returns
    -------
    CalibrationAssessment
    """
    msgs: list[str] = []
    recs: list[str] = []
    report = getattr(result, "report", result)  # fall back to result itself for Zernike fits

    # --- frame count ---
    n_frames = getattr(report, "n_stereo_frames", None) or getattr(report, "n_initialized_frames", None) or 0
    if n_frames < 4:
        msgs.append(f"Only {n_frames} stereo frames used (minimum 4 recommended).")
        recs.append("Capture more calibration images with the board in different poses.")
    if n_frames < 2:
        return CalibrationAssessment(status="failed", messages=msgs, recommendations=recs)

    # --- stereo RMS ---
    stereo_rms = getattr(report, "stereo_rms_px", None)
    if stereo_rms is not None:
        if stereo_rms > 1.0:
            msgs.append(f"Stereo RMS is high ({stereo_rms:.2f} px).")
            recs.append("Check corner detection quality; try method2d='rayfield_tps_robust'.")
        elif stereo_rms > 0.3:
            msgs.append(f"Stereo RMS is moderate ({stereo_rms:.2f} px).")
            recs.append("Consider rayfield-based calibration for better accuracy.")

    # --- mono RMS ---
    for side, attr in [("left", "mono_left_rms_px"), ("right", "mono_right_rms_px")]:
        rms = getattr(report, attr, None)
        if rms is not None and rms > 0.5:
            msgs.append(f"{side.capitalize()} mono RMS is high ({rms:.2f} px).")

    # --- skew / non-central health ---
    skew = getattr(report, "train_skew_p95_mm", None)
    if skew is not None and skew > 1.0:
        msgs.append(f"Ray skew P95 is high ({skew:.2f} mm).")
        recs.append("The optics may be non-central; try calibrate_noncentral().")

    # --- point-to-ray ---
    ptr = getattr(report, "train_point_to_ray_p95_mm", None)
    if ptr is not None and ptr > 0.5:
        msgs.append(f"Point-to-ray P95 is high ({ptr:.2f} mm).")
        recs.append("Increase Zernike order or use more calibration poses.")

    # --- decision ---
    if any("high" in m or "only" in m for m in msgs):
        status = "warning"
    else:
        status = "ok"

    if not msgs:
        msgs.append("All health checks passed.")
        recs.append("Calibration is usable for 3D reconstruction.")

    return CalibrationAssessment(status=status, messages=msgs, recommendations=recs)
