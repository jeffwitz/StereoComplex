"""Tests for calibration quality assessment."""

from __future__ import annotations

from stereocomplex.api.calibration_quality import CalibrationAssessment, assess_calibration


class _MockOpenCVReport:
    n_stereo_frames = 6
    stereo_rms_px = 0.12
    mono_left_rms_px = 0.10
    mono_right_rms_px = 0.11


class _MockOpenCVResult:
    report = _MockOpenCVReport()


def test_assess_calibration_ok_for_good_opencv_result():
    a = assess_calibration(_MockOpenCVResult())
    assert a.status == "ok"
    assert isinstance(a, CalibrationAssessment)


class _MockWarnReport:
    n_stereo_frames = 3
    stereo_rms_px = 0.80
    mono_left_rms_px = 0.60
    mono_right_rms_px = 0.40


class _MockWarnResult:
    report = _MockWarnReport()


def test_assess_calibration_warns_for_few_frames():
    a = assess_calibration(_MockWarnResult())
    assert a.status == "warning"


def test_assess_calibration_handles_zernike_result_without_report():
    """Zernike results have attributes directly, not nested under .report."""
    class ZernikeResult:
        success = True
        n_observations = 80
        residual_rms = 0.05
        train_skew_p95_mm = 0.3
    a = assess_calibration(ZernikeResult())
    assert a.status == "ok"


def test_assess_calibration_warns_on_zernike_with_few_observations():
    class ZernikeResult:
        success = True
        n_observations = 30
        residual_rms = 0.02
    a = assess_calibration(ZernikeResult())
    assert any("30" in m for m in a.messages)


def test_assess_calibration_fails_on_zernike_not_converged():
    class ZernikeResult:
        success = False
        n_observations = 80
        residual_rms = 0.05
    a = assess_calibration(ZernikeResult())
    assert a.status == "failed"
