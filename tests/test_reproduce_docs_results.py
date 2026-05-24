"""Unit tests for the documentation reproducibility helper."""

from __future__ import annotations

from pathlib import Path

from examples.reproduce_docs_results import (
    format_direct_vs_rayfield_summary,
    load_json_summary,
    require_paths,
)


def test_require_paths_accepts_existing(tmp_path):
    f = tmp_path / "exists.txt"
    f.write_text("ok")
    require_paths([f])  # should not raise


def test_require_paths_reports_missing(tmp_path):
    missing = tmp_path / "missing.txt"
    try:
        require_paths([missing])
    except FileNotFoundError:
        return
    raise AssertionError("should have raised FileNotFoundError")


def test_load_json_summary_returns_dict(tmp_path):
    f = tmp_path / "s.json"
    f.write_text('{"oracle":"test"}')
    d = load_json_summary(f)
    assert d["oracle"] == "test"


def test_load_json_summary_returns_empty_on_missing():
    d = load_json_summary(Path("/nonexistent/path.json"))
    assert d == {}


def test_format_summary_contains_key_fields():
    s = {
        "oracle": "CMO",
        "n_poses": 2,
        "noise_std_px": 0.0,
        "pipeline_A": {"rms_px": 0.04, "converged": True, "elapsed_s": 7.0},
        "pipeline_B": {"rayfield_winner": "cmo", "zernike_rms_mm": 0.001, "zernike_elapsed_s": 50.0},
        "rayfield_correct": True,
    }
    out = format_direct_vs_rayfield_summary(s)
    assert "CMO" in out
    assert "cmo" in out
    assert "0.04" in out
    assert "True" in out
