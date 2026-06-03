"""Golden-value regression guard for the CMO paper's headline numbers.

The manuscript quotes a small set of story-level numbers (Zernike 57p RMS, the
recovered CMO geometry, the 26p reprojection RMS, the corner-BA improvement).
``docs/assets/cmo_paper/AUDIT.md`` maps each to a tracked computation artefact.
This test pins those artefacts to the published values: if a re-fit regenerates
a JSON and the number drifts away from what the paper states, this fails -- so
the manuscript and the committed data cannot silently diverge.

Only assets that are tracked in git (and therefore present on a fresh clone) are
used here; the gitignored ``schur_ba/`` diagnostics are intentionally excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ASSETS = Path("docs/assets/pycaso_real_data")


def _load(name: str):
    return json.loads((ASSETS / name).read_text(encoding="utf-8"))


def test_zernike_57p_descriptors_match_paper():
    """Zernike O(0)+d(2) row: RMS and recovered geometry quoted in the paper."""
    row0 = _load("zernike_order_sweep.json")[0]
    assert row0["p"] == 57  # parameter count quoted throughout the paper
    assert row0["rms"] == pytest.approx(0.47, abs=0.02)  # "0.47 px" floor
    assert row0["b"] == pytest.approx(24.9, abs=0.1)  # baseline (mm)
    assert row0["WD"] == pytest.approx(64.7, abs=0.1)  # working distance (mm)
    assert row0["angle"] == pytest.approx(22.6, abs=0.1)  # convergence angle (deg)


def test_cmo_26p_reprojection_matches_paper():
    """26p CMO+SE(3): 1.06 px RMS (P50 0.87, P95 1.84) before corner BA."""
    ref = _load("corner_ba_refinement.json")
    before = ref["before_rayfield"]
    assert before["px_rms"] == pytest.approx(1.06, abs=0.02)
    assert before["px_p50"] == pytest.approx(0.87, abs=0.02)
    assert before["px_p95"] == pytest.approx(1.84, abs=0.02)


def test_corner_ba_improvement_matches_paper():
    """Corner BA: 1.06 -> 0.88 px, a ~17% improvement."""
    ref = _load("corner_ba_refinement.json")
    assert ref["after_joint_ba"]["px_rms"] == pytest.approx(0.88, abs=0.02)
    assert ref["improvement_pct"] == pytest.approx(17.0, abs=0.5)
