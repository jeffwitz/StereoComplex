"""Smoke tests: all optical diagrams must render without errors."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")


@pytest.mark.parametrize("builder,kwargs", [
    ("pinhole", {"O_left": (-6, 0), "O_right": (6, 0), "specimen": (0, 80)}),
    ("cmo_physical", {"f_obj": 80, "working_distance": 120, "b": 8, "exaggerated": True}),
    ("greenough", {"O_left": (-5, 0), "O_right": (5, 0), "specimen": (0, 60)}),
])
def test_diagram_renders_without_error(builder, kwargs):
    from stereocomplex.viz.figures import (
        diagram_cmo_physical,
        diagram_greenough,
        diagram_pinhole_stereo,
    )

    builders = {
        "pinhole": diagram_pinhole_stereo,
        "cmo_physical": diagram_cmo_physical,
        "greenough": diagram_greenough,
    }

    fig, ax = plt.subplots(figsize=(4, 3))
    builders[builder](ax=ax, **kwargs)
    # Beyond "does not raise": the builder must actually draw something.
    assert ax.has_data(), f"{builder} diagram drew no artists"
    plt.close(fig)


def test_cmo_diagram_from_real_model():
    """The CMO diagram must accept a real CMOPhysicalStereoModel instance."""
    from stereocomplex.physics.cmo_physical import CMOPhysicalStereoModel
    from stereocomplex.viz.figures import diagram_cmo_physical

    model = CMOPhysicalStereoModel(
        f_obj_mm=80, working_distance_mm=120, b_mm=8,
        f_tube_mm=50, cx_principal_px=320, cy_principal_px=240,
        pixel_pitch_mm=0.005, image_size=(640, 480),
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    diagram_cmo_physical(ax, model=model, exaggerated=False)
    assert ax.has_data(), "CMO diagram drew no artists"
    plt.close(fig)
