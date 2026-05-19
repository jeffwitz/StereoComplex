"""Optical-diagram generators driven by the physical model code.

Each diagram in :mod:`stereocomplex.viz.figures` is a pure matplotlib 2-D
function that takes the same model instances used for fitting, so the
diagrams stay numerically consistent with the calibration pipeline.

Usage::

    import matplotlib.pyplot as plt
    from stereocomplex.viz.figures import diagram_cmo_physical

    fig, ax = plt.subplots(figsize=(5, 4))
    diagram_cmo_physical(ax, f_obj=80, working_distance=120, b=8)
    fig.savefig("cmo_physical.svg")
"""

from stereocomplex.viz.figures import (  # noqa: F401
    diagram_cmo_physical,
    diagram_greenough,
    diagram_pinhole_stereo,
)
