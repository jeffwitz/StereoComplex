#!/usr/bin/env python3
"""Pareto frontier of gauge-regularized Zernike sweep (Figure 7).

Three-panel figure: (a) Pareto frontier of pixel RMS vs Z0 direction drift
with Pareto-optimal points highlighted, (b) baseline stability across
regularization strengths, (c) convergence angle stability.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10})
OUT = Path('paper/cmo/figures')

with open('docs/assets/pycaso_real_data/zernike_gauge_regularization_sweep.json') as f:
    sweep = json.load(f)

sweep_results = sweep.get('sweep', [])

# Find Pareto-optimal points (minimize both ray_rms_mm and drift_z0_deg)
pareto = []
for i, r in enumerate(sweep_results):
    dominated = False
    for j, r2 in enumerate(sweep_results):
        if i == j:
            continue
        if r2["ray_rms_mm"] <= r["ray_rms_mm"] and r2["drift_z0_deg"] <= r["drift_z0_deg"]:
            if r2["ray_rms_mm"] < r["ray_rms_mm"] or r2["drift_z0_deg"] < r["drift_z0_deg"]:
                dominated = True
                break
    if not dominated:
        pareto.append(r)

all_rms = [r["ray_rms_mm"] for r in sweep_results]
all_z0 = [r["drift_z0_deg"] for r in sweep_results]
pareto_rms = [r["ray_rms_mm"] for r in pareto]
pareto_z0 = [r["drift_z0_deg"] for r in pareto]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) Pareto frontier: RMS vs Z0 drift
ax = axes[0]
ax.scatter(all_rms, all_z0, c="steelblue", s=60, zorder=3, label="all runs")
ax.scatter(pareto_rms, pareto_z0, c="darkorange", s=120, zorder=4,
           edgecolors="black", linewidth=0.5, label="Pareto-optimal")
po_sorted = sorted(pareto, key=lambda r_: r_["ray_rms_mm"])
if len(po_sorted) >= 2:
    ax.plot([r_["ray_rms_mm"] for r_ in po_sorted],
            [r_["drift_z0_deg"] for r_ in po_sorted],
            "darkorange", lw=1.5, alpha=0.6, zorder=2)
constrained_rms = sweep.get('constrained_rms_mm', 0.000653)
ax.axvline(x=constrained_rms, color="gray", ls="--", alpha=0.5, label="constrained ref")
ax.set_xlabel("Ray RMS (mm)")
ax.set_ylabel("Z$_0$ drift (deg)")
ax.set_title("Pareto frontier: RMS vs gauge drift")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) Baseline stability
ax = axes[1]
for r in sweep_results:
    ax.plot(r["drift_z0_deg"], r.get("baseline_mm", 0), "o", ms=8, alpha=0.7)
ax.set_xlabel("Z$_0$ drift (deg)")
ax.set_ylabel("Baseline (mm)")
ax.set_title("Baseline stability")
ax.grid(True, alpha=0.3)

# (c) Convergence angle stability
ax = axes[2]
for r in sweep_results:
    ax.plot(r["drift_z0_deg"], r.get("convergence_angle_deg", 0), "o", ms=8, alpha=0.7)
ax.set_xlabel("Z$_0$ drift (deg)")
ax.set_ylabel("Convergence angle (deg)")
ax.set_title("Convergence angle stability")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'pareto_gauge_regularization.pdf', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('pareto_gauge_regularization.pdf OK')
