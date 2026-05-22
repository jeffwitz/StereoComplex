#!/usr/bin/env python3
"""Pipeline diagram: 4-stage calibration workflow (Figure 2).

Static flowchart showing the two-stage decomposition:
Rayfield measurement (Ray2D preprocessing + Zernike BA) → Physical model
identification (ray-space BIC → CMO+SE(3) model).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11})
OUT = Path('paper/cmo/figures')

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# Color scheme
BOX_STYLE = dict(boxstyle='round,pad=0.6', facecolor='#e8f0fe', edgecolor='#2a55c7', linewidth=2)
ARROW_COLOR = '#555555'
GROUP_STYLE = dict(facecolor='#f5f5f5', edgecolor='#aaaaaa', linewidth=1.5, linestyle='--')

# ── Stage boxes ──
# Row 1: Ray2D → Zernike BA
# Row 2: BIC selection → CMO+SE3 model
boxes = [
    (1.0, 4.5, 5.0, 1.8, 'Ray2D Preprocessing\nChArUco detection +\ndouble TPS denoising\n(165 corners × 10 frames)', '#cfe2ff'),
    (8.0, 4.5, 5.0, 1.8, 'Zernike Rayfield BA\nPer-pixel ray fitting\nO(u,v)+d(u,v), O·d=0\ntransverse gauge', '#cfe2ff'),
    (1.0, 1.5, 5.0, 1.8, 'Ray-Space BIC\nModel selection in ray space\nTelecentric CMO wins\n(ΔBIC > 40,000)', '#d4edda'),
    (8.0, 1.5, 5.0, 1.8, 'CMO + SE(3) Model\nJoint telecentric fit +\nper-channel arm alignment\n(26 params, 1.06 px RMS)', '#d4edda'),
]

for x, y, w, h, text, color in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                                     facecolor=color, edgecolor='#444444', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10,
            family='monospace' if 'BIC' not in text else 'serif')

# ── Arrows ──
# Row 1: Box1 → Box2
ax.annotate('', xy=(8.0, 5.4), xytext=(6.0, 5.4),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2.5))
ax.text(7.0, 5.7, 'pixel\ncorrespondences', ha='center', va='bottom', fontsize=8, color=ARROW_COLOR)

# Row 1→Row 2: Box2 → Box3
ax.annotate('', xy=(3.5, 3.3), xytext=(10.5, 4.5),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2.5,
                            connectionstyle='arc3,rad=-0.3'))
ax.text(6.5, 3.8, 'Zernike rayfield\n(per-pixel rays)', ha='center', va='center', fontsize=8, color=ARROW_COLOR)

# Row 2: Box3 → Box4
ax.annotate('', xy=(8.0, 2.4), xytext=(6.0, 2.4),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2.5))
ax.text(7.0, 2.7, 'CMO family +\nparameter seed', ha='center', va='bottom', fontsize=8, color=ARROW_COLOR)

# ── Group backgrounds ──
group1 = mpatches.FancyBboxPatch((0.3, 4.0), 13.4, 2.8, boxstyle='round,pad=0.2',
                                   facecolor='#f0f4ff', edgecolor='#2a55c7', linewidth=2, linestyle='--')
ax.add_patch(group1)
ax.text(0.6, 6.4, 'Rayfield Measurement (2D)', fontsize=13, fontweight='bold', color='#2a55c7', va='center')

group2 = mpatches.FancyBboxPatch((0.3, 1.0), 13.4, 2.8, boxstyle='round,pad=0.2',
                                   facecolor='#f0fff0', edgecolor='#2a7a3b', linewidth=2, linestyle='--')
ax.add_patch(group2)
ax.text(0.6, 3.4, 'Physical Model Identification (3D)', fontsize=13, fontweight='bold', color='#2a7a3b', va='center')

# ── Input/Output labels ──
ax.text(0.3, 7.0, 'INPUT', fontsize=10, fontweight='bold', color='#666666', va='center')
ax.text(0.3, 6.55, '10 stereo pairs\nChArUco board', fontsize=9, color='#666666', va='top')

ax.text(13.8, 0.5, 'OUTPUT', fontsize=10, fontweight='bold', color='#2a7a3b', va='center')
ax.text(13.8, 0.05, 'Calibrated rays\n+ physical model', fontsize=9, color='#666666', va='top')

fig.tight_layout(pad=0.5)
fig.savefig(OUT / 'pipeline.pdf', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('pipeline.pdf OK')
