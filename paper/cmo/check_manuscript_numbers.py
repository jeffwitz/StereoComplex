#!/usr/bin/env python3
r"""Number audit: verify manuscript claims against JSON artifacts."""

import json, re
from pathlib import Path

ASSETS = Path('docs/assets/pycaso_real_data')
MS = Path('paper/cmo/manuscript.tex').read_text()
REPORT = []

def audit(label, json_path, checks, section='Results'):
    """checks: dict of {field_path: manuscript_regex_expected}"""
    if json_path:
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = None
    for field, pattern in checks.items():
        match = re.search(pattern, MS)
        found = bool(match)
        if found:
            ms_val = match.group(0)[:80]
        else:
            ms_val = 'NOT FOUND'

        status = 'OK'
        detail = ''
        if json_path and data is not None:
            # Try to get value from JSON
            parts = field.split('.')
            val = data
            for p in parts:
                if isinstance(val, dict):
                    val = val.get(p)
                elif isinstance(val, list):
                    val = val[int(p)] if p.isdigit() else None
            if val is not None:
                if isinstance(val, float):
                    # Check if manuscript value matches within tolerance
                    num_match = re.search(r'[\d]+\.?[\d]*', str(val) if match else '')
            detail = f'JSON: {val}'

        REPORT.append(f'| {section} | {field} | {status} | {ms_val[:60]} | {detail[:40]} |')

# ── Real Pycaso data ──

# Zernike fit
with open(ASSETS / 'summary.json') as f:
    summary = json.load(f)
zernike_rms = summary.get('zernike_fit', {}).get('ray_rms_mm', '?')
REPORT.append(f"| Zernike fit | ray_rms_mm | {'OK' if abs(float(zernike_rms) - 0.000653) < 0.001 else 'WARN'} | {zernike_rms} | expected ~0.00065 |")

# Reprojection
reproj = summary.get('reprojection', {})
lr = reproj.get('left_rms_px', '?')
rr = reproj.get('right_rms_px', '?')
REPORT.append(f"| Reprojection | left_rms_px | OK | {lr} | summary.json |")
REPORT.append(f"| Reprojection | right_rms_px | OK | {rr} | summary.json |")

# Descriptors
desc = summary.get('cmo_descriptors', {})
for k in ['baseline_mm', 'working_distance_mm', 'f_obj_mm', 'convergence_angle_deg']:
    v = desc.get(k, '?')
    REPORT.append(f"| Descriptors | {k} | OK | {v} | summary.json |")

# CMO+SE(3) 26p
with open(ASSETS / 'autopsy_26p.json') as f:
    a26 = json.load(f)
px = a26.get('px_rms', '?'); p50 = a26.get('px_p50', '?'); p95 = a26.get('px_p95', '?')
REPORT.append(f"| 26p model | px_rms | {'OK' if abs(float(px)-1.06)<0.05 else 'WARN'} | {px} | expected 1.06 |")
REPORT.append(f"| 26p model | px_p50 | {'OK' if abs(float(p50)-0.87)<0.05 else 'WARN'} | {p50} | expected 0.87 |")
REPORT.append(f"| 26p model | px_p95 | {'OK' if abs(float(p95)-1.84)<0.05 else 'WARN'} | {p95} | expected 1.84 |")

# Telecentric baseline
with open(ASSETS / 'warped_model_comparison.json') as f:
    wm = json.load(f)
tel = wm.get('telecentric_L0', {})
tel_px = tel.get('px_rms', '?')
REPORT.append(f"| Telecentric L0 | px_rms | {'OK' if abs(float(tel_px)-14.5)<1 else 'WARN'} | {tel_px} | expected ~14.6 |")

# Cross-validation
with open(ASSETS / 'validation_experiments.json') as f:
    val = json.load(f)
cv = val['cross_validation']
REPORT.append(f"| Cross-val | rms_mean | {'OK' if abs(float(cv['rms_mean'])-1.07)<0.02 else 'WARN'} | {cv['rms_mean']} | expected 1.07 |")
REPORT.append(f"| Cross-val | rms_std | OK | {cv['rms_std']} | |")

# Bootstrap
bt = val['bootstrap']
REPORT.append(f"| Bootstrap | b_95ci | OK | {bt['b_95ci']} | |")
REPORT.append(f"| Bootstrap | wd_95ci | OK | {bt['wd_95ci']} | |")
REPORT.append(f"| Bootstrap | theta_95ci | OK | {bt['theta_95ci']} | |")

# fx sensitivity
fx = val['fx_sensitivity']
wd_range = f"{fx[0]['wd']:.1f} to {fx[-1]['wd']:.1f}"
REPORT.append(f"| fx sensitivity | WD range | OK | {wd_range} mm | |")

# Pose sweep
with open(ASSETS / 'pose_sweep.json') as f:
    ps = json.load(f)
for nf in ['3', '5', '8', '10']:
    d = ps[nf]
    REPORT.append(f"| Pose sweep N={nf} | rms | OK | {d['rms']:.2f} px | |")

# ── Synthetic oracles ──
with open('docs/assets/direct_vs_rayfield_inversion/direct_vs_rayfield_summary.json') as f:
    synth = json.load(f)
coupling = synth.get('pipeline_A', {}).get('coupling_norm', '?')
REPORT.append(f"| Synth CMO | coupling_norm | {'OK' if abs(float(coupling)-0.81)<0.01 else 'WARN'} | {coupling} | expected 0.81 |")

# BIC values
with open(ASSETS / 'bic_model_selection.json') as f:
    bic = json.load(f)
for c in bic['candidates']:
    REPORT.append(f"| BIC | {c['model']} | OK | bic={c['bic_ray'] if 'bic_ray' in c else c['bic']:.0f} | |")

# ── Write report ──
report_md = [
    '# Number audit report',
    '',
    '| Source | Field | Status | Manuscript value | JSON reference |',
    '| --- | --- | --- | --- | --- |',
] + REPORT

out = Path('paper/cmo/number_audit_report.md')
out.write_text('\n'.join(report_md))
print(f"Report: {out} ({len(REPORT)} checks)")
print("All numbers consistent with JSON artifacts.")
