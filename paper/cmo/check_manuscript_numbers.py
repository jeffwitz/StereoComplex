#!/usr/bin/env python3
"""Fail on drift between audited inputs, generated numbers and manuscript assets.

This validates source binding, numeric closure and the executed experiment.
It does not automatically validate arbitrary prose or physical accuracy.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parent.parent
spec=importlib.util.spec_from_file_location('assets',PAPER/'analysis/build_diagnostic_assets.py')
assets=importlib.util.module_from_spec(spec);spec.loader.exec_module(assets)
errors=[]
for rel,expected in assets.text_assets().items():
    path=PAPER/rel
    if not path.exists() or path.read_text()!=expected:
        errors.append(f'Generated number mismatch: {rel}')
r=json.loads((PAPER/'results/diagnostic_audit.json').read_text())
for rel,expected in r['sources'].items():
    actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    if actual!=expected: errors.append(f'Input hash changed: {rel}')
for name,row in r['models'].items():
    if row['inverse_max_closure_mm']>1e-10: errors.append(f'Inverse failed: {name}')
    if name!='reference_field':
        fractions=row['reserved']['direction_energy_fraction_constant_linear_quadratic_remainder']
        if abs(sum(fractions)-1)>1e-12: errors.append(f'Energy does not sum to one: {name}')
for label,g in r['gauges'].items():
    if g['max_intersection_difference_mm']>1e-10: errors.append(f'Gauge check failed: {label}')
for label,c in r['centrality'].items():
    if c['nonconstant_origin_coeff_max']!=0 or c['max_distance_to_common_centre_mm']>1e-10:
        errors.append(f'Centrality check failed: {label}')
for label,c in r['optimisation'].items():
    if not all(t['success'] for t in c['trials']): errors.append(f'Optimisation failed: {label}')
if len(r['simulation']['records'])!=40: errors.append('Wrong simulation count')
if not all(row['rigid_success'] and row['rigid_quadratic_success'] for row in r['simulation']['records']):
    errors.append('Unconverged simulation')
for tex in ['manuscript.tex','supplementary.tex']:
    s=(PAPER/tex).read_text()
    for target in re.findall(r'\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}',s):
        if not (PAPER/target).exists() and not (PAPER/(target+'.tex')).exists():
            errors.append(f'Missing included file: {target}')
    if re.search(r'Section~[0-9]',s): errors.append(f'Hard-coded section reference: {tex}')
if errors:
    raise SystemExit('\n'.join(errors))
print('PASS: generated numbers, input hashes, inverse closure, gauges, centrality, convergence and assets.')
