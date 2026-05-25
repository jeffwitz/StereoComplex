# Explanation

Background, design choices, and scientific positioning.  These pages explain
*why*, not *how*.  No code recipes — see the [How‑to guides](../how_to/INDEX) 
for task‑oriented instructions.

| Page | What it explains |
|---|---|
| [Why rayfields](why_rayfields) | The motivation: what problem does a rayfield solve that a pinhole matrix cannot |
| [Ray2D vs 3D](ray2d_vs_3d) | 2‑D refinement is not a 3‑D camera model — when to use each |
| [Central vs non‑central](central_vs_noncentral) | When the pinhole assumption fails, and what that costs |
| [Gauge choices](gauge_choices) | Transverse gauge, fixed f_x, effective descriptors, and regularisation |
| [Ray‑space BIC](ray_space_bic) | How model selection works in ray space, not pixel space |
| [Parallel‑plate origin field](parallel_plate_origin_field) | The inclined‑window problem and its non‑central signature |
| [Direct vs rayfield inversion](direct_vs_rayfield_inversion) | Why a rayfield is needed even with a direct pixel‑to‑3D mapping |
| [CMO case study](cmo_case_study) | The full Pycaso CMO story: physical model, fit, diagnostics, paper results |
| [Validation limits](validation_limits) | What we have NOT validated externally — the honest scope statement |
