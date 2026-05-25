# Why rayfields

This is an explanation page.  It does not contain commands you need to run.

A classical camera is modelled by a 3×3 matrix `K` and 5 distortion
coefficients.  This works when:

- The lens is well‑described by a pinhole model,
- The optical axis is perpendicular to the sensor,
- All rays converge to a single 3‑D point (the camera centre).

When these assumptions break — a protective glass plate, a microscope
objective, a prism, an inclined sensor — the single‑viewpoint model
fails.  Different pixels no longer share the same optical centre.

A **rayfield** is the minimal generalisation: for each pixel `(u,v)`,
store an independent 3‑D ray `(origin, direction)`.  The rayfield makes
no assumption about the optical system — it simply records what each
pixel observes.

StereoComplex uses **Zernike polynomials** as a compact basis for
rayfields.  Instead of storing millions of rays, we store ~100
coefficients per channel, from which the rayfield can be evaluated at
arbitrary sub‑pixel positions.

See also: [Central vs non‑central](central_vs_noncentral),
[Gauge choices](gauge_choices).
