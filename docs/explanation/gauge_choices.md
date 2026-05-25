# Gauge choices

This is an explanation page.  It does not contain commands you need to run.

A rayfield (origin O, direction d) has a gauge freedom: the same 3‑D ray
can be represented by many (O, d) pairs.  StereoComplex makes specific
choices to obtain a unique, interpretable rayfield:

1. **Transverse gauge.**  The origin O(u,v) is projected so that its offset
   from the camera centre is perpendicular to the ray direction d(u,v).
   This removes the ambiguity of sliding O along d.
2. **Fixed f_x.**  During the central→non‑central transition, the horizontal
   focal length is held constant.  This fixes the global scale.
3. **Origin‑Z regularisation.**  The Z‑component of O(u,v) is regularised
   during Zernike fitting, discouraging physically implausible origin depths.

These choices are arbitrary conventions, not physical laws.  A different
rayfield library using a different gauge would produce different O fields
that represent the SAME 3‑D rays.  The gauge matters for *interpreting*
the origin field as a physical map, but not for any downstream use
(triangulation, reconstruction) because those only use the full ray.
