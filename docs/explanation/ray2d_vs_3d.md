# Ray2D vs 3D

This is an explanation page.  It does not contain commands you need to run.

Ray2D is a 2‑D board‑plane correction, not a 3‑D camera model.

- **Ray2D** corrects detected corner positions *on the board plane*.
  It improves pixel‑level accuracy but does NOT change the camera model.
  You still get a pinhole matrix (K1, d1, K2, d2, R, T).
- **Ray3D** / **central rayfield** replaces the pinhole model entirely.
  Every pixel gets its own 3‑D ray direction.  You CANNOT export this
  as an OpenCV matrix (you can sample it, but it's a rayfield, not a
  pinhole).

When to use which: if your calibration RMS is acceptable but you suspect
non‑pinhole behaviour, skip Ray2D and go directly to the central rayfield
(Tutorial 3).
