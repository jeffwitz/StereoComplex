# Central vs non‑central

This is an explanation page.  It does not contain commands you need to run.

A **central** camera has a single optical centre: all rays pass through
one 3‑D point.  A pinhole camera is central.  A well‑corrected lens on a
DSLR is approximately central.

A **non‑central** camera does NOT have a single optical centre.  Different
pixels effectively see the world from different viewpoints.  Common causes:

- Protective glass or an inclined window in front of the sensor
- A microscope objective (back focal plane ≠ single point)
- A prism or beam splitter
- Stereo microscopes (Greenough or CMO)

StereoComplex detects non‑centrality by fitting a **Zernike origin field**
O(u,v) — a smooth 2‑D map of ray origins across the sensor.  If the origin
field is essentially constant, the camera is central.  If it varies
systematically, the camera is non‑central.
