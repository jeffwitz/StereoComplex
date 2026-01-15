# Architecture (MVP)

Objectif : boucler vite sur **dataset → géométrie → métriques**, sans bloquer sur OptiX.

## Modules

- `src/stereocomplex/meta.py` : schéma + validation des métadonnées (pitch obligatoire, crop/resize/binning).
- `src/stereocomplex/core/geometry.py` : conversions pixel↔capteur (µm), pinhole minimal, triangulation.
- `src/stereocomplex/sim/cpu/` : générateur CPU MVP (plan texturé + GT correspondances).
- `src/stereocomplex/eval/` : métriques/évaluations (oracle, détection ChArUco vs GT, compression sweeps).

## Évolution prévue

- `sim/optix/` viendra remplacer la source de données sans changer le format dataset.
- `ml/` utilisera le dataset v0 et produira un latent `z` + décodeur compact.
