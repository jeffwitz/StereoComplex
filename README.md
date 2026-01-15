# StereoComplex

Projet de calibration stéréo robuste image-based, entraînée sur jumeaux numériques, avec représentation compacte (ray-field/mapping) et prise en compte du flou (PSF compacte).

## Démarrage rapide (fallback CPU)

Sans installation (simple) :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli --help
```

Créer un dataset synthétique minimal :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/v0 --scenes 2 --frames-per-scene 16 --width 640 --height 480
```

ChArUco + blur (ex: 8 µm FWHM) :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_blur --pattern charuco --blur-fwhm-um 8
```

Blur plus fort sur les bords (approx PSF variable) :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_edgeblur --pattern charuco --blur-fwhm-um 6 --blur-edge-factor 3 --blur-edge-start 0.5
```

Interpolation texture (anti-alias) :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_interp --pattern charuco --tex-interp lanczos4
```

Aberrations géométriques (distorsion) :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_dist --pattern charuco --distort brown --distort-strength 0.5
```

Fond noir hors mire + WebP lossless :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_webp_black --pattern charuco --image-format webp --outside-mask hard
```

Valider la cohérence du dataset :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli validate-dataset dataset/v0
```

Oracle eval (sanity check synthèse : reprojection/triangulation très faibles) :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli eval-oracle dataset/v0
```

## Docs

- `docs/ARCHITECTURE.md`
- `docs/DATASET_SPEC.md`
- `docs/CONVENTIONS.md`
- `docs/START_HERE.md`
- `docs/CHARUCO_IDENTIFICATION.md`
- `docs/RAYFIELD_WORKED_EXAMPLE.md`

### Sphinx / ReadTheDocs

Build local HTML docs:

```bash
.venv/bin/python -m pip install -e .[docs]
make -C docs html
```
