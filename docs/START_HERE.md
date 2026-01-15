# Start Here

Objectif du démarrage : avoir une base **reproductible** et **mesurable** avant d’attaquer OptiX et le ML.

## Pourquoi cette structuration

- Le cahier des charges demande un modèle amorti (CalibNet) et un décodage compact, mais **le risque #1** est
  le sim→réel et le conditionnement. Donc on met en premier :
  1) un format dataset stable,
  2) une géométrie canonique (µm),
  3) des métriques automatiques.

## Ce qui est déjà en place

- Dataset v0 : `docs/DATASET_SPEC.md` + validateur (`validate-dataset`).
- Meta v0 (pitch obligatoire) : `src/stereocomplex/meta.py`.
- Géométrie minimale : `src/stereocomplex/core/geometry.py` (pixel↔capteur µm, pinhole, triangulation).
- Simulateur CPU MVP : `src/stereocomplex/sim/cpu/generate_dataset.py`.
- Oracle eval : `src/stereocomplex/eval/oracle.py` (sert de “sanity check”).
- Évaluation détection ChArUco (erreur 2D vs GT) : `docs/CHARUCO_IDENTIFICATION.md` + `eval-charuco-detection`.
- Exemple pédagogique complet (OpenCV brut vs ray-field + courbes) : `docs/RAYFIELD_WORKED_EXAMPLE.md`.

## Les prochaines briques (ordre recommandé)

1. Ajouter le **blur** au générateur CPU (gaussien puis spatialement variable) en unités physiques via `pitch_um`.
2. Définir la représentation compacte (bases) côté `core/model_compact/` (placeholder à créer ensuite).
3. Créer un premier `api/` (calibrate/reconstruct) qui ne dépend pas d’OptiX.
4. Remplacer la source des données par OptiX sans changer le dataset v0.

## Commandes utiles

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco --pattern charuco --blur-fwhm-um 4
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli validate-dataset dataset/charuco
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli eval-oracle dataset/charuco
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli eval-charuco-detection dataset/charuco --method rayfield
```
