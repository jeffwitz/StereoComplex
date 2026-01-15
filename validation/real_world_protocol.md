# Protocole réel (MVP)

But : commencer la validation sim→réel le plus tôt possible avec un petit set, sans attendre OptiX/ML.

## Capture

- 10–30 paires stéréo d’une mire (ChArUco ou texture), poses variées (tilt + translation + distance).
- Conserver les métadonnées par acquisition : `pitch_um`, W/H, binning, crop/ROI, resize, bit depth, gamma si connu.

## Mesures à calculer (au début)

- Erreur reprojection (px) sur points détectables (même si la calibration ML n’est pas prête).
- Stabilité vs flou/bruit (au moins 2 niveaux de focus/ISO/expo).
- Détection hors-domaine : score “qualité” minimal (à définir plus tard).

