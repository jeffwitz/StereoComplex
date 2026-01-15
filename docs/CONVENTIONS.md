# Conventions

## Repères

- Image : `u` vers la droite, `v` vers le bas (coordonnées pixels).
- Capteur : `x_um` vers la droite, `y_um` vers le bas, origine au centre du **crop** (avant resize).
- Monde (synthèse) : unités en **mm**.

## Pixels (centres)

Les coordonnées `(u_px, v_px)` sont des indices de pixel au centre du pixel. La conversion
interne utilise `(u + 0.5, v + 0.5)` pour passer en coordonnées continues.

Note pratique : certaines APIs OpenCV retournent des coordonnées de coins dans une convention
effectivement décalée de 0.5 px. Lorsqu’on compare à la GT du dataset, l’évaluation applique
la correction appropriée selon la méthode (voir `docs/CHARUCO_IDENTIFICATION.md`).
