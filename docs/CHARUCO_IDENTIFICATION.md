# ChArUco: stratégie d’identification 2D (baseline)

Objectif : obtenir des positions 2D de coins ChArUco aussi stables que possible (sub-pixel) pour mesurer l’impact du flou, de la compression et des aberrations, et préparer les étapes de calibration/reconstruction.

Le projet distingue volontairement :

- **une géométrie “prior”** (plan de mire + correspondances ArUco/ChArUco) ;
- **une observation image** (blur, compression, contrast, etc.) ;
- des méthodes qui exploitent un **modèle paramétrique** (pinhole + distorsion) ou un **modèle non-paramétrique** (champ lissé).

## Mesure d’erreur

Sur les datasets synthétiques, l’erreur est calculée vs la vérité terrain stockée dans `gt_charuco_corners.npz` :

- match par `corner_id` (ID stable) ;
- métriques par vue (left/right) : RMS, p50, p95, max, biais dx/dy.

Commande :

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli eval-charuco-detection dataset/v0 --method <METHOD>
```

## Convention pixel-center (important)

Le projet utilise une convention “pixel centers at integer coordinates” (voir `docs/CONVENTIONS.md`).
OpenCV reporte souvent les coins en convention décalée de 0.5 px ; le code d’évaluation corrige ce décalage pour `--method charuco`.

## Méthodes disponibles (CLI `--method`)

### 1) `charuco` (OpenCV direct)

- Pipeline OpenCV ArUco → interpolation ChArUco (coins internes).
- Avantage : simple, sans hypothèse de modèle caméra.
- Limite : précision souvent limitée (sensibilité blur/compression + conventions + heuristiques internes).

### 2) `homography` (2e passe géométrique plane)

- Détecte les coins ArUco, estime une homographie globale (RANSAC), puis projette tous les coins ChArUco.
- Très bon si l’image est bien décrite par une projection plane projective “simple”.
- Limite : se dégrade en présence de distorsions hors-modèle (ex: distorsion radiale notable).

### 3) `pnp` (2e passe paramétrique K + distorsion)

- Utilise `meta.json` (pitch/crop/resize + `f_um`) pour construire `K` et les coefficients de distorsion, puis :
  - `solvePnPRansac` sur coins ArUco 3D→2D,
  - `projectPoints` des coins ChArUco.
- Avantage : robuste si l’optique est modélisable par pinhole + distorsion (Brown).
- Limite : non applicable / biaisée si le système est non-pinhole (ex: modèles de microscope/CMO non-centraux).

**Point important (sur la focale)**

Dans le dataset synthétique actuel, `f_um` est connu car il est généré et stocké dans `meta.json` (champ `sim_params.f_um`).
La méthode `pnp` l’utilise donc comme un paramètre **connu** pour isoler l’effet “identification de points”.

En réel, `f_um` (et plus généralement `K` et la distorsion) ne sont pas connus a priori :

- soit ils sont estimés par une calibration classique multi-vues (ex: Zhang) avant de faire tourner `pnp`,
- soit ils font partie du problème d’auto-calibration (latent à estimer),
- soit on évite l’hypothèse pinhole en utilisant une méthode non-paramétrique (ex: `rayfield`).

### 4) `rayfield` (2e passe non-paramétrique “champ lissé” sur le plan)

But : remplacer un modèle pinhole par une hypothèse plus faible : le mapping du plan de la mire vers l’image est **basse fréquence**.

Implémentation (plane-only) :

- homographie globale `H` (RANSAC) comme “base” stable ;
- champ de résidus `r(x,y)` estimé sur une grille (bilinéaire), régularisé par un terme de lissage (Laplacien) et robuste aux outliers (Huber) ;
- prédiction : `u(x,y) = H(x,y) + r(x,y)`.

Avantage :

- ne dépend pas d’un modèle optique pinhole ;
- capture des variations lentes (aberrations complexes) tout en restant stable.

Limite :

- c’est un “ray-field” **restreint au plan** (warp 2D) ; pour un champ de rayons 3D complet par pixel, il faudra calibrer sur plusieurs poses/plans.

### 5) `kfield` (champ “K local” approché par affines lissées)

Cette méthode a été une étape intermédiaire : l’idée est de remplacer un `K` global par un champ spatialement variable,
sous hypothèse basse fréquence.

Attention : dans le code actuel, `kfield` n’interpole **pas** une matrice pinhole $K$ au sens strict. Elle construit plutôt
un champ lissé de modèles locaux **affines** (1er ordre), obtenu par **linéarisation** du mapping plan→image.

#### Linéarisation (Jacobien)

Soit un mapping inconnu (potentiellement complexe) entre le plan de mire et l’image :

- `u = u(x,y)`
- `v = v(x,y)`

Autour d’un point de référence `(x_q, y_q)`, on peut faire un développement de Taylor d’ordre 1 :

- `u(x,y) ≈ u_q + (∂u/∂x)_q · (x-x_q) + (∂u/∂y)_q · (y-y_q)`
- `v(x,y) ≈ v_q + (∂v/∂x)_q · (x-x_q) + (∂v/∂y)_q · (y-y_q)`

Le **Jacobien local** (la partie “linéaire”) est alors :

```
J(x_q,y_q) = [[∂u/∂x, ∂u/∂y],
             [∂v/∂x, ∂v/∂y]]  (évalué en (x_q,y_q))
```

L’idée de `kfield` est d’estimer ce Jacobien local (et l’offset) à partir des correspondances ArUco disponibles dans l’image,
puis de le lisser/interpoler pour obtenir une approximation basse-fréquence.

#### Construction (ce que fait le code)

- on fixe une grille d’ancrages en coordonnées mire `(x,y)` ;
- à chaque ancrage, on ajuste une affine locale par moindres carrés pondérés (voisins ArUco proches) :
  ```{math}
  u(x,y)=a_0 + a_1 x + a_2 y,\quad v(x,y)=b_0 + b_1 x + b_2 y
  ```
  où `a1,a2,b1,b2` sont une estimation du Jacobien local (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y).
- on lisse chaque paramètre $(a_0,a_1,a_2,b_0,b_1,b_2)$ sur la grille (Gaussian) ;
- pour un point $(x,y)$, on interpole bilinéairement ces paramètres puis on applique l’affine.

Pourquoi ce n’est pas suffisant :

- une affine locale ne capture pas bien la projectivité (et encore moins la distorsion) sur toute la mire ;
- interpoler directement une matrice $K$ n’est pas “géométriquement stable” (contraintes sur $f_x,f_y$ etc.).

Dans la pratique, `rayfield` (homographie + champ résiduel lissé) est plus proche de l’intuition “basse fréquence” tout en restant
numériquement stable.

## Hypothèses par méthode (ce que ça “suppose”)

Résumé des dépendances (dans l’état actuel du code) :

- `charuco`: ne requiert pas `K`/distorsion, mais dépend des heuristiques OpenCV.
- `homography`: ne requiert pas `K`/distorsion, suppose qu’une homographie globale explique bien l’image de la mire.
- `tps`: ne requiert pas `K`/distorsion ; suppose un warp 2D lisse (thin-plate spline) et peut extrapoler de façon instable si trop peu contraint.
- `pnp`: **requiert** un modèle optique (pinhole + distorsion) et ses paramètres (ou une étape préalable qui les estime).
- `rayfield`: ne requiert pas `K`/distorsion ; suppose une variation basse fréquence du warp sur le plan et utilise seulement des correspondances (Aruco) + régularisation.
- `rayfield_tps`: variante de `rayfield` où le résidu est reconstruit par TPS régularisée (au lieu d’une grille bilinéaire + Laplacien).

## Raffinements photométriques (CLI `--refine`)

Des raffinements basés sur tenseur de structure/gradients existent (`tensor`, `lines`, `lsq`, `noble`) mais, sur les datasets actuels, ils ont souvent déplacé les coins vers un optimum photométrique qui ne correspond pas au centre géométrique GT.
Ils sont donc à considérer comme ablations/expériences plutôt que comme méthode recommandée.

## Recommandation actuelle

- Si l’optique est bien approximée par pinhole + distorsion : préférer `pnp`.
- Si l’optique est complexe/non-centrale : préférer `rayfield` (hypothèse basse fréquence) et monter le niveau de régularisation si nécessaire.

## Comparaison pour l’article (script reproductible)

Le manuscrit inclut un tableau généré automatiquement (méthodes vs erreurs). Pour le régénérer :

```bash
PYTHONPATH=src .venv/bin/python paper/experiments/compare_charuco_methods.py dataset/v0_png --splits train
bash paper/build_pdflatex.sh
```

## Exemple pédagogique complet (OpenCV brut vs ray-field + courbes)

Voir `docs/RAYFIELD_WORKED_EXAMPLE.md` (inclut une explication détaillée de pourquoi une homographie globale + champ résiduel lissé permet de corriger une partie des aberrations/distorsions sur le plan de la mire).
