# Exemple complet : ChArUco “brut” OpenCV vs 2e passe ray-field (avec erreur et courbes)

Objectif : à partir de couples d’images `left/right`, comparer :

1) les coins ChArUco “bruts” obtenus par les fonctions usuelles OpenCV,
2) une 2e passe **non-paramétrique** type *plane ray-field* (homographie + champ résiduel lissé),
3) l’erreur 2D en pixels vs la vérité terrain (si disponible, typiquement en synthèse),
4) des courbes (ECDF + histogrammes) pour visualiser l’impact.

Cet exemple est conçu pour qu’un(e) étudiant(e) M2 puisse le refaire de A à Z.

## Pré-requis

- Python + `numpy`
- `opencv-contrib-python` (nécessaire pour `cv2.aruco`)

La génération de courbes ne dépend pas de `matplotlib` : elles sont dessinées directement avec OpenCV pour minimiser les dépendances.

## Données attendues

Le script fonctionne “out of the box” sur le format dataset v0 du repo (voir `docs/DATASET_SPEC.md`) :

```
dataset/v0_png/train/scene_0000/
  meta.json
  frames.jsonl
  left/000000.png
  right/000000.png
  gt_charuco_corners.npz
```

La **vraie erreur** (vs GT) nécessite `gt_charuco_corners.npz` (donc typiquement des données synthétiques).

## Rappel théorique : “plane ray-field” (warp non-paramétrique sur le plan)

On note :

- $(x,y)$ : coordonnées sur le plan de la mire (mm),
- $\mathbf{u}(x,y)=(u(x,y),v(x,y))^\top$ : coordonnées image (px),
- $\tilde{\mathbf{x}}=(x,y,1)^\top$ : coordonnées homogènes.

On définit l’opérateur de projection (coordonnées homogènes → inhomogènes) :

```{math}
:label: eq-pi
\pi\!\left(\begin{bmatrix}a\\b\\c\end{bmatrix}\right)=\begin{bmatrix}a/c\\b/c\end{bmatrix}.
```

L’idée est de ne **pas** imposer un modèle optique pinhole $(K,\mathbf d)$, mais d’exploiter :

- la structure “mire plane”,
- un a priori “variation basse fréquence” du mapping plan→image.

On écrit le mapping comme :

```{math}
:label: eq-rayfield
\mathbf{u}(x,y) = \pi(H\tilde{\mathbf{x}}) + \mathbf{r}(x,y)
```

- $H$ : homographie globale (base projective),
- $\mathbf{r}(x,y)$ : champ résiduel lissé (2D), appris à partir des marqueurs ArUco.

Remarque terminologique (“non-paramétrique”) :

- Ici, “non-paramétrique” ne veut **pas** dire “sans paramètres” : on estime bien un vecteur de paramètres.
- Cela veut dire “sans modèle optique **à faible dimension**” (pas de pinhole + distorsion $(K,\mathbf d)$), mais plutôt une **régression lisse** d’un warp 2D sur le plan, dont la complexité est réglée par la résolution de grille + la régularisation.
- Dans un vocabulaire plus strict, on peut aussi parler de méthode **semi-paramétrique** : une base projective paramétrique ($H$) + une correction lisse ($\mathbf r$).

### Pourquoi une homographie est une bonne base (mire plane)

Sans distorsion, une caméra pinhole observe une mire **plane** via une homographie.
En effet, si la mire est dans un plan $Z=0$ du repère mire et si la caméra a une pose $(R,t)$ et une matrice intrinsèque $K$, on peut écrire :

```{math}
:label: eq-homography-plane
s\,\tilde{\mathbf{u}} \;=\; K \,[\,\mathbf r_1\;\mathbf r_2\;\mathbf t\,]\;\tilde{\mathbf{x}}
\quad\Rightarrow\quad
\tilde{\mathbf{u}} \sim H\,\tilde{\mathbf{x}},
\;\; H = K [\,\mathbf r_1\;\mathbf r_2\;\mathbf t\,].
```

où $\tilde{\mathbf{u}}=(u,v,1)^\top$ et $\mathbf r_1,\mathbf r_2$ sont les deux premières colonnes de $R$.
Cela justifie le rôle de $\pi(H\tilde{\mathbf x})$ comme **approximation globale** (perspective + pose) du mapping mire→image.

### Lien avec les aberrations (distorsion) : “ce qui manque” à l’homographie

Dans ce repo, les images synthétiques incluent typiquement une distorsion de type Brown (radiale + tangentielle).
De manière générale, on peut voir la distorsion comme une application lisse $d(\cdot)$ agissant sur les coordonnées image idéales :

```{math}
:label: eq-dist-compose
\mathbf u(x,y) \;=\; d\!\left(\pi(H\tilde{\mathbf x})\right).
```

Si l’on réécrit $d(\mathbf u)=\mathbf u + \Delta(\mathbf u)$ (avec $\Delta$ l’offset de distorsion), alors :

```{math}
:label: eq-residual-definition
\mathbf u(x,y) \;=\; \pi(H\tilde{\mathbf x}) + \underbrace{\Delta\!\left(\pi(H\tilde{\mathbf x})\right)}_{\mathbf r(x,y)}.
```

Sur une optique réelle (ou un simulateur) :

- $\Delta$ est généralement **basse fréquence** (polynôme radial, décentrement, etc.) : elle varie doucement avec la position,
- donc le résidu $\mathbf r(x,y)$ est lui aussi **lisse** sur le plan de la mire.

L’homographie capture la “géométrie projective” dominante, et le champ $\mathbf r$ capture une correction **non paramétrique** des aberrations (au moins leur composante lisse sur le plan).

### Formulation d’estimation (objectif + régularisation)

À partir de correspondances ArUco $\{(x_i,y_i)\leftrightarrow \mathbf{u}_i\}_{i=1}^N$, on :

1) estime une homographie $H$ robuste (RANSAC),
2) calcule les résidus observés $\hat{\mathbf r}_i$,
3) ajuste un champ $\mathbf r(x,y)$ contraint “basse fréquence”.

Le champ $\mathbf r(x,y)$ est paramétré par des valeurs aux nœuds d’une grille régulière en coordonnées mire.

Définition de $\theta$ (paramètres du champ) :

- On fixe une grille de $M$ nœuds aux positions $\{(x_m,y_m)\}_{m=1}^M$ sur le plan de la mire.
- À chaque nœud, on associe un résidu 2D inconnu $\mathbf g_m=(g^x_m,g^y_m)^\top$ (en pixels, composantes horizontale/verticale dans l’image).
- On regroupe ces inconnues dans un vecteur de paramètres $\theta$ (minimiser “par rapport à $\theta$” est donc exactement minimiser “par rapport à tous les $\mathbf g_m$”) :

```{math}
:label: eq-theta
\theta=\begin{bmatrix}
g^x_1&\cdots&g^x_M&g^y_1&\cdots&g^y_M
\end{bmatrix}^\top
\in\mathbb{R}^{2M}.
```

Évaluation du champ (interpolation bilinéaire) :
pour un point $(x,y)$, on prend les 4 nœuds de la cellule contenant $(x,y)$ et des poids bilinéaires $\{w_m(x,y)\}$ (somme = 1),
ce qui donne :

```{math}
:label: eq-rtheta
\mathbf r_\theta(x,y)
=\sum_{m=1}^{M} w_m(x,y)\,\mathbf g_m
=\begin{bmatrix}
\sum_m w_m(x,y)\,g^x_m\\
\sum_m w_m(x,y)\,g^y_m
\end{bmatrix}.
```

Avec cette définition, $\mathbf r_\theta(x_i,y_i)$ est le résidu prédit au point $(x_i,y_i)$ et on résout :

```{math}
:label: eq-rayfield-objective
\min_{\theta}\sum_{i=1}^N \rho\!\left(\left\|\mathbf r_\theta(x_i,y_i)-\hat{\mathbf r}_i\right\|_2^2\right) + \lambda \|L\theta\|_2^2
```

- $L$ : Laplacien discret sur la grille (pénalise la “courbure”, impose un champ lisse),
- $\lambda$ : poids de régularisation,
- $\rho(\cdot)$ : perte robuste (Huber) pour limiter l’influence d’outliers.

La Huber (sur la norme) peut s’écrire :

```{math}
:label: eq-huber
\rho(t)=
\begin{cases}
t, & \sqrt{t}\le \delta,\\
2\delta \sqrt{t} - \delta^2, & \sqrt{t}>\delta,
\end{cases}
```

Elle se résout efficacement par IRLS (iteratively reweighted least squares). Les poids sont
$w_i=1$ si $\lVert\cdot\rVert\le\delta$ et $w_i=\delta/\lVert\cdot\rVert$ sinon.

### Mesures disponibles (AruCo)

Chaque coin ArUco détecté donne une correspondance bruitée :

```{math}
:label: eq-aruco-meas
\mathbf{u}_i \approx \mathbf{u}(x_i,y_i)
```

On estime d’abord $H$ via RANSAC, puis on calcule des résidus observés :

```{math}
:label: eq-residuals
\hat{\mathbf r}_i = \mathbf{u}_i - \pi(H\tilde{\mathbf{x}}_i)
```

### Pourquoi ça “réduit l’incertitude” sans modèle optique

On modélise une mesure comme :

```{math}
:label: eq-measurement-noise
\mathbf{u}_i = \mathbf{u}(x_i,y_i) + \boldsymbol\varepsilon_i
```

où $\boldsymbol\varepsilon_i$ regroupe bruit de détection, biais (blur/compression), et outliers.

Le point clé est que $\mathbf r(x,y)$ est estimé en utilisant **toutes** les mesures ArUco disponibles, sous contrainte de **lissage** :

- si les aberrations (et plus généralement la déviation au modèle projectif) sont bien dominées par une composante **lisse** sur le plan,
  alors $\mathbf r$ capture une correction systématique (biais) que l’homographie seule ne peut pas expliquer ;
- si le bruit de détection est localement non corrélé, le lissage joue le rôle d’un **débruitage** (réduction de variance) en imposant de la cohérence spatiale.

On échange donc variance vs biais :

- trop de lissage $\Rightarrow$ biais (on “aplatit” des variations réelles),
- pas assez de lissage $\Rightarrow$ variance (on suit le bruit/outliers).

Dans la pratique, la distorsion Brown et beaucoup d’effets de “non-perfection” optique sont suffisamment lisses pour que la correction $\mathbf r$ **diminue l’erreur** sur l’ensemble de la mire, y compris sur les coins ChArUco non directement utilisés pour estimer $\mathbf r$.

### Pourquoi pas “juste un flou gaussien des résidus” ?

L’idée “appliquer un gaussien pour virer les hautes fréquences” est correcte **si** l’on dispose déjà d’un champ résiduel **dense** (un résidu défini partout sur une grille régulière).

Ici, on n’a pas un champ dense : on n’observe des résidus $\hat{\mathbf r}_i$ qu’en un nombre **limité** de points (coins ArUco), avec :

- un échantillonnage **irrégulier** (géométrie des marqueurs),
- de grandes zones **sans mesure** (entre marqueurs / bords / masques),
- des **outliers** (detection ratée, compression, blur, etc.).

Avant même de “flouter”, il faut donc résoudre un problème d’**interpolation / inpainting** du champ à partir d’échantillons clairsemés. La formulation $\mathbf r_\theta$ (grille + interpolation) + $\|L\theta\|^2$ fournit précisément :

- une représentation explicite du champ sur une grille,
- un lissage contrôlé (via un opérateur Laplacien / régularisation de Tikhonov),
- une robustesse aux outliers (Huber/IRLS), ce qu’un flou gaussien classique ne gère pas.

Autrement dit, “gaussien sur une image de résidus” est une option **après** reconstruction d’un champ dense ; la méthode présentée intègre (1) reconstruction + (2) lissage + (3) robustesse dans un même estimateur.

### Variante TPS (thin-plate spline) pour reconstruire le résidu

Une alternative classique à la grille bilinéaire est de reconstruire le champ résiduel par **TPS régularisée** (spline de plaque mince), bien adaptée à des mesures clairsemées.

Principe : après l’homographie de base, on observe des résidus $\hat{\mathbf r}_i$ aux points ArUco $(x_i,y_i)$ et on ajuste deux TPS scalaires (pour $r^x$ et $r^y$) :

```{math}
:label: eq-tps-form
r(x,y)=a_0+a_1 x+a_2 y+\sum_{i=1}^{N} w_i\,U(\lVert (x,y)-(x_i,y_i)\rVert),
\qquad
U(r)=r^2\log(r^2)\;\; (U(0)=0).
```

En pratique, les coefficients $\{w_i\}$ et $\{a_k\}$ se trouvent en résolvant un système linéaire du type :

```{math}
:label: eq-tps-system
\begin{bmatrix}
K+\lambda I & P\\
P^\top & 0
\end{bmatrix}
\begin{bmatrix}
w\\ a
\end{bmatrix}
=
\begin{bmatrix}
\hat r\\ 0
\end{bmatrix},
```

où $K_{ij}=U(\lVert \mathbf x_i-\mathbf x_j\rVert)$, $P=[\mathbf 1,\;x,\;y]$, et $\lambda$ contrôle le lissage (plus $\lambda$ est grand, plus le champ est “rigide”).

Dans ce repo, cette variante est disponible via la méthode `rayfield_tps` (homographie + TPS sur les résidus). Sur `dataset/v0_png/train/scene_0000`, avec le réglage par défaut actuel (`tps_lam≈10`), elle est légèrement meilleure que la grille :

- gauche RMS: ~0.219 px (`rayfield_tps`) vs ~0.224 px (`rayfield`)
- droite RMS: ~0.153 px (`rayfield_tps`) vs ~0.161 px (`rayfield`)

### Ce que ça corrige (et ce que ça ne corrige pas)

Ce modèle améliore typiquement :

- des distorsions géométriques lisses (radiale/tangentielle), et plus largement toute déviation “basse fréquence” du mapping plan→image,
- une partie des biais de localisation induits par blur/compression, tant qu’ils se manifestent comme un **offset spatial cohérent**.

Il ne “répare” pas :

- des erreurs à haute fréquence (aliasing, artefacts localisés, occlusions),
- des effets 3D hors plan (si la mire n’est plus plane, l’hypothèse se casse),
- une vraie reconstruction physique des rayons 3D (ici on reste un **warp 2D sur le plan**).

### Points importants (ce que ce “ray-field” n’est pas)

- Ce modèle est un **warp 2D restreint au plan de la mire** : il ne reconstruit pas un champ $(\mathbf{o}(u,v),\mathbf{d}(u,v))$ de rayons 3D par pixel.
- Il est conçu comme une **2e passe de stabilisation 2D** quand un modèle pinhole (PnP) est inadapté (ex : systèmes non-centrés/CMO).

## Étapes (pipeline) : ce que fait exactement l’exemple

Pour chaque image `left` et `right` :

1) Construire l’objet `CharucoBoard` depuis `meta.json` (taille de carrés, taille de marqueur, dictionnaire ArUco).
2) Détecter les marqueurs ArUco (IDs + 4 coins 2D par marqueur).
3) Extraire les coins ChArUco “bruts” via OpenCV (interpolation à partir des marqueurs détectés).
4) Estimer une homographie $H$ entre les coins ArUco “objets” (mire) et “images”.
5) Calculer les résidus $\hat{\mathbf r}_i$, puis ajuster un champ $\mathbf r(x,y)$ **lissé** (grille + IRLS/Huber).
6) Prédire tous les coins ChArUco via $\pi(H\tilde{\mathbf{x}}) + \mathbf r(x,y)$.
7) Comparer aux coins GT (si disponibles) et produire :
   - un résumé (RMS, P50, P95),
   - des courbes ECDF + histogrammes,
   - des overlays visuels (optionnel).

### Où est le code dans ce repo ?

- Script exécutable : `docs/examples/rayfield_charuco_end_to_end.py`
- Implémentation du ray-field (utilisée par le script) : `src/stereocomplex/eval/charuco_detection.py` (`_predict_points_rayfield`)

### Convention de coordonnées (important)

Dans ce repo, la GT est en convention “pixel centers at integer coordinates”.

Dans l’implémentation actuelle (et dans le script) :

- les coins **ChArUco OpenCV** sont corrigés par `-0.5 px` (décalage typique OpenCV),
- les coins **AruCo** utilisés pour l’homographie / ray-field ne sont **pas** corrigés (ils sont déjà cohérents).

## Lancer l’exemple

Commande (sur le dataset de référence du repo) :

```bash
PYTHONPATH=src .venv/bin/python docs/examples/rayfield_charuco_end_to_end.py dataset/v0_png \
  --split train --scene scene_0000 \
  --out docs/assets/rayfield_worked_example \
  --save-overlays
```

Par défaut, l’exemple utilise la meilleure variante actuelle : `ray-field (H + TPS sur les résidus)` avec `tps_lam=10`.
Pour revenir au modèle historique “grille bilinéaire + Laplacien + Huber/IRLS”, utiliser :

```bash
PYTHONPATH=src .venv/bin/python docs/examples/rayfield_charuco_end_to_end.py dataset/v0_png \
  --split train --scene scene_0000 \
  --out docs/assets/rayfield_worked_example \
  --save-overlays \
  --rayfield-backend grid
```

Sorties :

- `docs/assets/rayfield_worked_example/summary.json` : métriques par vue,
- `docs/assets/rayfield_worked_example/plots/ecdf_left.png`, `ecdf_right.png`,
- `docs/assets/rayfield_worked_example/plots/hist_left.png`, `hist_right.png`,
- `docs/assets/rayfield_worked_example/overlays/left_frame000000.png`, `right_frame000000.png` (si `--save-overlays`).

## Courbes (exemples)

Ces figures sont générées automatiquement par la commande ci-dessus.

### ECDF (distribution cumulée de l’erreur)

Voir {numref}`fig-ecdf-left` et {numref}`fig-ecdf-right`.

```{figure} assets/rayfield_worked_example/plots/ecdf_left.png
:name: fig-ecdf-left
:alt: ECDF des erreurs (vue gauche)
:width: 95%

ECDF des erreurs 2D (vue gauche) : OpenCV “brut” vs 2e passe ray-field (TPS).
```

```{figure} assets/rayfield_worked_example/plots/ecdf_right.png
:name: fig-ecdf-right
:alt: ECDF des erreurs (vue droite)
:width: 95%

ECDF des erreurs 2D (vue droite) : OpenCV “brut” vs 2e passe ray-field (TPS).
```

### Histogrammes

Voir {numref}`fig-hist-left` et {numref}`fig-hist-right`.

```{figure} assets/rayfield_worked_example/plots/hist_left.png
:name: fig-hist-left
:alt: Histogramme des erreurs (vue gauche)
:width: 95%

Histogramme des erreurs 2D (vue gauche).
```

```{figure} assets/rayfield_worked_example/plots/hist_right.png
:name: fig-hist-right
:alt: Histogramme des erreurs (vue droite)
:width: 95%

Histogramme des erreurs 2D (vue droite).
```

### Sensibilité à $\lambda$ (TPS)

Le paramètre `tps_lam` contrôle le compromis lissage/ajustement : petit $\lambda$ suit davantage les points (et peut sur-ajuster le bruit), grand $\lambda$ rigidifie le champ (et peut sous-ajuster).

Courbes RMS et P95 en fonction de $\lambda$ (sur la scène d’exemple) :

Voir {numref}`fig-tps-lam-left` et {numref}`fig-tps-lam-right`.

```{figure} assets/rayfield_worked_example/plots/tps_lambda_sweep_left.png
:name: fig-tps-lam-left
:alt: Sensibilité lambda TPS (vue gauche)
:width: 95%

Sensibilité à $\lambda$ (TPS) : RMS et P95 en fonction de `tps_lam` (vue gauche).
```

```{figure} assets/rayfield_worked_example/plots/tps_lambda_sweep_right.png
:name: fig-tps-lam-right
:alt: Sensibilité lambda TPS (vue droite)
:width: 95%

Sensibilité à $\lambda$ (TPS) : RMS et P95 en fonction de `tps_lam` (vue droite).
```

Pourquoi un optimum différent gauche/droite ?

- Les deux caméras ont des aberrations et du bruit **différents** (distorsion + blur + détection ArUco), donc le compromis sur-ajustement / sous-ajustement n’est pas identique.
- Le nombre et la géométrie des marqueurs effectivement détectés peut varier légèrement entre vues → le “champ résiduel” est contraint différemment.

Quel $\lambda$ choisir ?

- En pratique, on choisit un **unique** $\lambda$ pour le système ; un bon choix est un plateau robuste proche du minimum des deux courbes.
- Sur cette scène, `tps_lam=10` est proche du minimum à gauche et très proche du minimum à droite ; c’est le défaut de l’exemple.

### Visualisation du champ d’aberrations (amplitude des résidus)

On visualise ici l’amplitude du champ résiduel appris sur le plan, $\lVert \mathbf r(x,y)\rVert$ (en pixels), qui met en évidence la structure “basse fréquence” des aberrations projetées sur la mire :

Voir {numref}`fig-residual-amp-left` et {numref}`fig-residual-amp-right`.

```{figure} assets/rayfield_worked_example/plots/residual_amp_left_frame000000.png
:name: fig-residual-amp-left
:alt: Champ ||r(x,y)|| sur la mire (vue gauche)
:width: 95%

Amplitude du champ résiduel $\lVert \mathbf r(x,y)\rVert$ (px) sur le plan de la mire (vue gauche, frame 0).
```

```{figure} assets/rayfield_worked_example/plots/residual_amp_right_frame000000.png
:name: fig-residual-amp-right
:alt: Champ ||r(x,y)|| sur la mire (vue droite)
:width: 95%

Amplitude du champ résiduel $\lVert \mathbf r(x,y)\rVert$ (px) sur le plan de la mire (vue droite, frame 0).
```

### Overlays (contrôle visuel)

Voir {numref}`fig-overlay-left` et {numref}`fig-overlay-right`.

```{figure} assets/rayfield_worked_example/overlays/left_frame000000.png
:name: fig-overlay-left
:alt: Overlay GT vs OpenCV vs ray-field (gauche)
:width: 95%

Overlay (vue gauche, frame 0) : GT (vert), OpenCV raw (rouge), ray-field (bleu).
```

```{figure} assets/rayfield_worked_example/overlays/right_frame000000.png
:name: fig-overlay-right
:alt: Overlay GT vs OpenCV vs ray-field (droite)
:width: 95%

Overlay (vue droite, frame 0) : GT (vert), OpenCV raw (rouge), ray-field (bleu).
```

### Micro-overlays (lisibilité sub-pixel)

Deux panneaux (gauche: `raw`, droite: `ray-field`) sur un voisinage de quelques pixels (grille pixel affichée) :

Voir {numref}`fig-micro-best-left` et {numref}`fig-micro-best-right`.

```{figure} assets/rayfield_worked_example/micro_overlays/left_best_frame000000.png
:name: fig-micro-best-left
:alt: Micro-overlay (best) gauche
:width: 95%

Micro-overlay (vue gauche, frame 0) : exemple de coin où la correction améliore fortement (panneaux raw vs ray-field).
```

```{figure} assets/rayfield_worked_example/micro_overlays/right_best_frame000000.png
:name: fig-micro-best-right
:alt: Micro-overlay (best) droite
:width: 95%

Micro-overlay (vue droite, frame 0) : exemple de coin où la correction améliore fortement (panneaux raw vs ray-field).
```

Coin de la mire (voisinage ±1 px) :

Voir {numref}`fig-micro-corner-left` et {numref}`fig-micro-corner-right`.

```{figure} assets/rayfield_worked_example/micro_overlays/left_corner_frame000000.png
:name: fig-micro-corner-left
:alt: Micro-overlay coin mire gauche
:width: 95%

Micro-overlay (vue gauche, frame 0) : coin de la mire (voisinage ±1 px).
```

```{figure} assets/rayfield_worked_example/micro_overlays/right_corner_frame000000.png
:name: fig-micro-corner-right
:alt: Micro-overlay coin mire droite
:width: 95%

Micro-overlay (vue droite, frame 0) : coin de la mire (voisinage ±1 px).
```

Notes de lecture (overlay) :

- Les erreurs étant souvent **sub-pixel**, l’overlay est une **zone recadrée** autour de la mire, puis **agrandie** (paramètre `--overlay-scale`).
- Les flèches représentent le résidu GT→prédiction. Quand le résidu est trop petit pour être visible, la flèche est **amplifiée** uniquement pour la visualisation (paramètres `--overlay-min-vector-len` et `--overlay-vector-scale`).
- Les micro-overlays se pilotent avec `--micro-radius` (par défaut 3 px), `--micro-corner-radius` (par défaut 1 px) et `--micro-scale` (par défaut 80×).

## Lire et interpréter les résultats

- Si le *ray-field* est meilleur, l’ECDF doit être “à gauche” (plus d’erreurs petites) et le P95 doit baisser.
- Si le lissage est trop fort (paramètre `--smooth-lambda`), on peut lisser des variations “réelles” → biais local (les points dérivent).
- Si le dataset contient des outliers de détection (marqueurs mal détectés), augmenter `--huber-c` ou `--iters` peut stabiliser.

## Pour aller plus loin (M2)

Idées d’extensions expérimentales :

- Faire varier `grid_size` / `smooth_lambda` et tracer RMS vs régularisation.
- Comparer aussi `homography` (base) vs `ray-field` pour isoler l’apport du champ résiduel.
- Tester sur plusieurs scènes/poses et étudier l’impact des zones “hors convex hull” (extrapolation).

## Références (pistes biblio)

Quelques références “classiques” utiles pour situer l’approche (robuste + régularisée + champ lisse à partir de correspondances clairsemées) :

- P. J. Huber, “Robust Estimation of a Location Parameter”, *Annals of Mathematical Statistics*, 1964. (perte de Huber, IRLS)
- A. N. Tikhonov, V. Y. Arsenin, *Solutions of Ill-posed Problems*, 1977. (régularisation, pénalités quadratiques)
- F. L. Bookstein, “Principal Warps: Thin-Plate Splines and the Decomposition of Deformations”, *IEEE TPAMI*, 1989. (interpolation lisse de déformations 2D)
- G. Wahba, *Spline Models for Observational Data*, 1990. (splines et lissage régularisé)
- S. Schaefer, T. McPhail, J. Warren, “Image Deformation Using Moving Least Squares”, *SIGGRAPH*, 2006. (MLS pour champs de déformation en 2D)
- B. K. P. Horn, B. G. Schunck, “Determining Optical Flow”, *Artificial Intelligence*, 1981. (champ dense + régularisation de lissage)
