# Ray-field 3D (central) sur données GT

Cette page introduit une première brique “ray-based 3D” qui vise les optiques **plus complexes que le modèle pinhole** (p. ex. CMO), mais que l’on commence volontairement à valider sur des données **synthétiques pinhole** afin d’avoir une référence (“oracle”) claire.

Ici, on **ne traite pas** la partie *débruitage* des points ChArUco : on part de correspondances **GT** (déjà parfaites) et on compare :

- une reconstruction “oracle” **pinhole + distorsion Brown** (paramètres exacts de synthèse),
- une reconstruction par **ray-field 3D central** basé sur une **base de Zernike**.

## Conventions et données

On travaille avec le format dataset v0 (voir `DATASET_SPEC.md`). Pour une scène donnée, on charge `gt_points.npz` (ou `gt_charuco_corners.npz`) :

- $P_i = (X_i, Y_i, Z_i)^\top$ : coordonnées 3D en mm dans le repère de la caméra gauche,
- $p^L_i = (u^L_i, v^L_i)^\top$ et $p^R_i = (u^R_i, v^R_i)^\top$ : projections pixels (avec distorsion) gauche/droite,
- baseline $B$ (en mm) via `meta.json` → `sim_params.baseline_mm`.

On utilise la convention de synthèse : le centre de la caméra gauche est $C_L=(0,0,0)^\top$ et le centre de la caméra droite exprimé dans le repère gauche est
```{math}
C_R = (B,0,0)^\top.
```

## Modèle 3D par champ de rayons (central)

Un pixel $p=(u,v)$ définit un **rayon** en 3D :
```{math}
\ell_p(t) = C + t\,\hat d(u,v), \quad t\ge 0
```
où :

- $C$ est une origine **constante** (modèle central) ;
- $\hat d(u,v)\in\mathbb{R}^3$ est une direction unitaire.

On paramètre la direction via des coordonnées “normalisées” :
```{math}
\tilde d(u,v) = \begin{bmatrix} x(u,v) \\ y(u,v) \\ 1 \end{bmatrix},
\qquad
\hat d(u,v)=\frac{\tilde d(u,v)}{\lVert\tilde d(u,v)\rVert}.
```

L’enjeu est donc d’apprendre deux champs scalaires $x(u,v)$ et $y(u,v)$.

## Base de Zernike (disque unité)

On ramène le plan image à un disque unité par :
```{math}
\tilde u = \frac{u-u_0}{R},\qquad \tilde v = \frac{v-v_0}{R},
```
où $u_0$ et $v_0$ sont les coordonnées du centre image, et où $R$ est un rayon couvrant toute l’image (cercle circonscrit).

On utilise des polynômes de Zernike réels $Z_k(\rho,\theta)$ (définis sur $\rho\in[0,1]$) et on approxime :
```{math}
x(u,v) = \sum_{k=1}^{K} a_k Z_k(\tilde u,\tilde v),
\qquad
y(u,v) = \sum_{k=1}^{K} b_k Z_k(\tilde u,\tilde v).
```

Dans l’implémentation (`CentralRayFieldZernike`), $K$ dépend de l’ordre radial max `nmax` (modes jusqu’à $n\le n_{\max}$).

## Fit sur GT (régression ridge / Tikhonov)

Comme les données sont GT, chaque point 3D $P_i$ est sur le rayon défini par son pixel. Donc, en coordonnées normalisées :
```{math}
x_i = \frac{X_i}{Z_i},\qquad y_i = \frac{Y_i}{Z_i}.
```

On construit la matrice de design $A\in\mathbb{R}^{N\times K}$ telle que $A_{ik}=Z_k(\tilde u_i,\tilde v_i)$, puis on estime les coefficients par **ridge** (aussi appelée régularisation $L^2$ ou **Tikhonov**) :
```{math}
\hat a = \arg\min_a\ \lVert Aa-x\rVert^2 + \lambda\lVert a\rVert^2,
\qquad
\hat b = \arg\min_b\ \lVert Ab-y\rVert^2 + \lambda\lVert b\rVert^2.
```

Cette étape est un **MVP** (*minimum viable prototype*) : elle permet d’obtenir un premier champ de rayons **compact** sans optimisation non-linéaire.

## Triangulation et métriques

Pour une paire $(p^L_i, p^R_i)$, on obtient deux rayons
```{math}
\ell^L_i(t)= C_L + t\,\hat d_L(p^L_i),
\qquad
\ell^R_i(s)= C_R + s\,\hat d_R(p^R_i).
```

On reconstruit $\hat P_i$ par triangulation “midpoint” (point au milieu du segment perpendiculaire commun). Les deux métriques associées sont :

- **erreur 3D** : $e_i = \lVert \hat P_i - P_i\rVert$ (en mm),
- **distance de “skew rays”** : $d^{\mathrm{skew}}_i = \mathrm{dist}(\ell^L_i,\ell^R_i)$ (en mm), égale à la longueur du segment perpendiculaire commun.

## Baseline “oracle” pinhole (référence)

Sur un dataset synthétique pinhole, on dispose des paramètres exacts :

- focale $f$ (via `sim_params.f_um`),
- distorsion Brown (via `sim_params.distortion_left/right`),
- pitch pixel (via `meta.json`).

On peut donc convertir un pixel (distordu) en rayon (non distordu) en :

1. pixel $(u,v)$ → coordonnées capteur $(x_{\mu m},y_{\mu m})$,
2. normalisation distordue : $x_d=x_{\mu m}/f_{\mu m}$, $y_d=y_{\mu m}/f_{\mu m}$,
3. inversion Brown : $(x,y)=\mathrm{undistort}(x_d,y_d)$,
4. direction : $\hat d = \mathrm{normalize}([x,y,1])$.

Cette reconstruction n’est pas un “fit” : c’est un **oracle** (borne basse attendue sur ce dataset).

## Exemple complet (GT) et comparaison

Commande :

```bash
.venv/bin/python paper/experiments/compare_pinhole_vs_rayfield3d_gt.py \
  --scene dataset/v0_png/train/scene_0000 \
  --gt gt_points.npz \
  --nmax 12 \
  --lam 1e-3
```

### Métriques (mm, px, %)

On rapporte :

- erreur de triangulation en mm : $e_i = \lVert \hat P_i - P_i\rVert$,
- erreur relative (pour l’ordre de grandeur) : $100\,e_i / \bar Z$ (en %) où $\bar Z$ est la profondeur moyenne,
- erreur de reprojection en pixels (gauche/droite), en reprojetant $\hat P_i$ via le **modèle pinhole Brown GT** et en comparant à $(u,v)$ GT.

Sorties (résumé, ordre de grandeur) :

```{table} Comparaison 3D sur GT (pinhole oracle vs ray-field 3D central)
:name: tab-rayfield3d-gt-summary

| Méthode 3D | Triangulation RMS (mm) | Triangulation RMS (% depth) | Reproj RMS L/R (px) | Skew RMS (mm) |
|---|---:|---:|---:|---:|
| Pinhole oracle (GT params) | $\approx 1\times 10^{-4}$ | $\approx 1\times 10^{-5}$ | $\approx 5\times 10^{-6}$ | $\approx 1\times 10^{-5}$ |
| Ray-field 3D central (Zernike) | $\approx 3.2\times 10^{-1}$ | $\approx 2.4\times 10^{-2}$ | $\approx 4\times 10^{-2}$ | $\approx 2.8\times 10^{-2}$ |
```

Lecture rapide :

- Sur données **pinhole**, l’oracle pinhole est quasi parfait (comme attendu).
- Le ray-field 3D central est une approximation compacte : sa performance dépend fortement de `nmax` (capacité) et de $\lambda$ (lissage), et sert surtout de point de départ pour les modèles “optique complexe” à venir.

## Références code

- Base de Zernike (modes réels + matrice de design) : `src/stereocomplex/core/model_compact/zernike.py`
- Modèle central `CentralRayFieldZernike` : `src/stereocomplex/core/model_compact/central_rayfield.py`
- Comparatif oracle pinhole vs ray-field 3D (GT) : `paper/experiments/compare_pinhole_vs_rayfield3d_gt.py`

## Depuis les images : détection + ray-field 2D, puis reconstruction

Cette section relie ce chapitre (ray-field 3D) au pipeline d’identification 2D :

1. détection ChArUco OpenCV sur les images (pixels mesurés),
2. correction des centres par **ray-field 2D** (`rayfield_tps_robust`),
3. reconstruction 3D par triangulation, avec deux méthodes :
   - **pinhole oracle** : rayons obtenus en inversant Brown avec les paramètres GT de synthèse,
   - **ray-field 3D central** : fit Zernike sur $(u,v)\leftrightarrow P$ (GT) puis triangulation.

### Script

```bash
.venv/bin/python paper/experiments/compare_3d_from_images_rayfield2d.py \
  dataset/v0_png \
  --split train --scene scene_0000 \
  --tps-lam 10 --tps-huber 3 --tps-iters 3 \
  --nmax 12 --lam3d 1e-3
```

Le script écrit un JSON de métriques (par défaut `paper/tables/3d_from_images_rayfield2d.json`) et imprime le même contenu sur `stdout`.

### Résultats (exemple)

Le tableau ci-dessous illustre un run sur `scene_0000` (5 frames). Sur ces images synthétiques, la correction 2D ray-field réduit nettement l’erreur 2D en pixels, et la triangulation s’améliore mécaniquement pour les deux reconstructions.

```{table} Reconstruction 3D depuis images (OpenCV raw vs ray-field 2D), avec deux méthodes 3D
:name: tab-rayfield3d-from-images

| Méthode 2D | RMS 2D L/R (px) | Pinhole oracle: RMS 3D (mm) | Ray-field 3D: RMS 3D (mm) |
|---|---:|---:|---:|
| OpenCV raw | $\approx 0.38 / 0.36$ | $\approx 3.82$ | $\approx 3.82$ |
| Ray-field 2D (`rayfield_tps_robust`) | $\approx 0.23 / 0.14$ | $\approx 1.28$ | $\approx 1.33$ |
```

Remarque : le “ray-field 3D” présenté ici est un prototype central (origine constante) et le fit utilise les correspondances GT 3D pour démarrer proprement. L’objectif ultérieur est de remplacer ce fit “assisté GT” par une calibration ray-based complète (multi-poses, optiques non centrales, etc.).

## Calibration ray-based (sans GT 3D) : bundle adjustment point↔rayon

Cette section remplace le fit “assisté GT 3D” par une calibration complète à partir :

- des correspondances $(u,v)\leftrightarrow (X,Y,0)$ de la mire (multi-poses),
- d’un champ de rayons central compact $d(u,v)$ (Zernike),
- et de poses par frame $(R_i,t_i)$.

### Résidu géométrique

Pour une observation $(u_{ij},v_{ij})$ du point mire $P_j$ dans l’image $i$ :

- point en repère caméra : $P^{\mathrm{cam}}_{ij}=R_i P_j + t_i$,
- direction unitaire : $\hat d_{ij}=\hat d(u_{ij},v_{ij})$.

Le rayon est $\ell_{ij}(t)=C+t\hat d_{ij}$ (ici $C$ constant, et on fixe $C=(0,0,0)^\top$).

On minimise la distance point↔rayon via le résidu vectoriel :

```{math}
r_{ij} = (I - \hat d_{ij}\hat d_{ij}^\top)\,P^{\mathrm{cam}}_{ij}.
```

Ce résidu est ensuite minimisé avec une perte robuste (Huber) et une régularisation $L^2$ sur les coefficients Zernike.

### Optimisation jointe (stéréo)

Dans la version stéréo, on optimise **simultanément** :

- les coefficients Zernike du champ $d_L(u,v)$ et $d_R(u,v)$,
- une pose rigide unique du couple $(R_{RL},t_{RL})$ telle que $P_R = R_{RL}P_L+t_{RL}$,
- les poses de la mire par image dans le repère de la caméra gauche $(R_i,t_i)$.

La résolution est faite via `scipy.optimize.least_squares` (Gauss-Newton/LM robuste) avec perte de Huber et régularisation $L^2$ sur les coefficients.

### Script (images → ray-field 2D → BA ray-field 3D → stéréo)

```bash
.venv/bin/python paper/experiments/calibrate_central_rayfield3d_from_images.py \
  dataset/v0_png \
  --split train --scene scene_0000 \
  --max-frames 5 \
  --method2d rayfield_tps_robust \
  --tps-lam 10 --tps-huber 3 --tps-iters 3 \
  --nmax 8 --lam-coeff 1e-3 --outer-iters 3 --fscale-mm 1.0
```

Sortie : JSON (par défaut `paper/tables/rayfield3d_ba_from_images.json`) avec :

- baseline estimée (mm + équivalent px à profondeur moyenne),
- erreurs 3D (mm et % profondeur),
- erreurs de reprojection (px), et skew rays (mm),
- diagnostics d’optimisation (coûts par itération).
- une section `opencv_pinhole_calib` : calibration pinhole OpenCV (intrinsèques + distorsion + rig stéréo) sur les **mêmes points 2D**.

### Résultats (exemple)

Sur `scene_0000` (5 frames), la reconstruction “pinhole oracle” reste une borne basse (pinhole + Brown GT). Le BA ray-field 3D central (Zernike, modèle central) est calibré **sans solvePnP** et **sans $K$ connu** : les poses initiales sont obtenues par homographies (Zhang-style) uniquement comme *initialisation*, puis le solveur optimise directement un coût point↔rayon (Gauss-Newton robuste via SciPy).

```{table} Calibration ray-based (central) depuis images : comparaison à l’oracle pinhole (exemple)
:name: tab-rayfield3d-ba-example

| Méthode 3D (mêmes points 2D) | Baseline abs. err. (mm) | Baseline abs. err. (px) | RMS 3D (mm) | RMS reproj L/R (px) |
|---|---:|---:|---:|---:|
| Pinhole oracle (GT params) | $0$ | $0$ | $\approx 1.28$ | $\approx 0.20 / 0.15$ |
| OpenCV pinhole calibré (images, non-GT) | $\approx 0.32$ | $\approx 0.29$ | $\approx 14.48$ | $\approx 3.02 / 2.77$ |
| Ray-field 3D BA (central, Zernike) | $\approx 0.21$ | $\approx 0.19$ | $\approx 1.55$ | $\approx 1.36 / 1.33$ |
```

Remarque : pour la ligne “Ray-field 3D BA”, le RMS 3D et les reprojections sont calculés **après** une mise en correspondance par similarité “origine fixée” (rotation + échelle, sans translation) entre la reconstruction et la référence GT. Sans cette étape, les erreurs “dans le repère GT” deviennent arbitrairement grandes car le coût point↔rayon ne fixe pas, à lui seul, le *choix de repère* global (gauge).

### Discussion : (i) baseline, (ii) reprojection, (iii) triangulation

Le tableau met en évidence deux faits importants :

1. **La baseline est désormais meilleure avec le Ray-field.** Ici, la calibration ray-based produit une erreur de baseline plus faible que la calibration pinhole OpenCV (mm et px équivalent). C’est cohérent avec le fait que l’optimisation ray-based est contrainte par un unique rig $(R_{RL},t_{RL})$ et un coût géométrique point↔rayon sur toutes les observations, ce qui limite les compensations “intrinsèques ↔ distorsion ↔ extrinsèques” typiques des calibrations pinhole sur mire plane.

2. **La baseline : norme vs direction.** Une erreur faible sur la norme $\lVert C_R\rVert$ ne garantit pas une direction parfaite. Sur cet exemple, les deux méthodes produisent une baseline légèrement “hors axe” (composantes $y,z$ non nulles) ; on reporte donc aussi l’angle à l’axe $x$ et la norme hors-axe (voir JSON du script).

   Par exemple sur `scene_0000` : l’angle est d’environ $3.38^\circ$ (Ray-field) contre $2.62^\circ$ (OpenCV), malgré une erreur de norme plus faible côté Ray-field. Cela illustre qu’il faut regarder **à la fois** la norme et la direction de la baseline.

3. **Pourquoi “pinhole non-GT” peut avoir baseline correcte mais reprojection/3D mauvaises.** Le solveur OpenCV minimise sa propre erreur image, mais les erreurs rapportées ici sont mesurées *vis-à-vis du modèle GT* (pinhole + Brown de synthèse). Une calibration pinhole peut donc être auto-consistante (faible `mono_rms_*`) tout en restant éloignée du modèle GT (reprojection GT élevée), en particulier à cause des couplages d’identifiabilité sur une mire plane.

Conclusion pratique :

- en robotique (rectification, stéréo dense), la précision de la baseline et la cohérence épipolaire dominent souvent le succès du matching ;
- en métrologie (Stereo DIC), une calibration ray-based peut stabiliser la géométrie stéréo lorsque le modèle pinhole global devient une approximation.

### Discussion : pourquoi le “Ray-field 3D BA” a besoin d’une comparaison “alignée”

Le coût point↔rayon
```{math}
r_{ij}=(I-\hat d_{ij}\hat d_{ij}^\top)\,P^{\mathrm{cam}}_{ij}
```
est invariant par transformation euclidienne globale du repère caméra (rotation) et, dans une moindre mesure, par un facteur d’échelle couplé à la profondeur (identifiabilité limitée sur mire plane). Autrement dit : **la calibration est définie à une jauge près**, alors que la GT impose un repère absolu (caméra gauche, axe $x$ aligné baseline, etc.). Pour éviter de confondre “mauvaise géométrie” et “différence de repère”, on reporte donc :

- un RMS 3D “aligné” (rotation + échelle) ;
- un RMS reprojection “aligné” (projection GT après alignement).

Ces métriques reflètent l’intérêt pratique en reconstruction (cohérence et stabilité), tandis que la baseline (mm et px équivalent) reste un indicateur directement lisible en stéréovision.

## Utilisation après identification (robotique / Stereo DIC)

Cette section explicite ce que coûte et ce que requiert l’utilisation d’un modèle ray-field **une fois calibré**, c’est-à-dire “après identification” des correspondances 2D (ChArUco, matching stéréo dense, flot optique, corrélation DIC, etc.).

### Entrées / sorties minimales

Pour reconstruire un champ 3D à partir d’une paire stéréo (gauche/droite), il faut :

- **Modèle stéréo** (calibration) :
  - rig $(R_{RL},t_{RL})$,
  - ray-field gauche $d_L(u,v)$ et ray-field droite $d_R(u,v)$ (coefficients Zernike, modèle central).
- **Correspondances 2D** :
  - soit des couples $(u_L,v_L)\leftrightarrow(u_R,v_R)$,
  - soit un champ de disparité $d(u,v)$ sur une image rectifiée (cas robotique classique).

En sortie, on produit :

- un nuage (ou champ) de points $\hat P$ en mm dans le repère de la caméra gauche,
- et éventuellement une métrique de qualité par point (distance “skew rays”).

### Calcul par point et coût algorithmique

À chaque correspondance, on exécute :

1. **Pixel → rayon** (gauche et droite) :
   - pinhole : normalisation + (dé)distorsion + normalisation → $\hat d$,
   - ray-field : évaluation de $x(u,v),y(u,v)$ (Zernike) puis $\hat d=\mathrm{normalize}([x,y,1])$.
2. **Triangulation** (intersection au sens moindres carrés) :
   - point milieu du segment perpendiculaire commun (quelques opérations vectorielles).

Du point de vue complexité, pour $N$ correspondances :

- pinhole : $\mathcal{O}(N)$ (avec un coût constant faible),
- ray-field Zernike : $\mathcal{O}(N\,K)$ si l’on évalue explicitement les $K$ modes (p. ex. $K=45$ pour `nmax=8`), puis $\mathcal{O}(N)$ pour la triangulation.

En pratique, **on peut ramener le coût au même ordre que pinhole** en pré-calculant une carte de rayons :

- pré-calcul (une fois) : $d(u,v)$ pour tous les pixels de l’image (coût amorti),
- temps réel : lookup $d$ + triangulation → $\mathcal{O}(N)$.

Ce pré-calcul peut stocker une carte $(H\times W\times 3)$ en `float32` (quelques Mo), ce qui est généralement acceptable en robotique.

### Pipeline “temps réel” (robotique)

Pour de la stéréo dense en robotique (depth map), un pipeline réaliste est :

1. pré-calculer $d_L(u,v)$ et $d_R(u,v)$ (ray directions) sur la grille image,
2. calculer les correspondances (stéréo matching) :
   - soit en rectifiant vers une caméra virtuelle (pinhole) puis disparité standard,
   - soit directement sur images non-rectifiées via un matcher plus général,
3. trianguler point par point et produire la profondeur / nuage 3D.

Si on utilise une rectification vers une caméra virtuelle, l’étape supplémentaire par rapport au pinhole est la construction des tables de remap (faite une fois), puis l’exécution de `cv2.remap` (temps réel, optimisée).

### Pipeline “deux instants” (Stereo DIC)

Avec deux paires d’images (référence + déformée), on peut obtenir un champ 3D de déplacement en :

1. identifiant des correspondances stéréo à $t_0$ et $t_1$ (ou en suivant les points entre $t_0\to t_1$),
2. triangulant $\hat P(t_0)$ et $\hat P(t_1)$ avec le même modèle stéréo,
3. calculant $\Delta \hat P = \hat P(t_1)-\hat P(t_0)$.

Ici encore, le surcoût du ray-field par rapport au pinhole est concentré dans l’évaluation pixel→rayon ; avec une carte pré-calculée, la reconstruction reste compatible avec des cadences élevées.

### Taille du modèle (complexité “paramètres”)

La taille “paramètres” d’un ray-field central Zernike reste compacte :

- par caméra : $2K$ coefficients (pour $x$ et $y$), p. ex. $2\times 45=90$ scalaires à `nmax=8`,
- stéréo : +6 paramètres de rig $(R_{RL},t_{RL})$.

C’est du même ordre de grandeur qu’un pinhole (focale, centre principal, distorsion), mais la représentation est plus flexible (elle n’impose pas une distorsion polynomiale particulière).

### Références code

- BA stéréo central point↔rayon : `src/stereocomplex/ray3d/central_stereo_ba.py`
- Driver expérimental (images → BA) : `paper/experiments/calibrate_central_rayfield3d_from_images.py`
