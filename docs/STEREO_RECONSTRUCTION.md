# Reconstruction 3D stéréo (OpenCV) et impact du “ray-field”

Objectif : montrer comment l’amélioration **2D** (coins ChArUco) se traduit en amélioration de la **calibration stéréo** et de la **triangulation 3D** avec les outils OpenCV “classiques”.

Cette page est volontairement séparée de `docs/RAYFIELD_WORKED_EXAMPLE.md` : elle se concentre sur la chaîne “calib trad + reconstruction 3D”.

## Pourquoi c’est surprenant sur un dataset *pinhole* ?

Même si les images sont générées par un modèle pinhole (avec distorsion Brown), les mesures 2D ne sont pas “parfaites” :

- flou (dont flou de bord), interpolation texture, bruit,
- compression/quantification possible selon le dataset,
- outliers de détection ArUco/ChArUco.

Dans ce contexte, la calibration OpenCV est souvent limitée par la **qualité de localisation** des coins (plus que par le modèle).
Le ray-field agit comme un **débruitage géométrique** sur le plan de la mire : on fournit à OpenCV des observations 2D plus cohérentes.

## Pipeline évalué

Pour chaque frame (couple gauche/droite) :

1) Détection des marqueurs ArUco (coins).
2) Deux variantes de coins ChArUco passées à OpenCV :
   - `raw` : coins OpenCV “bruts”,
   - `rayfield_tps_robust` : coins prédits par `H + TPS (λ) + IRLS (Huber)`.
3) Calibration mono : `cv2.calibrateCamera` (gauche puis droite).
4) Calibration stéréo : `cv2.stereoCalibrate` avec intrinsèques fixés, en estimant un **unique** $(R,T)$ sur **toutes** les paires retenues.
5) Triangulation : `cv2.triangulatePoints` après `cv2.undistortPoints`.
6) Comparaison à la vérité terrain 3D du dataset (`XYZ_world_mm` dans `gt_charuco_corners.npz`).

### Vues utilisées (important)

Les paramètres $(R,T)$ ne sont pas estimés sur une seule paire : OpenCV minimise l’erreur sur une **liste de vues** (une vue = une paire gauche/droite avec suffisamment de coins).
Le JSON exporté contient :

- `n_views_left`, `n_views_right` : nombre de vues mono utilisées par `calibrateCamera`,
- `n_views_stereo` : nombre de vues stéréo utilisées par `stereoCalibrate`,
- `view_stats.*.frame_ids` : quels `frame_id` ont effectivement contribué,
- `view_stats.*.n_corners` : stats sur le nombre de coins par vue (mean/p50/p95/min/max).

## “Baseline error” en pixels (disparity-equivalent)

La baseline est en mm, mais on peut exprimer son erreur comme une erreur de disparité (px) à profondeur $Z$ :

```{math}
:label: eq-baseline-px
\Delta d\;(\mathrm{px}) \approx \frac{f_x\;(\mathrm{px})\;\Delta B\;(\mathrm{mm})}{Z\;(\mathrm{mm})}.
```

Dans les résultats ci-dessous, on rapporte un résumé de $|\Delta d|$ sur les points GT (RMS/P95), ce qui donne une unité “image” plus intuitive.

## Script reproductible

Le script :

- compare `raw` vs `rayfield_tps_robust`,
- produit des métriques de calibration et de triangulation,
- compare aussi à la baseline GT (si présente dans `meta.json`).

Commande :

```bash
PYTHONPATH=src .venv/bin/python paper/experiments/compare_opencv_calibration_rayfield.py dataset/v0_png \
  --split train --scene scene_0000 \
  --out docs/assets/stereo_reconstruction_example/scene_0000_calib.json
```

Sortie : `docs/assets/stereo_reconstruction_example/scene_0000_calib.json`.

## Résultats (exemple)

Extrait (scene_0000, split `train`) :

```{list-table} Résumé calibration stéréo et triangulation (scene_0000, train).
:name: tab-stereo-calib-example
:header-rows: 1

* - Méthode 2D
  - Mono RMS L (px)
  - Mono RMS R (px)
  - Stereo RMS (px)
  - Baseline $\Delta B$ (mm)
  - Baseline $|\Delta d|$ RMS (px)
  - Triangulation RMS (mm)
* - raw
  - 0.306
  - 0.302
  - 0.381
  - 0.439
  - 0.424
  - 8.986
* - rayfield\_tps\_robust
  - 0.079
  - 0.061
  - 0.163
  - -0.212
  - 0.205
  - 7.161
```

Le résultat principal à lire est Tab. {numref}`tab-stereo-calib-example` : la méthode 2D change uniquement la qualité des points 2D fournis à OpenCV, et on observe ensuite son impact sur la calibration stéréo et la reconstruction.

### Intrinsèques et distorsion vs GT (en %)

Sur dataset synthétique, on peut aussi comparer les paramètres “physiques” estimés (focale et distorsion) à la vérité terrain. Le script exporte :

- `mono.percent_vs_gt.left.K.fx` / `fy` : erreur relative (%) sur $f_x, f_y$,
- (et aussi des erreurs relatives (%) par coefficient `k1,k2,p1,p2,k3`),
- `mono.distortion_displacement_vs_gt.*` : comparaison “champ de distorsion” en pixels (plus robuste/interprétable que comparer directement les coefficients).

Dans l’exemple ci-dessous, le déplacement RMS de distorsion GT vaut $\approx 1.404\,\mathrm{px}$ (gauche) et $\approx 0.947\,\mathrm{px}$ (droite) sur les cercles échantillonnés.

```{list-table} Erreurs relatives (%) sur focale et champ de distorsion (scene_0000, train).
:name: tab-mono-percent-example
:header-rows: 1

* - Méthode 2D
  - fx L (%)
  - fy L (%)
  - dist L err (%)
  - dist L err RMS (px)
  - fx R (%)
  - fy R (%)
  - dist R err (%)
  - dist R err RMS (px)
* - raw
  - 0.062
  - 0.010
  - 14.6
  - 0.205
  - 1.672
  - 1.544
  - 15.7
  - 0.149
* - rayfield\_tps\_robust
  - 0.251
  - 0.320
  - 22.6
  - 0.317
  - 0.688
  - 0.690
  - 16.9
  - 0.160
```

Remarque : ces pourcentages doivent être lus avec prudence, car l’optimisation OpenCV peut échanger “intrinsèques vs distorsion” tout en gardant un faible RMS de reprojection. Pour la reconstruction, Tab. {numref}`tab-stereo-calib-example` (RMS + baseline en px) reste l’indicateur le plus direct.

### Rectification : stabilité épipolaire (vertical disparity)

Pour objectiver l’impact sur un pipeline de stéréo dense, le script calcule aussi des métriques **après rectification** à partir du modèle estimé $(K_L, d_L, K_R, d_R, R, T)$ :

- `vertical_disparity_measured_px` : $|y_L^{rect}-y_R^{rect}|$ sur les points détectés,
- `vertical_disparity_gt_px` : idem sur les points GT (même rectification estimée, donc “erreur modèle”),
- `disparity_error_measured_px` : erreur sur la disparité rectifiée $|(x_L^{rect}-x_R^{rect})-(x_{L,GT}^{rect}-x_{R,GT}^{rect})|$.

```{list-table} Métriques de rectification (scene_0000, train).
:name: tab-rectification-example
:header-rows: 1

* - Méthode 2D
  - |Δy| RMS (px)
  - |Δy| GT RMS (px)
  - |Δd| RMS (px)
  - ray skew RMS (mm)
* - raw
  - 0.379
  - 0.244
  - 0.369
  - 0.400
* - rayfield\_tps\_robust
  - 0.218
  - 0.195
  - 0.138
  - 0.250
```

Tab. {numref}`tab-rectification-example` rend explicite le “compromis” : même si certains paramètres intrinsèques/distorsion peuvent dériver, la **cohérence épipolaire** (erreur verticale et erreur de disparité) s’améliore fortement — ce qui est critique pour des algorithmes stéréo qui supposent des correspondances **ligne à ligne**.

### Discussion : stabilité épipolaire vs “vérité” des paramètres

Sur des mires planes et un nombre limité de poses, l’optimisation OpenCV est connue pour présenter un couplage entre :

- les intrinsèques ($f_x,f_y,c_x,c_y$),
- la distorsion (ex. Brown $k_1,k_2,p_1,p_2,k_3$),
- et la pose relative stéréo ($R,T$).

Le ray-field modifie uniquement les observations 2D, et peut donc déplacer l’optimum vers une solution où la **géométrie épipolaire** est plus stable (Tab. {numref}`tab-rectification-example`), sans nécessairement reproduire coefficient-par-coefficient le modèle Brown GT.

Pour la reconstruction, l’équation stéréo rectifiée

```{math}
:label: eq-stereo-depth
Z = \frac{f_x\,B}{d}
```

montre qu’une erreur relative sur $f_x$ (ou $B$) induit principalement une erreur d’échelle sur $Z$, alors qu’une erreur de rectification (vertical disparity) et une erreur sur la disparité $d$ perturbent directement la qualité des correspondances et le bruit sur la 3D.

## Point théorique : de la baseline à l’intersection des rayons

En stéréovision métrique (robotique, stéréo dense) comme en métrologie (stéréo-DIC), il est tentant de penser que la précision 3D dépend uniquement de la qualité du matching 2D. En pratique, la précision dépend aussi — et souvent surtout — de la capacité du **modèle géométrique** à faire **quasi-intersecter** les deux rayons optiques associés aux pixels correspondants.

### 1) Deux rayons 3D associés à une correspondance 2D

Soit une correspondance 2D $\mathbf u_L=(u_L,v_L)$ dans l’image gauche et $\mathbf u_R=(u_R,v_R)$ dans l’image droite.
On définit les coordonnées homogènes $\tilde{\mathbf u}=(u,v,1)^\top$ et les coordonnées normalisées :

```{math}
:label: eq-normalized-coords
\mathbf x_L \sim \mathbf K_L^{-1}\tilde{\mathbf u}_L,\qquad
\mathbf x_R \sim \mathbf K_R^{-1}\tilde{\mathbf u}_R.
```

Dans le repère de la caméra gauche, un rayon peut s’écrire sous la forme d’une droite :

```{math}
:label: eq-rays
\mathcal D_L(\lambda)=\mathbf C_L+\lambda\,\mathbf d_L,\qquad
\mathcal D_R(\mu)=\mathbf C_R+\mu\,\mathbf d_R,
```

où $\mathbf C_L=(0,0,0)^\top$, $\mathbf d_L$ est $\mathbf x_L$ normalisé, et $\mathbf d_R$ est $\mathbf x_R$ ramené dans le repère gauche.
Avec la convention OpenCV de `stereoCalibrate` ($\mathbf X_R=\mathbf R\,\mathbf X_L+\mathbf T$), le centre de la caméra droite dans le repère gauche vaut :

```{math}
:label: eq-right-center
\mathbf C_R = -\mathbf R^\top \mathbf T.
```

### 2) Le cas réel : des droites gauches (skew lines)

Dans un monde parfait, $\mathcal D_L$ et $\mathcal D_R$ se coupent exactement au point 3D $\mathbf X$.
En pratique (calibration imparfaite, bruit 2D résiduel), les deux droites ne sont pas sécantes : elles sont **gauches**.

Les algorithmes de triangulation (ex. `cv2.triangulatePoints`) reviennent alors à choisir un point $\hat{\mathbf X}$ “au mieux”, typiquement en minimisant un critère d’erreur de reprojection ou en trouvant le point le plus proche des deux droites.
Une quantité géométrique utile est la **distance minimale entre les deux droites**, qui mesure directement “à quel point les rayons se ratent”.
Pour $\mathbf C_L=\mathbf 0$, on peut écrire cette distance (par point) :

```{math}
:label: eq-skew-distance
d_{\mathrm{skew}} = \frac{\left|(\mathbf C_R)\cdot(\mathbf d_L\times \mathbf d_R)\right|}{\lVert \mathbf d_L\times \mathbf d_R\rVert}.
```

Le script exporte cette métrique en mm : `stereo.ray_skew_distance_mm` (RMS/P95/max). Elle ne remplace pas une erreur vs GT, mais elle explique *pourquoi* une calibration peut produire une triangulation instable même si les correspondances 2D semblent bonnes.

### 3) Contrainte épipolaire et rôle de la baseline

La condition idéale pour qu’une paire $(\mathbf x_L,\mathbf x_R)$ corresponde à un même point 3D avec un modèle $(\mathbf R,\mathbf T)$ est la **contrainte épipolaire** :

```{math}
:label: eq-epipolar
\mathbf x_R^\top \mathbf E\,\mathbf x_L = 0,
\qquad \mathbf E = [\mathbf T]_{\times}\mathbf R,
```

où $[\mathbf T]_{\times}$ est la matrice antisymétrique associée au produit vectoriel. Pour $\mathbf T=(t_x,t_y,t_z)^\top$ :

```{math}
:label: eq-cross-matrix
[\mathbf T]_{\times} =
\begin{bmatrix}
0 & -t_z & t_y\\
t_z & 0 & -t_x\\
-t_y & t_x & 0
\end{bmatrix},
\qquad
[\mathbf T]_{\times}\,\mathbf a = \mathbf T \times \mathbf a.
```

Une erreur sur la baseline (ou sur la rotation) rend $\mathbf E$ incohérente : les paires $(\mathbf x_L,\mathbf x_R)$ observées ne satisfont plus la contrainte, ce qui se traduit par des rayons plus “gauches” (hausse de $d_{\mathrm{skew}}$) et par une rectification moins fiable (hausse de $|\Delta y|$ et des erreurs de disparité, cf. Tab. {numref}`tab-rectification-example`).

### 4) Pourquoi ce résultat compte pour robotique et stéréo-DIC

- **Robotique / stéréo dense** : la rectification suppose des correspondances quasi-horizontales. Une baisse de $|\Delta y|$ et de l’erreur de disparité après rectification facilite des méthodes de matching “ligne à ligne” et réduit le bruit de profondeur.
- **Métrologie / stéréo-DIC** : même si l’on évite parfois la rectification (pour limiter l’interpolation), la reconstruction repose toujours sur la triangulation via $(\mathbf K,\mathbf d,\mathbf R,\mathbf T)$. Stabiliser la géométrie épipolaire réduit l’incohérence des rayons et donc le biais/bruit 3D introduit par la calibration.

### Définitions des métriques (colonnes)

- **Méthode 2D** : la manière dont on produit les coins 2D $(u,v)$ passés à OpenCV.
  - `raw` : coins ChArUco “bruts” OpenCV.
  - `rayfield_tps_robust` : coins prédits par `H + TPS (λ) + IRLS (Huber)` à partir des coins ArUco.
- **Mono RMS L (px)** : RMS de reprojection (en pixels) retourné par `cv2.calibrateCamera` sur la caméra gauche, en utilisant les coins 2D de la méthode.
- **Mono RMS R (px)** : idem caméra droite.
- **Stereo RMS (px)** : RMS de reprojection (en pixels) retourné par `cv2.stereoCalibrate` (avec intrinsèques fixés), en utilisant les coins 2D de la méthode sur les paires gauche/droite.
- **Baseline $\Delta B$ (mm)** : erreur sur la norme de la translation estimée,

  ```{math}
  :label: eq-baseline-mm
  \Delta B = \lVert \mathbf T\rVert - B_{\mathrm{GT}}.
  ```

- **Baseline $|\Delta d|$ RMS (px)** : conversion de l’erreur de baseline en une erreur “équivalente disparité” (pixels) aux profondeurs GT,

  ```{math}
  :label: eq-baseline-px-abs
  |\Delta d| = \left|\frac{f_x\,\Delta B}{Z}\right|,
  ```

  puis on résume $|\Delta d|$ sur les points GT (RMS/P95/max). C’est l’unité la plus intuitive pour juger l’impact sur la reconstruction.
- **Triangulation RMS (mm)** : RMS de l’erreur 3D $\lVert \hat{\mathbf X}-\mathbf X_{\mathrm{GT}}\rVert$ (en mm) après triangulation par `cv2.triangulatePoints` (sur points undistortés), résumée sur l’ensemble des coins triangulés.

Pour que cette valeur soit interprétable, le script exporte aussi :

- **depth\_mm** : distribution des profondeurs $Z$ (mm) des points GT utilisés (P05/P50/P95).
- **triangulation\_error\_rel\_z\_percent** : erreur relative $100\,\lVert \hat{\mathbf X}-\mathbf X_{\mathrm{GT}}\rVert/Z$ (RMS/P95/max).

Ainsi, un “RMS = 7.4 mm” peut être lu comme “$\approx 0.55\%$ à $Z \approx 1.3\,\mathrm{m}$” sur cette scène.

### Interpréter la triangulation (mm) selon la distance

L’erreur absolue (mm) dépend fortement de la distance de travail : à erreur de disparité $\sigma_d$ (px) constante, l’approximation stéréo classique donne :

```{math}
:label: eq-depth-error
\sigma_Z \approx \frac{Z^2}{f_x\,B}\,\sigma_d
\quad\Longrightarrow\quad
\frac{\sigma_Z}{Z} \approx \frac{Z}{f_x\,B}\,\sigma_d \approx \frac{\sigma_d}{d},
```

où $Z$ est la profondeur, $B$ la baseline, $f_x$ la focale (px), $d$ la disparité (px).
On rapporte donc aussi une métrique relative ($\%$ de $Z$), qui permet de comparer des scènes à distances différentes.

Sur l’exemple de Tab. {numref}`tab-stereo-calib-example`, on obtient :

```{list-table} Profondeur et erreur 3D normalisée (même exemple).
:name: tab-stereo-triang-interpret
:header-rows: 1

* - Méthode 2D
  - Depth P50 (mm)
  - Depth [P05, P95] (mm)
  - Triang RMS (%Z)
  - Triang P95 (%Z)
* - raw
  - 1539
  - [909, 1612]
  - 0.615
  - 1.057
* - rayfield\_tps\_robust
  - 1539
  - [909, 1612]
  - 0.518
  - 0.589
```

Tab. {numref}`tab-stereo-triang-interpret` montre que, malgré des erreurs mm “visuellement grandes”, l’erreur relative reste de l’ordre de $10^{-2}$ (pourcents), et que l’amélioration due au ray-field est cohérente avec la forte baisse de l’erreur de baseline en pixels.

Remarque : sur les datasets synthétiques v0, `gt_charuco_corners.npz` fournit `XYZ_world_mm`. Le script suppose que cette 3D est cohérente avec la convention utilisée par la triangulation (cadre caméra gauche), ce qui est le cas pour `dataset/v0_png/train/scene_0000` (vérifié par reprojection).

Lecture :

- La baisse de RMS reprojection (mono + stéréo) montre que la calibration OpenCV “absorbe” beaucoup moins d’erreurs de localisation.
- L’erreur de baseline en pixels (équivalent disparité) chute fortement : cela illustre que la géométrie stéréo (échelle) devient nettement plus stable.
- La triangulation 3D s’améliore aussi, mais dépend en plus de la qualité des intrinsèques/distorsions estimés et de la géométrie des poses.
