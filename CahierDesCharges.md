# Cahier des charges de développement

**Auto-calibration stéréo robuste image-based, entraînée sur jumeaux numériques OptiX, avec modèle direct compact (type Pycaso) + flou (PSF compacte)**

---

## 0) Objectif général

Développer un logiciel capable de :

1. **Phase calibration (robuste, image-based)**
   À partir d’un petit ensemble de paires d’images stéréo d’une mire **ChArUco** (ou texture équivalente), estimer un **vecteur latent compact** (z) décrivant le système de capture (géométrie de projection non-pinhole + effets de flou), **sans** solvePnP OpenCV fragile et sans dépendre d’une extraction parfaite de coins.

2. **Phase inférence (production)**
   Utiliser (z) pour produire un **mapping direct compact** permettant de reconstruire le 3D avec haute précision :
   [
   (u_L,v_L,u_R,v_R)\rightarrow(X,Y,Z)
   ]
   en garantissant :

   * robustesse aux capteurs différents (taille/pitch), binning, crop, resize,
   * prise en compte du flou (via une PSF compacte et/ou robustesse au flou),
   * stabilité (pas de “problème inverse” fragile en production).

3. **Stratégie de données**
   Utiliser **OptiX** comme moteur de jumeau numérique géométrique (interfaces planes, off-axis, etc.) pour générer un grand volume de données réalistes (images + vérités terrain) afin d’entraîner un modèle d’apprentissage “amorti”.

---

## 1) Périmètre fonctionnel

### Entrées

* **Calibration** : (N) paires stéréo ((I_L^i, I_R^i)) (mire ChArUco), poses variées (tilt + translation + distance).
* **Métadonnées capteur** (obligatoires pour la portabilité réelle) :

  * W,H (résolution image),
  * crop/ROI (offsets), resize (facteurs), binning (bx,by),
  * **pixel pitch** (µm) si disponible (fortement recommandé pour le flou).
* **Inference** :

  * soit paires d’images stéréo (si on intègre le matching),
  * soit correspondances ((u_L,v_L,u_R,v_R)) (si matching externe).

### Sorties

* Un **vecteur latent** (z) (dimension faible, typ. 8–64).
* Un **modèle direct compact** (coefficients d’une base faible dimension) permettant :

  * back-projection sous forme de **champ de rayons compact** (recommandé), ou
  * mapping direct type Pycaso (polynômes / Zernike sur variables ((u_L,v_L,u_R,v_R))).
* Optionnel : paramètres de PSF compacts (\phi) (faible dimension) et/ou module de correction/“défloutage” ciblé.

---

## 2) Exigences clés

### Robustesse

* ✓ Calibration stable sur optiques non-pinhole (CMO/off-axis, interfaces planes, etc.).
* ✓ Tolérance au flou, contraste variable, bruit, illumination non homogène.
* ✓ Transférabilité entre matériels : capteurs différents + binning/crop/resize.

### Compacité / conditionnement

* ✓ Représentation finale **faible dimension** (pas de LUT massive), base lisse (type Zernike sur disque) + coefficients.
* ✓ Inference rapide (temps quasi-réel possible).

### Validation

* ✓ Mesures synthétiques + réelles, métriques claires (erreur 3D, reprojection, sensibilité).
* ✓ Détection “hors domaine” + score de confiance (incertitude sur (z)).

---

## 3) Architecture logicielle cible

### 3.1. Bloc “Simulateur OptiX”

* Génère des paires d’images stéréo (Charuco/texture) avec :

  * géométrie stéréo (baseline/angles),
  * interfaces planes (verre/air/liquide), indices, épaisseurs, inclinaisons,
  * distorsions géométriques résiduelles (paramétrables),
  * **PSF géométrique compacte** (modèle paramétrique), convolution, bruit, vignettage.
* Doit produire aussi la **vérité terrain** :

  * poses du plan, paramètres physiques simulés,
  * et surtout des labels utiles à l’apprentissage :

    * correspondances pixel↔point 3D (sur le plan),
    * ou champ de rayons (optionnel),
    * ou projection parfaite des coins/points.

### 3.2. Bloc “Calibration image-based”

* **CalibNet** : encode un set de paires d’images ({(I_L^i, I_R^i)}) → latent (z).
* Le modèle doit être invariant à l’ordre, accepter (N) variable, et sortir :

  * (z),
  * éventuellement un score d’incertitude (\sigma_z).

### 3.3. Bloc “Décodage compact”

Deux options (à développer en parallèle, choisir ensuite) :

**Option A (recommandée) : latent → champ de rayons compact**

* Décodeur (D(z)) produit coefficients d’un champ :
  [
  o(u,v), d(u,v)
  ]
  représentés par bases lisses (Zernike/polynômes sur disque) et quelques coefficients.
* Reconstruction 3D par triangulation analytique (moindres carrés sur deux droites).

**Option B : latent → mapping direct compact**

* Décodeur (D(z)) produit coefficients d’un mapping direct :
  [
  (X,Y,Z) = A(z),M(u_L,v_L,u_R,v_R)
  ]
  où (M) est une base (monômes/“Zernike” au sens base compacte), et (A) petit.

### 3.4. Bloc “Flou / PSF”

* PSF **compacte** et **spatialement variable** via champs de paramètres décrits par peu de modes (Zernike sur disque).
* Deux usages :

  1. **data augmentation** (robustesse au flou),
  2. optionnel : module de correction / matching PSF-aware / déconvolution légère.

### 3.5. Matching stéréo (selon périmètre)

* MVP : accepter des correspondances externes.
* V2 : intégrer un module de matching robuste (dense/semi-dense) entraîné sur simulation.

---

## 4) Décomposition en tâches et “agents de codage”

### Agent 1 — **Opticien / Simulateur OptiX**

**Compétences**

* C++/CUDA, OptiX (raygen/hit programs), shading, performance GPU.
* Optique géométrique (Snell, interfaces planes), modélisation caméra off-axis.
* Génération de textures (Charuco) + mapping UV.
* Génération de labels (GT) robustes.

**Tâches**

1. Implémenter la scène paramétrique (2 caméras + plan Charuco + interfaces).
2. Générer des images réalistes (noise, vignettage, quantization, gamma).
3. Implémenter un modèle PSF compact (paramétrique) + convolution (GPU).
4. Produire un format dataset unifié + outils de génération massive.
5. Fournir une API (CLI + Python binding) pour piloter la simulation.

**Livrables**

* `sim/optix/` : projet compilable + docker/conda + exemples.
* `sim/generate_dataset.py` (ou CLI) : génération dataset versionné.
* Documentation des paramètres du jumeau (distributions, bornes).

---

### Agent 2 — **Vision/ Géométrie / Modèle compact**

**Compétences**

* Géométrie stéréo, triangulation, modèles non-pinhole, bases polynomiales.
* Conception d’API de reconstruction stable (sans solve inverse fragile).

**Tâches**

1. Définir la représentation compacte (bases Zernike sur disque / polynômes).
2. Implémenter :

   * conversion coordonnées pixel ↔ coordonnées normalisées / µm (gestion W,H, crop, resize, binning, pitch),
   * triangulation analytique robuste,
   * métriques de confiance (distance entre rayons, conditionnement).
3. Définir l’interface du décodeur (D(z)) (coeffs → fonctions).
4. Fournir une librairie inference minimale (CPU/GPU) exportable (TorchScript/ONNX).

**Livrables**

* `core/geometry/` : fonctions de base, tests unitaires.
* `core/model_compact/` : bases + évaluation rapide.
* `api/infer.py` : API stable.

---

### Agent 3 — **Data Scientist / ML**

**Compétences**

* PyTorch/JAX, architectures set-based, domain randomization, calibration/inference amortie.
* Supervision via labels synthétiques, régularisations, incertitude.

**Tâches**

1. Définir les tâches d’apprentissage :

   * CalibNet : images → latent (z),
   * Décodeur : (z) → coefficients du champ (ray-field/mapping),
   * option : module PSF-aware / matching.
2. Définir les pertes :

   * erreur 3D (sur points du plan / hors plan si dispo),
   * reprojection via modèle compact,
   * régularisation (faible ordre / faible amplitude, éviter compensation).
3. Mettre en place domain randomization (illumination, blur, bruit, compression, défauts mire).
4. Scripts d’entraînement + validation + export modèle.

**Livrables**

* `ml/train_calibnet.py`, `ml/train_decoder.py` (+ config YAML).
* `ml/eval/` : notebooks/scripts de métriques.
* Modèles exportés + “model card” (périmètre, limites).

---

### Agent 4 — **MLOps / Infra**

**Compétences**

* CI/CD, packaging, docker, gestion de données, tracking, reproducibility.

**Tâches**

1. Définir format dataset + versioning (hash, manifests).
2. Pipeline de génération + entraînement reproductible (Docker + lock deps).
3. Tracking expérimentations (MLflow/W&B optionnel).
4. CI : tests unitaires + lint + build OptiX + smoke training.

**Livrables**

* `docker/`, `env/`, `ci/` (GitHub Actions/GitLab CI).
* Guide installation GPU + dépendances.

---

### Agent 5 — **Validation “vraie vie” / QA**

**Compétences**

* Acquisition expérimentale, protocole métrologie, analyse d’erreurs.
* Tests de robustesse, OOD, instrumentation.

**Tâches**

1. Définir un protocole de capture réel minimal (poses mire, distances, etc.).
2. Construire un dataset réel de validation (même petit) + GT partielle (règle, jauge, déplacement connu, plan).
3. Mesurer :

   * erreur 3D sur points connus,
   * stabilité vs flou/bruit,
   * sensibilité à crop/binning.
4. Définir critères d’acceptation (pass/fail).

**Livrables**

* `validation/real_world_protocol.md`
* `validation/report_template.md`
* Scripts d’évaluation sur réel.

---

## 5) Spécifications dataset

### Formats recommandés

* Images : PNG/TIFF 16 bits si possible, sinon 8 bits + gamma connu.
* Métadonnées : JSON/YAML par séquence + par image.
* Labels :

  * poses du plan (optionnel),
  * correspondances pixel↔3D sur plan (recommandé),
  * paramètres de simulation (pour audit, pas nécessaire à l’inférence).

### Organisation

* `dataset/{version}/train|val|test/scene_x/`

  * `left/0001.png`, `right/0001.png`
  * `meta.json`
  * `gt_points.npz` (points 3D + pixels) ou équivalent

---

## 6) API logicielle (proposition)

### Calibration

```python
z, quality = calibrate_stereo(images_left, images_right, meta)
```

* `meta` contient W,H, pitch, binning, crop/resize, taille carrés.

### Inference (si correspondances disponibles)

```python
XYZ, conf = reconstruct_points(uL, vL, uR, vR, z, meta)
```

### Inference end-to-end (option V2)

```python
XYZ_map, conf_map = reconstruct_dense(image_left, image_right, z, meta)
```

---

## 7) Critères de succès (à fixer selon ton niveau de précision visé)

Exemples (à adapter) :

* Erreur 3D RMS sur plan mire (synthèse) : < 0.05–0.2 mm selon échelle.
* Robustesse blur : performance stable jusqu’à FWHM ~ 3–8 px.
* Robustesse crop/binning : dégradation contrôlée (< 10–20%).
* Temps :

  * calibration < 1–5 s pour N=10–30 images (GPU),
  * reconstruction points : > 1e5 pts/s (CPU) ou quasi temps réel.

---

## 8) Plan de développement (jalons)

### M0 — Socle

* Spec dataset, API, repo, CI, environnements.

### M1 — Simulateur OptiX MVP

* 2 caméras + plan Charuco + bruit simple + GT points.

### M2 — Baseline ML

* CalibNet + décodeur (ray-field ou mapping), entraînement sur synthèse.

### M3 — Robustesse (domain randomization + flou)

* PSF compacte + blur, variations illumination, compression, défauts mire.

### M4 — Validation réelle

* protocole réel, adaptation sim→réel, métriques et seuils.

### M5 — Produit

* packaging, export ONNX/TorchScript, doc utilisateur.

---

## 9) Contraintes / risques principaux (et mitigations)

* **Sim→réel** : risque majeur. → domain randomization agressif + validation réelle très tôt.
* **Non-identifiabilité** (physique ↔ résidu Zernike ↔ flou) → régularisation forte + architecture hiérarchique (latent compact + bases faibles).
* **Métadonnées manquantes** (pitch/binning) → mode dégradé “pixel units” + avertissement + score de confiance.
* **OptiX dépendances** → docker + scripts build + fallback CPU (optionnel).

---

## 10) Organisation du dépôt (proposition)

* `sim/optix/` : code C++/CUDA OptiX
* `sim/python/` : bindings + génération dataset
* `core/geometry/` : triangulation, normalisation, métadonnées
* `core/model_compact/` : bases (Zernike/polynômes) + evaluation
* `ml/` : entraînement, modèles, pertes, configs
* `api/` : calibration + inference
* `validation/` : protocole réel, scripts, rapports
* `docs/` : guide utilisateur + guide dev + specs

---

Si tu veux, je peux te donner ensuite :

* une **liste minimale** de paramètres du latent (dimension et signification “soft” sans physique explicite),
* un choix de base compacte (ordre, nombre de modes) compatible Pycaso,
* et une spécification précise des **labels** à sortir d’OptiX pour que l’apprentissage soit bien conditionné (points sur plan, ray-field, etc.).
