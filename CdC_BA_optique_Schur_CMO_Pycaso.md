# Cahier des charges — BA optique régularisée par complément de Schur pour le cas CMO Pycaso

## 0. Résumé exécutif

L'objectif est d'ajouter à StereoComplex une étape de **bundle adjustment optique contrôlée par l'observabilité**.

Le pipeline actuel a déjà une structure forte :

1. les images ChArUco réelles du microscope CMO Pycaso sont traitées ;
2. un rayfield 3D Zernike est identifié ;
3. un modèle physique CMO télécentrique avec alignement SE(3) des deux bras est construit à partir de ce rayfield ;
4. ce modèle fournit une très bonne initialisation de la bundle adjustment directe ;
5. la BA directe peut ensuite faire baisser l'erreur de rétroprojection.

Le problème est que la BA directe reste un problème fortement couplé : les paramètres optiques et les poses peuvent se compenser. Ce couplage est quantifié par le complément de Schur de la matrice de Fisher. Le diagnostic existant indique un couplage fort, typiquement \(c \simeq 0.81\), pour une formulation directe CMO-like.

Le présent CdC propose d'ajouter une régularisation qui empêche la BA directe de réduire la rétroprojection en activant des directions optiques faiblement observables, c'est-à-dire des directions qui appartiennent au quasi-noyau du complément de Schur.

L'idée centrale est :

\[
\text{rayfield 3D}
\;\longrightarrow\;
\theta_0 \text{ physique}
\;\longrightarrow\;
\text{analyse Schur}
\;\longrightarrow\;
\text{BA directe régularisée}
\]

où \(\theta_0\) est le modèle CMO identifié par rayfield.

---

## 1. Objectifs scientifiques

### 1.1 Objectif principal

Montrer que l'estimation par rayfield ne sert pas uniquement à identifier un modèle optique, mais fournit aussi :

- une initialisation physique de la BA directe ;
- une métrique d'observabilité des paramètres optiques ;
- un prior régularisant les directions optiques mal observables.

La thèse à tester est :

> La BA directe initialisée par rayfield permet de réduire l'erreur de rétroprojection ; la régularisation par complément de Schur permet de conserver ce gain tout en évitant les dérives dans les directions optiques couplées aux poses.

### 1.2 Objectifs secondaires

Le développement doit permettre de répondre quantitativement aux questions suivantes :

1. La BA directe initialisée par le modèle CMO issu du rayfield converge-t-elle systématiquement mieux qu'une BA directe initialisée par un modèle pinhole/OpenCV ?
2. La régularisation Schur réduit-elle le déplacement des paramètres optiques dans les modes faibles du complément de Schur ?
3. La régularisation Schur stabilise-t-elle les descripteurs physiques :
   \[
   b,\quad WD,\quad f_{\mathrm{obj}},\quad \theta,\quad R_L,\quad R_R
   \]
   sous bootstrap, leave-one-frame-out, ou sous-échantillonnage des poses ?
4. La pénalisation Schur permet-elle d'obtenir une erreur de rétroprojection proche de la BA non régularisée, mais avec une meilleure interprétabilité physique ?
5. Les modes faibles identifiés par le complément de Schur correspondent-ils effectivement aux directions dans lesquelles les paramètres optiques dérivent lorsque la BA est non régularisée ?

---

## 2. Notations

### 2.1 Observations

On dispose de \(P\) poses de mire. Pour chaque pose \(p\), chaque canal \(c \in \{L,R\}\), et chaque coin ChArUco \(j\), on connaît :

- le pixel observé :
  \[
  y_{pcj} = (u_{pcj}, v_{pcj})
  \]
- le point 3D correspondant dans le repère de la mire :
  \[
  X_j^{B} \in \mathbb{R}^3
  \]
- la pose de la mire dans le repère caméra :
  \[
  \eta_p = (R_p, t_p) \in SE(3)
  \]

Le point de mire exprimé dans le repère de référence est :

\[
X_{pj} = R_p X_j^{B} + t_p.
\]

### 2.2 Modèle optique

Le modèle optique physique est paramétré par un vecteur :

\[
\theta \in \mathbb{R}^{n_\theta}.
\]

Dans le cas CMO Pycaso, \(\theta\) correspond au modèle compact :

- squelette CMO télécentrique ;
- paramètres de direction / télécentricité ;
- cisaillement de pupille ;
- correction SE(3) par bras optique ;
- éventuellement paramètres additionnels si le modèle courant les expose.

Pour chaque canal \(c\), le modèle associe à un pixel \(y=(u,v)\) une droite 3D :

\[
\mathcal{L}_c(y;\theta)
=
\left(
O_c(y;\theta),
d_c(y;\theta)
\right),
\]

avec :

\[
O_c(y;\theta) \in \mathbb{R}^3,
\qquad
d_c(y;\theta) \in \mathbb{S}^2,
\qquad
\|d_c\| = 1.
\]

La droite géométrique est :

\[
X(\lambda)
=
O_c(y;\theta)
+
\lambda d_c(y;\theta).
\]

### 2.3 Résidu point-rayon

Pour un point de mire \(X_{pj}\) et le rayon associé au pixel observé \(y_{pcj}\), le résidu géométrique est la distance orthogonale du point au rayon.

On définit :

\[
r_{pcj}(\theta,\eta_p)
=
\left(I - d d^T\right)
\left(
X_{pj} - O
\right),
\]

où :

\[
O = O_c(y_{pcj};\theta),
\qquad
d = d_c(y_{pcj};\theta).
\]

Le résidu est un vecteur de \(\mathbb{R}^3\). Sa norme est la distance point-rayon :

\[
e_{pcj}
=
\|r_{pcj}\|.
\]

L'objectif de BA géométrique est :

\[
\mathcal{L}_{\mathrm{ray}}(\theta,\eta)
=
\sum_{p,c,j}
\rho\!\left(
\frac{\|r_{pcj}(\theta,\eta_p)\|^2}{\sigma_r^2}
\right),
\]

où \(\rho\) est une fonction robuste, par exemple Huber ou soft-\(L^1\).

### 2.4 Résidu pixel-équivalent

Pour comparer avec une erreur de rétroprojection en pixels, on peut convertir la distance point-rayon locale en pixel-équivalent :

\[
e_{\mathrm{px},pcj}
\approx
\frac{f_x}{Z_{pj}}
\,
\|r_{pcj}\|,
\]

où \(Z_{pj}\) est la profondeur locale du point dans le repère de référence et \(f_x\) l'échelle focale de référence.

Cette quantité n'est pas une reprojection pinhole exacte. Elle est une approximation locale, utile pour comparer les modèles avec une unité familière en pixels.

Dans le cas où une vraie fonction de projection inverse ou implicite existe pour le modèle optique, on pourra aussi définir une erreur de reprojection directe :

\[
\epsilon_{pcj}^{\mathrm{img}}
=
\pi_c(X_{pj};\theta) - y_{pcj}.
\]

Pour le développement initial, on recommande d'utiliser le résidu point-rayon, plus général pour les modèles non centraux.

---

## 3. Problème de BA directe

### 3.1 Formulation standard

La BA directe cherche :

\[
(\theta^\star,\eta^\star)
=
\arg\min_{\theta,\eta}
\mathcal{L}_{\mathrm{data}}(\theta,\eta),
\]

avec :

\[
\mathcal{L}_{\mathrm{data}}
=
\sum_k
\|r_k(\theta,\eta)\|^2,
\]

où \(k\) indexe toutes les observations \((p,c,j)\).

Le vecteur de paramètres complet est :

\[
x =
\begin{bmatrix}
\theta\\
\eta
\end{bmatrix}.
\]

Le Jacobien des résidus est :

\[
J
=
\frac{\partial r}{\partial x}
=
\begin{bmatrix}
J_\theta & J_\eta
\end{bmatrix}.
\]

La matrice de Fisher, ou approximation de Gauss-Newton de la Hessienne, est :

\[
\mathcal{I}
=
J^T W J,
\]

où \(W\) est une matrice de poids, généralement diagonale, qui incorpore les poids robustes, les incertitudes et les normalisations.

Elle s'écrit par blocs :

\[
\mathcal{I}
=
\begin{bmatrix}
\mathcal{I}_{\theta\theta} & \mathcal{I}_{\theta\eta}\\
\mathcal{I}_{\eta\theta} & \mathcal{I}_{\eta\eta}
\end{bmatrix}.
\]

### 3.2 Interprétation des blocs

- \(\mathcal{I}_{\theta\theta}\) mesure l'information apparente sur les paramètres optiques si les poses étaient parfaitement connues.
- \(\mathcal{I}_{\eta\eta}\) mesure l'information sur les poses si l'optique était parfaitement connue.
- \(\mathcal{I}_{\theta\eta}\) mesure le couplage entre optique et poses.

Si \(\mathcal{I}_{\theta\eta}\) est important, une modification des paramètres optiques peut être compensée par une modification des poses. Le problème devient mal identifiable.

### 3.3 Complément de Schur

L'information effective sur les paramètres optiques après marginalisation des poses est :

\[
S_\theta
=
\mathcal{I}_{\theta|\eta}
=
\mathcal{I}_{\theta\theta}
-
\mathcal{I}_{\theta\eta}
\mathcal{I}_{\eta\eta}^{-1}
\mathcal{I}_{\eta\theta}.
\]

Cette matrice est le complément de Schur du bloc pose.

Elle répond à la question :

> Quelle information reste-t-il sur les paramètres optiques lorsque les poses sont autorisées à s'ajuster librement ?

Si une direction \(v\) vérifie :

\[
v^T S_\theta v \approx 0,
\]

alors cette direction optique est faiblement observable une fois les poses marginalisées.

Autrement dit, déplacer \(\theta\) dans la direction \(v\) ne produit pas un effet mesurable clairement distinct d'un déplacement des poses.

### 3.4 Couplage normalisé

Un indicateur global de couplage peut être défini par :

\[
c
=
\frac{
\left\|
\mathcal{I}_{\theta\eta}
\mathcal{I}_{\eta\eta}^{-1}
\mathcal{I}_{\eta\theta}
\right\|_F
}{
\left\|
\mathcal{I}_{\theta\theta}
\right\|_F
}.
\]

Interprétation :

- \(c \approx 0\) : les paramètres optiques et les poses sont peu couplés ;
- \(c \approx 1\) : une grande partie de l'information optique apparente disparaît quand on marginalise les poses ;
- \(c > 1\) peut apparaître numériquement si l'échelle des paramètres est mal contrôlée ou si le problème est très mal conditionné.

Important : \(c\) dépend de l'échelle des paramètres. Il doit être utilisé comme diagnostic interne, pas comme propriété physique absolue.

---

## 4. Principe de la régularisation Schur

### 4.1 Idée générale

On dispose d'une estimation initiale :

\[
\theta_0
\]

issue de l'identification par rayfield 3D.

La BA directe non régularisée minimise :

\[
\mathcal{L}_{\mathrm{data}}(\theta,\eta).
\]

Mais si certaines directions optiques sont mal observables, l'optimiseur peut réduire l'erreur en déplaçant \(\theta\) dans ces directions tout en compensant par les poses. Cela peut améliorer la RMS tout en détruisant l'interprétabilité physique.

On ajoute donc un prior anisotrope :

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{data}}
+
\mathcal{L}_{\mathrm{Schur}}.
\]

Ce prior pénalise fortement les déplacements de \(\theta\) dans les directions faibles de \(S_\theta\), mais laisse libres les directions bien observables.

### 4.2 Décomposition spectrale

On calcule le complément de Schur au point initial :

\[
S_0
=
S_\theta(\theta_0,\eta_0).
\]

On diagonalise :

\[
S_0
=
V \Lambda V^T,
\]

avec :

\[
\Lambda = \mathrm{diag}(\lambda_1,\ldots,\lambda_{n_\theta}),
\qquad
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{n_\theta}.
\]

Les directions \(v_i\) associées aux petites valeurs propres \(\lambda_i\) sont les directions optiques faiblement observables.

### 4.3 Prior sur les modes faibles

On définit le déplacement optique normalisé :

\[
\delta\theta
=
D_\theta^{-1}(\theta-\theta_0),
\]

où \(D_\theta\) est une matrice diagonale d'échelles de paramètres.

Exemples d'échelles :

- rotation SE(3) : \(1^\circ\) en radians ;
- translation SE(3) : \(1\,\mathrm{mm}\) ;
- distance / baseline / working distance : \(10\,\mathrm{mm}\) ;
- principal point : \(100\,\mathrm{px}\) ;
- slopes ou coefficients sans dimension : ordre de grandeur initial ou valeur fixée, par exemple \(10^{-2}\) à \(10^{-1}\).

Le prior Schur est :

\[
\mathcal{L}_{\mathrm{Schur}}
=
\alpha
\sum_{i=1}^{n_\theta}
w_i
\left(
v_i^T \delta\theta
\right)^2.
\]

Les poids \(w_i\) doivent être grands pour les directions faibles et petits pour les directions fortes.

Une définition pratique est :

\[
w_i
=
\left(
\frac{\lambda_{\max}}
{\lambda_i + \varepsilon \lambda_{\max}}
\right)^p.
\]

Avec :

- \(p = 1\) pour une pénalisation modérée ;
- \(p = 2\) pour une pénalisation plus agressive ;
- \(\varepsilon = 10^{-6}\) ou \(10^{-8}\).

On peut aussi ne pénaliser que les modes faibles :

\[
w_i =
\begin{cases}
1 & \text{si } \lambda_i/\lambda_{\max} < \tau,\\
0 & \text{sinon}.
\end{cases}
\]

Typiquement :

\[
\tau \in [10^{-4},10^{-2}].
\]

La version continue est préférable pour commencer, car elle évite les seuils trop arbitraires.

### 4.4 Résidus ajoutés à `least_squares`

Pour intégrer cette pénalisation dans `scipy.optimize.least_squares`, on n'ajoute pas directement \(\mathcal{L}_{\mathrm{Schur}}\), mais des résidus supplémentaires :

\[
r_i^{\mathrm{Schur}}
=
\sqrt{\alpha w_i}
\,
v_i^T \delta\theta.
\]

Le vecteur de résidus total devient :

\[
r_{\mathrm{total}}
=
\begin{bmatrix}
r_{\mathrm{data}}\\
r_{\mathrm{Schur}}
\end{bmatrix}.
\]

Ce format permet de conserver l'optimiseur Gauss-Newton / trust-region standard.

### 4.5 Variante simple : prior projecteur faible

Une version plus robuste pour un premier développement est :

\[
P_{\mathrm{weak}}
=
V_{\mathrm{weak}} V_{\mathrm{weak}}^T,
\]

où \(V_{\mathrm{weak}}\) contient les vecteurs propres faibles.

La pénalisation devient :

\[
\mathcal{L}_{\mathrm{weak}}
=
\alpha
\left\|
P_{\mathrm{weak}}
D_\theta^{-1}(\theta-\theta_0)
\right\|^2.
\]

C'est plus simple à interpréter :

> On autorise les mouvements optiques, mais pas dans les directions que le complément de Schur juge non observables.

### 4.6 Variante avancée : pénalisation dynamique du couplage

On peut pénaliser directement le couplage courant :

\[
C
=
\mathcal{I}_{\theta\theta}^{-1/2}
\mathcal{I}_{\theta\eta}
\mathcal{I}_{\eta\eta}^{-1/2}
\]

et ajouter :

\[
\mathcal{L}_{\mathrm{coupling}}
=
\beta \|C\|_F^2.
\]

Cette stratégie est conceptuellement élégante, mais elle est plus lourde :

- il faut recalculer le Jacobien au cours de l'optimisation ;
- il faut différencier ou approximer le gradient de cette pénalité ;
- les racines inverses matricielles sont coûteuses et sensibles au rang.

Cette variante ne doit pas être implémentée en premier. Elle peut être documentée comme extension.

---

## 5. Architecture logicielle proposée

### 5.1 Nouveaux modules

Créer un sous-module dédié :

```text
src/stereocomplex/optical_ba/
    __init__.py
    residuals.py
    fisher.py
    schur.py
    priors.py
    regularized_ba.py
```

### 5.2 `residuals.py`

Responsabilités :

- construire les résidus point-rayon ;
- construire les résidus pixel-équivalents ;
- gérer les poids robustes ;
- normaliser les résidus.

API proposée :

```python
def point_to_ray_residuals(
    *,
    model,
    theta: np.ndarray,
    poses: dict[int, np.ndarray],
    observations,
    residual_scale_mm: float = 1.0,
) -> np.ndarray:
    ...
```

```python
def pixel_equivalent_residuals(
    *,
    model,
    theta: np.ndarray,
    poses: dict[int, np.ndarray],
    observations,
    fx_ref_px: float,
) -> np.ndarray:
    ...
```

### 5.3 `fisher.py`

Responsabilités :

- calculer un Jacobien par différences finies ;
- gérer la mise à l'échelle des paramètres ;
- construire la matrice de Fisher ;
- partitionner les blocs optique / pose.

API proposée :

```python
@dataclass(frozen=True)
class FisherBlocks:
    I_tt: np.ndarray
    I_tp: np.ndarray
    I_pt: np.ndarray
    I_pp: np.ndarray
    J: np.ndarray
    residuals: np.ndarray
    parameter_scales_theta: np.ndarray
    parameter_scales_pose: np.ndarray
```

```python
def finite_difference_jacobian_scaled(
    fun,
    x0: np.ndarray,
    scales: np.ndarray,
    rel_step: float = 1e-6,
    method: str = "central",
) -> np.ndarray:
    ...
```

```python
def build_fisher_blocks(
    *,
    residual_fun,
    theta0: np.ndarray,
    pose0: np.ndarray,
    theta_scales: np.ndarray,
    pose_scales: np.ndarray,
    robust_weights: np.ndarray | None = None,
) -> FisherBlocks:
    ...
```

### 5.4 `schur.py`

Responsabilités :

- calculer le complément de Schur ;
- ajouter un amortissement sur le bloc pose ;
- calculer la norme de couplage ;
- extraire les modes faibles.

API proposée :

```python
@dataclass(frozen=True)
class SchurDiagnostic:
    S_theta: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    coupling_norm: float
    weak_mode_indices: np.ndarray
    condition_number: float
    rank_effective: int
```

```python
def schur_complement_theta(
    I_tt: np.ndarray,
    I_tp: np.ndarray,
    I_pp: np.ndarray,
    *,
    damping_pose: float = 1e-8,
    pinv_rcond: float = 1e-10,
) -> np.ndarray:
    ...
```

```python
def coupling_norm_schur(
    I_tt: np.ndarray,
    I_tp: np.ndarray,
    I_pp: np.ndarray,
    *,
    damping_pose: float = 1e-8,
) -> float:
    ...
```

```python
def diagnose_schur_modes(
    fisher: FisherBlocks,
    *,
    weak_threshold: float = 1e-3,
) -> SchurDiagnostic:
    ...
```

### 5.5 `priors.py`

Responsabilités :

- construire les résidus de régularisation Schur ;
- construire les résidus de prior isotrope classique ;
- fournir des fonctions de continuation sur \(\alpha\).

API proposée :

```python
@dataclass(frozen=True)
class SchurPrior:
    theta0: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    theta_scales: np.ndarray
    alpha: float
    epsilon: float = 1e-6
    power: float = 1.0
    weak_only: bool = False
    weak_threshold: float = 1e-3
```

```python
def schur_prior_residuals(
    theta: np.ndarray,
    prior: SchurPrior,
) -> np.ndarray:
    ...
```

### 5.6 `regularized_ba.py`

Responsabilités :

- lancer une BA non régularisée ;
- lancer une BA régularisée Schur ;
- enregistrer les métriques ;
- exporter les résultats.

API proposée :

```python
@dataclass(frozen=True)
class OpticalBAResult:
    theta: np.ndarray
    poses: dict[int, np.ndarray]
    success: bool
    message: str
    nfev: int
    rms_px: float
    p50_px: float
    p95_px: float
    rms_mm: float
    schur_coupling_before: float
    schur_coupling_after: float
    weak_mode_norm: float
    theta_drift_norm: float
    descriptor_dict: dict[str, float]
    diagnostics: dict[str, float]
```

```python
def run_optical_ba(
    *,
    model,
    theta0: np.ndarray,
    poses0,
    observations,
    loss: str = "soft_l1",
    f_scale_px: float = 1.0,
    max_nfev: int = 200,
) -> OpticalBAResult:
    ...
```

```python
def run_schur_regularized_optical_ba(
    *,
    model,
    theta0: np.ndarray,
    poses0,
    observations,
    schur_prior: SchurPrior,
    loss: str = "soft_l1",
    f_scale_px: float = 1.0,
    max_nfev: int = 200,
) -> OpticalBAResult:
    ...
```

---

## 6. Protocole expérimental Pycaso CMO

### 6.1 Données

Utiliser le protocole existant du notebook :

```text
examples/notebooks/09_pycaso_real_data.py
```

Les données de référence sont :

```text
docs/assets/pycaso_real_data/intermediate_state.npz
```

Ce fichier doit permettre de relancer les étapes de modèle physique, BIC, SE(3), ablation et BA sans dépendre des images brutes.

Le protocole doit éviter de coder les valeurs numériques en dur. Les valeurs existantes doivent être relues depuis :

```text
docs/assets/pycaso_real_data/
    summary.json
    aligned_cmo_fit.json
    se3_ablation.json
    bic_model_selection.json
    zernike_conditioning_summary.json
    zernike_conditioning_diagnostic.json
```

### 6.2 Configurations à comparer

#### Cas A — BA directe depuis initialisation faible

Objectif : établir le comportement de référence de la BA directe mal initialisée.

Initialisations à tester :

1. OpenCV standard ;
2. CMO perspective naïf ;
3. CMO télécentrique sans SE(3) ;
4. éventuellement pinhole + distortion.

Mesures :

- convergence ou non ;
- RMS pixel finale ;
- \(P50\), \(P95\) ;
- nombre d'itérations ;
- drift des paramètres ;
- couplage Schur au départ et à l'arrivée.

Attendu :

- convergence mauvaise ou instable ;
- erreur de rétroprojection élevée ;
- fort couplage pose / optique.

#### Cas B — BA directe depuis rayfield, sans régularisation

Objectif : confirmer que l'identification par rayfield donne une initialisation dans le bon bassin.

Initialisation :

\[
\theta_0 = \theta_{\mathrm{CMO+SE(3)}}^{\mathrm{rayfield}}.
\]

Mesures :

- RMS initiale ;
- RMS après BA ;
- amélioration relative ;
- drift de \(\theta\) ;
- drift des descripteurs physiques ;
- projection de \(\theta-\theta_0\) sur les modes faibles de Schur ;
- couplage Schur après BA.

Attendu :

- baisse de RMS ;
- convergence stable ;
- paramètres optiques peu modifiés si le résultat existant est confirmé ;
- la BA ajuste surtout les poses.

#### Cas C — BA directe depuis rayfield, avec prior isotrope

Objectif : disposer d'un témoin simple.

Prior :

\[
\mathcal{L}_{\mathrm{iso}}
=
\alpha
\|D_\theta^{-1}(\theta-\theta_0)\|^2.
\]

Tester plusieurs valeurs :

\[
\alpha \in \{10^{-4},10^{-3},10^{-2},10^{-1},1,10\}.
\]

Mesures :

- RMS ;
- drift total ;
- drift modes faibles ;
- stabilité des descripteurs.

Attendu :

- régularisation efficace mais trop uniforme ;
- risque de bloquer des directions optiques réellement observables.

#### Cas D — BA directe depuis rayfield, avec prior Schur continu

Prior :

\[
\mathcal{L}_{\mathrm{Schur}}
=
\alpha
\sum_i
\left(
\frac{\lambda_{\max}}
{\lambda_i + \varepsilon\lambda_{\max}}
\right)^p
(v_i^T D_\theta^{-1}(\theta-\theta_0))^2.
\]

Paramètres à balayer :

\[
\alpha \in \{10^{-4},10^{-3},10^{-2},10^{-1},1,10\},
\]

\[
p \in \{0.5,1,2\},
\]

\[
\varepsilon \in \{10^{-8},10^{-6},10^{-4}\}.
\]

Mesures :

- RMS ;
- drift dans les modes faibles ;
- drift dans les modes forts ;
- stabilité des descripteurs ;
- couplage Schur après BA ;
- temps de calcul.

Attendu :

- RMS proche de la BA non régularisée ;
- déplacement beaucoup plus faible dans les modes faibles ;
- meilleure stabilité bootstrap ;
- meilleure interprétabilité.

#### Cas E — BA directe depuis rayfield, avec prior projecteur faible

Prior :

\[
\mathcal{L}_{\mathrm{weak}}
=
\alpha
\|P_{\mathrm{weak}}D_\theta^{-1}(\theta-\theta_0)\|^2.
\]

Seuils :

\[
\tau \in \{10^{-4},10^{-3},10^{-2}\}.
\]

Paramètre :

\[
\alpha \in \{10^{-3},10^{-2},10^{-1},1,10\}.
\]

Attendu :

- interprétation très claire ;
- comparaison utile avec le prior continu ;
- peut être plus robuste pour le papier.

---

## 7. Métriques à produire

### 7.1 Métriques d'erreur

Pour chaque run :

\[
\mathrm{RMS}_{px}
=
\sqrt{
\frac{1}{N}
\sum_k e_{px,k}^2
}.
\]

En plus :

- \(P50_{px}\) ;
- \(P95_{px}\) ;
- RMS en mm ;
- nombre de points ;
- nombre de frames ;
- nombre d'évaluations de fonction ;
- succès / échec de l'optimiseur.

### 7.2 Métriques de drift optique

Déplacement optique normalisé :

\[
\Delta_\theta
=
D_\theta^{-1}(\theta-\theta_0).
\]

Norme totale :

\[
\|\Delta_\theta\|.
\]

Projection sur les modes faibles :

\[
\Delta_{\mathrm{weak}}
=
\|P_{\mathrm{weak}}\Delta_\theta\|.
\]

Projection sur les modes forts :

\[
\Delta_{\mathrm{strong}}
=
\|(I-P_{\mathrm{weak}})\Delta_\theta\|.
\]

Ratio :

\[
r_{\mathrm{weak}}
=
\frac{
\Delta_{\mathrm{weak}}
}{
\|\Delta_\theta\|+\varepsilon
}.
\]

Une bonne régularisation Schur doit réduire \(\Delta_{\mathrm{weak}}\) sans bloquer excessivement \(\Delta_{\mathrm{strong}}\).

### 7.3 Métriques de stabilité physique

Extraire après chaque run :

- baseline :
  \[
  b = \|O_R - O_L\|
  \]
- working distance :
  \[
  WD
  \]
- focale effective objectif :
  \[
  f_{\mathrm{obj}} = WD - z_p
  \]
- angle de convergence :
  \[
  \theta = \arccos(d_L\cdot d_R)
  \]
- rotation SE(3) gauche :
  \[
  \|r_L\|
  \]
- rotation SE(3) droite :
  \[
  \|r_R\|
  \]
- translations SE(3), avec prudence car elles sont partiellement non identifiables.

Métriques :

\[
\Delta b = b - b_0,
\qquad
\Delta WD = WD - WD_0,
\qquad
\Delta \theta = \theta - \theta_0.
\]

### 7.4 Métriques Schur

Avant et après optimisation :

- norme de couplage :
  \[
  c
  \]
- valeurs propres de \(S_\theta\) ;
- conditionnement effectif :
  \[
  \kappa_{\mathrm{eff}} =
  \frac{\lambda_{\max}}{\lambda_{\min,\mathrm{eff}}}
  \]
- nombre de modes faibles ;
- variation des modes faibles.

### 7.5 Métriques de généralisation

Pour éviter de sur-interpréter la baisse de RMS sur les points d'entraînement :

1. leave-one-frame-out ;
2. train/test split par frames ;
3. train/test split par coins de mire ;
4. bootstrap sur frames ;
5. bootstrap sur coins.

Pour chaque split :

- calibrer sur train ;
- évaluer sur test ;
- reporter RMS train et RMS test.

Une bonne régularisation Schur devrait donner :

- RMS train proche de la BA non régularisée ;
- RMS test égale ou meilleure ;
- drift physique plus faible ;
- variance bootstrap plus faible.

---

## 8. Figures et tableaux attendus

### 8.1 Tableau principal

Créer un tableau :

| Méthode | Init | Prior | RMS px | P50 | P95 | \(\Delta_{\mathrm{weak}}\) | \(\Delta b\) | \(\Delta \theta\) | succès |
|---|---|---|---:|---:|---:|---:|---:|---:|---|

Lignes minimales :

1. OpenCV / pinhole direct ;
2. CMO télécentrique direct ;
3. CMO+SE(3) rayfield initial ;
4. BA depuis rayfield sans prior ;
5. BA depuis rayfield avec prior isotrope ;
6. BA depuis rayfield avec prior Schur continu ;
7. BA depuis rayfield avec prior Schur projecteur faible.

### 8.2 Courbe Pareto

Tracer :

\[
\mathrm{RMS}_{px}
\quad \text{vs} \quad
\Delta_{\mathrm{weak}}.
\]

Chaque point correspond à une valeur de \(\alpha\).

Interprétation attendue :

- BA non régularisée : RMS bas, \(\Delta_{\mathrm{weak}}\) potentiellement élevé ;
- prior isotrope : \(\Delta_{\mathrm{weak}}\) bas mais RMS possiblement dégradé ;
- prior Schur : meilleur compromis.

### 8.3 Spectre Schur

Tracer les valeurs propres normalisées :

\[
\lambda_i/\lambda_{\max}.
\]

Afficher :

- modes faibles ;
- seuil \(\tau\) ;
- projection du drift de la BA non régularisée sur chaque mode.

### 8.4 Stabilité bootstrap

Pour chaque méthode :

- boxplot de \(b\) ;
- boxplot de \(WD\) ;
- boxplot de \(\theta\) ;
- boxplot de \(\mathrm{RMS}_{test}\).

### 8.5 Évolution pendant l'optimisation

Tracer en fonction des itérations ou des évaluations :

- RMS ;
- \(\Delta_{\mathrm{weak}}\) ;
- \(\|\theta-\theta_0\|\) ;
- norme du gradient si disponible.

---

## 9. Critères d'acceptation

### 9.1 Tests unitaires

Les tests doivent valider :

1. `schur_complement_theta` retourne une matrice symétrique.
2. Si \(\mathcal{I}_{\theta\eta}=0\), alors :
   \[
   S_\theta = \mathcal{I}_{\theta\theta}.
   \]
3. Si le couplage augmente artificiellement, les valeurs propres faibles de \(S_\theta\) diminuent.
4. `schur_prior_residuals(theta0, prior)` retourne zéro.
5. Un déplacement exactement dans un mode faible est plus pénalisé qu'un déplacement de même norme dans un mode fort.
6. Le prior projecteur faible ne pénalise pas les modes forts.
7. Les résultats sont invariants à une mise à l'échelle cohérente des paramètres si les scales sont correctement fournis.

### 9.2 Tests synthétiques minimaux

Créer un petit problème jouet :

- deux paramètres optiques ;
- une pose ;
- un résidu linéaire construit pour avoir un couplage connu.

Vérifier que :

- le complément de Schur identifie le mode faible ;
- la BA non régularisée dérive dans ce mode ;
- la BA régularisée Schur limite cette dérive.

### 9.3 Tests intégration Pycaso

Sur `intermediate_state.npz` :

1. reconstruire le modèle initial CMO+SE(3) ;
2. calculer le diagnostic Schur initial ;
3. lancer BA non régularisée ;
4. lancer BA Schur ;
5. exporter `summary_schur_ba.json`.

Le test d'intégration ne doit pas imposer des valeurs exactes trop strictes, mais des relations :

- la BA depuis rayfield doit réduire ou ne pas dégrader la RMS initiale ;
- la BA Schur doit avoir une RMS finale à moins de 5 à 10 % de la BA non régularisée ;
- la BA Schur doit réduire \(\Delta_{\mathrm{weak}}\) d'au moins 30 % par rapport à la BA non régularisée ;
- les descripteurs physiques doivent être plus stables ou au moins non dégradés.

### 9.4 Critère fort pour le papier

Un résultat serait très convaincant si :

\[
\mathrm{RMS}_{px}^{\mathrm{Schur}}
\leq
1.05
\,
\mathrm{RMS}_{px}^{\mathrm{BA}}
\]

et :

\[
\Delta_{\mathrm{weak}}^{\mathrm{Schur}}
\leq
0.5
\,
\Delta_{\mathrm{weak}}^{\mathrm{BA}}.
\]

Encore mieux si, en bootstrap :

\[
\mathrm{std}(b)_{\mathrm{Schur}}
<
\mathrm{std}(b)_{\mathrm{BA}},
\]

\[
\mathrm{std}(\theta)_{\mathrm{Schur}}
<
\mathrm{std}(\theta)_{\mathrm{BA}},
\]

avec une RMS test équivalente ou meilleure.

---

## 10. Stratégie de développement

### Étape 1 — Diagnostic offline

Objectif : ne pas modifier la BA au début.

Actions :

1. charger `intermediate_state.npz` ;
2. reconstruire \(\theta_0,\eta_0\) ;
3. construire les résidus ;
4. calculer \(J\) par différences finies ;
5. construire la Fisher ;
6. calculer le complément de Schur ;
7. exporter :
   - coupling norm ;
   - valeurs propres ;
   - modes faibles ;
   - figures.

Livrable :

```text
docs/assets/pycaso_real_data/schur_ba_diagnostic.json
docs/assets/pycaso_real_data/schur_spectrum.png
```

### Étape 2 — BA sans prior unifiée

Objectif : disposer d'un wrapper propre de BA directe.

Actions :

1. implémenter `run_optical_ba` ;
2. reproduire les résultats connus ;
3. vérifier l'amélioration de RMS depuis l'initialisation rayfield ;
4. exporter les paramètres avant/après.

Livrable :

```text
docs/assets/pycaso_real_data/optical_ba_unregularized.json
```

### Étape 3 — Prior isotrope témoin

Objectif : avoir une baseline de régularisation.

Actions :

1. ajouter un prior isotrope \(\|\theta-\theta_0\|^2\) ;
2. balayer \(\alpha\) ;
3. tracer RMS vs drift.

Livrable :

```text
docs/assets/pycaso_real_data/optical_ba_isotropic_prior_sweep.json
```

### Étape 4 — Prior Schur fixe

Objectif : tester l'idée principale.

Actions :

1. calculer \(S_0\) au point initial ;
2. construire les poids \(w_i\) ;
3. ajouter les résidus Schur à la BA ;
4. balayer \(\alpha\), \(p\), \(\varepsilon\) ;
5. comparer à la BA non régularisée et au prior isotrope.

Livrable :

```text
docs/assets/pycaso_real_data/optical_ba_schur_prior_sweep.json
docs/assets/pycaso_real_data/optical_ba_schur_pareto.png
```

### Étape 5 — Bootstrap / généralisation

Objectif : démontrer l'apport réel.

Actions :

1. répéter les calibrations sur sous-ensembles de frames ;
2. faire du leave-one-frame-out ;
3. comparer RMS train/test ;
4. comparer stabilité des descripteurs.

Livrable :

```text
docs/assets/pycaso_real_data/optical_ba_schur_bootstrap.json
docs/assets/pycaso_real_data/optical_ba_schur_bootstrap.png
```

### Étape 6 — Documentation

Créer :

```text
docs/SCHUR_REGULARIZED_BA.md
```

Cette page doit expliquer :

- pourquoi la BA directe est couplée ;
- ce que mesure le complément de Schur ;
- pourquoi le rayfield donne une initialisation et un prior ;
- comment lire les résultats sur Pycaso ;
- quelles limites ne pas dépasser dans l'interprétation.

---

## 11. Points de vigilance

### 11.1 Mise à l'échelle des paramètres

Le complément de Schur dépend fortement de l'échelle des paramètres.

Obligation :

- travailler dans un espace de paramètres normalisé ;
- documenter les scales ;
- exporter les scales dans les JSON ;
- ne jamais comparer deux coupling norms calculées avec des scales différentes.

### 11.2 Amortissement du bloc pose

\(\mathcal{I}_{\eta\eta}\) peut être mal conditionné.

Utiliser :

\[
\mathcal{I}_{\eta\eta}^{\lambda}
=
\mathcal{I}_{\eta\eta}
+
\lambda I.
\]

Tester :

\[
\lambda \in \{10^{-10},10^{-8},10^{-6},10^{-4}\}.
\]

Exporter la sensibilité du diagnostic à \(\lambda\).

### 11.3 Ne pas confondre prior et vérité physique

Le prior Schur ne dit pas que \(\theta_0\) est la vérité. Il dit :

> dans les directions non observables par la BA directe, on préfère rester proche de la solution rayfield, car elle est physiquement interprétable et obtenue par une étape découplée.

### 11.4 Ne pas sur-pénaliser

Si \(\alpha\) est trop grand, la BA ne pourra plus améliorer les paramètres optiques même dans les directions bien observées.

Il faut donc produire une courbe Pareto, pas seulement une valeur unique.

### 11.5 Ne pas annoncer une baisse massive de RMS comme objectif

Le but n'est pas nécessairement de battre la BA non régularisée en RMS train.

Le succès est :

- RMS quasi identique ;
- moins de drift faible ;
- meilleurs descripteurs ;
- meilleure stabilité hors entraînement.

---

## 12. Sorties JSON recommandées

Chaque run doit produire un enregistrement du type :

```json
{
  "method": "schur_prior",
  "init": "rayfield_cmo_se3_26p",
  "alpha": 0.1,
  "power": 1.0,
  "epsilon": 1e-6,
  "weak_threshold": 1e-3,
  "success": true,
  "nfev": 183,
  "rms_px_initial": 1.06,
  "rms_px_final": 0.90,
  "p50_px_final": 0.70,
  "p95_px_final": 1.65,
  "theta_drift_norm": 0.12,
  "weak_mode_norm": 0.02,
  "strong_mode_norm": 0.11,
  "coupling_initial": 0.81,
  "coupling_final": 0.35,
  "descriptors_initial": {
    "baseline_mm": 24.9,
    "working_distance_mm": 64.7,
    "f_obj_mm": 62.2,
    "convergence_angle_deg": 22.6
  },
  "descriptors_final": {
    "baseline_mm": 24.8,
    "working_distance_mm": 64.7,
    "f_obj_mm": 62.1,
    "convergence_angle_deg": 22.5
  }
}
```

Les valeurs ci-dessus sont illustratives. Le code ne doit pas les imposer.

---

## 13. Structure de commande souhaitée

Créer un script reproductible :

```text
examples/pycaso_schur_regularized_ba.py
```

CLI proposée :

```bash
PYTHONPATH=src python examples/pycaso_schur_regularized_ba.py \
  --input docs/assets/pycaso_real_data/intermediate_state.npz \
  --out docs/assets/pycaso_real_data/schur_ba \
  --mode all \
  --max-nfev 200
```

Options :

```text
--mode diagnostic
--mode ba
--mode isotropic-sweep
--mode schur-sweep
--mode bootstrap
--mode all

--alpha-list 1e-4,1e-3,1e-2,1e-1,1,10
--weak-threshold 1e-3
--schur-power 1.0
--schur-epsilon 1e-6
--loss soft_l1
--f-scale-px 1.0
--max-nfev 200
--seed 0
```

---

## 14. Résultat attendu si l'idée fonctionne

Le résultat idéal serait :

```text
Rayfield CMO+SE(3) initial
    RMS ≈ valeur actuelle du modèle rayfield-fit

BA directe non régularisée
    RMS diminue
    mais activation possible de modes faibles

BA directe Schur-régularisée
    RMS presque aussi basse
    activation des modes faibles fortement réduite
    descripteurs physiques plus stables
    meilleure stabilité bootstrap/test
```

La conclusion scientifique serait :

> Le rayfield joue un double rôle : il fournit une initialisation physique dans le bon bassin d'attraction de la BA directe, et il fournit une base d'observabilité permettant de régulariser la BA pour éviter les compensations pose/intrinsèques.

---

## 15. Formulation possible pour l'article

Phrase courte :

> The rayfield estimate is not only an initializer for direct bundle adjustment; it also defines an observability-aware prior that penalizes Schur-null optical directions, allowing reprojection refinement while preventing pose-intrinsic compensation.

Version française :

> L'estimation par rayfield ne sert pas seulement d'initialisation à la bundle adjustment directe ; elle définit aussi un prior d'observabilité qui pénalise les directions optiques quasi-nulles du complément de Schur, permettant d'améliorer la rétroprojection sans réintroduire de compensation pose/intrinsèques.

Version plus technique :

> We compute the Schur complement of the Fisher information matrix with respect to pose variables at the rayfield-initialized solution. Weak eigenmodes of this Schur complement define optical directions that are poorly observable after pose marginalization. Penalizing displacement along these modes yields a Schur-regularized optical BA that preserves the physical CMO interpretation while retaining most of the reprojection improvement.

---

## 16. Définition du succès du développement

Le développement est considéré réussi si les livrables suivants existent :

```text
src/stereocomplex/optical_ba/fisher.py
src/stereocomplex/optical_ba/schur.py
src/stereocomplex/optical_ba/priors.py
src/stereocomplex/optical_ba/regularized_ba.py
examples/pycaso_schur_regularized_ba.py
docs/SCHUR_REGULARIZED_BA.md
tests/test_schur_complement.py
tests/test_schur_prior.py
tests/test_pycaso_schur_ba_smoke.py
```

et si la commande :

```bash
PYTHONPATH=src python examples/pycaso_schur_regularized_ba.py --mode all
```

produit :

```text
docs/assets/pycaso_real_data/schur_ba/
    diagnostic.json
    unregularized_ba.json
    isotropic_prior_sweep.json
    schur_prior_sweep.json
    bootstrap.json
    schur_spectrum.png
    pareto_rms_vs_weak_drift.png
    bootstrap_descriptors.png
    summary.md
```

avec un `summary.md` lisible et directement intégrable dans le manuscrit.

---

## 17. Priorité de mise en œuvre

Ordre recommandé :

1. diagnostic Schur offline ;
2. wrapper BA directe unifié ;
3. prior isotrope ;
4. prior Schur continu ;
5. prior projecteur faible ;
6. bootstrap / train-test ;
7. documentation.

Ne pas commencer par la pénalisation dynamique du couplage. Elle est intéressante mais non nécessaire pour démontrer l'apport principal.
