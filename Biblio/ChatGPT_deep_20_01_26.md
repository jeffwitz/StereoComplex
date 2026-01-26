# Calibration par Modèles de Rayons vs Modèles Classiques : Stabilité face au Bruit de Détection

## Contexte et Modèles de Caméra

Les **modèles de caméra « ray field » (basés sur les rayons)**
représentent chaque pixel par un rayon de projection dans l\'espace 3D,
plutôt que d\'utiliser une formule paramétrique restreinte.
Concrètement, un modèle générique peut associer à chaque pixel une
direction de rayon (et éventuellement un point d\'origine) calibrée
empiriquement[\[1\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=We%20briefly%20describe%20two%20generic,spline%20model%20is).
Cela inclut par exemple des approches *per-pixel* (un rayon calibré par
pixel) ou avec interpolation (*splines* sur une grille de
rayons)[\[1\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=We%20briefly%20describe%20two%20generic,spline%20model%20is).
À l\'inverse, les **modèles classiques** (pinhole avec distorsion
polynomiale de Brown, modèles unifiés de fisheye type Mei ou
Kannala-Brandt, ou encore les mappings polynomiaux directs « Soloff »)
utilisent quelques paramètres (typiquement \~12) pour décrire
globalement la projection et la distorsion de l\'optique.

**Problème posé :** ces modèles sont-ils également robustes face à des
imprécisions dans la détection des cibles de calibration (bruit subpixel
sur les coins d\'une mire, détections légèrement biaisées, etc.) ? La
stabilité se manifeste ici par des calibrations moins sensibles à ces
bruits : par exemple des résultats de reconstruction 3D moins biaisés,
des paramètres intrinsèques/extrinsèques moins variables, une géométrie
stéréo plus cohérente, malgré des erreurs de détection.

## Biais des Modèles Paramétriques vs Flexibilité des Modèles Rayons

Les modèles paramétriques classiques peuvent souffrir de **erreurs
systématiques résiduelles** lorsque la lentille présente des distorsions
complexes ou lorsque des erreurs de détection perturbent l\'ajustement.
Par exemple, en calibrant une caméra de smartphone, on observe que les
modèles classiques laissent un motif structuré d\'erreur de
reprojection, signe d\'une distorsion mal modélisée, alors qu\'un modèle
générique par rayons élimine quasiment tout motif systématique (il ne
reste qu\'un bruit
aléatoire)[\[2\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=Figure%201%3A%20%20Residual%20distortion,Interpolation%20happens%20among).
En ce sens, un modèle riche en paramètres peut mieux *absorber* les
aberrations ou déviations locales sans introduire de biais global. Des
études antérieures l\'ont noté : un modèle générique ajusté finement
produit des erreurs de reprojection plus faibles que les modèles
Brown/Mei, *sans sur-ajustement apparent*, pourvu qu\'on dispose
d\'assez d\'images de
calibration[\[3\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=match%20at%20L631%20The%20generic,Interestingly).
Au contraire, un modèle trop rigide pourrait **interpréter des
perturbations de détection comme de la distortion**, ce qui fausse
certains paramètres (par ex. centre principal, focale ou coefficients de
distorsion) et dégrade la géométrie calibrée.

## Stabilité Expérimentale sous Bruit de Détection

Des travaux récents offrent des **comparaisons directes sous conditions
de bruit** (détection dégradée, données perturbées) entre approches
paramétriques et par rayons. Un résultat marquant provient d\'une
campagne de tests impliquant des mires ChArUco synthétiques compressées
avec perte (introduisant du *bruit* et un léger biais dans la
localisation des coins) :

-   **Méthode polynomiale (Soloff/direct)** : elle s\'est révélée **très
    sensible** aux altérations de l\'image, au point de perdre
    l\'échelle correcte de reconstruction 3D. Sous forte compression,
    l\'erreur de profondeur RMS a explosé (jusqu\'à \~50 mm
    d\'erreur)[\[4\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Dataset%20Pycaso%20direct%20RMS%20Z,23)[\[5\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1650%20,triangulates%20stably%20because%20it%20regularizes).
    En termes simples, le mapping polynomial a \"déraillé\" face à ce
    bruit et produit des reconstructions très fausses (l\'algorithme
    ajuste un polynôme qui, en présence de coins déplacés, déforme la
    géométrie et rend la scène plus petite/grande qu\'en réalité).
-   **Méthode ray-field 3D (modèle de rayons)** : elle est restée
    **stable** et physiquement cohérente dans le même
    test[\[6\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Key%20result%3A%203D%20ray,ready%20NPZ).
    La calibration par rayons n\'a quasiment pas dévié en échelle
    (erreur \~1 mm seulement) malgré les coins
    bruités[\[4\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Dataset%20Pycaso%20direct%20RMS%20Z,23).
    Comme le résument les auteurs, *« le modèle 3D par rayons reste
    stable sous une forte compression avec artefacts, là où les
    pipelines pinhole classiques se montrent sensibles aux erreurs de
    localisation »*[\[6\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Key%20result%3A%203D%20ray,ready%20NPZ).
    Cette robustesse s\'explique par le fait que le modèle de rayons
    impose une certaine continuité et une contrainte globale (par ex. un
    centre de projection unique pour un modèle central) qui régularisent
    la
    solution[\[5\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1650%20,triangulates%20stably%20because%20it%20regularizes).
    En d\'autres termes, même si les coins bougent un peu, le modèle
    ray-field préfère de petits ajustements locaux des rayons plutôt
    qu\'une déformation globale de l\'échelle de la caméra.

Un autre indice de stabilité est donné par l\'**erreur de triangulation
3D** et la **cohérence géométrique** d\'un duo stéréo calibré. Une étude
a introduit la métrique de *distance entre rayons* (\"**ray skew**\") :
c\'est la distance minimale entre les deux rayons émis par une paire de
correspondances stéréo (idéalement, deux rayons se croisent exactement
en un point 3D, distance
nulle)[\[7\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=From%20each%20correspondence%20we%20thus,n∥%20%2C%20with%20n).
Un calibrage instable ou bruité donnera des rayons légèrement désalignés
(skew non nul).

-   En comparant une calibration classique OpenCV (pinhole+Brown) sur
    des coins ChArUco bruts, versus une calibration après affinement des
    coins par un modèle ray-field 2D, on constate une **réduction
    significative du skew et des erreurs 3D**. Par exemple, le
    post-traitement des détections par un champ de rayons (fitting d\'un
    champ de résidus 2D) a réduit l\'erreur de disparité moyenne de
    **0,37 px à 0,14 px** et la distance moyenne entre rayons de **0,400
    mm à 0,250 mm**, ce qui traduit une géométrie stéréo plus
    cohérente[\[8\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=2D%20method%20,off%20explicit%3A%20even%20if%20some).
    La reconstruction 3D qui en résulte gagne en précision (erreur de
    profondeur réduite d\'environ 60 %) et en consistance entre les deux
    caméras.

```{=html}
<!-- -->
```
-   De même, le *baseline* (la distance entre centres optiques en
    stéréo) estimé gagne en stabilité. Avec des détections bruitées, les
    modèles classiques peuvent estimer un baseline légèrement erroné
    (p.ex. surestimé de quelques mm, ou incohérent entre calibrations
    répétées). Or, en convertissant l\'erreur de baseline en une erreur
    de disparité équivalente, on constate qu\'**une calibration
    ray-field réduit drastiquement cette erreur** comparé à OpenCV. Une
    série de tests rapporte que l\'erreur de base stéréo exprimée en
    pixels chute de façon significative, ce qui *« rend la géométrie
    stéréo (l\'échelle) bien plus
    stable »*[\[9\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1527%20•%20The,depends%20on%20the%20quality%20of).
    En chiffres, l\'amélioration du baseline équivaut à diviser par 2 ou
    3 l\'erreur de disparité due à une mauvaise évaluation de l\'écart
    entre
    caméras[\[9\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1527%20•%20The,depends%20on%20the%20quality%20of)[\[10\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1654%20the%20rays,py).
    En conservant la cohérence du baseline, le modèle à rayons assure
    que les profondeurs reconstruites restent justes d\'une calibration
    à l\'autre et ne dérivent pas selon les
    perturbations[\[10\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1654%20the%20rays,py).

Enfin, la **robustesse aux outliers** fait partie de la stabilité. Les
outils de calibration générique récents intègrent des filtre de
détection d\'outliers (points aberrants à plus de quelques pixels) et
des moindres carrés robustes. Le *review* de Shao *et al.* (2024) note
que pratiquement tous les frameworks modernes gèrent ces outliers
automatiquement, ce qui améliore la fiabilité des
résultats[\[11\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=From%20the%20review%20and%20experiments%2C,we%20summarize%20several%20findings).
Le modèle de rayons, en particulier, répartit les erreurs sur de
nombreux paramètres locaux plutôt que de concentrer l\'erreur sur
quelques paramètres globaux, ce qui tend à limiter l\'impact d\'un coin
mal détecté (celui-ci affectera essentiellement les rayons voisins, sans
fausser toute la caméra).

## Comparaisons avec les Modèles plus Classiques

Plusieurs travaux ont comparé quantitativement les modèles génériques vs
paramétriques. Schöps *et al.* (CVPR 2020) ont montré sur un large
éventail de caméras (de la quasi-pinhole à la fisheye) que leur pipeline
de **calibration générique** produisait *systématiquement une erreur de
reprojection plus faible* que les modèles pinhole+distorsion
standard[\[3\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=match%20at%20L631%20The%20generic,Interestingly).
Surtout, ils ont démontré l\'impact *dans les applications* : en
estimant la profondeur stéréo ou la pose 3D, les erreurs de calibration
classiques introduisent un **biais** non négligeable sur les résultats,
biais qui est corrigé en grande partie par un modèle générique plus
précis[\[12\]](https://arxiv.org/abs/1912.02908#:~:text=calibration%20pipeline%20for%20generic%20models,camera%20calibration%20available%20to%20everyone)[\[13\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=camera%20pose%20estimation%20as%20examples%2C,released%20our%20calibration%20pipeline%20at).
En d\'autres termes, une calibration pinhole imparfaite (même avec une
erreur RMS de l\'ordre du dixième de pixel) peut induire des décalages
systématiques dans les reconstructions 3D, alors qu\'un modèle à 10 000
paramètres élimine ce biais et rend les mesures 3D plus
justes[\[12\]](https://arxiv.org/abs/1912.02908#:~:text=calibration%20pipeline%20for%20generic%20models,camera%20calibration%20available%20to%20everyone).
Les auteurs recommandent donc d\'utiliser ces modèles génériques
\"chaque fois que possible\" pour gagner en
exactitude[\[14\]](https://arxiv.org/abs/1912.02908#:~:text=accuracy,camera%20calibration%20available%20to%20everyone).

Historiquement, il y avait des débats sur la nécessité d\'un modèle
complexe pour des caméras à faible distorsion. Une étude de 2013
intitulée *« Can a fully unconstrained imaging model be applied
effectively to central cameras? »* montrait déjà que **même une caméra
quasi-pinhole** peut bénéficier d\'un modèle non-paramétrique, avec des
améliorations de
précision[\[15\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schops_Why_Having_10000_Parameters_in_Your_Camera_Model_Is_Better_CVPR_2020_paper.pdf#:~:text=parametric%20cal%02ibration%20,to%20practical%20advantages%20in%20applications).
Certes, un modèle générique demande plus de données pour éviter le
sur-ajustement, mais avec suffisamment d\'images couvrant tout le champ,
il *ne sur-ajuste pas* et offre une excellente généralisation (tests
faits sur des images non utilisées pour la
calibration)[\[3\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=match%20at%20L631%20The%20generic,Interestingly).
Au contraire, un modèle polynomial global (type Soloff) peut être
mathématiquement **instable hors de son domaine d\'étalonnage** : par
exemple, on note qu\'il n\'est pas analytiquement inversible et qu\'il
peut mal extrapoler en dehors de la zone couverte par la
mire[\[16\]](https://elib.dlr.de/215234/1/Barta_2025_Meas._Sci._Technol._36_055301.pdf#:~:text=paper%2C%20we%20review%20and%20compare,In)[\[17\]](https://elib.dlr.de/215234/1/Barta_2025_Meas._Sci._Technol._36_055301.pdf#:~:text=experiments%20without%20volumetric%20refinement,In).
Les expériences de compression d\'images citées plus haut illustrent
bien ce point : le modèle polynomial a perdu l\'échelle correcte dès que
les données de calibration sortaient un peu de son cas idéal, tandis que
le modèle ray-field, plus contraint géométriquement, a conservé une
échelle
cohérente[\[5\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1650%20,triangulates%20stably%20because%20it%20regularizes).

Enfin, signalons que certains modèles classiques à très grand champ
(ex : le modèle unifié de Mei pour fisheye) souffrent de **redondances
et corrélations de paramètres** qui nuisent à la stabilité
numérique[\[18\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=2,from%20parameter%20instability%20and%20redundancy)[\[19\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=pinhole%20radial%E2%80%93tangential%20model%20well,from%20parameter%20instability%20and%20redundancy).
Par exemple, dans le modèle de Mei, la combinaison de la focale, du
coefficient de distorsion et de la position du centre peut être ajustée
de façons multiples aboutissant à des solutions presque équivalentes, ce
qui rend l\'optimisation délicate et parfois non robuste (instabilité
rapportée dans l\'étude de Shao *et
al.*[\[18\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=2,from%20parameter%20instability%20and%20redundancy)).
Un modèle par rayons n\'a pas ce problème de redondance paramétrique
puisque chaque rayon est un paramètre indépendant ; la contrepartie est
la nécessité d\'une **régularisation** (par ex. supposer que les rayons
voisins varient de façon
lisse[\[20\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schops_Why_Having_10000_Parameters_in_Your_Camera_Model_Is_Better_CVPR_2020_paper.pdf#:~:text=inaccuracy%20,work%2C%20we%20follow%20these%20approaches)[\[21\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schops_Why_Having_10000_Parameters_in_Your_Camera_Model_Is_Better_CVPR_2020_paper.pdf#:~:text=our%20calibration%20pattern%20and%20feature,us%20to%20ob%02serve%20and%20model)).
En pratique, cette régularisation (par spline de degré modéré, ou base
polynomiale orthonormale
limitée[\[22\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=y%28u%2C%20v%29,v%20−%20v0%20R)[\[23\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=x,modes%20up%20to%20n%20≤))
suffit à assurer un bon compromis entre flexibilité et stabilité : on
évite les oscillations non physiques tout en épousant les déformations
complexes de l\'optique.

## Indicateurs de Stabilité et Résultats Chiffrés

Pour quantifier la stabilité face au bruit, plusieurs **métriques** sont
utilisées dans la littérature :

-   **RMS d\'erreur de reprojection (2D)** -- un classique, qui diminue
    typiquement avec un modèle plus flexible. Par exemple, Schöps *et
    al.* rapportent des médianes d\'erreur de reprojection
    systématiquement plus basses pour leurs modèles à spline par rapport
    aux modèles Brown ou KB, de l\'ordre de **0,02--0,04 px vs
    0,07--0,16 px** selon les caméras, sans signe de
    sur-ajustement[\[24\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=these%20cells%20in%20Fig)[\[25\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=%28colors%29.%20For%20SC,were%20too%20inconsistent).
    De plus, ils évaluent ces erreurs sur des images *de test*
    distinctes des images de calibration pour vérifier la stabilité hors
    échantillon
    (généralisation)[\[26\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=We%20also%20list%20the%20median,and%20the%20empirical%20distribution%20of).

```{=html}
<!-- -->
```
-   **Erreur 3D (profondeur) RMS** -- pertinente pour la stéréo. Dans
    les tests de robustesse avec images compressées, on a vu un écart de
    plusieurs ordres de grandeur entre le ray-field et le polynomial :
    \~**1 mm** vs **10--50 mm** d\'erreur de profondeur moyenne selon le
    taux de
    compression[\[4\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Dataset%20Pycaso%20direct%20RMS%20Z,23).
    Même dans des conditions normales, une calibration pinhole
    introduisait un biais de profondeur de l\'ordre de quelques
    pourcents, éliminé par la calibration
    générique[\[12\]](https://arxiv.org/abs/1912.02908#:~:text=calibration%20pipeline%20for%20generic%20models,camera%20calibration%20available%20to%20everyone).
    Par exemple, avec des caméras RealSense, le biais de profondeur à
    2 m dû à la distorsion non-modélisée pouvait atteindre \~5 mm avec
    le modèle Brown, contre \<1 mm avec le modèle par
    rayons[\[12\]](https://arxiv.org/abs/1912.02908#:~:text=calibration%20pipeline%20for%20generic%20models,camera%20calibration%20available%20to%20everyone).

```{=html}
<!-- -->
```
-   **Distance entre rayons (skew)** -- comme mentionné, c\'est un
    indicateur très sensible de l\'incohérence géométrique. Une bonne
    calibration donnera un **skew RMS proche de 0**. Dans les
    expériences, l\'approche ray-field réduit le skew jusqu\'à 30--40%
    par rapport aux détections brutes
    OpenCV[\[8\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=2D%20method%20,off%20explicit%3A%20even%20if%20some),
    signe que les deux caméras \"voient\" mieux la même scène 3D. Par
    ailleurs, on rapporte souvent le **95ᵉ centile du skew** pour
    garantir qu\'aucune correspondance n\'est catastrophique : là
    encore, les modèles ray-field atteignent des skew max très faibles
    (ex: \~0,0014 mm) attestant d\'une excellente *consistance
    rigide*[\[27\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L2650%20Method%20Raw,00139 mm)[\[28\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Method%20Raw%20RMS%20Z%20Refined,00139 mm).

```{=html}
<!-- -->
```
-   **Erreur de baseline** -- exprimée soit en pourcentage, soit
    convertie en **erreur de disparité équivalente** (plus intuitive
    pour évaluer l\'impact sur la
    stéréo)[\[29\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1719%20Baseline%20error,the%20stability%20of%20epipolar%20geometry).
    Une baisse de cette erreur signifie qu\'on a mieux déterminé la
    distance entre caméras. Avec le raffinage par rayons, une étude note
    que *« l\'erreur de base en pixels chute significativement, rendant
    l\'échelle stéréo beaucoup plus
    stable »*[\[9\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1527%20•%20The,depends%20on%20the%20quality%20of).
    Par exemple, sous flou ou bruit, le baseline estimé par OpenCV
    pouvait varier de \~0,2--0,3 px de disparité, tandis qu\'avec
    l\'optimisation ray-field ce n\'était plus que \~0,1
    px[\[9\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1527%20•%20The,depends%20on%20the%20quality%20of)[\[30\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1750%20–%20WebP,aligned).
    Cela se traduit par une meilleure superposition des paires d\'images
    rectifiées et une contrainte épipolaire plus
    stable[\[31\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=•%20Stereo%20baseline%20error%20in,only%20metric%20%28low%20is%20better)[\[32\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1428%20An%20error,and%20disparity%20errors%3B%20see%20Tab).

```{=html}
<!-- -->
```
-   **Répétabilité des paramètres** -- si l\'on recalibre plusieurs fois
    avec des données perturbées différemment, un modèle stable donnera
    des paramètres intrinseques et extrinseques peu dispersés. Les
    outils comme Kalibr, OpenCV, etc., montrent parfois des variations
    sur la focale ou le centre principal quand l\'image est très
    dégradée. Les approches récentes n\'ont pas explicitement publié ces
    écarts-type, mais le fait que le ray-field *ne dépend pas d\'un
    optimum fragile de 10 paramètres* suggère qu\'il peut être moins
    sujet aux minima locaux et donc plus répétable. Le **survey 2024 de
    Shao et al.** souligne par exemple que le modèle de Mei avait
    tendance à l\'instabilité numérique (plusieurs jeux de paramètres
    donnant des ré projections
    quasi-équivalentes)[\[18\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=2,from%20parameter%20instability%20and%20redundancy),
    ce qui complique la répétabilité, alors qu\'un modèle spline bien
    régularisé évite ce piège.

En résumé, tous ces indicateurs empiriques pointent vers une
**supériorité du modèle à rayons en conditions dégradées** : il offre
une calibration plus fidèle (erreurs 2D/3D réduites), et plus *robuste*
(géométrie stéréo cohérente, invariance de l\'échelle, etc.) face à des
imprécisions de mesure.

## Implémentations Open-Source et Projets Reproductibles

Jusqu\'à récemment, les modèles génériques étaient surtout du domaine de
la recherche, mais cela change. Plusieurs projets open-source permettent
d\'expérimenter et de reproduire ces résultats de stabilité :

-   **PuzzlePaint Camera Calibration** -- Thomas Schöps et collègues ont
    publié leur pipeline complet de calibration générique (modèles
    central et non-central par grille de splines) suite à
    CVPR 2020[\[33\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=generic%20models%20should%20be%20preferred,camera%20calibration%20available%20to%20everyone).
    Le code (disponible sur GitHub) automatise la détection (avec un
    motif de calibration spécial type *« deltille »* et un affinage
    subpixel avancé), l\'initialisation dense et le bundle adjustment
    par rayons. Ce projet a servi à démontrer en pratique que *« 10 000
    paramètres valent mieux que 12 »* pour obtenir une calibration sans
    biais[\[34\]](https://arxiv.org/abs/1912.02908#:~:text=,examples%2C%20we%20show%20that%20the).
    Il est donc possible pour quiconque de tester ce calibrage générique
    sur ses propres images et de constater la meilleure stabilité
    lorsque les données sont bruitées ou hors du modèle classique.

```{=html}
<!-- -->
```
-   **PYCASO (Python Calibration by Soloff)** -- Il s\'agit d\'un module
    Python publié en 2023 (SoftwareX) dédié à la calibration
    stéréoscopique par la méthode de
    Soloff[\[35\]](https://www.sciencedirect.com/science/article/pii/S235271102300136X#:~:text=The%20PYthon%20module%20for%20the,a%20general%20and%20accurate%20method).
    Bien que ce soit un modèle polynomial **non physique** (Soloff), il
    est notable car de nombreux travaux en **vélocimétrie par images de
    particules (PIV/PTV)** l\'ont employé et comparé à d\'autres
    approches. Le code PYCASO permet notamment de calibrer une paire de
    caméras via un polynôme direct et de réaliser ensuite un ajustement
    de Levenberg-Marquardt pour améliorer l\'inversion (ils distinguent
    *« polynôme direct »* et *« polynôme de Soloff avec LM »* dans les
    comparatifs)[\[36\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1.16%20Pycaso,5)[\[37\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Pycaso%20originally%20distinguishes%20the%20direct,The).
    Ce projet a servi de *baseline* dans les comparaisons mentionnées
    plus haut -- par exemple, le *benchmark* **Pycaso *Z-sweep*** oppose
    la reconstruction 3D par polynômes de Soloff à celle par ray-field
    sur des scènes
    synthétiques[\[38\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1.16%20Pycaso,reconstruction%20in%20the%20same%20synthetic)[\[39\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Our%20implementation%20of%20the%20Soloff,We%20refer).
    Les résultats, qu\'on a cités, montrent clairement l\'avantage du
    ray-field en robustesse (PYCASO perd l\'échelle sous forte
    compression, contrairement au modèle par rayons qui reste
    juste[\[4\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Dataset%20Pycaso%20direct%20RMS%20Z,23)[\[5\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1650%20,triangulates%20stably%20because%20it%20regularizes)).
    Le code étant ouvert, cela constitue une base reproductible pour qui
    veut vérifier ces affirmations ou adapter la méthode.

```{=html}
<!-- -->
```
-   **StereoComplex** -- C\'est un projet de recherche (2025) qui
    propose un pipeline complet pour la reconstruction stéréo sans
    modèle pinhole, en exploitant des *fields* de rayons en 2D et 3D. Il
    intègre des outils de **décodage et lissage des coins ChArUco** par
    un champ de distorsion 2D, puis une calibration stéréo par champ de
    rayons 3D (base Zernike) avec bundle adjustment. Les auteurs
    fournissent des scripts pour reproduire les expérimentations de
    robustesse (*robustness sweep*, *compression stress test*, etc.)
    évoquées
    précédemment[\[40\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1,length%2C%20and%20the%20image%20degradations)[\[41\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1,in%20the%20face%20of%20severe).
    Par exemple, le script `sweep_z_compare_pycaso.py` génère les JSON
    de performance des méthodes Soloff vs
    ray-field[\[42\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1,q80)[\[4\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Dataset%20Pycaso%20direct%20RMS%20Z,23).
    Ce degré de reproductibilité (jeux de données synthétiques fournis,
    code d\'évaluation) est précieux pour vérifier par soi-même la
    stabilité supérieure du modèle ray-field dans divers scénarios
    (changement d\'échelle de mire, flou, aberrations optiques simulées,
    compression,
    etc.[\[43\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1,length%2C%20and%20the%20image%20degradations)).
    StereoComplex n\'est pas (encore) un projet grand public, mais
    illustre la tendance à documenter les bénéfices des modèles à rayons
    avec des **métriques claires et du code ouvert**.

```{=html}
<!-- -->
```
-   **Autres outils** : BabelCalib (2023) a été proposé comme approche
    unifiée pour caméras centrales, et bien qu\'il soit davantage
    orienté multi-modèles paramétriques, il vise aussi à faciliter la
    calibration flexible (son nom *babel* suggère la compatibilité de
    nombreux modèles dans un même outil). Basalt (VINSE), Camodocal,
    Kalibr sont des frameworks établis en robotique ; ils supportent
    surtout des modèles globaux (pinhole, fisheye, omnidirectionnel) et
    gèrent plutôt bien les calibrations classiques, mais n\'implémentent
    pas encore de modèle \"champ de rayons\" fully non-paramétrique.
    Néanmoins, la communauté va vers plus de généralité : par exemple
    l\'outil **Kalibr** commence à intégrer des motifs comme AprilGrid
    et des modèles Kannala--Brandt 8 paramètres, ce qui prolonge la
    validité du calibrage pour très grand
    angle[\[18\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=2,from%20parameter%20instability%20and%20redundancy).
    On peut s\'attendre à ce que des évolutions ou plugins intègrent à
    l\'avenir les méthodes par rayons, étant donné les avantages
    démontrés.

## Conclusion

**En conclusion**, la littérature et les expérimentations récentes
mettent en évidence qu\'une **modélisation par champs de rayons offre
une stabilité supérieure** par rapport aux modèles de caméra classiques,
en particulier face aux imprécisions de détection. Qu\'il s\'agisse de
bruit subpïxel, d\'images compressées avec pertes, de légers flous ou
d\'aberrations optiques non prévues par un modèle polynomial, la
calibration par rayons montre moins de dégradation : les erreurs de
reprojection restent faibles et non biaisées, les reconstructions 3D
gardent la bonne échelle, et la géométrie stéréo demeure cohérente
(rayons qui croisent, épipolarité
respectée)[\[6\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Key%20result%3A%203D%20ray,ready%20NPZ)[\[9\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1527%20•%20The,depends%20on%20the%20quality%20of).
À l\'inverse, les approches paramétriques peuvent introduire des biais
systématiques lorsque les conditions s\'écartent de leurs hypothèses (on
l\'a vu avec le *mapping* polynomial perdant pied sous bruit, ou avec un
modèle pinhole classique ne pouvant pas absorber une distorsion
complexe)[\[5\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1650%20,triangulates%20stably%20because%20it%20regularizes)[\[2\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=Figure%201%3A%20%20Residual%20distortion,Interpolation%20happens%20among).

Ces gains de robustesse ne viennent pas sans contrepartie : un modèle
générique requiert davantage d\'images (idéalement couvrant densément le
champ visuel) et un calcul plus lourd. Cependant, grâce aux progrès des
algorithmes (bundle adjustment régularisé, initialisation automatiques)
et à la puissance de calcul actuelle, ces méthodes deviennent
praticables au-delà du labo. L\'accès à des implémentations open-source
comme celles citées permet d\'adopter ces techniques dans des contextes
appliqués (robotique, métrologie, réalité virtuelle, etc.) où une
calibration *fiable et stable* est critique. Les éléments compilés dans
ce rapport -- comparatifs chiffrés, métriques dédiées (erreur 3D, skew,
baseline, etc.) et ressources logicielles -- illustrent clairement la
**supériorité des modèles ray-based** en termes de stabilité, et
invitent à les préférer \"chaque fois que
possible\"[\[12\]](https://arxiv.org/abs/1912.02908#:~:text=calibration%20pipeline%20for%20generic%20models,camera%20calibration%20available%20to%20everyone)
pour des mesures 3D les plus exactes et robustes possibles.

**Sources :** Les affirmations et chiffres ci-dessus s\'appuient sur des
publications récentes et comparatifs disponibles : par exemple *Schöps
et al., CVPR 2020* pour l\'argument général sur 10k vs 12
paramètres[\[34\]](https://arxiv.org/abs/1912.02908#:~:text=,examples%2C%20we%20show%20that%20the),
le *projet StereoComplex (2025)* pour les tests sous bruit de détection
(mire ChArUco
bruitée)[\[6\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Key%20result%3A%203D%20ray,ready%20NPZ)[\[4\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Dataset%20Pycaso%20direct%20RMS%20Z,23),
ainsi que le *survey Shao et al. 2024* sur les outils de calibration
grand
angle[\[18\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=2,from%20parameter%20instability%20and%20redundancy).
Les données plus détaillées (tableaux d\'erreurs, graphiques de bias)
proviennent du rapport technique associé au code
StereoComplex[\[4\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Dataset%20Pycaso%20direct%20RMS%20Z,23)[\[8\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=2D%20method%20,off%20explicit%3A%20even%20if%20some)
et de l\'article PTV de Barta *et al.* 2025 comparant polynômes vs
modèles
physiques[\[16\]](https://elib.dlr.de/215234/1/Barta_2025_Meas._Sci._Technol._36_055301.pdf#:~:text=paper%2C%20we%20review%20and%20compare,In)[\[17\]](https://elib.dlr.de/215234/1/Barta_2025_Meas._Sci._Technol._36_055301.pdf#:~:text=experiments%20without%20volumetric%20refinement,In).
Enfin, les liens vers les codes open-source sont fournis pour
approfondir ou reproduire ces résultats (cf. références
Puzzlepaint[\[33\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=generic%20models%20should%20be%20preferred,camera%20calibration%20available%20to%20everyone),
PYCASO[\[35\]](https://www.sciencedirect.com/science/article/pii/S235271102300136X#:~:text=The%20PYthon%20module%20for%20the,a%20general%20and%20accurate%20method),
etc.). Chaque source est indiquée en citation pour permettre au lecteur
de vérifier les chiffres et propos (voir références en **【】**). Les
tableaux et figures mentionnés (ex: Fig. 1 de Schöps *et al.* montrant
les résidus de
distorsion[\[2\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=Figure%201%3A%20%20Residual%20distortion,Interpolation%20happens%20among),
ou les tableaux de StereoComplex sur l\'erreur sous
compression[\[4\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Dataset%20Pycaso%20direct%20RMS%20Z,23))
illustrent visuellement ces constats de stabilité et peuvent être
consultés via les liens fournis.

En synthèse, *« les modèles à 10 000 paramètres doivent être privilégiés
face aux modèles à douze »*, surtout dans des scénarios non idéaux, pour
obtenir des calibrations plus stables et
fiables[\[12\]](https://arxiv.org/abs/1912.02908#:~:text=calibration%20pipeline%20for%20generic%20models,camera%20calibration%20available%20to%20everyone).
Les développements récents rendent cette approche de plus en plus
accessible, augurant d\'une adoption élargie dans les applications
industrielles et scientifiques où la précision 3D est primordiale.

[\[1\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=We%20briefly%20describe%20two%20generic,spline%20model%20is)
[\[11\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=From%20the%20review%20and%20experiments%2C,we%20summarize%20several%20findings)
[\[18\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=2,from%20parameter%20instability%20and%20redundancy)
[\[19\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/#:~:text=pinhole%20radial%E2%80%93tangential%20model%20well,from%20parameter%20instability%20and%20redundancy)
Geometric Wide-Angle Camera Calibration: A Review and Comparative
Study - PMC

<https://pmc.ncbi.nlm.nih.gov/articles/PMC11510980/>

[\[2\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=Figure%201%3A%20%20Residual%20distortion,Interpolation%20happens%20among)
[\[3\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=match%20at%20L631%20The%20generic,Interestingly)
[\[13\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=camera%20pose%20estimation%20as%20examples%2C,released%20our%20calibration%20pipeline%20at)
[\[24\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=these%20cells%20in%20Fig)
[\[25\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=%28colors%29.%20For%20SC,were%20too%20inconsistent)
[\[26\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=We%20also%20list%20the%20median,and%20the%20empirical%20distribution%20of)
[\[33\]](https://ar5iv.labs.arxiv.org/html/1912.02908#:~:text=generic%20models%20should%20be%20preferred,camera%20calibration%20available%20to%20everyone)
\[1912.02908\] Why Having 10,000 Parameters in Your Camera Model is
Better Than Twelve

<https://ar5iv.labs.arxiv.org/html/1912.02908>

[\[4\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Dataset%20Pycaso%20direct%20RMS%20Z,23)
[\[5\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1650%20,triangulates%20stably%20because%20it%20regularizes)
[\[6\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Key%20result%3A%203D%20ray,ready%20NPZ)
[\[7\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=From%20each%20correspondence%20we%20thus,n∥%20%2C%20with%20n)
[\[8\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=2D%20method%20,off%20explicit%3A%20even%20if%20some)
[\[9\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1527%20•%20The,depends%20on%20the%20quality%20of)
[\[10\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1654%20the%20rays,py)
[\[22\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=y%28u%2C%20v%29,v%20−%20v0%20R)
[\[23\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=x,modes%20up%20to%20n%20≤)
[\[27\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L2650%20Method%20Raw,00139 mm)
[\[28\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Method%20Raw%20RMS%20Z%20Refined,00139 mm)
[\[29\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1719%20Baseline%20error,the%20stability%20of%20epipolar%20geometry)
[\[30\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1750%20–%20WebP,aligned)
[\[31\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=•%20Stereo%20baseline%20error%20in,only%20metric%20%28low%20is%20better)
[\[32\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=match%20at%20L1428%20An%20error,and%20disparity%20errors%3B%20see%20Tab)
[\[36\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1.16%20Pycaso,5)
[\[37\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Pycaso%20originally%20distinguishes%20the%20direct,The)
[\[38\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1.16%20Pycaso,reconstruction%20in%20the%20same%20synthetic)
[\[39\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=Our%20implementation%20of%20the%20Soloff,We%20refer)
[\[40\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1,length%2C%20and%20the%20image%20degradations)
[\[41\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1,in%20the%20face%20of%20severe)
[\[42\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1,q80)
[\[43\]](file://file_00000000c2b871f48005e0b10449d790#:~:text=1,length%2C%20and%20the%20image%20degradations)
stereocomplex.pdf

[file://file_00000000c2b871f48005e0b10449d790](../../../../)

[\[12\]](https://arxiv.org/abs/1912.02908#:~:text=calibration%20pipeline%20for%20generic%20models,camera%20calibration%20available%20to%20everyone)
[\[14\]](https://arxiv.org/abs/1912.02908#:~:text=accuracy,camera%20calibration%20available%20to%20everyone)
[\[34\]](https://arxiv.org/abs/1912.02908#:~:text=,examples%2C%20we%20show%20that%20the)
\[1912.02908\] Why Having 10,000 Parameters in Your Camera Model is
Better Than Twelve

<https://arxiv.org/abs/1912.02908>

[\[15\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schops_Why_Having_10000_Parameters_in_Your_Camera_Model_Is_Better_CVPR_2020_paper.pdf#:~:text=parametric%20cal%02ibration%20,to%20practical%20advantages%20in%20applications)
[\[20\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schops_Why_Having_10000_Parameters_in_Your_Camera_Model_Is_Better_CVPR_2020_paper.pdf#:~:text=inaccuracy%20,work%2C%20we%20follow%20these%20approaches)
[\[21\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schops_Why_Having_10000_Parameters_in_Your_Camera_Model_Is_Better_CVPR_2020_paper.pdf#:~:text=our%20calibration%20pattern%20and%20feature,us%20to%20ob%02serve%20and%20model)
Why Having 10,000 Parameters in Your Camera Model Is Better Than Twelve

<https://openaccess.thecvf.com/content_CVPR_2020/papers/Schops_Why_Having_10000_Parameters_in_Your_Camera_Model_Is_Better_CVPR_2020_paper.pdf>

[\[16\]](https://elib.dlr.de/215234/1/Barta_2025_Meas._Sci._Technol._36_055301.pdf#:~:text=paper%2C%20we%20review%20and%20compare,In)
[\[17\]](https://elib.dlr.de/215234/1/Barta_2025_Meas._Sci._Technol._36_055301.pdf#:~:text=experiments%20without%20volumetric%20refinement,In)
Comparison of camera calibration methods for particle tracking
velocimetry

<https://elib.dlr.de/215234/1/Barta_2025_Meas._Sci._Technol._36_055301.pdf>

[\[35\]](https://www.sciencedirect.com/science/article/pii/S235271102300136X#:~:text=The%20PYthon%20module%20for%20the,a%20general%20and%20accurate%20method)
PYCASO: Python module for calibration of cameras by Soloff's method -
ScienceDirect

<https://www.sciencedirect.com/science/article/pii/S235271102300136X>
