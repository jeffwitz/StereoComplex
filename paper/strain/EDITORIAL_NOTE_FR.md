# Positionnement du manuscrit court

Le manuscrit livré porte sur une question utile à la communauté CIN/DIC :
**le modèle qui restitue le mieux la profondeur est-il aussi celui qui préserve
le mieux les longueurs lors d'un mouvement rigide ?** Sur le balayage fin
disponible, la réponse est non. Cette observation est étayée par trois méthodes,
des plans réservés à l'évaluation, un contrôle spatial supplémentaire et des
simulations dont la vérité est connue.

## Ce qui a été effectivement réalisé

Les 404 images des deux séries PYCASO ont été retraitées. Les comparaisons
utilisent les coins directement observés, sans compléter les points manquants.
Les modèles sont ajustés sur dix plans par série et évalués sur les 91 autres.
Les 488 ajustements simulés couvrent deux étendues de profondeur, plusieurs
niveaux de bruit et une extension connue de 1000 microdéformations. Les données,
scripts, sorties détaillées et figures sont fournis.

Les erreurs de profondeur du balayage fin sont 9,41 µm pour le champ de rayons,
5,89 µm pour le modèle de type Soloff et 5,20 µm pour le polynôme direct. Les
déformations apparentes des jauges de 1,2 mm sont respectivement 185, 268 et
338 µε. Le champ de rayons présente donc le meilleur résultat sur les longueurs,
mais pas sur la profondeur ni sur le déplacement du centre de la mire.

## Originalité et solidité scientifique

L'originalité défendable est **circonscrite** : une représentation de rayons
référencée sur un plan fini, ajustable linéairement, est évaluée avec des critères
qui font apparaître un compromis concret entre profondeur et déformation
apparente. La formulation et le protocole sont suffisamment explicites pour
être repris dans une chaîne de mesure.

Le principe d'une calibration par rayons existe déjà, notamment chez
[Bothe et al. (2010)](https://doi.org/10.1364/AO.49.005851). L'importance de la
propagation des erreurs de calibration vers les grandeurs mesurées existe aussi,
notamment chez [Reu (2013)](https://doi.org/10.1007/s11340-013-9746-1).
L'inversion du classement entre critères ne doit donc pas être vendue comme un
principe scientifique inédit. La nouveauté est dans la formulation pratique,
la comparaison reproductible et son résultat documenté pour ces données.

Le principal risque de relecture est la portée de la validation : les positions
du déplacement sont nominales, et la déformation est évaluée sur une mire
rigide par des jauges entre coins. Il ne s'agit pas d'une validation complète de
la CIN sur un matériau déformé avec une mesure indépendante. La différence de
classement observée est réelle dans les sorties du protocole, mais son origine
ne peut être attribuée sans ambiguïté à la seule géométrie du modèle.

Les simulations contrôlent les erreurs de calcul et la restitution d'une
extension. Elles montrent aussi que les méthodes ont des performances presque
identiques sous bruit indépendant dans le cas central simulé. Ce résultat
limite utilement la généralité des conclusions expérimentales.

## Choix de revue

**Strain est la cible prioritaire que je recommande.** La question du choix de
calibration en fonction de l'observable et de la déformation parasite parle
directement à la métrologie expérimentale. Présenter ce travail comme une
contribution méthodologique compacte est plus cohérent que promettre une
identification physique complète du microscope. Les
[instructions aux auteurs](https://onlinelibrary.wiley.com/page/journal/14751305/homepage/forauthors.html)
devront guider l'habillage final et les déclarations de soumission.

**JTCAM reste envisageable, mais le positionnement est moins direct.** Son
[périmètre](https://jtcam.episciences.org/) comprend la mécanique expérimentale,
les articles de données et de logiciels, tout en recherchant des contributions
d'importance fondamentale. Avec les données actuelles, l'argument le plus
plausible serait un protocole ouvert d'évaluation et un outil réutilisable. Le
manuscrit ne démontre pas aujourd'hui une nouvelle méthode générale de mesure
des champs mécaniques. Aucune probabilité d'acceptation sérieuse ne peut être
chiffrée sur cette seule analyse.

## Ce qui a été retiré du récit principal

L'identification des paramètres physiques du microscope, la sélection BIC des
architectures, l'interprétation des modes comme aberrations optiques et la
régularisation de Schur ne sont pas nécessaires à la question posée. Leur
maintien multiplierait les affirmations à défendre sans renforcer la preuve
sur les grandeurs mécaniques. Le manuscrit long reste intact dans `paper/cmo`.

La reconstruction de pièce de monnaie n'est pas ajoutée comme illustration de
plus. L'article PYCASO publié comporte déjà une comparaison avec profilométrie,
mais les valeurs numériques de cette référence n'ont pas été trouvées dans les
dépôts inspectés. Une nouvelle carte de relief sans référence supplémentaire
allongerait l'article sans valider davantage la précision.

## Débouchés accessibles sans nouvelles acquisitions

1. Propager les covariances de localisation jusqu'aux déplacements et aux
   jauges, pour construire un outil d'aide au choix de calibration et de
   longueur de jauge.
2. Étudier un ajustement commun tenant compte des erreurs sur les coordonnées
   image et objet : les trois modèles actuels minimisent des résidus dans des
   espaces différents, ce qui laisse une question méthodologique précise.
3. Évaluer sur les images archivées l'effet des algorithmes de correspondance
   et du lissage, en imposant les mêmes points et masques à toutes les méthodes.
4. Diffuser les observations et les partitions comme petit benchmark de
   calibration pour la communauté DIC, avec scores de position, mouvement et
   longueur, plutôt qu'un classement fondé sur un seul résidu image.

Ces prolongements constituent des projets distincts. Ils ne sont pas annoncés
comme déjà démontrés dans le manuscrit et ne sont pas nécessaires à la lecture
de sa contribution actuelle.
