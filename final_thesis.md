2. Codage neuronal
2.1 Cadre
2.1.1 Enjeux

L2F concentre actuellement son activité dans création d'une librairie d'analyse
topologique des données et son utilisation dans des cas concrets d'étude de
données complexes tels que la prédiction en environnement chaotiques
(attracteur de Lorenz par exemple), détection de changement de régime sur un
signal bruité, prédictions de crash financiers basées sur des séries
temporelles.

Nous avons alors choisi d'étudier un problème actuel proposé par GNT ENS
qui est proposé sous la formé d'un challenge où l'objectif d'un point de vue
du candidat est de produire les meilleures prédictions sur un problème donné.

Il consiste à prédire l'activité cérébrale, binaire, dont la signification des
étiquettes est non indiquée à partir de séries temporelles de pics neuronaux.

Ce problème de classification comprend donc un jeu de données avec des
étiquettes, le candidat doit prédire de manière logicielle la classe d'un jeu
de données test sans étiquettes.

Ce challenge est orienté recherche, le jeu de données est public (inscription
requise) et est ouvert à tout public.

2.1.2 Métrique d'évaluation

Le score utilisé pour le classement est le cohen_kappa_score:

TODO: mettre formule

La répartition des étiquettes étant inégale entre les 0 et 1, la métrique a
été choisie pour prendre ce paramètre en compte.

Le jeu de données d'entraînement comporte 16000 éléments contre 11000 pour
celui de test.

TODO : image

2.1.3 Base d'évaluation

Une classement sur le jeu de données de test est également accessible
publiquement, la base d'évaluation fixée par les organisateurs est produite à
partir de caractéristiques statistiques (211 au total) calculées sur les séries
temporelle des intervalles inter-pics.

Un algorithme de forêts aléatoires est ensuite utilisé pour le modèle de
prédiction et le cohen-kappa score annoncé est de 0.3273.

Les paramètres utilisés ne sont pas communiqués.

2.2 Présentation des données
2.2.1 Séquence de potentiels d'action

Chaque exemple du jeu de données contient le temps d'arrivée en unité arbitraire
de 50 pics consécutifs.

Nous n'avons aucune information sur la forme du potentiel d'action, on a donc
uniquement le choix d'étudier la séquence d'intervalles entre les pics ou les
temps d'arrivée.

2.2.2 Connectivité des neurones

De plus, chaque exemple appartient à un neurone numéroté et un neurone peut
comporter plusieurs exemples.

Il existe plusieurs théories sur le fonctionnement cérébral, l'un est basé sur
la théorie d'un réseau de neurones, l'autre qui ne considère pas la
connectivité et s'intéresse uniquement à l'évolution de codage neuronal en
présence d'un stimuli externe. TODO: 1605.01905

Les deux approches sont valides et compatibles entre elles mais nous nous
intéresserons uniquement à la seconde approche.

Notons également que le jeu de données de test contient uniquement des neurones
non présents dans le jeu de données d'entraînement.

3. Topologie algébrique de séries temporelles

3.1 Introduction (pourquoi ?)
3.1.1 Propriétés, définitions et procédure
L'analyse topologique des données consiste à étudier la structure de graphes,
leur connectivité ou la structure d'un nuage de points.

Les définitions introduites par la suite seront de nature informelle, sans
entrer dans le détail afin de conserver la notion d'intuition.

La procédure usuelle consiste en plusieurs étapes:
On définit un complexe simplicial comme un graphe pouvant contenir des
triangles en plus de sommets et arêtes. Ces objets peuvent ainsi être créés en
plus grande dimension.

On définit une filtration de manière informelle comme une séquence de complexes
simpliciaux donnés par un algorithme de création de complexes simpliciaux à
partir d'un nuage de points.

Plus concrètement, nous nous concentrerons sur la filtration Vietoris-Rips
dont l'algorithme de création étant le suivant: étant donné un nuage de point
et E, on crée une arête entre deux points si et seulement si la distance entre
ces deux points est inférieure à 2E.

De cette manière, on obtient une séquence de Vietoris-Rips complexes en
fonction de E.

On définit ensuite le k-ème nombre de Betti comme le nombre de surfaces
k-dimensionnelles indépendantes. De manière plus intuitive, b0 correspond au
nombre de composantes connexes, b1 aux trous en dimension 2.

TODO: image

De cette séquence on peut quantifier la durée de vie d'une composante connexe
ainsi que des "trous" à l'aide d'un diagramme de persistence qui a pour
abscisse la naissance et la mort en ordonnée du trou ou de la composante
connexe selon la dimension choisie.

TODO: image

A partir de ces diagrammes, il sera alors possible de créer des
caractéristiques qu'on pourra étudier d'un point de vue statistique.

3.1.2 Intuition
Dans notre cas, on cherche dans un premier temps à construire un nuage de
points à partir d'une série temporelle.
Le time delay embedding a été prouvé comme étant un moyen efficace pour
caractériser les séries temporelles chaotiques.

L'objectif est de pouvoir quantifier la périodicité d'un système dynamique, et
plus particulièrement d'un signal.

TODO : mettre image

3.2 Méthodologie et caractérisation (comment ?)
3.2.0 Gaussian KDE
Cette première étape expérimentale consiste à faire une estimation par noyau
gaussien dans le but de transformer la série temporelle binaire en signal.
Les paramètres de cette méthode sont le nombre de points par pic et la variance.

3.2.1 Takens embedding
Nous appliquons ensuite un changement de représentation temporisé (time delay
embedding) sur le signal

Cette fonction est incluse dans la librairie, des tests unitaires ont été
implémentés par la suite dans le cadre de l'amélioration de la librairie.

3.2.2 Homologie persistente
Nous calculons ensuite les diagrammes de persistence donnés par l'algorithme
précédemment décrit.

Notre implémentation en Python fera appel à une méthode en C++ de la librairie
GUDHI, développée par l'INRIA, pour des raisons de performance.

De plus, le choix des dimensions sera limité à 0 et 1 étant donné le nombre de
points élévé après l'introduction du signal.

3.2.3 Amplitude
On définit l'amplitude d'un diagramme de persistence comme la distance entre ce
diagramme et le diagramme neutre (qui correspond à une infinité de points sur
la diagonale)

Plusieurs métriques sont actuellement supportées par la librairie dont:
wasserstein, betti, heat, landscape et bottleneck.

3.2.4 Algebraic functions and tropical functions
Basé sur les travaux effectués par Adcock et al., 2016, plusieurs combinaisons
algébriques sont calculées à partir des diagrammes de persistence.

Ces fonctions sont actuellement en cours d'intégration dans la librairie.

3.2.5 Betti curves
Les courbes ou signature de Betti correspondent aux nombres de betti en
fonction de e, leur calcul permet d'obtenir de nouvelles données, adaptés à une
étude statistique.

3.2.5.1 Area
Après visualisation des courbes et moyennes, nous avons trouvé pertinent de
calculer la somme des nombres de betti sur un intervalle fixé visuellement.

3.2.5.2 Moments
De la même manière, la répartition asymétrique des nombres de betti a motivé
notre choix de calculer les moments d'ordre 1, 2 et de les ajouter à la liste
des caractéristiques topologiques.

On obtient alors environ 15 nouveaux descripteurs des données, nous allons donc
procéder à l'évaluation.

4. Modèles de prédiction
4.1 Equilibrage

Comme mentionné précédemment, le jeu de données est deséquilibré avec un ratio
1:4.4, nous allons donc choisir une méthode de rééquilibrage en utilisation la
librairie imblearn.

Il existe 3 types de rééquilibrage: sous-échantillonage, sur-échantillonage et
la combbinaison des deux.

Parmi les méthodes de sous-échantillonage disponibles, AllKNN s'est révélé
meilleur que le choix aléatoire d'exemples.

Les trois méthodes de sur-échantillonage testés sont SMOTE, ADASYN et la copie
d'exemples en minorité choisis aléatoirement.

Aucune méthode combinée n'a été testée, les résultats montrent une meilleure
performance des méthodes de sous-échantillonage à jeu de donnée et modèle de
prédiction identique.

0.288 AllKNN
0.271 random under
0.203 random over
0.265 SMOTE
0.262 ADASYN

4.2 Choix du modèle
Le modèle retenu pour l'évaluation est un algorithme de forêts aléatoire pour
son interprétabilité et la possibilité de classer l'importance des descripteurs.

Dans tous les essais effectués, le score indiqué est basé sur une validation
croisée à 5 plis.

4.3 Perf
4.3.1 Apport de la TDA
En ajoutant 6 combinaisons de descripteurs topologiques aux 211 descripteurs
statistiques, on obtient une amélioration du score de 0.006, ce qui peut
sembler négligeable mais le classement de l'importance des descripteurs
montre que le gain est provoqué par quelques descripteurs topologiques.

5. Conclusion
Gains de la TDA

Faiblesses de la TDA

Ce qui a été négligé
Features manuelles (threshold), pas très robuste mathématiquement
Homologie de dim > 1 à cause de la puissance de calcul
restreindre le temps
normalisation neuron

Travail futur
