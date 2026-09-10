# RHO rapide et superviseur d'endurance à plusieurs minutes

## But et séparation des calculs

Le RHO d'un cycle assure le mouvement. Un superviseur plus lent choisit la
répartition de l'effort en regardant 120 à 300 cycles devant lui. La période
des archives utilisées est d'une seconde : ces horizons représentent deux
et cinq minutes. Le superviseur fournit quelques poids numériques, fixes
pendant une résolution RHO. Le NLP pourra conserver son graphe compilé et
IPOPT/MA57; la latence inférieure à une seconde reste à vérifier sur la boucle
complète.

Le superviseur est conçu pour pouvoir travailler en arrière-plan. Son délai
de calcul, l'âge des observations et le changement d'état pendant son calcul
doivent être pris en compte avant d'appliquer une proposition. L'implémentation
présente prépare ce contrat; elle ne lance pas de commande clinique.

## Ce qui change par rapport au preview à deux phases

Le dernier preview conditionnel donnait les mêmes PW que l'allocation rapide
sur les cas testés. Son coût local reproduisait aussi les mêmes valeurs aux
41 points de chaque ajustement, avec un temps de construction de 17–18 s.
Cela ne justifiait pas son intégration au RHO.

La nouvelle étape cherche directement une répartition de stimulation dont les
conséquences futures sont meilleures. Il faut d'abord que les poids modifient
effectivement la commande. Pondérer seulement l'écart à une répartition de
référence déjà réalisable peut laisser cet écart nul pour tous les poids.

Le prototype pénalise donc le recrutement normalisé de chaque muscle au carré,
avec une faible préférence supplémentaire pour la répartition de référence.
Augmenter le poids d'un muscle encourage à utiliser les autres lorsque le
moment demandé et leurs capacités le permettent. Les muscles antagonistes
gardent leurs contributions signées. Le moment total original et les bornes
de PW restent des contraintes, vérifiées aux extrémités des phases du modèle
compact. Cela ne valide pas des adaptations d'inertie ou de roue libre entre
les phases, notamment autour de la zone commune de moments négatifs.

Ce coût de stimulation est un choix de politique, pas une loi d'endurance.
Le cas « tous les poids égaux » est une nouvelle référence et ne doit pas
être confondu avec l'ancien allocateur centré sur les moments archivés.

### Priorité après lecture de l'article de handcycling

La demande de réutiliser les formules de l'annexe change la référence
scientifique de la prochaine expérimentation : partir des poids physiologiques
et mécaniques calculés, puis tester leur sensibilité aux paramètres musculaires,
avant de rechercher des corrections lentes. Le petit coût de recrutement reste
un test technique de redistribution, pas un substitut au coût de l'article.

L'article rapporte, pour son RHO de deux cycles, 1230 cycles sans pondération,
1692 avec les poids calculés et 1729 avec les poids bayésiens. Ces résultats
concernent son modèle et sa tâche, pas les archives actuelles. La méthode de
calcul des poids ne nécessite pas de FHO.

Le coût de fatigue de l'équation 16 intègre une racine de somme pondérée des
pertes de capacité au carré. Il porte sur la perte accumulée de capacité,
pas sa dérivée. Remplacer ce coût par le recrutement au carré, supprimer la
racine, ou normaliser chaque perte par la capacité au repos à poids inchangés
change le coût et impose une comparaison distincte. Pour conserver exactement
l'expression RMS en passant aux pertes relatives, il faudrait multiplier
chaque poids brut par le carré de la capacité au repos correspondante.
Le poids nul du triceps dans l'article
ne peut pas non plus entrer tel quel dans le superviseur expérimental limité
aux poids strictement positifs entre 0,25 et 4.

Le protocole détaillé, les paramètres disponibles et les informations encore
manquantes sont dans [le plan de validation des poids de l'article](physiological_weights_validation_plan.md).
L'implémentation fidèle des formules attend le Supplementary Material 1,
absent du PDF principal fourni. Le benchmark du coût de recrutement ne doit
pas être présenté comme cette validation.

## Trois briques à vérifier séparément

### Le cycle musculaire contrôlé par les poids

La génération de force utilise la carte compacte Ding existante. Chaque phase
conserve le calcium, la force résiduelle et les trois états de fatigue; les
PW sont réévaluées en fonction de l'état et des poids. Les écarts initiaux de
fatigue présents dans les archives ne sont pas effacés.

Cette première version propage explicitement toutes les phases. Elle sert de
référence pour vérifier une future version qui avancera par blocs de cycles.
Elle permet déjà de tester si des poids différents changent les forces et le
nombre de phases réalisables.

### La carte de fatigue sous un cycle de force imposé

Les équations lentes Ding peuvent être composées exactement quand le profil
de force est imposé. On résume chaque phase par une intégrale de force qui
tient compte de la récupération jusqu'à la fin de la phase. Une simple force
moyenne ne conserve pas exactement cette information.

Si ce même profil de force est répété, la fatigue après des centaines de
cycles se calcule directement. Les états lents à la fin des phases du premier
et du dernier cycle bornent aussi ceux des cycles intermédiaires, sous cette
hypothèse précise. Les états algébriques devenus non physiques sont signalés.

Cette accélération ne démontre pas que les PW pourront encore produire le
profil de force imposé. Avant de l'utiliser pour sauter des cycles dans une
projection d'endurance, il faudra reconstruire la commande, vérifier la
faisabilité mécanique et contrôler les erreurs entre les points de contrôle.
Le premier prototype garde cette carte séparée du classement des politiques.

### Le superviseur de poids

Le superviseur examine un petit ensemble de poids positifs. Leur échelle
commune est supprimée pour ne comparer que les différences entre muscles.
Le réglage courant figure toujours parmi les candidats. Le premier essai
annonce un facteur de modification de 1,5 pour chaque muscle, en augmentation
et en diminution; ces choix ne seront pas réajustés après les résultats.

Si plusieurs candidats terminent l'horizon, le superviseur favorise celui qui
conserve la meilleure marge de moment dans sa phase la plus fragile. Si aucun
ne termine, il compare les préfixes réalisés parmi les échecs de capacité
explicitement comparables. Les erreurs de calcul et les refus du domaine du
modèle sont exclus par défaut du classement et ne deviennent pas de faux
scores d'endurance. Un échec de cette politique ne prouve pas que toute autre
stratégie serait impossible.

Un budget de calcul arrête le lancement de nouveaux candidats. Un candidat
en cours peut dépasser ce budget : ce mécanisme n'est pas une interruption
temps réel. Une proposition conserve son état de départ, son contexte, ses
résultats et son âge. L'acceptation ultérieure utilise des tolérances explicites
sur l'état, le retard en cycles et le changement des poids courants.

## Validation avant une liaison au RHO

1. Vérifier l'optimum du petit problème de répartition avec un calcul
   indépendant, les signes des moments et les limites de stimulation.
2. Vérifier la carte de fatigue répétée contre une intégration indépendante,
   y compris les écarts initiaux non nuls et un cas devenant non physique.
3. Comparer neuf réglages de poids sur les mêmes états terminaux archivés,
   aux horizons 120 et 300 cycles. Mesurer tous les calculs de candidats.
4. Rejouer les PW sélectionnées avec les équations Ding complètes. Distinguer
   explicitement une vérification de préfixe d'un exercice complet.
5. Vérifier que le véritable RHO répond aux poids comme la projection le
   prévoit, avec les mêmes termes de coût et la même normalisation.
6. Installer ensuite la mise à jour asynchrone, mesurer les délais du RHO,
   puis comparer des exercices effectivement exécutés avec et sans superviseur.

Les étapes 5 et 6 sont nécessaires pour conclure à un bénéfice du contrôleur.
Les simulations présentes ne nécessitent aucune donnée FHO.

## Fondements et périmètre

La séparation des dynamiques rapides et lentes s'appuie sur le principe du
[MPC à plusieurs échelles de temps](https://arxiv.org/abs/2205.10433).
L'évaluation de politiques futures pour guider un horizon court relève du
[cadre rollout et programmation dynamique](https://web.mit.edu/dimitrib/www/Bertsekas_NMPC_IFAC.pdf).
Ces références fournissent des principes d'architecture; elles ne valident pas
ce modèle musculaire ni les performances cliniques du prototype.
