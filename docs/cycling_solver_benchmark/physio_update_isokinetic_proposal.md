# Physio-U : réévaluer les poids physiologiques depuis la fatigue courante

Proposition expérimentale, 25 septembre 2026. Aucun gain d'endurance n'est
encore démontré et aucun lanceur RHO existant n'active cette proposition.

## Ce qui change par rapport à l'annexe

Le calcul de l'annexe part de muscles au repos, impose 80 % de Fmax pendant
une fraction du cycle et attribue un poids élevé aux muscles fatigables qui
contribuent aux zones mécaniques difficiles. Son score brut est la contribution
mécanique multipliée par le carré d'une pente de fatigue. Les facteurs sont
ensuite normalisés par le maximum, conformément à la demande de ne plus
mettre artificiellement le plus petit poids à zéro.

Le recalcul proposé conserve ces deux questions : « quel muscle risque de
perdre de la capacité ? » et « cette capacité contribue-t-elle au travail
demandé ? ». Il les pose depuis l'état effectivement atteint, tous les 10 ou
20 cycles certifiés. Les paramètres biologiques alpha et tau ne changent pas
artificiellement avec la fatigue : ce sont les états A, tau1 et Km qui évoluent.
Le dénominateur de la fatigue reste A_rest. Remplacer A_rest par A_courant
remettrait implicitement chaque muscle à zéro de fatigue à chaque mise à jour.

La formule de l'annexe ne doit pas être relancée aveuglément pendant 1500
cycles depuis cet état : une force prescrite qui excède la capacité restante
peut produire A négatif. Ce défaut existait déjà pour les paramètres locaux.
Le nouveau noyau utilise seulement un challenge d'un cycle ; sa cadence de
réévaluation est indépendante de cette durée. Il ne prétend pas prédire les
PW sur plusieurs minutes.

## Pourquoi le cas isocinétique change la criticité

Dans le modèle actuel, la vitesse est imposée et l'OCP doit produire un travail
total par cycle. Le couple instantané est une sortie de la dynamique inverse,
avec ses bornes. Une phase peut donc produire moins et une autre plus. Le seuil
instantané constant de 0,20 Nm de l'annexe ne représente pas cette tâche.

Une contribution plus cohérente consiste à estimer le travail disponible de
chaque muscle dans les phases où il contribue au mouvement, puis sa contribution
au travail demandé lorsqu'il est combiné avec les autres muscles. Pour quatre
muscles il existe seulement 16 groupes possibles. On moyenne l'apport marginal
d'un muscle dans ces groupes : c'est une contribution de Shapley.

Cette opération est une somme numérique très courte, sans OCP supplémentaire.
Elle attribue une contribution positive à un triceps utile même si d'autres
muscles peuvent toujours le remplacer à chaque angle. Un muscle dont la
contribution mécanique est réellement nulle reste nul ; il n'y a aucune
soustraction du score minimal des autres muscles.

Dans le noyau actuel, le travail d'un groupe est le minimum entre le travail
demandé et la somme de ses opportunités de travail. Cette enveloppe néglige
les transitions de PW, la persistance de force et les bornes de couple. Elle
est donc un score mécanique approximatif, pas une borne atteignable certifiée.
Les forces passives et la contribution mécanique commune doivent être traitées
une seule fois. Il faut éviter le calcul historique qui les attribue à chaque
profil musculaire individuel.

## Une fatigue qui tient compte de l'état courant

Le challenge applique un profil de force court et calcule exactement les
équations lentes A, par morceaux à force constante. Il compare ce résultat
à une récupération pendant la même durée sans force. Cette comparaison est
essentielle : un muscle peut récupérer au total tout en subissant un coût
physiologique dû à sa stimulation.

Le candidat exploratoire utilise l'augmentation de la fatigue quadratique par
rapport à cette récupération. Au repos, elle est égale au carré de la diminution
normalisée de A, ce qui rejoint la structure de la formule de l'annexe. Après
fatigue, elle dépend de A courant et de la récupération. Le score brut combine
ce coût physiologique avec la contribution au travail.

Il existe un risque réel de double pondération : le RHO minimise déjà une
fatigue quadratique. Un score qui augmente lui-même avec la fatigue peut
surprotéger un muscle fatigué. Il faut donc comparer au minimum ces ablations :

1. poids physiologiques fixes recalculés correctement pour le cas isocinétique ;
2. contribution mécanique actualisée, facteur de fatigabilité initial conservé ;
3. contribution et facteur physiologique actualisés, noyau exploratoire fourni.

La variante 2 est la première candidate prudente ; la variante 3 n'est pas
automatiquement meilleure parce qu'elle exploite plus d'état.

## Donnée manquante pour un branchement scientifique au RHO

Le callback actuel transmet surtout A/A_rest. Cela ne suffit pas à estimer
la force encore disponible. L'enveloppe de force doit tenir compte des états
Cn, F, A, tau1, Km, des limites de PW, de la fréquence et de l'historique de
stimulation. Une enveloppe simplement proportionnelle à A serait une
approximation à mesurer explicitement.

En isocinétique, les profils de longueur, de vitesse musculaire et de bras
de levier peuvent être mis en cache, puisque la cinématique est prescrite.
La prochaine brique est un court replay Ding à PW bornées depuis l'état
courant, avec deux ou trois niveaux de recrutement. Il faut comparer sa force
disponible à DOP853 aux checkpoints avant de l'utiliser pour les poids. Ce
calcul ne cherche aucune séquence optimale et n'utilise aucune donnée FHO/BO.
Un replay à PW maximal partout ne suffit pas non plus à certifier les moments
réalisables, puisqu'il ne choisit pas les phases et peut violer les bornes
de couple. Le score doit rester étiqueté comme approximation.

La première partie de cette brique est maintenant disponible dans
`build_isokinetic_max_pw_envelope`. Depuis une frontière certifiée complète,
elle rejoue les cinq états Ding phase par phase avec la PW maximale autorisée,
puis produit `available_positive_power` avec les bras de levier et la vitesse
isocinétique du cycle. `reference_force` reste la force du cycle certifié et
le travail demandé est reconstruit à partir de ses moments, jamais à partir du
contre-factuel PW-max. Elle refuse explicitement une entrée réduite à A seul ou
un cycle sans travail positif. Le pas Ding utilisé est le même que celui déjà
comparé à DOP853 pour le propagateur de PW ; le test de cette passerelle
vérifie en plus la conservation des unités de travail et le point terminal de
la propagation maximale.

Cette enveloppe ne résout toujours pas les contraintes de moment couplées :
elle n'est donc pas branchée silencieusement dans la boucle clinique ni
présentée comme une preuve de viabilité. Le raccordement au RHO devra extraire
une archive isocinétique certifiée, fournir les gains de géométrie par phase,
et conserver dans le journal les entrées de l'enveloppe avant toute mise à
jour du coût.

## Normalisation, cadence et audit

Le score brut est conservé et exposé en normalisation par le maximum. Pour
préserver l'échelle globale du coût RHO, le vecteur envoyé au solveur aurait
une moyenne géométrique égale à un, après projection bornée dans [0,25 ; 4].
Cette projection peut modifier les ratios : le journal l'indique explicitement.
La normalisation GM elle-même ne modifie pas les ratios. Le lissage logarithmique
est de 0,2, le changement maximal par mise à jour de ×1,1 et la zone morte de
1 %. Les cadences 10 et 20 sont comparées ; aucun changement intermédiaire.

Chaque calcul conserve les capacités d'entrée, les profils utilisés, les
facteurs physiologiques et mécaniques, le score brut, la cible projetée et le
poids proposé. Si le cycle source n'est pas certifié, si le challenge sort du
domaine, ou si un score est nul/invalide, le noyau conserve les poids précédents.
Une proposition ne signifie pas que le coût OCP a été modifié : le futur
branchement devra utiliser `ParametricFatigueWeightBinding` et vérifier son
reçu d'application, sans recompiler le NLP.

## Validation et ordre de travail

Les tests fournis comparent l'équation lente exacte à DOP853 à plusieurs
durées de phase, vérifient le retour au cas repos, la récupération nette avec
coût de stimulation positif, les permutations des muscles, la conservation
du crédit mécanique total, le cas de muscles redondants et de muscles inutiles,
les limites de variation et les cadences 10/20.

Le script `scripts/audit_physio_update.py` prépare un pilote synthétique de
40 cycles de mesures, avec 30 phases et quatre muscles, aux deux cadences.
Il mesure séparément le noyau et la nouvelle enveloppe Ding cinq états à PW
maximale. Sur l'audit du 25 septembre 2026, l'enveloppe plus proposition est
à 53 ms de médiane et 80 ms au p95 (100 répétitions synthétiques, 30 phases,
quatre muscles) ; c'est compatible avec une mise à jour toutes les 10 ou 20
secondes, mais ne comprend ni extraction de solution ni OCP. Il ne résout
aucun RHO et ne valide aucune amélioration physiologique. Reproduction :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/audit_physio_update.py --output /tmp/physio-update-audit.json
```

Après validation de l'enveloppe et du branchement paramétrique :

1. Utiliser des états isocinétiques certifiés au repos, à fatigue modérée et
   proche de l'arrêt. Comparer les propositions aux poids fixes sur des branches
   RHO réelles de 1/5/20 cycles, avec même état initial et historique.
2. Vérifier le travail, les bornes de couple, le replay DOP853, les PW et la
   réserve de force. Conserver les coûts de calcul complets, pas seulement IPOPT.
3. Comparer unitaire, physio fixe isocinétique, update-10 et update-20 jusqu'à
   un arrêt confirmé avec la sonde de viabilité ; seuil de 3000 cycles explicite.
   Reporter aussi les arrêts numériques et les essais censurés.
4. Comparer les trois ablations du score avant de chercher une amplitude plus
   agressive. L'endurance observée, pas la fatigue AUC locale seule, tranche.
5. Transférer ensuite au cas à résistance constante avec une criticité de
   moment par phase adaptée. Ne pas appliquer tel quel le score de travail
   isocinétique : la vitesse doit alors être maintenue par les muscles.

La cible de latence clinique demeure moins d'une seconde en moyenne par cycle,
sur la chaîne complète. Le coût d'un noyau NumPy ne mesure ni la construction
des enveloppes ni le surcoût éventuel de convergence dû aux changements de poids.
