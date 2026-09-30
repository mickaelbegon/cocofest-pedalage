# PACE : apprendre les cycles restants d'une vraie politique RHO

Proposition Astra du 25 septembre 2026. Ce document est une proposition
d'expérience ; aucune nouvelle campagne ni modification de politique n'est
annoncée comme exécutée.

## Ce que les résultats permettent de conclure

À 0,22 Nm, le meilleur poids fixe exploré par le BO valide 579 cycles sur le
modèle nominal, contre 481 pour les poids unitaires, 489 pour Physio, 485 pour
PACE historique et 482–483 pour les deux pilotes QP-Ding corrigés. Les 20 essais
BO ne prouvent pas que 579 soit l'optimum global. Leur terminaison est encore
qualifiée par le proxy capacité/saturation, et non par une preuve
d'impossibilité physiologique.

Le défaut décisif de PACE est le remplacement de la politique RHO par une
allocation QP qui fait évoluer les muscles sous une cinématique imposée. Une
intégration musculaire précise ne suffit pas à prédire les décisions du RHO.
Les corrections de normalisation et de régularisation ont résolu des problèmes
numériques réels, sans démontrer un gain d'endurance substantiel.

Il faut aussi corriger l'interprétation du test de fidélité :

- `validate_pace_decision_fidelity.py` classe la fatigue AUC sur 1/5/20 cycles.
  Ce classement mesure bien la fidélité à ce coût court. Il ne classe pas
  directement l'endurance restante, qui est notre cible.
- Au checkpoint 300, H20, `epsilon_1em05` donne 20,5518906896, les poids PACE
  courants 20,5518906174 et la proposition historique exactement la même AUC
  que les poids courants. L'écart de 7,2e-8 entre epsilon et courant est un
  ex æquo numérique pratique, pas une différence scientifique entre rangs
  3, 4 et 5. L'écart avec les poids unitaires (20,4768986611) et BO
  (20,4695182926) est en revanche nettement plus grand.
- À ce même horizon, BO a une capacité minimale de 0,515465 contre 0,516311
  pour les poids unitaires et 0,519218 pour PACE. Optimiser systématiquement
  la capacité minimale peut donc rejeter BO, malgré son meilleur résultat
  depuis l'état initial. Ce constat illustre un conflit de métriques ; il ne
  démontre pas encore que BO gagne depuis ce checkpoint particulier.

Sources locales :
`pace-regularization-audit-20260925b/continuations/summary.json` et
`four-model-r022-1500-dynamics-20260923/bo-fixed-weights-nominal-v4-seed-aligned/summary.json`.

## Trois directions différentes

| Direction | Données et décision | Intérêt | Risque principal |
|---|---|---|---|
| Amélioration d'une politique de référence par sa valeur restante | Vraies branches RHO, puis continuation avec les meilleurs poids fixes ; prédiction du gain de cycles par action | Cible exactement l'endurance opérationnelle et conserve la mécanique RHO | Coût de collecte hors ligne ; erreur du modèle de valeur |
| BO hors ligne d'une règle adaptative très courte | Optimiser une règle à un ou deux changements de poids, déclenchés par l'état de fatigue, sur des simulations RHO complètes | Première preuve simple qu'une politique variable fait mieux que des poids fixes ; très faible coût en ligne | Surapprentissage du modèle nominal ; beaucoup de simulations si trop de paramètres |
| Modèle appris de transition d'un cycle RHO | Apprendre l'état suivant en fonction de l'état, des poids et des paramètres ; projeter 100–300 cycles avec cette carte | Conserve implicitement les changements de recrutement et de mécanique du contrôleur réellement utilisé | Accumulation d'erreur près des changements d'ensemble actif et de l'arrêt |

Le BO contextuel en ligne est une variante de la deuxième direction, mais
l'endurance fournit une récompense très retardée et l'état change entre deux
évaluations. Il ne faut donc pas essayer des poids sur le patient comme dans
un bandit ordinaire. Les explorations se font dans le modèle identifié ; en
ligne, une règle déjà validée sélectionne ou conserve les poids.

La première direction est recommandée. Elle commence par un test entièrement
en RHO réel, sans approximation apprise, pour vérifier qu'une adaptation peut
dépasser la bonne politique fixe. Si ce test ne trouve aucun avantage,
construire un modèle de valeur complexe est prématuré.

## Première expérience : intervention brève, puis retour à BO

Nom de travail : **PACE-V**, pour valeur d'endurance restante. La valeur ne
vient ni d'une FHO ni du QP compact précédent.

1. Rejouer le meilleur BO fixe depuis la même graine et archiver les états
   complets aux cycles 180, 300 et 420. Vérifier la répétabilité du résultat
   et des reprises avant de comparer les interventions.
2. Depuis chaque checkpoint, évaluer quatre branches : maintenir les poids
   BO, et trois changements prédéclarés. Les changements proposés sont une
   interpolation logarithmique à mi-chemin entre BO et unitaire, une
   redistribution Biceps–Triceps, et une redistribution entre deltoïdes.
   Projeter les propositions sur le même domaine de poids admissibles,
   archiver les poids effectivement appliqués et supprimer les doublons.
3. Appliquer chaque changement pendant 20 cycles, puis rétablir les poids BO
   pour toute la suite. La branche de référence reste BO pendant toute la
   continuation. Les branches partent du même état complet, du même
   historique PW et du même primal de départ.
4. Aller jusqu'au même critère d'arrêt avec un plafond de sécurité commun.
   Compter le nombre de cycles valides restants, y compris les 20 premiers.
   Rapporter séparément fatigue AUC, état terminal, saturation et temps.
5. Examiner les arrêts depuis le dernier état certifié. Une reprise réussie
   réfute cet arrêt comme limite de tâche ; l'échec d'un solveur local ne
   prouve toujours pas l'infaisabilité. Conserver explicitement les issues
   indéterminées et les trajectoires censurées.

Cela donne neuf branches challengers et trois contrôles de reprise. À titre
d'ordre de grandeur, si les branches terminent vers le cycle 579, les queues
de référence représentent 399 + 279 + 159 = 837 cycles ; quatre branches par
état représentent environ 3 348 cycles calculés. À une seconde par cycle,
cela représente environ une heure CPU, divisible entre trois workers, plus
initialisation et reprises. C'est une estimation de budget, pas un benchmark.

Le résultat utile est un avantage apparié en cycles restants. Une différence
de fatigue courte ne constitue pas le critère de sélection. Un seuil de gain
pratique, par exemple 10 cycles, doit être fixé avant les expériences et
comparé à la variabilité numérique des reprises. Si aucun challenger
n'améliore BO, élargir une fois l'espace des interventions ou leur durée avec
un budget prédéclaré ; ne pas déclarer une amélioration grâce à un autre proxy.

## Comment rendre ce principe économique en clinique

La simulation complète de chaque futur sert à collecter les données hors
ligne. Ensuite, on apprend la valeur de continuer avec la politique BO, ou
directement l'avantage d'un changement de poids tenu pendant K cycles puis
suivi de BO. La prédiction rend des **cycles supplémentaires attendus**, avec
une incertitude calibrée. Elle n'apprend pas la somme des fatigues futures.

Le premier modèle peut être une régression régularisée ou un petit processus
gaussien ; le choix dépendra de la quantité de données et d'une validation
groupée. Un réseau profond n'est pas justifié par les 20 trajectoires BO et
quelques continuations actuelles. Apprendre directement les différences
appariées entre candidat et référence peut être plus facile que prédire la
durée restante absolue.

Les features candidates comprennent :

- les quatre capacités A/Arest et les écarts indépendants de Tau1 et Km,
  conservés selon la réduction exacte de Ding déjà documentée ;
- les quatre forces et le calcium à une phase de cycle définie ;
- l'état mécanique réduit, les écarts de fermeture et de vitesse ;
- le dernier profil PW, ou une représentation validée de ce profil, ses
  saturations et les tendances de fatigue sur quelques cycles ;
- les poids actuels, les paramètres musculaires identifiés, la résistance,
  la cadence et la fréquence de stimulation.

Le checkpoint complet reste la source pour les branches physiques. Une
réduction de features ne doit pas effacer une mémoire PW ou mécanique qui
change la réponse. Le simple numéro du cycle n'est pas un état suffisant.

Les actions restent les trois coordonnées de log-poids centrés, à moyenne
géométrique 1, dans le domaine actuel [0,25 ; 4]. On démarre avec quelques
actions discrètes autour de BO ; une optimisation continue viendra seulement
si ce premier espace montre un avantage reproductible. La durée K de l'action
fait partie de sa définition et des données d'entraînement.

En fonctionnement, un superviseur lent peut évaluer quelques vraies branches
RHO en arrière-plan et enrichir le modèle. Des workers persistants réutilisent
le NLP compilé de même structure, avec leurs propres paramètres et warm starts.
Il faut benchmarker compilation, copie des checkpoints et concurrence CPU,
pas seulement le temps solveur chaud.

**Les workers mettent à jour un modèle ; au moment d'appliquer des poids,
le classement est recalculé à partir de l'état vivant actuel.** Appliquer
directement le gagnant d'une branche issue d'un état vieux de dix cycles
introduirait un nouveau décalage. Une simple limite d'âge ne remplace pas
cette vérification. L'évaluation d'un petit modèle sur quelques candidats
devrait être peu coûteuse, mais le budget doit être mesuré.

L'objectif temporel reste le cycle complet : résolution, certification,
callback et transport. Un modèle lent hors délai ne doit pas bloquer le RHO.
L'inférence et les mises à jour de poids se font hors dérivées IPOPT ; la
dimension du NLP rapide reste inchangée. Introduire ensuite une valeur
terminale différentiable dans le NLP est une expérience distincte, avec son
propre benchmark de convergence.

## Garde et validation

Le candidat de référence est toujours présent. Une alternative est admise
uniquement si l'avantage prédit, diminué de son erreur empirique calibrée,
dépasse un seuil pratique ; sinon on conserve la politique de référence.
Cette règle est une précaution empirique et non une garantie mathématique
de sûreté pour le pédalage non convexe.

Les contraintes mécaniques et de stimulation restent vérifiées par le RHO.
Le garde ne doit plus imposer simultanément « aucune hausse de fatigue
totale » et « aucune baisse de la capacité minimale » : ces deux proxies
peuvent exclure les échanges nécessaires à une meilleure endurance.

La validation doit suivre quatre niveaux :

1. **Oracle d'intervention** : gain restant observé contre BO depuis un état
   identique ; répétabilité des reprises et classification des arrêts.
2. **Prédiction hors échantillon** : erreur en cycles, classement des paires
   réellement différentes, regret et couverture des intervalles. Grouper
   les données par trajectoire et variante ; ne pas répartir aléatoirement
   les cycles voisins d'une même trajectoire entre train et test.
3. **Boucle fermée nominale** : comparaison PACE-V, BO fixe, unitaire et PACE
   historique, mêmes graines/solveurs/résistance et même protocole d'arrêt.
4. **Généralisation** : autres variantes Ding et résistances tenues à l'écart,
   perturbations d'identification, mesure du temps total et des délais
   manqués. Un résultat nominal n'est pas une validation clinique.

Les plafonds atteints fournissent une borne inférieure sur l'endurance.
Un échec numérique non résolu ne fournit pas un temps de fatigue exact.
Ne pas entraîner une régression naïve en attribuant arbitrairement ces
deux issues à un nombre de cycles terminal certain.

## Liens conceptuels

L'amélioration d'une politique par des rollouts suivis de la politique de
référence et une valeur terminale relève de la programmation dynamique :
[Bertsekas, MPC et apprentissage par renforcement](https://web.mit.edu/dimitrib/www/Bertsekas_NMPC_IFAC.pdf).
Les garanties classiques demandent une valeur et une dynamique appropriées ;
elles ne se transfèrent pas automatiquement à notre approximation.

L'apprentissage de coûts terminaux à partir de trajectoires réellement
exécutées est étudié dans
[Rosolia et Borrelli, Learning MPC](https://arxiv.org/abs/1609.01387).
L'idée de conserver une politique de référence dans les régions incertaines
est reliée à
[Laroche et al., Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf).
Ce sont des inspirations méthodologiques ; leurs hypothèses de garantie ne
sont pas démontrées pour le présent modèle FES.
