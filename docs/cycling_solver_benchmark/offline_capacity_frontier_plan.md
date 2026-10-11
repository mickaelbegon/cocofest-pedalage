# Frontière hors ligne de capacité d'endurance

## Statut

Ce document décrit une expérience **hors ligne**. Aucun résultat de cette
frontière ne doit encore modifier le RHO clinique. Le premier code associé,
`cocofest.optimization.capacity_frontier`, ne résout aucun OCP : il scelle un
plan de directions de support et impose les règles de provenance et de
certification des observations. `ding_slow_manifold` représente la réduction
exacte des trois états lents de Ding utilisée par le futur atlas.

Il y a deux objets à ne pas confondre. L'objet principal est une **frontière
locale de faisabilité d'un cycle** : pour un état rapide/historique conditionné,
quelles capacités initiales minimales permettent encore le cycle exact ?
L'autre est un Pareto descriptif des continuations RHO jusqu'à l'arrêt. Ce
second objet est intéressant pour sélectionner des politiques, mais ne remplace
pas l'atlas de faisabilité. Ni l'un ni l'autre n'est un FHO.

## Correction de l'objectif proposé

L'objectif brut

\[
\max_{PW,A}\sum_m c_m(1-A_m/A_{m,rest})
\]

récompense directement la fatigue. Il est donc utile au plus comme *stress
test*, mais pas comme méthode pour construire une stratégie d'endurance : il
peut préférer épuiser un muscle même lorsqu'une solution plus durable existe.

La campagne de continuations proposée fait de l'endurance la valeur principale :

1. à tâche, modèle, conditions initiales et contraintes identiques, poursuivre
   le RHO sous une politique candidate jusqu'au dernier cycle certifié ;
2. à ce point, figer les états musculaires, mécaniques et l'historique de PW,
   puis certifier séparément qu'un cycle admissible est ou n'est pas possible ;
3. parmi les politiques dont les durées diffèrent d'au plus la tolérance
   déclarée, comparer secondairement la capacité terminale
   `c · (A / A_rest)`, avec `c >= 0`.

Une direction `c` privilégie donc une capacité **qui reste**, et non une
fatigue à produire. Le front retenu est le sous-ensemble non dominé de
`(cycles_certifiés, A_1/A_1,rest, ..., A_M/A_M,rest)`. Un point domine un autre
s'il a au moins autant de cycles et au moins autant de capacité pour chaque
muscle, avec une amélioration stricte dans au moins une composante.

Un arrêt IPOPT isolé est `indéterminé`, pas une limite du front. Une campagne
qui atteint son plafond de cycles est censurée à droite, pas une preuve de
capacité infinie ni de frontière atteinte.

## Atlas principal : capacités initiales qui gardent un cycle faisable

Pour une tâche strictement figée et des états rapides/historique `h` préparés,
l'ensemble recherché est `C(T,h)`: les vecteurs de capacité initiale pour
lesquels il existe un PW rendant le cycle complet admissible. On cherche ses
points minimaux au sens composante par composante. Pour chaque direction
positive `w`, le solve de support est donc : minimiser
`w · (A(0)/A_rest)` sous les contraintes effectives du cycle, et non maximiser
la fatigue produite à la fin du cycle. Les directions epsilon-contrainte
compléteront ensuite les branches non supportées possibles d'un front non
convexe.

Les contraintes conservent la fermeture géométrique modulo `2π`, cadence,
travail net, bornes de couple/assistance, PW/slew et la garde du demi-cycle
suivant. `A`, `Tau1` et `Km` sont des conditions initiales communes à tous les
nœuds du front ; ils ne sont pas forcés à être périodiques en fin de cycle.
Imposer cette périodicité définirait un régime soutenable distinct.

`Cn`, `F` et l'historique complet de PW sont conditionnés par un reçu exact ;
ils ne peuvent pas être laissés libres, sinon le solveur pourrait bénéficier
d'une force initiale non préparée. Plusieurs atlas pourront être calculés pour
plusieurs états rapides représentatifs.

## État lent cohérent avec Ding

Dans `ding2007_with_fatigue`, les états lents `A`, `Tau1` et `Km` ont le même
forçage de force et la même relaxation. En l'absence de reset ou de clipping,
depuis le repos ils restent sur une variété de dimension un par muscle :

\[
q_m=(A_m-A_{m,rest})/\alpha_A
    =(Tau1_m-Tau1_{m,rest})/\alpha_{Tau1}
    =(Km_m-Km_{m,rest})/\alpha_{Km}.
\]

La première frontière ne doit donc pas perturber `A` seul en gardant `Tau1` et
`Km` fixes : cet état est en général non atteignable. La capacité déclarée
reste `A/A_rest` pour être lisible, mais le contexte/certificat conserve les
trois états, `Cn`, `F` et l'historique de stimulation. Le code vérifie déjà
l'aller-retour et l'intersection des bornes sur cette variété. La prochaine
étape P0 est de vérifier l'invariant numériquement dans les archives et replays
réellement employés ; hors variété, l'atlas conservera les trois états lents.

**Contrôle P0 réalisé.** Sur les huit checkpoints gauches c20 à c160 de la
campagne instrumentée, le résidu maximal observé vaut `2.013e-8` en coordonnée
`q` (c160). Il augmente progressivement avec les cycles et est compatible avec
la transcription/intégration numérique ; il ne justifie pas de prétendre à une
égalité bit à bit. La tolérance de précondition de l'adaptateur est donc
`5e-8`, soit plus de deux fois ce maximum observé. Elle protège seulement le
passage à l'atlas : les contraintes OCP et les replays gardent leurs propres
tolérances bien plus strictes. Toute nouvelle campagne doit republier ce même
audit et est rejetée si elle dépasse sa tolérance déclarée.

`SlowManifoldInitialStateAdapter` réalise déjà la partie sûre de P0 : depuis
un checkpoint et un programme préparé identiques, il dérive les indices
symboliques de `A`, `Tau1`, `Km`, vérifie le contexte, les paramètres et
l'appartenance de l'état source à la variété, puis produit des copies de
`lbx`, `ubx` et `x0` où ces trois états initiaux co-varient. Il ne construit
pas encore le solveur de support, ne certifie aucune faisabilité et ne modifie
pas le RHO.

**Smoke P1 réel.** Au checkpoint gauche c140, l'adaptateur a restauré les
`4 288` variables et `4 171` contraintes originales, puis le problème de
support uniforme a ajouté `8` égalités de variété lente. Sur la boîte de
capacité `[0.2, 1]`, IPOPT/MA57 converge en 40 itérations (`1.11 s` de solve,
`5.52 s` de construction locale, après `6.92 s` pour le solve nominal qui
matérialise le graphe). Le défaut discret maximal original est
`5.93e-11` et celui de variété `8.33e-17`. Le candidat atteint les ratios
`[0.200002, 0.200002, 0.200004, 0.613227]` pour
`[Delt_ant, Delt_post, Biceps, Triceps]`. C'est un **smoke auxiliaire**, pas
un point de front : il faut reconstruire le NLP de manière indépendante,
répéter avec d'autres seeds et faire le replay DOP853 avant toute conclusion.

**Validation fraîche du même smoke.** Une seconde reconstruction depuis le
reçu, avec les capacités candidates fixées, converge en 24 itérations et passe
l'audit discret complet (`5.05e-7` maximal, tolérance `1e-6`). Le replay
DOP853 indépendant est disponible : travail `6.0280765 J` contre la cible
`6.0242998 J`, soit une erreur relative `6.27e-4` (0,063 %) ; son erreur
d'extrémité maximale normalisée est `7.14e-4`. Ces chiffres sont rapportés,
pas automatiquement acceptés : le seuil DOP853 du front doit être fixé avant
la grille des directions et appliquer la même règle à chaque candidat.

**Calibration DOP853.** Le même replay du solve nominal reconstruit donne une
erreur relative de travail `6.28e-4` et une erreur d'extrémité normalisée
`7.18e-4`; le candidat uniforme n'est donc pas moins fidèle que la référence
Radau-5. Le pilote pré-enregistre des seuils conservateurs égaux à `1.5×` ces
valeurs, soit `9.5e-4` pour le travail et `1.1e-3` pour l'état. Ces seuils sont
des gates de fidélité de transcription, non des critères de faisabilité ou de
supériorité physiologique.

**Campagne locale c140 terminée.** Quinze directions de support (uniforme,
axes, paires et quatre directions intérieures) ainsi que neuf coupes
epsilon-contrainte valides ont passé la reconstruction fraîche, l'audit
discret et les mêmes gates DOP853. Les graphiques produits dans le dossier de
campagne représentent les projections `Delt_ant/Triceps`,
`Delt_post/Triceps`, `Biceps/Triceps`, les profils en coordonnées parallèles
et les trois coupes epsilon. Ils doivent être lus comme des **témoins locaux**
de faisabilité à c140, non comme une frontière de Pareto globale.

Le motif robuste à cette étape est que le minimum local du triceps requiert de
garder les trois autres muscles presque au repos : la direction triceps seule
donne `[0.997988, 0.999983, 0.963987, 0.318884]` sur la boîte `[0.2, 1]`.
Les premières coupes epsilon sur le deltoïde postérieur et le biceps avaient
été mal paramétrées (une borne nulle involontaire sur le deltoïde antérieur).
Elles sont exclues du résumé et remplacées par les neuf coupes vérifiant leur
propre contrainte epsilon à `<= 1e-6`.

**Sensibilité à la borne basse.** Trois directions c140 ont été répétées sur
`[0.1, 1]`, avec reconstruction fraîche et replay DOP853. Le support triceps
reste inchangé à la précision pratique (`Triceps=0.318847`), ce qui rend le
minimum local du triceps robuste à ce test. En revanche le support uniforme
atteint maintenant `[0.100002, 0.100002, 0.100007, 0.689642]` au lieu de
heurter `0.2` pour les trois premiers muscles : cette branche était tronquée
par la boîte initiale. Il est donc interdit de conclure que `0.2` est une
réserve physiologique limite ; les prochains atlas devront documenter une
analyse de sensibilité de domaine avant toute interpolation.

**Réplication de conditionnement rapide.** Les deux directions informatives de
cette sensibilité (`support uniforme` et `minimum triceps`) ont été répétées
sur les checkpoints gauches c120, c140 et c160, tous avec la boîte `[0.1, 1]`
et les mêmes vérifications fraîches/DOP853. La tendance ci-dessous n'est pas
un suivi physiologique direct des capacités effectivement propagées : chaque
ligne décrit la capacité *initiale minimale hypothétique* compatible avec le
cycle lorsque les états rapides et l'historique du checkpoint sont gelés.

| Checkpoint | Support uniforme : `[Delt_ant, Delt_post, Biceps, Triceps]` | Minimum triceps : même ordre |
| --- | --- | --- |
| c120 | `[.100001, .100001, .100003, .679963]` | `[.997947, .999981, .960008, .316812]` |
| c140 | `[.100002, .100002, .100007, .689642]` | `[.999107, .999994, .965216, .318847]` |
| c160 | `[.100002, .100002, .100005, .698408]` | `[.998061, .999983, .967007, .321914]` |

Les trois replays restent sous `6.63e-4` d'erreur relative de travail et
`7.11e-4` d'erreur d'extrémité normalisée. La branche uniforme demeure donc
tronquée sur les trois états rapides; à l'inverse, le minimum local de triceps
reste entre `.3168` et `.3219` lorsque les autres muscles sont proches du
repos. C'est une réplication qualitative utile, mais il faut encore sonder
des directions intermédiaires et des branches epsilon à c120/c160 avant de
construire une interpolation multi-checkpoints.

**Pentes intermédiaires epsilon.** Pour commencer à distinguer une pente locale
d'une simple comparaison d'extrêmes, le triceps a ensuite été minimisé sous
une borne `0.6` appliquée séparément à chacun des trois autres muscles, à c120
et c160. Les six points ont une violation epsilon nulle, passent leur audit
frais et leur replay DOP853. Les capacités de triceps obtenues sont :

| Borne active à 0.6 | c120 | c140 | c160 |
| --- | ---: | ---: | ---: |
| Delt_ant | .344220 | .346462 | .349562 |
| Delt_post | .342633 | .344373 | .347562 |
| Biceps | .364945 | .368522 | .372589 |

La hiérarchie est stable : imposer une réserve au biceps coûte davantage de
réserve de triceps que l'imposer à un deltoïde, et le coût augmente légèrement
dans cet intervalle de checkpoints. Ce résultat reste **conditionnel aux états
rapides figés** ; il motive une interpolation conservative par contexte, pas
une unique courbe utilisable hors domaine.

**Audit d'interpolation c120--c140--c160.** Les quatre reçus bruts de c120 et
c160 ont été archivés avec leurs hashes; les témoins c140 à `epsilon=0.6`, dont
les reçus bruts n'étaient plus présents, ont été reconstruits uniquement depuis
le résumé immuable qui conserve les hashes, les vecteurs de support et les gates
numériques. Une interpolation linéaire n'utilise que c120 et c160; c140 est un
vrai holdout. Les erreurs relatives des pentes d'échange observées à c140 sont
`0.18 %` (Delt_ant), `0.96 %` (Delt_post) et `0.35 %` (Biceps), sous le gate
pré-déclaré de 10 %. La sortie conserve pour chaque pente l'intervalle
conservatif qui absorbe l'erreur de holdout. Elle reste une hypothèse hors
ligne : l'artefact l'étiquette explicitement
`forbidden_pending_causal_RHO_holdout`.

Pour une première ablation très bornée, ces bornes prudentes sont transformées
en poids de fatigue normalisés à moyenne un, inversement proportionnels à la
pente d'échange, puis le triceps est amplifié. Dans l'ordre
`[Delt_ant, Delt_post, Biceps, Triceps]`, cela donne
`[.8403, .9104, .4285, 1.8208]`. C'est une **proposition diagnostique**, non
une loi physiologique ni une contrainte : une grande pente du biceps signifie
seulement que, dans cet état rapide gelé et pour cette coupe, relâcher le
biceps achète localement davantage de réserve de triceps.

Les branches appariées `unit` et `frontier-diagnostic` depuis l'ancre gauche
c140 ont chacune certifié leurs trois cycles courts. Toutes ont conservé le
travail nominal (résidu de travail nul), ont passé l'audit du NLP complet sous
`1e-5`, ont été restaurées dans un processus frais, et ont réutilisé le même
solveur compilé au sein de chaque branche. Les résidus maximums observés sont
`9.10e-9` (unitaire) et `3.04e-8` (frontière, premier cycle). Cela valide le
câblage paramétrique et la comparabilité locale, mais ne démontre pas encore
un gain d'endurance. La continuation longue doit partir des endpoints certifiés
et s'arrêter uniquement sur un test de faisabilité gelée avec redémarrages
alternatifs; un simple arrêt IPOPT demeure indéterminé.

**Courbure des échanges epsilon.** Les douze coupes complémentaires
(`epsilon=.4` et `.8`, trois muscles, c120/c160) ont toutes terminé avec une
reconstruction fraîche et un replay DOP853, et les six courbes respectent la
monotonie attendue : relâcher la borne epsilon ne demande jamais davantage de
réserve de triceps. Cependant, elles ne sont pas affines. Par exemple, à c120
la capacité minimale du triceps sous borne biceps vaut `.417984`, `.364945`,
`.327751` pour epsilon `.4`, `.6`, `.8`; les valeurs correspondantes à c160
sont `.426971`, `.372589`, `.334012`. Le biceps demeure le meilleur échange
vers du triceps, mais son ratio sécant dépend fortement de l'intervalle. Les
deltoïdes ont une courbure plus modérée. Cette réplication **confirme la
hiérarchie qualitative**, mais interdit de présenter l'inverse de la seule
pente epsilon `.6` comme une loi de poids générale.

La continuation longue est donc volontairement gelée après le test de câblage
de trois cycles. La prochaine gate est une règle de poids issue de pentes
locales/intervalle conservatif, validée à c140 comme holdout, puis seulement
une continuation appariée jusqu'au test de faisabilité gelée. Les courbes
actuelles sont un atlas conditionnel à l'état rapide et à l'historique de PW,
pas une table transférable à un autre sujet ou à une autre phase du cycle.

**Pentes locales, vrai holdout c140.** Les neuf coupes de c140 ont ensuite été
recalculées dans la même boîte `[.1, 1]`, y compris epsilon `.6`, plutôt que
d'hériter des reçus historiques. Pour chaque muscle, deux sécantes adjacentes
`.4→.6` et `.6→.8` encadrent la pente locale; le minimum de cet intervalle est
la valeur prudente. L'interpolation c120/c160 prédit alors à c140 des
intervalles qui recouvrent les trois observations. Les erreurs relatives des
milieux sont `0.24 %` (Delt_ant), `0.46 %` (Delt_post) et `1.14 %` (Biceps).
Les poids diagnostiques correspondants deviennent
`[.8123, .9508, .3353, 1.9016]`. Le biceps reçoit donc une pénalité moindre,
le triceps la plus forte, mais sur la base d'une sensibilité **locale et
conservatrice** explicitement validée.

Cette seconde politique passe aussi les branches appariées c140 de trois
cycles : contrôle unitaire `<=9.10e-9`, politique locale `<=8.23e-9`, travail
nominal exact, solveur compilé réutilisé et vérification fraîche des endpoints.
Elle franchit le gate de comparabilité locale. Le prochain essai est désormais
justifié : une continuation fixe appariée jusqu'au plafond déclaré, puis le
test gelé de faisabilité multi-départs au dernier cycle certifié. Cela mesurera
l'endurance; ce sera le premier test qui puisse invalider ou soutenir un
bénéfice de la règle locale.

**Continuation fixe initiale.** Depuis les endpoints appariés c143, la
politique unitaire a certifié jusqu'à c169 et la politique à pentes locales
jusqu'à c171 avant un arrêt IPOPT. Les deux endpoints ont été restaurés dans
un processus frais et une résolution nominale suivante échoue à nouveau
(violations normalisées respectivement `5.30e-3` et `7.60e-3`). Le signal
préliminaire est donc `+2` cycles certifiés pour la politique locale sur cette
trajectoire. Il ne doit pas encore être nommé endurance ou fatigue maximale :
la résolution fraîche infructueuse est une réplication numérique utile, non le
test gelé d'infaisabilité. Les probes explicites à objectif nul ont été lancés
sur les deux archives; seule leur combinaison éventuelle avec un témoin
contre-factuel de repos de fatigue pourra justifier une attribution locale à la
fatigue.

## Campagne de continuations complémentaire

Pour quatre muscles, le plan générera 15 directions positives déterministes :
une uniforme, quatre directions individuelles, six paires et quatre points
internes du simplexe. Les directions négatives sont exclues : elles ne
représentent pas une préférence de conservation de capacité.

Chaque direction devient une politique candidate **hors ligne** (poids de
fatigue initialisés à partir de la direction, avec bornes et normalisation
explicitement archivées). Toutes les autres conditions restent strictement
identiques : résistance, tâche isocinétique, solveur IPOPT/MA57, Radau-5,
fréquence, discretisation, limites PW/slew, critère d'arrêt et politique de
reprise.

Pour chaque candidat :

1. vérifier la reprise exacte à froid ;
2. exécuter les RHO jusqu'au premier endpoint ou au plafond, en sauvegardant
   chaque frontière certifiée ;
3. depuis le dernier état certifié, exécuter le test de faisabilité gelée avec
   départs alternatifs ;
4. rejouer les points publiés avec DOP853 ;
5. extraire les capacités et appliquer le filtre de dominance seulement entre
   points de même `context_sha256`.

Le premier résultat acceptable est descriptif : il doit montrer au moins deux
points non dominés, reproductibles, sur une tâche donnée. Il ne démontre pas
encore qu'un poids dépendant de l'état augmente l'endurance, ni qu'il constitue
la frontière de faisabilité locale.

## Passage éventuel au RHO en ligne

Seulement après un holdout : interpoler localement les pentes du front dans les
coordonnées lentes atteignables, produire des poids/une cible comme **paramètre
numérique** du graphe RHO précompilé, puis vérifier l'effet par re-solves
appariés. Le RHO ne doit ni résoudre le front ni attendre son calcul. Hors du
domaine d'interpolation, il conserve la politique de référence et journalise
le refus.

Le contrat `TaskReserveObjectiveBinding` et le canal paramétrique de poids de
fatigue sont les points de raccordement pertinents. Toute priorité de réserve
doit d'abord être testée hors ligne, idéalement de manière lexicographique
(valeur de continuation puis fatigue comme départage), car un simple grand
poids numérique ne garantit pas la priorité scientifique voulue.

## Commande de planification

Cette commande ne lance aucune simulation et ne modifie pas le RHO :

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
"$PY" scripts/plan_capacity_frontier.py \
  --context-sha256 <sha256-du-contexte-physique-scelle> \
  --fast-condition-sha256 <sha256-des-etats-rapides-et-historique-scelles> \
  --muscle Biceps --muscle Triceps --muscle DeltoidAnterior --muscle DeltoidPosterior \
  --random-directions 4 --seed 19 \
  --output /tmp/capacity-frontier-cold/plan.json
```

Le contexte et le conditionnement rapide ne peuvent pas être inventés : ils
doivent provenir du manifeste/reçu de la campagne exacte à comparer. La
commande crée le plan d'atlas de faisabilité ; elle ne lance pas de solve. Son
exécuteur sera ajouté après le test P0 d'invariant et après que les OCP de
support/epsilon-contrainte et le budget de calcul auront été fixés.
