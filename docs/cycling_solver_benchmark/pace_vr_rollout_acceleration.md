# Accélération et audit numérique de PACE-VR — 29 septembre 2026

Les optimisations du cœur du rollout sont implémentées. Elles conservent les
huit sous-pas exponentiels, la formule analytique du calcium, les équations de
fatigue et toutes les vérifications du replay. Le processus RHO et sa politique
asynchrone n'ont pas été modifiés. Aucune campagne d'endurance n'a été lancée.

## Modifications retenues

### QP persistant (30 septembre 2026)

Le QP de recrutement PACE-VR est désormais résolu par défaut par une instance
**qpOASES** persistante, accessible via CasADi déjà présent dans
`cocofest-rho32`. La structure du QP est créée une fois par dimension de
contraintes puis seules ses matrices, bornes et son warm-start sont mis à jour.
La résolution stricte conserve l'égalité de travail. Si cette égalité est hors
de la région de confiance au premier pas SQP, une unique formulation avec un
écart de travail normalisé et fortement pénalisé fournit seulement un pas de
progrès; les pas suivants repassent à l'égalité stricte. Le replay physique,
les bornes de PW/force/puissance et la tolérance de travail restent les
critères d'acceptation.

`qp_backend` accepte `auto` (défaut, qpOASES), `casadi_qpoases` et `scipy`.
Le dernier permet une reproduction explicite de l'ancien chemin SLSQP.

HPIPM reste le backend QP approprié pour les sous-problèmes structurés des RHO
acados. Il n'est **pas** activé pour ce QP condensé : le pont CasADi→HPIPM de
l'environnement a produit une segmentation après des appels répétés. Il ne
serait pas sûr de l'exposer à une campagne clinique sans un binding HPIPM
natif validé ou une vraie transcription acados du QP.

Sur le snapshot certifié droit, cycle 1, de la campagne asymétrique à 1,92 Nm,
la configuration qpOASES donne le même préfixe réduit de 86 cycles pour les
cartes séquentielle et condensée; la carte condensée prend respectivement
4,82 s et 1,66 s (gain ×2,91). Le préfixe reste une propriété du rollout
réduit, pas un certificat d'échec physiologique.

1. **Condensation exacte par intervalle de stimulation.** Les coefficients de
   force étant figés pendant cet intervalle dans le prédicteur existant, sa
   récurrence affine peut être évaluée par une petite matrice triangulaire.
   Les huit sous-pas sont calculés ensemble avec NumPy, ainsi que leur intégrale
   de force, leur convolution de fatigue et les sensibilités. Cela retire la
   boucle Python interne. `_cycle_reference` conserve la récurrence séquentielle
   pour les comparaisons indépendantes. Le replay sans sensibilités évite leur
   calcul.
2. **Retrait des seules contraintes de force mathématiquement redondantes.**
   Avec recrutement non négatif et coefficients Ding positifs, un sous-pas de
   gain mécanique positif préserve une force positive. Dans un bloc de gain
   négatif, une force devenue négative ne peut pas redevenir positive; imposer
   la positivité à la fin du bloc implique donc celle de tous les sous-pas du
   bloc. Les lignes nécessaires sont conservées. Le mode revient aux lignes
   complètes lorsque les hypothèses de domaine ne sont pas satisfaites.
   Dans l'exemple étudié, tous les gains sont positifs: les 960 lignes `F>=0`
   sont redondantes, et le QP passe de 1 320 à 360 lignes. **Le replay contrôle
   toujours les 960 forces**, ainsi que toutes les bornes de puissance et de PW.
3. **LP d'enveloppe différé.** Le score n'utilise que l'enveloppe du dernier
   modèle SQP. Les LP des itérations intermédiaires, auparavant calculés puis
   jetés, sont supprimés.
4. **Secours numérique audité.** Si la suppression des lignes redondantes
   laisse SLSQP sortir des bornes, le même QP est réessayé avec toutes les
   contraintes. Ce secours n'est pas exécuté lorsqu'un LP prouve que l'égalité
   de travail est hors de l'enveloppe de ce modèle affine. Cela ne certifie
   aucun échec physiologique. Les tolérances de replay restent inchangées.

L'option `reduce_redundant_force_constraints=false` permet l'ablation. Les
résultats enregistrent les nombres de lignes, de secours et d'itérations SQP.

## Benchmark sur les snapshots réels

Source: campagne `pace-vr-h100-parallel-smoke-20260929`, à 1,92 Nm bilatéraux,
répartition fixe 50/50, muscles asymétriques, 30 Hz. Les snapshots des cycles
1 et 20 sont extraits des fichiers `right/result.json` et `left/result.json`,
avec vérification du cycle certifié et de l'empreinte du contexte. Quatre muscles,
30 intervalles, quatre nœuds de phase par muscle, soit 16 variables de QP.

Exécution avec un thread numérique par processus, CPU 18 à droite et 19 à gauche.
La référence conserve la carte séquentielle et toutes les lignes de force; elle
partage le pilote SQP/LP courant. Les gains ci-dessous sont donc conservateurs
par rapport au code historique qui calculait aussi les LP intermédiaires.

| Snapshot | Temps référence | Temps optimisé | Gain | Préfixe référence / optimisé |
|---|---:|---:|---:|---:|
| Droit, cycle 1 | 21,44 s | 4,86 s | ×4,41 | 86 / 86 |
| Gauche, cycle 1 | 13,30 s | 5,43 s | ×2,45 | 67 / 67 |
| Gauche, cycle 20 | 11,48 s | 3,10 s | ×3,71 | 60 / 60 |
| Droit, cycle 20 | 1,90 s | 2,46 s | Non comparable | 0 / 13 |

La dernière ligne révèle une sensibilité numérique préexistante de SLSQP près
des PW actives: la référence refuse immédiatement une borne, tandis que la
formulation équivalente valide 13 cycles avant un autre refus numérique. Le
script refuse de calculer un facteur d'accélération lorsque les quantités de
travail effectivement réalisées diffèrent. Il ne faut pas interpréter ce cas
comme une différence de capacité musculaire.

**Ces temps ne représentent pas un fit valide complet à H=100.** Le programme
demande un fit, mais le rollout nominal s'arrête avant 100 cycles; aucun
échantillon directionnel n'est donc lancé. Il serait incorrect d'affirmer que
100 cycles plus trois perturbations sont désormais calculés en 5 secondes.

Les sorties détaillées sont dans:

- `asymmetric-sides-r192-rho-bo-20260928/pace-vr-acceleration-20260929/validated-right-cycle1.json`
- `asymmetric-sides-r192-rho-bo-20260928/pace-vr-acceleration-20260929/validated-left-cycle1.json`
- `asymmetric-sides-r192-rho-bo-20260928/pace-vr-acceleration-20260929/validated-left-cycle20.json`
- `asymmetric-sides-r192-rho-bo-20260928/pace-vr-acceleration-20260929/checked-right-cycle20.json`

## Précision

À **PW identiques**, la différence ajoutée par l'accélération est de l'ordre de
l'arrondi machine: calcium identique, erreur maximale de force inférieure à
`5e-11 N`, erreur maximale de A inférieure à `7e-10`, Tau1 et Km inférieures à
`9e-14`. Les profils sous-pas et les intégrales de travail sont comparés, pas
seulement le dernier état.

Un replay DOP853 indépendant (`rtol=1e-10`, `atol=1e-11`) suit chaque PW optimisée
sur **tout le préfixe validé**, avec la même géométrie échantillonnée. Il mesure
l'erreur du prédicteur compact déjà présente avant l'accélération. Le gel de
géométrie par intervalle n'est donc pas validé par cette comparaison.

| Source cycle 1 | Bras droit, 86 cycles | Bras gauche, 67 cycles |
|---|---:|---:|
| Calcium | 0 | 0 |
| Force max sur les sous-pas | 2,04 N | 1,56 N |
| Force / pic de force de référence | 0,461 % | 0,396 % |
| A, erreur / pic de référence | 0,189 % | 0,180 % |
| Tau1, erreur / pic de référence | 0,344 % | 0,321 % |
| Km, erreur / pic de référence | 0,259 % | 0,230 % |
| Travail: écart maximal à la cible | 0,395 % | 0,383 % |

Les erreurs des états lents sont relevées aux frontières des cycles. Les forces
sont comparées à chacun des huit sous-pas: 20 640 échantillons temporels à droite
et 16 080 à gauche. Pour les snapshots du cycle 20, le script ajoute les erreurs
par muscle normalisées par **son propre** pic: moins de 0,442 % pour la force sur
ces trajectoires. Les bornes travail/puissance du modèle compact restent
vérifiées à leurs tolérances d'origine; ce n'est pas une certification de ces
bornes pour la dynamique DOP853, dont l'écart de travail ci-dessus reste visible.

Tests: **71 tests passent**, couvrant le cœur PACE-VR, le prédicteur compact,
l'adaptateur asynchrone, les scripts de benchmark et de décision. Les nouveaux
tests comparent les cartes à 30/50 Hz, 1/4/8/16 sous-pas, avec gains positifs et
négatifs, les jacobiens, le travail, les ensembles admissibles du QP, les LP
d'enveloppe et un fit H=100 synthétique complet. Le cas synthétique ne remplace
pas la validation du fit sur l'exemple réel.

## Le parallélisme doit employer deux processus

Le dispatcher actuel utilise deux **threads** dans le processus superviseur.
Une mesure appariée du même snapshot avec le nouveau cœur donne:

| Dispatcher, CPU 18 et 19 | Temps mural, démarrage/fermeture inclus |
|---|---:|
| Dispatcher actuel à deux threads | 10,31 s |
| Deux processus numériques `spawn` | 6,00 s |

Les deux modes donnent les mêmes préfixes 86/67 et les mêmes refus scientifiques.
Un essai supplémentaire du dispatcher à threads a pris 18,64 s; sa variabilité
et sa contention imposent de vérifier le temps mural réel, au-delà des temps
mono-bras. Cette mesure indique un bénéfice de l'isolation en processus, sans
attribuer toute la différence au seul GIL. Le script de mesure isole chaque
processus sur son CPU et inclut son démarrage. **La production n'a pas encore
été basculée vers ce dispatcher à processus**; sa politique asynchrone a été
laissée intacte dans cette intervention.

Sortie: `pace-vr-acceleration-20260929/bilateral-dispatch.json` dans le même dossier
de campagne. Aucun RHO ne tournait dans ce test; il ne mesure donc pas un effet
sur la latence du RHO.

## Décisions scientifiques restantes

1. Remplacer les threads mono-bras par deux processus numériques dédiés, avec
   le même contrôle de délai et d'ancienneté. Le gain est désormais mesuré.
2. Stabiliser la résolution du QP aux bornes de PW. Un démarrage admissible
   obtenu par un petit LP, puis une optimisation depuis ce point, est une piste
   plus pertinente qu'une relaxation des bornes physiques.
3. Résoudre le refus du rollout avant l'horizon. Dès le cycle 1, même la
   référence sans limite de temps s'arrête à 86/67 cycles. Les déficits de
   travail du dernier essai sont environ 0,017 J à droite et 0,082 J à gauche,
   avec enveloppes affines négatives. C'est une limite de cette allocation
   réduite/localisée, pas une preuve d'épuisement du bras. Les quatre nœuds de
   phase autour d'une PW initiale très saturée peuvent restreindre fortement
   les redistributions admissibles. Le classement prédictif doit être validé
   avant de transmettre les poids au RHO.
4. Mesurer ensuite le **fit effectivement accepté**, comprenant ses rollouts
   perturbés. Réduire H uniquement pour une contrainte de budget ne répare pas
   les points 2 et 3. La formulation actuelle refuse en outre tout fit lorsque
   le rollout de base n'atteint pas H; à l'approche d'une limite réelle située
   dans H, il faudra définir explicitement une valeur lisse de déficit/marge
   ou une autre règle d'acceptation validée, plutôt que prolonger silencieusement
   une trajectoire irréalisable.

## Reproduction

Depuis la racine du dépôt:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/benchmark_pace_vr_rollout.py \
  --source asymmetric-sides-r192-rho-bo-20260928/pace-vr-h100-parallel-smoke-20260929/results/pace_vr/right/result.json \
  --source-cycle 1 --cpu 18 --dop853-cycles 100 \
  --output /tmp/pace-vr-right-benchmark.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/benchmark_pace_vr_parallel.py \
  --right-source asymmetric-sides-r192-rho-bo-20260928/pace-vr-h100-parallel-smoke-20260929/results/pace_vr/right/result.json \
  --left-source asymmetric-sides-r192-rho-bo-20260928/pace-vr-h100-parallel-smoke-20260929/results/pace_vr/left/result.json \
  --source-cycle 1 --cpu-ids 18 19 --budget-seconds 16 \
  --output /tmp/pace-vr-dispatch-benchmark.json
```
