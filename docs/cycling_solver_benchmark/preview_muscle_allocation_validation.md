# Validation numérique de l'allocation musculaire avec anticipation courte

## Résultat

L'allocation gloutonne H1 et les previews de 2 et 3 phases complètent les dix
cycles aux deux frontières RHO testées. À l'ancre 0, les trois politiques
complètent aussi 30 cycles. À l'ancre 112, aucune ne complète H30 : H1 s'arrête
après 607 phases, preview-2 après 636 et preview-3 après 605.

Ces nombres de phases ne sont **pas** des cycles d'endurance. Preview-2 et
preview-3 peuvent refuser un QP anticipé une ou deux phases avant qu'une action
courante ne devienne impossible. Ils décrivent seulement les politiques
numériques testées avec leur modèle affine gelé. Aucun gain d'endurance, aucune
performance clinique et aucune infaisabilité Ding globale ne sont établis.

Les neuf trajectoires compactes complètes ont été rejouées avec Ding complet.
Toutes passent les gates numériques de `1e-3 N·m` sur le moment aux extrémités
de phase et de `1e-3` sur les états lents normalisés. Aucun replay H30 complet
n'existe à l'ancre 112, puisque les trois séquences PW y sont incomplètes.

## Protocole pré-déclaré

La source, le profil, les ancres 0 et 112, les états terminaux archivés et le
contexte de coordonnées sont ceux des validations précédentes. Le runner refuse
un baseline incompatible avant calcul : schéma, empreintes SHA-256 source et
profil, sous-pas, seuils, horizon, échelle de moment, centre et empreinte de
contexte doivent correspondre.

Les réglages sont :

- horizons 10 et 30 cycles, 30 phases par cycle;
- `16` sous-pas pour la carte compacte et le replay Ding;
- preview `1`, `2` ou `3` phases;
- cibles de moment total originales exactes, tolérance numérique `1e-8 N·m`;
- `60` itérations SLSQP maximum, tolérance d'optimalité `1e-6`;
- régularisation du recrutement `1e-8`;
- limite `0,5 s` dans les évaluations d'objectif du QP;
- aucun repli vers H1 en cas d'échec.

H1 passe directement par l'allocateur glouton existant. Pour K2/K3, le QP
minimise la somme des carrés des écarts de moments musculaires pondérés par la flexibilité
nominale, plus `0,5e-8 ||r||²`, sous les égalités de moment total des phases
prévues et les bornes PW. Seule la première commande est appliquée, puis le QP
est reconstruit. À la fin de l'horizon, sa profondeur est tronquée au nombre de
phases restantes. Les états lents futurs employés pour construire chaque QP
sont une graine nominale gelée; ils ne constituent pas une propagation Ding
complète à l'intérieur du QP.

Le pré-écran d'un cycle projetait linéairement les coûts H30 entre `0,70 s`
pour H1 et `2,93 s` pour K3, sous la limite fixée de `15 s`. Aucun cas n'a donc
été arrêté pour coût. Cette projection n'est qu'un gate de campagne; les temps
mesurés ci-dessous sont une seule passe des rollouts complets exécutés ensuite,
sans mesure concurrente; ce ne sont pas des médianes de répétitions.

## Progression et coût hors NLP

| Ancre | Horizon | H1 glouton | Preview-2 | Preview-3 |
|---:|---:|---|---|---|
| 0 | 10 | complet, `0,224 s` | complet, `0,741 s` | complet, `1,008 s` |
| 0 | 30 | complet, `0,681 s` | complet, `2,219 s` | complet, `3,037 s` |
| 112 | 10 | complet, `0,227 s` | complet, `0,735 s` | complet, `1,005 s` |
| 112 | 30 | échec après 607, `0,459 s` | échec après 636, `1,573 s` | échec après 605, `2,076 s` |

H1 est ici mesurée via l'enveloppe `PreviewMuscleAllocation(K1)`, pas via le
backend batch optimisé du rapport précédent. Cette enveloppe rend H1 elle-même
un peu plus coûteuse que l'appel direct du prédicteur compact; les rapports
ci-dessus comparent les trois politiques dans la même interface. K2 coûte
environ 3,3 fois H1 et K3
environ 4,4 fois H1 sur les cas complets. Ces QP et leur coût restent hors NLP.
Le temps total de la campagne, replays Ding de validation inclus, est `58,83 s`.

Aucun fit local de 41 points n'a été chronométré pour la politique preview.
Les `0,73–1,01 s` H10 mesurent un seul rollout K2/K3, sans les évaluations de
marge répétées, les points tenus à l'écart ni les éventuels refits. Cette
campagne ne démontre donc pas une latence de mise à jour prête pour un usage
clinique ou temps réel.

Les coûts objectifs cumulés du QP sont des diagnostics dont la normalisation
de flexibilité est reconstruite à chaque phase. Les fenêtres de preview se
recouvrent et leur profondeur diffère également; ces cumuls ne sont donc pas
directement comparables entre profondeurs comme une métrique d'endurance :

| Cas | Itérations K2 / K3 | Objectif cumulé K2 / K3 |
|---|---:|---:|
| ancre 0, H10 | `546 / 841` | `0,520 / 0,529` |
| ancre 0, H30 | `1656 / 2659` | `4,521 / 4,446` |
| ancre 112, H10 | `414 / 629` | `1,134 / 1,659` |
| ancre 112, préfixe H30 | `907 / 1284` | `9,289 / 11,774` |

Sur tous les cas complets, les égalités finales du QP ont un résidu maximal
`1,66e-15 N·m`, les bornes PW n'ont aucune violation, la stationnarité KKT reste
sous `9,58e-7` et la complémentarité sous `1,20e-10`. Toutes les commandes et
les frontières d'état attendues sont finies et dans le domaine positif. Aucun
repli greedy n'a été employé. Les cibles totales originales sont suivies dans
la carte compacte à `1,67e-15 N·m` près au maximum.

## Échecs anticipés à l'ancre 112, H30

H1 refuse la phase 7 du cycle futur d'index 20 car la cible totale est sous la
borne courante du modèle compact. Preview-2 modifie l'allocation et progresse
jusqu'au cycle 21, puis refuse à la phase 6 un QP qui inclut la phase suivante.
Preview-3 refuse dès la phase 5 du cycle 20, son QP couvrant alors les phases 5,
6 et 7. K3 ne démontre donc pas une endurance moindre que H1 : il détecte une
incompatibilité de son problème anticipé deux phases plus tôt.

Les deux refus preview ont le statut SLSQP 8 et des résidus d'égalité de
`2,714e-5 N·m` pour K2 et `1,319e-4 N·m` pour K3. Un audit indépendant avec
HiGHS, tolérances primale et duale `1e-10`, confirme que les égalités de ces
**QP affines gelés** sont incompatibles avec leurs bornes. Le problème minimax
donne un résidu incompressible de `2,6494889e-5 N·m` pour K2 et
`1,2879356e-4 N·m` pour K3, sans violation de bornes, soit respectivement
environ 2649 et 12879 fois la tolérance de suivi `1e-8`. Ce contrôle montre que
l'échec n'est pas seulement un arrêt du solveur SLSQP. Il ne prouve toujours
pas l'infaisabilité du modèle Ding complet, d'une autre approximation future
ou d'une autre politique d'allocation.

Commande de reproduction de cet audit indépendant :

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-value-review-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python - <<'PY'
import numpy as np
from scipy.optimize import linprog
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.preview_muscle_allocation import PreviewMuscleAllocation
from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
from scripts.validate_local_endurance_value import DEFAULT_SOURCE, DEFAULT_PROFILE, _source_terminal_states

p = build_rho_adaptive_moment_policy(
    DEFAULT_SOURCE, DEFAULT_PROFILE, cycle_index=112, cycle_period=1.0
)
c, _ = select_certified_rho_cycle(DEFAULT_SOURCE, cycle_index=112, cycle_period=1.0)
initial = _source_terminal_states(c, p.muscle_names)
predictor = CompactMusclePredictor(p.intervals, p.parameters, substeps=16)
options = dict(
    primal_feasibility_tolerance=1e-10,
    dual_feasibility_tolerance=1e-10,
    ipm_optimality_tolerance=1e-12,
)
for depth in (2, 3):
    policy = PreviewMuscleAllocation(predictor, preview_phases=depth)
    result = policy.rollout(initial, horizon_cycles=30)
    step = result.completed_intervals
    problem = policy.build_preview_qp(
        result.state_history[step], step % len(p.intervals)
    )
    A, b = problem.equality_matrix, problem.equality_target
    n = A.shape[1]
    bounds = list(zip(np.zeros(n), problem.upper_bounds))
    strict = linprog(
        np.zeros(n), A_eq=A, b_eq=b, bounds=bounds,
        method="highs", options=options,
    )
    inequalities = np.block([
        [A, -np.ones((len(b), 1))],
        [-A, -np.ones((len(b), 1))],
    ])
    slack = linprog(
        np.r_[np.zeros(n), 1.0],
        A_ub=inequalities,
        b_ub=np.r_[b, -b],
        bounds=bounds + [(0, None)],
        method="highs",
        options=options,
    )
    print(depth, step, strict.status, slack.status, slack.fun, A @ slack.x[:-1] - b)
PY
```

## Replay Ding complet des trajectoires complètes

Le replay applique les PW compactes aux équations Ding complètes, RK4 à 16
sous-pas. Il audite le suivi aux extrémités de phase et l'écart des états lents;
il ne ferme pas à nouveau la boucle preview sur les états Ding rejoués.

| Ancre | Horizon | Erreur moment max H1 / K2 / K3 | Erreur lente max / repos H1 / K2 / K3 | Gate |
|---:|---:|---:|---:|---|
| 0 | 10 | `0,4804 / 0,4801 / 0,4801 mN·m` | `3,814e-5 / 3,756e-5 / 3,737e-5` | passé |
| 0 | 30 | `0,5116 / 0,5108 / 0,5107 mN·m` | `1,128e-4 / 1,108e-4 / 1,099e-4` | passé |
| 112 | 10 | `0,4827 / 0,4828 / 0,4828 mN·m` | `4,427e-5 / 4,420e-5 / 4,422e-5` | passé |

Les petites différences entre H1, K2 et K3 ne suffisent pas à départager une
politique d'endurance. Les seuils vérifient seulement la fidélité numérique des
PW sélectionnées sur les trajectoires complètes disponibles.

## Artefacts et reproduction

Les artefacts finaux sont dans `.cache/preview-muscle-allocation-validation/` :

- `report.json` : réglages, compatibilité du baseline, progression, échecs,
  audits QP, coûts, latences, gates Ding et empreintes SHA-256;
- `validation_arrays.npz` : états, PW, moments, enveloppes, diagnostics temporels
  et séries de replay Ding;
- `validation.png` : progression, coût hors NLP, moment Ding et erreur des états
  lents.

Commande reproductible :

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mplconfig \
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -u \
scripts/validate_preview_muscle_allocation.py
```

Tests ciblés :

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mplconfig \
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest -q \
tests/test_preview_muscle_validation.py tests/test_preview_muscle_allocation.py
```
