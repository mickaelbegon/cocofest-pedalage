# Budget initial ACADOS et récupération du même RHO

L'option `--acados-rho-initial-iteration-budget N` limite la première tentative
ACADOS de chaque RHO physique à partir du RHO 2. Elle est désactivée par défaut
(`null` dans la configuration JSON). Le RHO 1 initialise la capsule native avec
le budget nominal. Les retries du même RHO et les résolutions auxiliaires gardent
le budget nominal, restauré dans un bloc `finally`, même après une exception.

Cette politique détermine un budget à l'avance. Elle ne détecte pas une stagnation
en cours de SQP et ne prédit pas les fenêtres qui échoueront. Une fenêtre facile
peut toujours converger avant le budget; une fenêtre qui aurait convergé plus tard
peut désormais nécessiter une récupération.

## Conditions scientifiques

Le budget doit être positif et inférieur à `--max-acados-iterations`. L'option
exige explicitement le mode RHO ACADOS, `--acados-ipopt-recovery`,
`--retry-failed-rho-without-advance` et au moins une tentative de récupération.
Les caps artificiels de test et les retries MAXITER internes sont incompatibles
avec cette politique, pour empêcher un cumul de budgets ambigu.

Les critères de convergence et de faisabilité restent ceux du runner. Un statut
MAXITER, même primalement faisable, ne certifie pas un RHO. La récupération reprend
le RHO figé depuis le primal préparé avant l'échec, issu du dernier état certifié.
Le backend de récupération IPOPT et son solveur linéaire existants sont conservés;
la campagne de référence utilise MA57. La nouvelle option ne sélectionne pas à
elle seule un autre solveur linéaire.

Sans `--acados-ipopt-fallback-advance`, une nouvelle certification ACADOS reste
nécessaire après une graine IPOPT. Avec cette option hybride, seule la récupération
Radau finale convergée et indépendamment faisable peut avancer le RHO. Les niveaux
de récupération servant seulement de graine ne peuvent pas l'avancer.

## Audit

`result.json` exporte la configuration et `acados_rho_initial_iteration_budget`.
L'audit indique, pour chaque RHO concerné, le budget demandé et appliqué par le
setter natif, les itérations effectuées, le statut et les résidus au retour,
les temps solveur et mural, la restauration du budget nominal, les fenêtres de
récupération correspondantes et leur coût mural total. Les exceptions natives
laissent également une entrée avec l'erreur et l'état de restauration.

Le champ `unused_nominal_iteration_headroom` est un solde de budget, pas une
économie mesurée. `estimated_time_saved_s` reste `null`: le temps d'un scénario
nominal qui n'a pas été exécuté n'est pas observable. Un replay apparié doit
comparer aussi les récupérations, la fatigue, la mécanique, le slew et le nombre
de RHO certifiés.

## Estimation rétrospective 50 Hz, ΔPW ≤ 100 µs

Source locale:
`frequency-slew-100-acados-hybrid-corrected-20260912/acados-50hz-100us/result.json`,
entrée `solver_attempt_accounting.attempts`. Le run certifie 100 RHO, dont 23 par
IPOPT/MA57. Les 23 tentatives ACADOS en échec atteignent chacune 100 SQP et cumulent
914,795 s dans le champ de temps mural des tentatives. Le temps bout en bout est
1661,647 s. Parmi les 76 succès ACADOS après RHO 1, les budgets courts
interrompraient aussi certains succès historiques:

| Budget initial | Économie brute théorique sur les 23 échecs | Succès historiques dépassant le budget |
| --- | ---: | ---: |
| 20 | 731,836 s | 21 / 76 |
| 30 | 640,357 s | 16 / 76 |
| 50 | 457,398 s | 8 / 76 |

Le calcul est `Σ temps_échec × (100 − budget) / 100`: il suppose un coût constant
par itération et les mêmes fenêtres en échec. Il ignore les récupérations
supplémentaires, les changements de trajectoire et les coûts d'installation.
Ces valeurs ne constituent donc ni un gain net mesuré ni une garantie de gain.
Le budget 50 préserve davantage de succès historiques que 20 et constitue un
point de comparaison moins agressif pour un futur replay.

Vérifications automatisées: défauts inchangés, validation des combinaisons,
propagation de configuration, cap consommé une seule fois par RHO, restoration
après exception, certification inchangée et sérialisation de l'audit avec les
liens vers les récupérations. Aucun nouveau benchmark de 100 cycles n'est inclus
dans cette estimation.

Un pilote natif ultérieur sur la nouvelle formulation `delta_pw_v1` compare huit
RHO appariés : 8/8 certifiés dans les deux variantes, trois récupérations aux
mêmes RHO 2/4/6, et boucle RHO de 190,274 s avec le budget nominal à 84,073 s avec
le budget initial 30. Les indicateurs de fatigue, audits physiques et slew
exportés sont identiques. Ce gain mesuré sur une seule paire courte ne prédit
pas celui d'une campagne de 100 cycles. Le rapport et la commande de replay
sont dans `acados-budget30-delta-pilot-20260912T1452/REPORT.md`.
