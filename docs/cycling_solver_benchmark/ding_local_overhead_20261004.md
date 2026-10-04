# Temps hors IPOPT du RHO Ding local prescrit — 4 octobre 2026

## Résultat mesuré

Même RHO unilatéral prescrit, 30 Hz, Radau-5, objectif fatigue unitaire,
IPOPT–MA57, Hessienne exacte compilée, 100 cycles propagés indépendamment.
Deux exécutions séquentielles sur CPU 12, un thread numérique, cache compilé
identique. Aucun export ni certificat n'a été retiré.

| Médiane chaude | Scalaire | Vectorisé |
|---|---:|---:|
| Itération RHO complète | 174.852 ms | 167.407 ms |
| IPOPT | 77.556 ms | 76.530 ms |
| Préparation de la réduction | 3.634 ms | 2.195 ms |
| Reconstruction primale/duale | 4.692 ms | 1.408 ms |
| Audit espace original | 1.172 ms | 1.217 ms |
| Archive NPZ synchrone | 1.650 ms | 1.654 ms |

Les 100 cycles sont validés dans chaque condition. Les vecteurs `x`, `g`,
`lam_x` et `lam_g` sont **identiques bit à bit** à chaque cycle. Le gain sur
l'itération complète est de 4,26 %. Une seule paire séquentielle ne caractérise
pas toute la variabilité du matériel; la différence IPOPT n'est pas attribuée
à la vectorisation. Le gain mesuré directement sur préparation/reconstruction
est de 4,72 ms/cycle.

Résultats : `local-results/ding-overhead-20261004/{scalar100,vector100}/result.json`,
archives et mesures détaillées dans `symbolic-audit/`.

## Changement exact et validation

`scripts/benchmark_ding_radau5_local_ab.py` intersecte désormais les bornes
affines répétées avec `numpy.maximum.at` / `minimum.at`. Il attribue les
multiplieurs de bornes au premier candidat d'origine par une réduction
vectorisée. Les coefficients positifs, négatifs, nuls, les bornes partagées,
les égalités et les priorités en cas d'ex æquo suivent la même règle scalaire.
`DING_LOCAL_VECTORIZED=0` restaure le chemin historique pour les comparaisons.
L'objectif, les équations, les tolérances et les certificats sont conservés.

Neuf tests ciblés passent, dont 30 configurations aléatoires reproductibles de
bounds/duals avec cibles répétées, coefficients de signes différents, égalités
et rejet des intersections impossibles; le test de reconstruction Radau
et les tests de compilation exacte passent aussi.

## Goulet principal restant

Le profilage de **toute la boucle RHO** sur 10 cycles identifie la construction
répétée des vecteurs Bioptim de bornes et d'initialisation, avant IPOPT :

- 11 760 appels à `_compute_value_for_node`;
- 360 618 appels à `PathCondition.__array_finalize__`;
- 118 146 recherches de variables et 117 630 appels `evaluate_at`;
- temps sous cProfile : 1,153 s pour `bounds_vectors`, 0,655 s pour
  `init_vector`, sur dix cycles.

Ces temps **profilés ne sont pas des latences de benchmark** : cProfile
pénalise beaucoup les petits appels Python. Ils localisent cependant le
travail évitable. Les ~80 ms restants hors adaptateur ne sont pas principalement
du calcul physiologique, du certificat ou de la factorisation MA57.

Fichiers : `.benchmark-deps/bioptim/bioptim/optimization/vector_utils.py`,
`optimization_vector.py`, `bound_vector.py`, `init_vector.py`.

La prochaine optimisation à fort potentiel est un chemin numérique groupé
pour `CONSTANT`, `CONSTANT_WITH_FIRST_AND_LAST`, `EACH_FRAME` et `ALL_POINTS` :
extraire une fois par variable ses indices et son échelle, créer les colonnes
par diffusion/indexation, puis empiler suivant `vector_layout`. Les points
internes du premier intervalle doivent conserver les règles d'interpolation
actuelles. Toute interpolation personnalisée ou disposition variable par
nœud doit reprendre le chemin standard. Les valeurs des bornes doivent être
relues à chaque appel : mettre les nombres en cache pourrait ignorer une
limite terminale, une nouvelle demande de travail ou un état initial changé.

## Parallélisme, compilation et exports

- Les deux bras indépendants peuvent rester sur deux processus et deux CPU.
  Les cycles successifs d'un même bras dépendent de la solution certifiée
  précédente et ne sont pas indépendants.
- La granularité des appels par nœud et le GIL rendent leur distribution à
  des threads peu attractive. La vectorisation élimine le coût de dispatch.
- Compiler un noyau numérique n'a de sens qu'après extraction des métadonnées
  Python; commencer par NumPy groupé permet une validation directe face au
  chemin standard. Les callbacks de l'objectif et de la dynamique sont déjà
  traités séparément par CasADi.
- L'audit d'admissibilité et le certificat restent sur le chemin causal. Les
  exports NPZ/JSON peuvent être confiés à un écrivain après copie immuable,
  avec file bornée, propagation d'erreurs et vidage à l'arrêt. Aucun gain de
  cette option n'est revendiqué ici.
- La réécriture de toute la liste JSON à chaque cycle est quadratique en
  nombre de cycles; un journal JSONL et une synthèse en fin de run éviteraient
  cette croissance. Les consommateurs historiques doivent être adaptés avant
  changement de format.
- Le run scalaire passe aussi 78,45 s dans `summary_build` **après la fin des
  100 cycles**. Ce coût n'entre pas dans les 174,85 ms d'itération. Il peut
  devenir une analyse hors ligne sans retarder la commande RHO, mais les
  résultats scientifiques finaux doivent être produits et leurs erreurs
  signalées.

Profilage reproductible : `scripts/profile_ding_local_overhead.py --scope rho`
ou `--scope adapter`, suivi de `--` et des arguments usuels de
`benchmark_ding_local_compiled.py`. Profils enregistrés dans
`local-results/ding-overhead-20261004/{profile,rho-profile}.prof`.
