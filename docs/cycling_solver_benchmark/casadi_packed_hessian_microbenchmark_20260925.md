# Hessienne exacte MX par paquets : audit CasADi local du 25 septembre 2026

## Décision provisoire

Un graphe MX construit avec `Function.map(..., "thread", 4)` **conserve un
`ThreadMap` après `casadi.hessian`** dans l'environnement local. Son évaluation
est plus rapide sur le problème synthétique ci-dessous. Une Hessienne exacte
calculée indépendamment par stage, puis assemblée aux frontières d'état,
reproduit la Hessienne globale à une erreur absolue inférieure à `9e-17`.

La taille de paquet **10 cycles n'est pas retenue d'avance** : avec la méthode
de construction testée ici, elle est plus lente que les paquets de 1 ou 5 cycles.
Il faut inspecter `nlp_hess_l` du FHO réel avant de modifier Bioptim ou la
callback IPOPT. Aucun solve FHO ni compilation C n'a été effectué par cet audit.

## Environnement observé

| Runtime | Version CasADi | Mapping `thread` | Mapping `openmp` |
|---|---|---|---|
| `cocofest-rho32` | 3.7.2 | `ThreadMap` actif | `OmpMap` créé, mais repli série avec avertissement `WITH_OPENMP=OFF` |
| `cocofest-madnlp32` | 3.7.2+ | `ThreadMap` actif | même repli série |

La roue CasADi 3.8.0 utilisée dans les pilotes du 12 septembre résidait dans
`/tmp/bioptim-casadi38-venv`, actuellement absent. Les nombres ci-dessous
ne caractérisent **pas** CasADi 3.8.0. Le graphe MX, la fonction de stage et
les vérifications peuvent être rejoués dès que cet environnement est restauré.

Les mesures principales utilisent 4 CPU logiques `12–15`, qui correspondent à
quatre cœurs distincts sur cette machine. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` et `NUMEXPR_NUM_THREADS` valent 1. Un seul processus de mesure
est lancé, sans FHO concurrent démarré par ce protocole. Chaque temps est la
médiane de 12 appels après un échauffement. Le JSON garde minima et P90.

## Problème et vérification

Le problème synthétique contient une chaîne de 20 ou 60 stages. Chaque stage
voit quatre états à gauche, quatre états à droite et quatre contrôles. Les états
de frontière sont partagés. Son Lagrangien local contient un objectif non
linéaire (`sin`, `exp`) et quatre contraintes non linéaires avec multiplicateurs.
Le vecteur global a respectivement 164 ou 484 variables. Il possède une
Hessienne creuse avec 1 932 ou 5 772 non-zéros structurels.

La référence est `casadi.hessian(sum(mapped_stage_lagrangian), x)` calculée
globalement. Deux autres constructions sont mesurées : Hessienne de chaque
stage, puis scatter-add creux ; Hessienne d'un paquet de 1, 5 ou 10 stages,
puis scatter-add. La frontière d'état de deux paquets apparaît dans les deux
Hessiennes locales et ses contributions sont additionnées. L'erreur numérique
maximale de toutes les variantes est inférieure à `9e-17` pour les points
déterministes testés. C'est une vérification d'une évaluation, pas une preuve
sur tous les états physiologiques ni de l'ordre de sparsité exigé par IPOPT.

## Mesures locales

Tous les temps du tableau sont en millisecondes par appel. « Assemblage »
inclut la conversion `coo_matrix → csr_matrix`, mais exclut les entrées IPOPT,
la résolution linéaire, la recherche de pas et la construction initiale du graphe.

| Stages | Hessienne globale série | Hessienne globale `ThreadMap/4` | Stages locaux série + assemblage | Stages locaux `ThreadMap/4` + assemblage |
|---:|---:|---:|---:|---:|
| 20 | 16,01 | 6,17 | 2,38 | 1,59 |
| 60 | 49,20 | 13,44 | 7,02 | 4,56 |

Le gain de la Hessienne globale `ThreadMap/4` est ainsi ~3,7× à 60 stages,
sur cette fonction synthétique. Le stage local est beaucoup moins coûteux même
en série ; la différence ne peut donc pas être attribuée au seul parallélisme.
Elle inclut un changement de structure du graphe AD et un assemblage Python
qui serait à réimplémenter avec le contrat exact de `hess_lag` pour IPOPT.

La variante par paquets utilise une construction MX avec appels de fonctions
imbriquées, identique entre tailles de paquets :

| 60 stages, taille de paquet | Nombre de paquets | Dimension Hessienne locale | Série + assemblage (ms) | `ThreadMap/4` + assemblage (ms) |
|---:|---:|---:|---:|---:|
| 1 | 60 | 12 | 32,28 | 9,64 |
| 5 | 12 | 44 | 33,45 | 10,66 |
| 10 | 6 | 84 | 46,75 | 14,83 |

Ce tableau ne doit pas être confondu avec le tableau « stages locaux » : le
paquet de 1 est dérivé *à travers* un appel `stage_lag`, alors que le stage
local direct dérive son expression avant le mapping. Le résultat montre surtout
qu'augmenter la taille des paquets peut perdre la granularité parallèle et
alourdir la différentiation. La variante `openmp` se replie sur la série dans
les deux runtimes présents ; ses temps ne constituent pas un test OpenMP.

## Suite pour conclure sur FHO65

1. Capturer les fonctions `nlp_hess_l` exactes du FHO65 MX, sans compilation ni
   solve supplémentaire, et relever leurs `find_functions()`, classes des
   mappings, dimensions et non-zéros. Un `ThreadMap` conservé dans le graphe
   dérivé permet un premier A/B `--n-threads 1/4/8` sans callback nouvelle.
2. Si l'évaluation de cette fonction ne bénéficie pas du mapping, construire
   une callback Hessienne locale expérimentale hors campagne. Contrôler
   numériquement valeurs, sparsité et ordre des triplets contre la callback
   native à plusieurs `x`, `lambda`, `sigma` d'un certificat FHO65.
3. Mesurer séparément build, `nlp_hess_l`, temps total IPOPT, itérations,
   admissibilité, objectif et pic RSS, avec mêmes CPU, seed, MA57 et tolérances.
   Essayer des tailles 1, 5 puis 10 seulement si la précédente le justifie.
4. Retenir la variante uniquement si elle produit le même certificat et baisse
   le temps total. Le microbenchmark actuel établit la faisabilité
   mathématique et le potentiel de `ThreadMap`, pas un gain FHO.

## Reproduction

Le script est [benchmark_casadi_packed_hessian.py](../../scripts/benchmark_casadi_packed_hessian.py).
Les sorties brutes sont [CasADi 3.7.2](casadi_packed_hessian_microbenchmark_20260925.json)
et [CasADi 3.7.2+](casadi_packed_hessian_madnlp_20260925.json).

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
taskset -c 12-15 /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/benchmark_casadi_packed_hessian.py --stages 20 60 --workers 4 \
  --packet-sizes 1 5 10 --repeats 12 \
  --output docs/cycling_solver_benchmark/casadi_packed_hessian_microbenchmark_20260925.json
```

La mesure a été réalisée en un seul passage, dans un ordre fixe des variantes.
Une décision de performance doit ajouter des répétitions alternées et tenir
compte de la charge de la machine. Les petits temps du script ne se
transposent pas aux millions de variables et contraintes d'un FHO long.
