# FHO65 : coût de construction du graphe avec ThreadMap

Date : 25 septembre 2026. Ce profil utilise le même FHO65 MX, la même graine et
la même affinité CPU24–31 que le benchmark de résolution, avec OMP/BLAS à un
thread. Le harnais `scripts/profile_fho_construction.py` s'arrête à l'entrée de
`generic_solve` : il ne crée pas IPOPT et ne produit aucun certificat.

## Mesures

| Étape (temps inclusif) | 1 thread natif | 8 threads natifs | 8 threads, `map` au premier nœud |
|---|---:|---:|---:|
| Avant `generic_solve` | 59,55 s | 230,76 s | 64,64 s |
| `_finalize_penalties` | 41,54 s | 210,53 s | 45,81 s |
| `_declare_continuity` | 7,87 s | 173,23 s | 8,40 s |
| `_set_penalty_function` (5 915 appels) | 18,04 s | 184,27 s | 20,48 s |
| `update_constraints` | 28,46 s | 31,56 s | 30,45 s |
| `_prepare_dynamics` | 0,73 s | 0,72 s | 0,85 s |
| Pic RSS avant solve | 1,27 GiB | 1,34 GiB | 1,28 GiB |

Les durées des méthodes imbriquées ne s'additionnent pas. La création des
pénalités de continuité explique presque tout le coût supplémentaire du mode
8 threads : la seule méthode `_declare_continuity` ajoute 165,36 s, pour un
surcoût total de 171,21 s avant IPOPT.

## Mécanisme et essai isolé

Dans la version Bioptim locale,
`PenaltyOption.add_or_replace_to_penalty_pool` visite chaque nœud de
`Node.ALL_SHOOTING`. Pour chaque visite, `_set_penalty_function` appelle deux
fois `Function.map(n_nodes, "thread", n_threads)` lorsque la pénalité est
threadée, pour `function[node]` et `weighted_function[node]`. Sur la
continuité du FHO65, il y a 1 950 nœuds. Pourtant, le dispatch threadé
`interface_utils.py` lit `penalty.weighted_function[0]` pour toute la série,
et la version `weighted_function_non_threaded[node]` reste disponible pour
chaque nœud.

Le commutateur expérimental `--map-first-node-only` du harnais laisse la
fonction mappée au premier nœud et construit les fonctions non threadées aux
autres nœuds. Il réduit la préparation 8 threads de **166,12 s (72,0 %)** et
la ramène à 5,09 s du temps série. Le commutateur ne modifie ni le dépôt
Bioptim installé ni une campagne active.

Une mise en cache de `Function.map` par identité de la fonction locale ne
résoudrait pas ce cas : `_set_penalty_function` crée une nouvelle `Function`
pour chaque nœud, donc les identités diffèrent. Une empreinte structurelle
pour partager les fonctions entre nœuds ou entre OCP devrait intégrer le graphe
et ses constantes numériques, la taille des entrées/sorties, le nombre de
nœuds, le nombre de threads, MX/SX et la version CasADi. Elle coûterait elle
même une sérialisation/hachage de milliers de fonctions et risquerait une
réutilisation périmée après changement de dynamique, profil ou horizon. Le
calcul paresseux d'un seul `ThreadMap` par pénalité et par construction OCP est
le correctif le plus simple : le graphe mappé a exactement la durée de vie de
son OCP. Pour répéter un solve au **même horizon** dans un même processus, il
faut ensuite conserver l'interface et son `nlpsol` déjà construit ; le cache
de `generic_solve` le permet tant que le graphe et les constantes de pénalité
ne changent pas. Un nouvel horizon `N+3` exige un nouveau NLP.

## Validation sur la résolution exacte bornée

Le harnais `scripts/benchmark_fho_threaded_solve.py` accepte aussi
`--map-first-node-only`. Avec 8 threads, MA57, même graine, mêmes CPU24–31 et
un plafond de 30 itérations :

| Mesure | Bioptim natif 8 threads | `map` au premier nœud |
|---|---:|---:|
| Préparation application/OCP | 258,74 s | 76,25 s |
| Construction `nlpsol` | 98,71 s | 98,84 s |
| Solve IPOPT | 208,11 s | 206,38 s |
| Temps total du harnais | 566,32 s | 382,21 s |
| Pic RSS | 22,93 GiB | 22,87 GiB |

La comparaison numérique dans
`local-results/fho65-threaded-solve-20260925/comparison-map-first.json`
confirme : mêmes 265 223 variables et 259 414 contraintes, mêmes 1 833 064
non-zéros de Jacobienne et 1 072 504 non-zéros de Hessienne, même hash de
sparsité Hessienne, même `x0` et mêmes bornes, même objectif initial, mêmes
triplets `(lbg, ubg, g(x0))`, même historique d'objectif IPOPT et **même
vecteur final `x` bit à bit**. Le gain de 184,11 s sur le total vaut **32,5 %**
pour ce budget de 30 itérations. La différence de 1,74 s de solve relève de
la variabilité de mesure ; l'accélération vient de la préparation.

Ce test finit à `Maximum_Iterations_Exceeded` avec une violation de 63,0 :
il vérifie le chemin numérique à budget fixé, pas un certificat convergé. Avant
une modification générale de Bioptim, il reste à comparer le gradient,
la Jacobienne et la Hessienne à des points perturbés, ainsi qu'une résolution
convergée. Le correctif doit conserver les fonctions non threadées pour chaque
nœud et ne supprimer que les `ThreadMap` non utilisés du dispatch.

## Jacobienne par paquets

Sur les 30 itérations FHO65 mesurées auparavant, la Jacobienne globale native
prend 24,74 s sur 208,12 s de solve à 8 threads (11,9 %). Elle bénéficie déjà
du `ThreadMap` des pénalités Bioptim : ses 32 évaluations tombent de 85,49 s
à 24,74 s entre 1 et 8 threads. Une Jacobienne par paquets aurait donc, même
si son évaluation devenait gratuite, un plafond de gain de 11,9 % sur ce solve.
Elle ajouterait le coût de construction et d'assemblage des paquets, ainsi
qu'une vérification stricte de l'ordre des lignes de contraintes. La priorité
est de supprimer les `ThreadMap` inutilisés à la construction et de mesurer
la résolution exacte à 12 threads. Une Jacobienne par paquets n'est justifiée
que si, sur un horizon plus grand ou une autre formulation, son coût redevient
une part importante du temps total.

Un profil de construction additionnel avec 12 threads et `map` au premier
nœud, sur CPU20–31, atteint `generic_solve` en **59,81 s** (pic RSS 1,28 GiB).
C'est uniquement le coût avant IPOPT. Un second test exact borné a été exécuté
sur les mêmes CPU16–27 et avec les mêmes options que la référence native
12 threads :

| Mesure FHO65, 12 threads | Bioptim natif | `map` au premier nœud |
|---|---:|---:|
| Préparation application/OCP | 223,32 s | 71,38 s |
| Construction `nlpsol` | 96,28 s | 95,99 s |
| Solve IPOPT, 30 itérations | 193,79 s | 188,85 s |
| Total du harnais | 514,12 s | 356,95 s |
| Hessienne, 30 appels | 144,68 s | 140,60 s |
| Jacobienne, 32 appels | 20,46 s | 20,02 s |
| Pic RSS | 22,91 GiB | 22,90 GiB |

Le temps total baisse de **157,17 s (30,6 %)** ; le gain de préparation est
de **151,94 s**. Les écarts plus petits dans le solve relèvent d'un seul
passage et ne sont pas attribués au correctif. Le fichier
`local-results/fho65-threaded-solve-20260925/comparison-12-map-first.json`
confirme les mêmes dimensions, sparsités, données initiales, historique
d'objectif et vecteur final `x` bit à bit. Les deux solves sont non certifiés :
plafond de 30 itérations et violation finale de 63,0.

## Reproduction

Les données brutes sont dans
`local-results/fho65-construction-profile-20260925/threads-{1,8,8-map-first,12-map-first}/report.json`.
Pour chaque variante, exécuter `scripts/profile_fho_construction.py` avec
`--base-command local-results/fho65-native-hessian-audit-20260925/threads-8-ones/command.json`,
`--threads 1` ou `8`, et un nouveau `--output-dir`. Ajouter
`--map-first-node-only` pour l'essai accéléré. Le harnais exige un répertoire
de sortie inexistant.
