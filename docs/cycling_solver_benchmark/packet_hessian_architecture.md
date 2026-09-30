# Hessienne exacte par paquets : prototype et limites

Date : 2026-09-25. Module expérimental :
`cocofest/optimization/packet_hessian.py`. Aucun branchement à une campagne
active ni modification du driver FHO.

## Ce qui a été vérifié

Environnement testé : `/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python`,
CasADi **3.7.2**. Le mécanisme ne nécessite donc pas CasADi 3.8. Il reste à
mesurer les différences entre versions sur le vrai graphe FHO.

Le prototype accepte une liste de contributions locales
`kernel(z,p) -> (f_local,g_local)`. Chaque contribution connaît les indices de
ses variables dans le vecteur NLP **mis à l'échelle** et les indices de ses
contraintes dans le vecteur de multiplicateurs global. Une contrainte appartient
à exactement un paquet ; les variables peuvent appartenir à plusieurs paquets.

La Hessienne locale de `sigma*f_local + dot(lambda_local,g_local)` est dérivée
avant la création du `Function.map`. Les noyaux identiques sont évalués ensemble
via `map(count, "thread", min(threads,count))`. Les seuls coefficients non nuls
du triangle supérieur local sont retournés. Une matrice constante creuse réalise
ensuite le placement et la somme dans le triangle supérieur global. Les indices
locaux peuvent avoir un ordre différent de l'ordre global.

Ce calcul conserve les couplages de frontière et les variables globales
partagées. Il ne construit pas une approximation bloc-diagonale. La formule est
`H = sum(P_b.T * H_b * P_b)`. Les termes dont les variables traversent une
frontière restent intégralement dans un paquet, avec toutes leurs variables.

La fonction fournie à CasADi/IPOPT est exactement :

```python
hess_lag(x, p, lam_f, lam_g) -> triu_hess_gamma
solver = casadi.nlpsol("solver", "ipopt", nlp, {"hess_lag": result.function})
```

`hess_lag` est une option CasADi au premier niveau, pas `ipopt.hess_lag`.
Le test de résolution réel valide ce contrat, sans compilation C.

## Tests reproductibles

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cocofest-packet-mpl /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest tests/test_packet_hessian.py -q
```

Résultat initial : **5 tests réussis**. Ils vérifient l'égalité numérique et la
sparsité avec une dérivation globale indépendante, avec 1 et 3 threads, huit
points et multiplicateurs aléatoires, des indices permutés, un terme bilinéaire
entre paquets et une variable globale partagée. Ils vérifient aussi la détection
de contraintes oubliées/doublées, le mélange de paquets linéaires et non linéaires,
et une résolution IPOPT comparée à sa Hessienne automatique.

Ces tests établissent le contrat mathématique et logiciel. Ils ne mesurent pas
un gain sur le FHO et ne certifient aucune trajectoire de la campagne.

## Point d'intégration Bioptim identifié

La version locale est dans `.benchmark-deps/bioptim/bioptim/`.
`interfaces/ipopt_interface.py::solve` délègue à
`interfaces/interface_utils.py::generic_solve`. Celui-ci :

1. collecte `raw_objectives` et `raw_g` via les dispatchers ;
2. applique `_shake_penalties_tree` ;
3. construit `interface.nlp` depuis `v`, `sum1(shaked_objectives)` et
   `shaked_constraints` ;
4. appelle `nlpsol(..., options)` en branche non compilée.

L'injection de la fonction personnalisée doit se faire avant cet appel. Mais
**l'extraction des paquets est le vrai travail restant** : il faut conserver le
registre des contributions durant le dispatch, avec l'ordre global exact des
contraintes, les facteurs d'échelle, les poids, les quadratures et les paramètres
de temps. Après le "shake", il faut dériver les expressions transformées, ou
prouver qu'elles ont exactement les mêmes dérivées vis-à-vis des variables
optimisées. Une injection de Hessienne du graphe avant transformation dans le NLP
transformé serait incorrecte sans cet audit.

Le callback devient obsolète dès que la structure, les constantes incorporées
ou l'ordre NLP changent : le cache doit inclure l'identité du partitionnement et
du graphe, pas seulement le nombre de cycles.

## Choix de tailles et audit requis

Sur FHO65, 5 cycles donnent 13 paquets ; 10 cycles donnent 7 paquets dont un
reliquat. Avec 12 threads, 10 cycles ne peuvent pas remplir tous les threads si
l'unité de travail est le paquet. Il faut donc **mesurer 5 et 10**, et éventuellement
un paquet plus petit, plutôt que supposer que 10 est optimal. Le reliquat peut
former un second groupe de noyaux. Si chaque paquet devient un noyau distinct,
le regroupement par `Function` ne fournit aucun parallélisme entre ces groupes :
la réutilisation d'un noyau paramétré est essentielle.

Les couplages ne sont pas nécessairement limités aux continuités : durée optimisée,
paramètres globaux, critères multi-nœuds et périodicité peuvent aussi relier des
paquets. Il faut inclure ces termes dans des paquets de frontière/globaux. Une
continuité affine seule apporte une Hessienne nulle.

Les noyaux locaux doivent être séquentiels pour éviter un ThreadMap imbriqué avec
la parallélisation Bioptim. Fixer BLAS/OpenMP à 1, l'affinité et le nombre de threads
externes pour le premier benchmark. Conserver le même solveur linéaire, même seed,
mêmes tolérances et même état `(x,lambda,sigma,p)` pour comparer les évaluations.

Avant un solve FHO expérimental : comparer `f`, `g`, `grad(f)`, `Jac(g)` et Hessienne
sur plusieurs points incluant seed, résultat et points perturbés ; vérifier la
différence absolue/relative et les produits Hessienne-vecteur ; tester des
multiplicateurs aléatoires et `lam_f=0` pour isoler la partie contraintes. Contrôler
la couverture unique des contributions objectif, qui ne peut pas être déduite des
seuls indices de contraintes.

Le module construit l'union des sparsités locales. Des annulations algébriques
entre contributions peuvent rendre cette union plus large que la sparsité du
graphe monolithique simplifié : ne pas supprimer ces entrées par simple supposition.
Documenter `nnz` et accepter seulement une Hessienne numériquement équivalente.

Conclusion actuelle : prototype exact fonctionnel et testable ; extraction du
vrai NLP et gain FHO encore à établir. Aucune revendication d'accélération issue
des tests synthétiques.
