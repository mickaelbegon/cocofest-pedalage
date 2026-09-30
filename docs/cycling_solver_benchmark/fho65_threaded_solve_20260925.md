# FHO65 : benchmark de résolution exacte bornée, 1 contre 8 threads

## Protocole

Harnais : `scripts/benchmark_fho_threaded_solve.py`. Données brutes :
`local-results/fho65-threaded-solve-20260925/`.

Le harnais intercepte le NLP MX final après le dispatch et le shake Bioptim.
Il lance directement IPOPT/MA57 avec Hessienne exacte et un plafond identique
de 30 itérations. Les variables, tolérances, préfixe FHO62 et graine FHO65
proviennent du même `command.json` de l'audit natif antérieur. Les multiplicateurs
initiaux sont nuls dans les deux variantes afin de respecter la permutation
des contraintes entre le dispatch série et le dispatch threadé.

Les variantes sont exécutées séquentiellement sur les mêmes CPU16–23.
OMP/OpenBLAS/MKL/NumExpr sont limités à un thread. Aucun C généré, aucun
fichier de campagne modifié. Le résultat est un benchmark borné, pas un
certificat FHO, même si IPOPT venait à converger avant le plafond.

La préparation OCP, la construction `nlpsol`, l'audit numérique initial et
la résolution sont chronométrés séparément. Les statistiques CasADi conservent
le nombre et le temps des callbacks, ainsi que le statut et l'historique IPOPT.
Les vecteurs initiaux/finals et les bornes des contraintes sont conservés pour
contrôler l'équivalence en tenant compte de leur permutation.

Les temps muraux de callbacks threadés doivent être comparés avec leurs nombres
d'appels ; une divergence d'itérés due à l'ordre différent des contraintes peut
changer le temps total sans invalider le modèle. Un plafond de 30 itérations
mesure le coût d'un budget fixe et ne prédit pas à lui seul le temps de convergence.

## Résultats mesurés

Les deux solves terminent à `Maximum_Iterations_Exceeded`, 30 itérations.
Les rapports valides sont dans `threads-1-hsl` et `threads-8-hsl`.

| Mesure | 1 thread | 8 threads |
|---|---:|---:|
| Préparation application/OCP | 62.053 s | 258.738 s |
| Dont dispatch/shake | 11.339 s | 22.097 s |
| Construction nlpsol | 112.567 s | 98.709 s |
| Préparation + construction | 174.620 s | 357.447 s |
| Solve, chronométrage externe | 396.489 s | 208.115 s |
| Solve, chronométrage CasADi interne | 396.293 s | 207.228 s |
| Hessienne, 30 appels | 281.291 s | 154.059 s |
| Hessienne, moyenne par appel | 9.376 s | 5.135 s |
| Jacobienne, 32 appels | 85.494 s | 24.743 s |
| Contraintes, 32 appels | 5.153 s | 2.391 s |
| Total harnais, audit et exports compris | 571.877 s | 566.322 s |
| Pic RSS | 27.165 GiB | 22.931 GiB |

Le solve borné est **47.5 % plus court** sur 8 threads. La Hessienne baisse
de **45.2 %**, la Jacobienne de **71.1 %**. Le surcoût de préparation et de
construction est de **182.827 s** : sur ce budget de 30 itérations, le total
ne baisse que de **5.554 s (1.0 %)**. Un écart de cette ampleur en un seul
passage ne démontre pas un avantage robuste de bout en bout ; on est proche
du point d'amortissement. La projection linéaire locale donne environ **29.1
itérations** pour amortir le surcoût grâce à H/J et au reste du solve. Cette
projection diffère des 43 appels du précédent audit, qui ne comptait que H.
Une résolution convergée plus longue doit être mesurée avant de chiffrer son
gain de convergence complet.

L'objectif final vaut **21547.08906839423**, et la violation maximale des
contraintes non mises à l'échelle vaut **63.00230049263256**, identiques dans
les deux variantes. Cette violation confirme que ces sorties ne sont pas
des solutions FHO certifiées. Le plafond était volontaire pour mesurer un
budget comparable.

## Comparabilité numérique et structurelle

Le script `scripts/compare_fho_threaded_solve.py` reproduit la comparaison
dans `comparison.json`. Les contrôles suivants passent exactement :

- 265223 variables, 259414 contraintes, 1833064 non-zéros de Jacobienne,
  1072504 non-zéros du triangle supérieur de Hessienne CasADi ;
- même sparsité de Hessienne, mêmes x0 et bornes des variables ;
- même objectif initial (**25749.139193584993**) ;
- même ensemble de triplets `(borne inférieure, borne supérieure, g(x0))`
  après tri, ce qui prend en compte la permutation des contraintes ;
- même historique d'objectif aux 31 points IPOPT ;
- même vecteur final x, différence maximale **0.0**.

La violation initiale auditée est **0.006000236490365296** dans les deux cas.
Ces observations vérifient l'équivalence des entrées et du chemin numérique
de ce test, sans constituer une preuve pour toute formulation/initialisation.

## Coût réel de MA57

Les statistiques internes IPOPT mesurent séparément les étapes linéaires :

| Temps mural | 1 thread | 8 threads |
|---|---:|---:|
| Symbolique | 2.662 s | 2.885 s |
| Factorisation | 15.117 s | 16.027 s |
| Substitution | 2.635 s | 2.925 s |
| Total de ces trois postes | 20.414 s | 21.837 s |
| Algorithme IPOPT hors évaluations | 25.800 s | 25.454 s |
| Évaluations des fonctions | 370.452 s | 181.732 s |

La factorisation n'est donc pas la source du gain. Sur ce test, elle ne
représente avec symbolique/substitution que 5.2 % du temps du solve série.
Les threads numériques étant fixés à 1, ce test isole le parallélisme CasADi.

## Incident de lancement conservé

Le premier répertoire `threads-1` est invalide pour la comparaison : après
construction du NLP, le chargement implicite de `libhsl.so` a échoué, avec
`Invalid_Option` et zéro appel à H. Il reste conservé pour la traçabilité.
Le harnais a été corrigé avec `--hsl-library` explicite et un minuscule solve
MA57 de préflight avant tout grand build. Les deux mesures valides utilisent
la même bibliothèque
`/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so`.

## Reproduction

Exécuter successivement pour `--threads 1` puis `--threads 8`, en changeant
le répertoire de sortie (le harnais refuse les répertoires existants) :

```bash
env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cocofest-mpl-fho65 \
  LD_LIBRARY_PATH=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/lib \
  taskset -c 16-23 /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/benchmark_fho_threaded_solve.py \
  --base-command local-results/fho65-native-hessian-audit-20260925/threads-8-ones/command.json \
  --output-dir local-results/fho65-threaded-solve-reproduction/threads-1 \
  --threads 1 --iterations 30 \
  --hsl-library /home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
```

Comparaison :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/compare_fho_threaded_solve.py \
  local-results/fho65-threaded-solve-20260925/threads-1-hsl \
  local-results/fho65-threaded-solve-20260925/threads-8-hsl
```

Le benchmark valide un gain substantiel dans le solve natif threadé, et montre
que sa préparation plus lente consomme ce gain pour un petit budget. La
Hessienne par paquets devra battre cette référence native à 8 threads ; ces
résultats n'évaluent pas encore une callback par paquets.
