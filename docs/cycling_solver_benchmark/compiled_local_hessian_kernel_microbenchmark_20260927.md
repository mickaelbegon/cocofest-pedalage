# Noyau Hessien local exact compilé, FHO global MX : microbenchmark du 27 septembre 2026

## Question évaluée

Peut-on compiler les très petits noyaux répétés d'un FHO tout en gardant le
graphe global FHO en MX, non compilé ? Oui, pour une **callback Hessienne
explicite** : on peut générer en C une fonction locale qui prend les variables
de deux frontières et les contrôles d'un stage, ainsi que les multiplicateurs
locaux et `sigma`, et qui renvoie directement la Hessienne exacte locale du
Lagrangien. Les blocs sont ensuite évalués en parallèle et additionnés aux
frontières d'état partagées.

Ce n'est pas l'équivalent de remplacer une sous-expression MX primaire par
`external(...)` puis demander à CasADi de la différencier : cette seconde
approche ne fournit pas les dérivées analytiques de l'externe. Le résultat de
ce test est déjà différencié avant codegen et devient la sortie de la callback
`hess_l`; il préserve donc le caractère exact de la Hessienne.

## Protocole

Le noyau, identique au microbenchmark de paquets précédent, a 12 variables
locales (4 états gauche, 4 états droite, 4 contrôles) et 4 contraintes
non-linéaires, avec termes `sin` et `exp`. Il est répété sur 180 stages, soit
1 444 variables globales et 23 056 non-zéros après coalescence. Les blocs
locaux se chevauchent seulement aux frontières d'état, et sont additionnés
dans une matrice CSR.

CasADi construit la Hessienne locale exacte symboliquement, puis le seul
noyau de sortie Hessienne est généré en C et compilé avec `gcc -O3 -fPIC
-shared`. Les deux variantes reçoivent strictement les mêmes valeurs
`(y, lambda, sigma)`. Les temps sont la médiane de 15 appels après
échauffement, avec OMP/BLAS/MKL/NUMEXPR fixés à un thread. Ce n'est ni un
solve IPOPT, ni une transcription FHO physiologique.

La commande reproductible est :

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/benchmark_compiled_local_hessian_kernel.py \
  --stages 180 --workers 1 8 12 --repeats 15 \
  --output docs/cycling_solver_benchmark/compiled_local_hessian_kernel_microbenchmark_20260927.json
```

Les mesures brutes sont dans
[`compiled_local_hessian_kernel_microbenchmark_20260927.json`](compiled_local_hessian_kernel_microbenchmark_20260927.json).

## Résultats

| Workers demandés | VM : évaluation locale | C : évaluation locale | Gain local | VM : éval. + assemblage | C : éval. + assemblage | Gain bout-en-bout |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18,19 ms | 0,592 ms | ×30,7 | 20,07 ms | 2,249 ms | ×8,93 |
| 8 | 9,98 ms | 0,484 ms | ×20,6 | 12,21 ms | 2,270 ms | ×5,38 |
| 12 | 20,57 ms | 0,554 ms | ×37,1 | 20,08 ms | 2,313 ms | ×8,68 |

L'erreur absolue maximale VM/C est **exactement 0** pour les trois cas. La
compilation génère une bibliothèque de seulement 82,6 kB et prend 1,95 s.
Selon le niveau de parallélisme, ce coût est amorti après 97–205 appels de la
callback seule, ou 109–196 appels avec l'assemblage Python inclus. Un FHO à
plusieurs centaines/milliers d'itérations peut donc l'amortir *si* le chemin
de production possède un assemblage sparse natif assez léger.

Le 12-workers est moins bon que 8 pour le noyau interprété sur cette très
petite charge : c'est le surcoût des tâches. Le noyau C est déjà si court que
son bénéfice de parallélisation est marginal. Ce résultat suggère de choisir
la granularité (stages par paquet) après mesure du FHO réel, plutôt que de
forcer douze petites tâches à chaque appel.

## Conséquence pour le FHO

Le coût de Hessienne de FHO_91 (74,6 % du solveur) fait de cette architecture
une piste à fort potentiel, mais **ce gain n'est pas encore un gain FHO**. Une
implémentation utilisable doit fournir, avec le même ordre/sparsité que IPOPT :

1. `f`, `g`, `grad_f`, `jac_g` et une `hess_l` exacte, pas seulement le bloc
   Hessien; chaque callback peut compiler ses propres noyaux locaux exacts;
2. une addition sparse native (C++/CasADi callback), sans conversion
   Python/Scipy et sans densifier les blocs;
3. une vérification aux états et multiplicateurs de plusieurs certificats FHO
   que valeurs, triplets et ordre de `hess_l` sont identiques à la callback
   native MX;
4. un A/B FHO identique : même graine, MA57, tolérances, CPU, itérations,
   faisabilité, objectif et pic mémoire.

Tant que ces quatre points ne sont pas validés, le FHO doit conserver le
chemin MX exact actuel. La prochaine expérimentation raisonnable est une
capture **sans solve** de la callback `nlp_hess_l` de FHO65, puis une
implémentation prototype de ses seuls blocs de dynamique/collocation. Les
termes globaux (bornes, raccords de cycles, objectifs) restent assemblés MX.

## Fichiers

- Script : [`scripts/benchmark_compiled_local_hessian_kernel.py`](../../scripts/benchmark_compiled_local_hessian_kernel.py)
- Données brutes : [`compiled_local_hessian_kernel_microbenchmark_20260927.json`](compiled_local_hessian_kernel_microbenchmark_20260927.json)
- Expérience antérieure de paquets MX : [`casadi_packed_hessian_microbenchmark_20260925.md`](casadi_packed_hessian_microbenchmark_20260925.md)
