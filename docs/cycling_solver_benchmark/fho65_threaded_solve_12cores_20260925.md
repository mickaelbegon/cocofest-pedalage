# FHO65 exact MA57 : extension du benchmark à 12 threads

## Protocole et périmètre

Extension du [benchmark 1/8 threads](fho65_threaded_solve_20260925.md), avec le
même harnais `scripts/benchmark_fho_threaded_solve.py`, le même NLP MX final,
la même graine, les mêmes tolérances IPOPT et 30 itérations. CasADi 3.7.2,
Bioptim dans `.benchmark-deps/bioptim`, MA57 et Hessienne exacte. Aucun code C
généré, aucune compilation du FHO. Ce test borné ne produit aucun certificat.

Le run à 12 threads est confiné à CPU16–27 ; les anciennes références 1/8
threads étaient confinées à CPU16–23. `lscpu -e` confirme que CPU16–27
correspondent à 12 cœurs physiques distincts, via un seul sibling SMT par
cœur (siblings CPU0–11). Aucun autre solve lourd n'était actif au lancement.
OMP, OpenBLAS, MKL et NumExpr sont limités à un thread. Les mesures ne sont
pas répétées : la variation de construction peut inclure les effets de charge,
de fréquence CPU et de cache ; elle n'est pas attribuée à une optimisation
de code.

## Mesures

| Temps mural / métrique | 1 thread | 8 threads | 12 threads |
|---|---:|---:|---:|
| Préparation application/OCP | 62.053 s | 258.738 s | 223.318 s |
| Dont dispatch/shake | 11.339 s | 22.097 s | 20.063 s |
| Construction nlpsol | 112.567 s | 98.709 s | 96.280 s |
| Préparation + construction | 174.620 s | 357.447 s | 319.598 s |
| Résolution IPOPT externe, 30 itérations | 396.489 s | 208.115 s | 193.791 s |
| Résolution CasADi interne | 396.293 s | 207.228 s | 192.873 s |
| Hessienne, 30 appels | 281.291 s | 154.059 s | 144.683 s |
| Hessienne, moyenne/appel | 9.376 s | 5.135 s | 4.823 s |
| Jacobienne, 32 appels | 85.494 s | 24.743 s | 20.464 s |
| Contraintes, 32 appels | 5.153 s | 2.391 s | 2.168 s |
| Total harnais, audits/exports compris | 571.877 s | 566.322 s | 514.116 s |
| Pic RSS | 27.165 GiB | 22.931 GiB | 22.912 GiB |

Le passage de 8 à 12 threads réduit le temps de solve de **6.88 %**, celui de
la Hessienne de **6.09 %** et celui de la Jacobienne de **17.29 %**. Le total
mesuré baisse de **9.22 %**. La baisse de préparation de 37.849 s doit être
distinguée de la parallélisation des évaluations : aucune modification de
construction n'a été faite pour ce run. Par rapport au run série, le solve
baisse de **51.12 %** et le total de **10.10 %**.

Le parallélisme natif a donc un rendement décroissant au-delà de 8 threads
sur ce cas. Le choix 12 threads reste un candidat utile quand 12 cœurs sont
disponibles ; il ne garantit pas un gain de 50 % par rapport à 8 threads et
ne démontre pas un temps de convergence complet. La Hessienne représente
encore 74.7 % du solve externe à 12 threads. La Jacobienne représente 10.6 %,
ce qui borne le levier d'une optimisation portant uniquement sur elle.

Le log `initial_guess_preparation_time_s = 203.167618` précède le dispatch
et représente la majorité de la préparation application/OCP. Cette étiquette
inclut le travail effectué depuis le début de préparation de l'application ;
il faut profiler ce périmètre avant de l'attribuer exclusivement à la graine.

## Équivalence vérifiée

Les comparaisons 1/12 et 8/12 passent exactement sur les sondes sauvegardées :

- 265223 variables, 259414 contraintes, 1833064 non-zéros de Jacobienne,
  1072504 non-zéros de Hessienne triangulaire supérieure ;
- mêmes x0, bornes de variables et sparsité de Hessienne ;
- mêmes triplets bornes/valeurs initiales des contraintes après permutation ;
- même objectif initial, même historique IPOPT d'objectif aux 31 points ;
- même vecteur final x, différence maximale **0.0**.

Les trois runs atteignent `Maximum_Iterations_Exceeded`, avec objectif
21547.08906839423 et violation maximale de contraintes 63.00230049263256.
Les sorties demeurent explicitement non certifiées.

## MA57 à 12 threads CasADi

Les temps muraux internes IPOPT sont : symbolique **2.798 s**, factorisation
**15.929 s**, substitution **2.737 s**, soit **21.464 s**. Les évaluations de
fonctions totalisent **167.953 s** selon IPOPT. Le parallélisme des évaluations
CasADi ne change donc pas sensiblement le coût de MA57, maintenu séquentiel.

## Données et reproduction

Rapport : `local-results/fho65-threaded-solve-20260925/threads-12-hsl/report.json`.
Log : `local-results/fho65-threaded-solve-20260925/threads-12-hsl.log`.
Sondes : `initial-probe.npz` et `final-probe-uncertified.npz` dans le run.
Comparaisons : `comparison-1-vs-12.json`, `comparison-8-vs-12.json`, même dossier.
Le champ de projection d'amortissement 8/12 est négatif parce que le run 12
présente déjà une préparation mesurée plus courte ; il signifie qu'aucun
surcoût de préparation n'est à amortir dans cette paire de mesures.

```bash
env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cocofest-mpl-fho65 \
  LD_LIBRARY_PATH=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/lib \
  taskset -c 16-27 /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/benchmark_fho_threaded_solve.py \
  --base-command local-results/fho65-native-hessian-audit-20260925/threads-8-ones/command.json \
  --output-dir local-results/fho65-threaded-solve-reproduction/threads-12 \
  --threads 12 --iterations 30 \
  --hsl-library /home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
```

Le harnais refuse un dossier existant. Pour reproduire les contrôles numériques,
utiliser `scripts/compare_fho_threaded_solve.py` avec la référence 1 ou 8 threads
puis le dossier 12 threads, et un nouveau chemin `--output`.
