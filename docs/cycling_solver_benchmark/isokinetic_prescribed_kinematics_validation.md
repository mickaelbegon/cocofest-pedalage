# Validation de la cinématique isocinétique prescrite

La variante `isokinetic_kinematics=prescribed` remplace les états `theta` et
`omega` par `theta(t) = theta_start + omega_target * (t - time_origin)` et
`omega(t) = omega_target`. Elle conserve les cinq états Ding par muscle et
`E_prod`. La référence de comparaison est le modèle
`isokinetic_kinematics=states`, avec le même profil réduit, les mêmes paramètres
Ding, commandes, origine angulaire, horizon et transcription Radau 5.

## Contrôles numériques

1. Vérifier les dimensions, l'ordre des états et la sérialisation : pour quatre
   muscles, 23 états en mode `states` et 21 en mode `prescribed` ; les 21 états
   restants doivent être dans le même ordre. Contrôler aussi le cas bilatéral
   (43 et 41 états) et la combinaison avec les états de slew si elle est exposée.
2. À plusieurs instants, y compris des temps de collocation à l'intérieur d'un
   intervalle, évaluer les deux membres droits sur les mêmes 20 états Ding,
   `E_prod`, forces et PW. Reconstituer `theta/omega` de la référence par la
   loi prescrite, puis comparer toutes les 21 dérivées conservées. Le test doit
   utiliser des coefficients Fourier non constants pour détecter une erreur de
   phase ; tolérance absolue et relative de 1e-10 pour ce calcul symbolique.
3. Avec un jeu de PW figé, intégrer un cycle complet dans les deux formulations
   à Radau 5. Comparer `E_prod`, les états Ding, la charge inverse, le travail
   cumulé et la fatigue à chaque nœud. Toute trajectoire reconstruite doit
   respecter les mêmes contraintes physiques, notamment la cible de travail et
   les bornes de charge.
4. Résoudre un petit OCP avec le **même seed physique** et les mêmes options
   IPOPT/MA57. Comparer objectif, résidu de contraintes, certification dense,
   trajectoires PW et fatigue. Un autre optimum local est possible : dans ce
   cas, comparer d'abord l'objectif et la faisabilité de la solution de chaque
   formulation réévaluée dans l'autre formulation.

## Temps

Pour la comparaison de performance, lancer au moins 100 cycles RHO par variante
sur les mêmes CPU, avec un seul thread BLAS/OpenMP par solveur et le même seed
certifié. Exclure la compilation, la construction de l'OCP et le premier cycle
des statistiques chaudes. Rapporter séparément la médiane et le P90 du temps
solveur, de l'itération RHO complète, le nombre d'itérations IPOPT et le nombre
de cycles certifiés. Alterner l'ordre des variantes ou effectuer deux paires de
campagnes afin de mesurer la variabilité du CPU. Le gain n'est établi que si les
trajectoires et contraintes physiques restent comparables et si le temps RHO
complet diminue.

## Smoke vérifié le 30 septembre 2026

Le même cas unilatéral réduit isocinétique a été résolu avec IPOPT/MA57, SX,
Radau 5, 30 stimulations/cycle, couple énergétique équivalent de 0,2 Nm,
vitesse de −2π rad/s et bornes de charge de [−3, 3] Nm. Les deux variantes
emploient la même valeur initiale physique issue du générateur standard.
Quatre fenêtres RHO sont certifiées dans chaque cas, y compris les audits
DOP853 de cadence et de charge.

| Cinématique | États | Solveur chaud médiane / P90 | Itération RHO complète médiane / P90 |
|---|---:|---:|---:|
| `states` | 23 | 0,767 / 0,836 s | 0,853 / 0,922 s |
| `prescribed` | 21 | 0,442 / 0,490 s | 0,526 / 0,571 s |

Le premier cycle est exclu des statistiques chaudes. Le temps d'itération
complète mesure un cycle de la boucle RHO, incluant transfert, solveur et
certification ; le temps solveur est celui rapporté par IPOPT. Sur ces quatre
cycles, l'objectif cumulé vaut respectivement 0,178203774563 et
0,178203774190. L'écart maximal des PW est de 3,48·10⁻⁶ µs, celui de la
charge inverse de 1,56·10⁻⁹ Nm, et celui de l'énergie de 7,73·10⁻¹⁰ J.
L'écart maximal des capacités finales `A` est de 1,57·10⁻⁸. Cet échantillon
court confirme la cohérence, pas encore la stabilité du gain sur 100 cycles.

Commande pour reproduire chaque variante, depuis la racine du dépôt ; choisir
successivement `states` et `prescribed` pour `MODE` et un fichier de sortie
distinct :

```bash
MODE=prescribed
env PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/mpl-cocofest \
  python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
  --solvers ipopt --mechanical-formulation reduced \
  --formulation isokinetic --isokinetic-kinematics "$MODE" \
  --n-windows 4 --cycles-per-window 1 --stimulations-per-cycle 30 \
  --compact-rho-output --ipopt-linear-solver ma57 \
  --ipopt-hsl-library /chemin/vers/libhsl.so --n-threads 1 \
  --output-json "/tmp/isokinetic-${MODE}-rho4.json"
```
