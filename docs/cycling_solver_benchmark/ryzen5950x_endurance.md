# Benchmark d'endurance sur AMD Ryzen 9 5950X

Ce protocole compare les trois meilleures chaînes reduced actuelles jusqu'à
`2000` RHO ou jusqu'à deux échecs consécutifs du même RHO physique :

| Chaîne | Formulation et transcription | Recovery |
|---|---|---|
| IPOPT | reduced, SX, Radau 5, évaluateurs C, MUMPS | seconde chance sur le même RHO; aucun autre solveur |
| MadNLP | reduced, SX, Radau 5, évaluateurs C, MUMPS, plafond chaud `73` itérations et `20 s` | IPOPT/MUMPS Radau 5 certifie le même RHO après les tentatives MadNLP |
| ACADOS | reduced, SQP/IRK, 4 stages, 5 steps, borne terminale absolue `omega = -2*pi +/- 0.3 rad/s` | IPOPT/MUMPS Radau 5 certifie exceptionnellement le même RHO, puis ACADOS reprend |

Le script est
[`run_ryzen5950x_endurance_sweep.sh`](../../.github/scripts/run_ryzen5950x_endurance_sweep.sh).
Il est reprenable : un résultat ayant atteint l'horizon, un arrêt de fatigue
accepté ou un double échec sérialisé n'est pas relancé, sauf avec
`FORCE_RERUN=true`.

## Convention de couple

La vitesse nominale du pédalier est négative. Une valeur CLI
`signed:+0.15` est donc un couple **résistant** de `0.15 N.m`; elle ne doit pas
être remplacée par `0.15`, car la forme non signée représente une assistance.
Le sweep par défaut utilise `+0.10`, `+0.15` et `+0.20 N.m`. Le dernier cas
peut échouer très tôt : c'est un résultat du benchmark, pas une erreur CI.

Chaque JSON distingue :

- RHO demandés, tentés et certifiés;
- première fenêtre échouée et classe de l'arrêt;
- temps solveur, boucle RHO et préparation initiale;
- temps et nombre de recoveries/fallbacks;
- objectif de fatigue, AUC et capacité finale des quatre muscles;
- résidus, continuité, phase absolue, bornes de cadence et patrons de PW.

## Faut-il utiliser 30 threads ?

Le 5950X possède `16` cœurs physiques et `32` threads SMT. La baseline utilise
donc `RHO_THREADS=16`, avec BLAS, OpenMP imbriqué et Julia à un thread. Le
paramètre `--n-threads` parallélise surtout les évaluations CasADi par
stage/map; il ne transforme pas automatiquement MUMPS en factoriseur à 30
threads. Utiliser directement `30` peut accélérer certaines évaluations, mais
peut aussi ralentir les factorisations et augmenter la variabilité par
contention SMT.

Avant l'endurance, comparer `16` et `30` sur 30 RHO, trois répétitions, sans
autre charge sur la machine. Retenir `30` seulement si sa médiane et son P90
de boucle RHO sont meilleurs sans changer les itérations ni le résultat
scientifique. Ne jamais faire tourner deux solveurs simultanément pendant
cette calibration ou la campagne finale.

## Environnements

Suivre d'abord les sections 4 à 10 de
[`linux_32core_setup.md`](linux_32core_setup.md). Deux environnements restent
nécessaires à cause des ABI CasADi différentes :

- `cocofest-rho32` pour IPOPT et ACADOS;
- `cocofest-madnlp32` pour MadNLP/MUMPS et son recovery IPOPT.

Sur la branche actuelle :

```bash
git clone --branch codex/full-horizon-homotopy \
  https://github.com/mickaelbegon/cocofest-pedalage.git cocofest
export COCOFEST_ROOT="$PWD/cocofest"
cd "$COCOFEST_ROOT"
```

Les trois fichiers `benchmark-seed/common-reduced.npz`,
`benchmark-seed/common-full.npz` et
`benchmark-seed/reduced-cycling-fourier12.npz` doivent exister. Le script
reconstruit en plus un seed ACADOS natif pour **chaque résistance** afin de ne
pas attribuer à un solveur une branche initiale provenant d'un autre couple.

## Calibration 16 contre 30 threads

Dans l'environnement IPOPT/ACADOS :

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for repeat in 1 2 3; do
  for threads in 16 30; do
    OUTPUT_ROOT="$COCOFEST_ROOT/thread-calibration/r${repeat}-t${threads}" \
    MAX_RHOS=30 RHO_THREADS="$threads" RESISTANCES_NM="0.15" \
    STRATEGIES="ipopt acados-ipopt" \
      bash .github/scripts/run_ryzen5950x_endurance_sweep.sh
  done
done
```

Dans l'environnement MadNLP :

```bash
conda activate cocofest-madnlp32
cd "$COCOFEST_ROOT"
export PATH="${HOME}/.juliaup/bin:${HOME}/.julia/bin:$PATH"
export LD_LIBRARY_PATH="$COCOFEST_ROOT/.cache/madnlp-mumps/lib:$COCOFEST_ROOT/.cache/madnlp-mumps/share/julia/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for repeat in 1 2 3; do
  for threads in 16 30; do
    OUTPUT_ROOT="$COCOFEST_ROOT/thread-calibration/r${repeat}-t${threads}" \
    MAX_RHOS=30 RHO_THREADS="$threads" RESISTANCES_NM="0.15" \
    STRATEGIES="madnlp" \
      bash .github/scripts/run_ryzen5950x_endurance_sweep.sh
  done
done
```

Comparer les six fichiers `endurance-summary.csv`, mais vérifier également
dans les JSON que les itérations, l'objectif, la fatigue et les audits sont
équivalents. La compilation initiale et la construction de l'OCP sont hors de
la métrique `execution_timing.rho_solve_loop_wall_time_s`.

## Campagne d'endurance

Choisir le meilleur nombre de threads mesuré, ici noté `T`. Les deux commandes
écrivent dans le même répertoire sans écraser les autres stratégies.

IPOPT et ACADOS avec fallback IPOPT :

```bash
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

OUTPUT_ROOT="$COCOFEST_ROOT/benchmark-results/ryzen5950x-resistance-sweep" \
MAX_RHOS=2000 RHO_THREADS=T RESISTANCES_NM="0.10 0.15 0.20" \
STRATEGIES="ipopt acados-ipopt" \
  bash .github/scripts/run_ryzen5950x_endurance_sweep.sh
```

MadNLP/MUMPS avec fallback IPOPT :

```bash
conda activate cocofest-madnlp32
cd "$COCOFEST_ROOT"
export PATH="${HOME}/.juliaup/bin:${HOME}/.julia/bin:$PATH"
export LD_LIBRARY_PATH="$COCOFEST_ROOT/.cache/madnlp-mumps/lib:$COCOFEST_ROOT/.cache/madnlp-mumps/share/julia/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

OUTPUT_ROOT="$COCOFEST_ROOT/benchmark-results/ryzen5950x-resistance-sweep" \
MAX_RHOS=2000 RHO_THREADS=T RESISTANCES_NM="0.10 0.15 0.20" \
STRATEGIES="madnlp" \
  bash .github/scripts/run_ryzen5950x_endurance_sweep.sh
```

Le tableau final est
`benchmark-results/ryzen5950x-resistance-sweep/endurance-summary.csv`. Les
répertoires conservent aussi les JSON, logs, trajectoires NPZ, checkpoints aux
RHO `100, 300, 600, 1000, 1500, 2000` et la configuration matérielle.

## Interprétation

L'arrêt après deux échecs est une donnée, mais n'est déclaré
`fatigue_limited_candidate` que si le classifieur trouve aussi une capacité
réduite, une stimulation saturée ou une autre preuve de recrutement/fatigue.
Un `unconfirmed_endurance_stop` indique un problème numérique non attribuable
à la fatigue. Le nombre de cycles maximal ne doit jamais être comparé sans
ces deux étiquettes.

Pour MadNLP, rapporter séparément les succès natifs et les RHO avancés par
IPOPT. Pour ACADOS, rapporter séparément le temps nominal, le temps des
recoveries et les changements de branche PW au passage ACADOS--IPOPT--ACADOS.
Le coût moyen recovery inclus est plus pertinent pour la robustesse; la
médiane ACADOS seule décrit seulement le fast path.
