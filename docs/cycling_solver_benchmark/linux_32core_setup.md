# Installation et benchmark sur Linux 32 cœurs

Cette procédure reproduit les piles numériques du workflow
[`cycling_solver_benchmark_linux.yml`](../../.github/workflows/cycling_solver_benchmark_linux.yml)
sur une machine x86-64 dédiée.

## 0. Chemin rapide

Les sections suivantes décrivent chaque étape en détail et restent la référence
pour diagnostiquer une installation. Sur un hôte déjà préparé (section 4) avec
Miniforge (section 5), l'installation complète tient en trois commandes :

```bash
git clone --branch codex/full-horizon-homotopy \
  https://github.com/mickaelbegon/cocofest-pedalage.git cocofest
cd cocofest

bash .github/scripts/install_benchmark_stack_linux.sh rho32
bash .github/scripts/install_benchmark_stack_linux.sh madnlp32   # requiert Julia, section 8.1
```

`install_benchmark_stack_linux.sh` enchaîne le clone des sources épinglées, la
création de l'environnement Conda, CasADi, biorbd-CasADi, Bioptim, puis ACADOS
ou libMad/MadNLP selon la pile. Chaque étape est ignorée si son artefact existe
déjà, donc le script se relance sans refaire les compilations longues :

```bash
bash .github/scripts/install_benchmark_stack_linux.sh rho32 --list-steps
bash .github/scripts/install_benchmark_stack_linux.sh rho32 --force-step acados
bash .github/scripts/install_benchmark_stack_linux.sh rho32 --jobs 16
```

Il termine par la validation de la section 7.3 ou 8.4. Ensuite :

```bash
source .github/scripts/benchmark_env.sh rho32     # section 9.1
bash .github/scripts/build_benchmark_seed_linux.sh  # section 10
```

En cas d'échec, reprendre la section correspondante puis relancer le script.
La section 16 recense les pannes déjà rencontrées et leurs signatures.

## 1. Architecture recommandée

Utiliser deux environnements Conda construits depuis le même fichier de base :

| Environnement | Solveurs | CasADi | ABI C++ |
|---|---|---|---:|
| `cocofest-rho32` | IPOPT, FATROP, ACADOS | roue officielle `3.7.2` | `0` |
| `cocofest-madnlp32` | MadNLP/MUMPS et IPOPT de préparation | source `3.7.2` avec libMad | `1` |

Il ne faut pas installer la roue CasADi officielle dans l'environnement
MadNLP après sa compilation : elle écraserait le plugin et pourrait rendre
biorbd-CasADi incompatible. De même, biorbd-CasADi doit être recompilé dans
chaque environnement avec l'ABI de son CasADi.

## 2. Versions épinglées

| Composant | Version ou commit |
|---|---|
| Ubuntu | `24.04` recommandé, `26.04` validé |
| GCC | `13` (CI) ou `15` (Ubuntu 26.04) |
| Python | `3.11` |
| CasADi officiel | `3.7.2` |
| CasADi MadNLP | `973b086f4dcda9f49cd9c1948432ae4b7ee54886` |
| Bioptim | `f7a0d722526967d9a81a8ad596ddb911d32a0bfe` |
| Branche Bioptim | `codex/cocofest-acados-v055-exploration` |
| ACADOS | `59d93e17d2985fdd73fc58b8a83ed8f83a024171` |
| libMad | `5529f23a6bff33c566ad954da38d352f1f172356` |
| Julia | `1.12.6` |
| JuliaC | `73be8587a80bbb65dab7acd71d406f72867a3571` |
| biorbd | `Release_1.12.2` |
| RBDL-CasADi | `93475e2ea9bc87f37709a2312533ce3187f054b9` |

La branche MadNLP exige un `libgcc_s.so.1` exportant `GCC_13.0.0`. Ubuntu
24.04 satisfait le contrat utilisé par la CI, Ubuntu 26.04 également. Une
distribution plus ancienne doit réussir le script `check_libmad_host_linux.sh`
avant toute compilation.

Sur un hôte plus récent que l'image CI, deux points de compatibilité sont déjà
pris en charge par les scripts et ne demandent aucune intervention manuelle :

- `biorbd Release_1.12.2` utilise `SIZE_MAX` sans inclure `<cstdint>`, que GCC
  15 ne fournit plus transitivement. `install_biorbd_casadi_linux.sh` corrige
  son checkout temporaire;
- le `qpOASES_e` vendu par ACADOS `0.5.5` passe un `Constraints**` à une
  fonction attendant un `Constraints*`. GCC 13 le signalait par un
  avertissement, GCC 14 en fait une erreur. `install_acados_linux.sh` rétablit
  le diagnostic en avertissement, donc exactement la traduction produite par la
  CI.

## 3. Dimensionnement de la machine

- x86-64 Linux;
- 32 CPU disponibles dans le cpuset du processus;
- au moins 32 Gio de RAM pour le benchmark RHO; 64 Gio donnent davantage de
  marge pour les compilations simultanées;
- environ 40 Gio libres pour Miniforge, Julia, les sources, builds et
  artefacts;
- accès réseau à GitHub, conda-forge, PyPI et aux serveurs Julia.

Le sweep full horizon n'est pas inclus dans cette estimation. Son problème
actuel est numérique avant d'être mémoire; une machine 128 Gio ne le corrige
pas sans meilleure seed multi-cycle.

## 4. Préparer Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl git git-lfs jq \
  librhash-dev ninja-build pkg-config tar unzip wget xz-utils

git lfs install

uname -a
uname -m
lscpu
nproc
free -h
df -h
cc --version
```

Vérifier que `nproc` retourne 32 dans le shell qui lancera le benchmark. Sur
un ordonnanceur, `nproc --all` peut afficher la machine entière tandis que
`nproc` reflète correctement le cpuset alloué.

## 5. Installer Miniforge

Si Conda/Miniforge est déjà installé, ignorer cette section et vérifier que
`conda config --show channels` utilise uniquement `conda-forge`.

```bash
export MINIFORGE_PREFIX="${HOME}/miniforge3"

curl -fsSLo /tmp/Miniforge3.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash /tmp/Miniforge3.sh -b -p "$MINIFORGE_PREFIX"
source "$MINIFORGE_PREFIX/etc/profile.d/conda.sh"

conda config --set channel_priority strict
conda config --remove-key channels 2>/dev/null || true
conda config --add channels conda-forge
conda --version
```

Ajouter la ligne suivante au fichier d'initialisation du shell si nécessaire :

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
```

## 6. Cloner les sources épinglées

Choisir un chemin absolu accessible par l'utilisateur du benchmark :

```bash
export RHO_WORK_ROOT="/chemin/absolu/vers/rho-work"
mkdir -p "$RHO_WORK_ROOT"
cd "$RHO_WORK_ROOT"

git clone --branch codex/full-horizon-homotopy \
  https://github.com/mickaelbegon/cocofest-pedalage.git cocofest
export COCOFEST_ROOT="$RHO_WORK_ROOT/cocofest"

mkdir -p "$COCOFEST_ROOT/.benchmark-deps"

git clone --recurse-submodules \
  https://github.com/mickaelbegon/BiorbdOptim.git \
  "$COCOFEST_ROOT/.benchmark-deps/bioptim"
git -C "$COCOFEST_ROOT/.benchmark-deps/bioptim" checkout \
  f7a0d722526967d9a81a8ad596ddb911d32a0bfe
git -C "$COCOFEST_ROOT/.benchmark-deps/bioptim" submodule update \
  --init --recursive

git clone https://github.com/mickaelbegon/libMad.git \
  "$COCOFEST_ROOT/.benchmark-deps/libMad"
git -C "$COCOFEST_ROOT/.benchmark-deps/libMad" checkout \
  5529f23a6bff33c566ad954da38d352f1f172356

test "$(git -C "$COCOFEST_ROOT/.benchmark-deps/bioptim" rev-parse HEAD)" = \
  f7a0d722526967d9a81a8ad596ddb911d32a0bfe
test "$(git -C "$COCOFEST_ROOT/.benchmark-deps/bioptim/external/acados" rev-parse HEAD)" = \
  59d93e17d2985fdd73fc58b8a83ed8f83a024171
test "$(git -C "$COCOFEST_ROOT/.benchmark-deps/libMad" rev-parse HEAD)" = \
  5529f23a6bff33c566ad954da38d352f1f172356
```

Avant ce clone, les changements locaux doivent être commités et poussés. Une
nouvelle machine ne peut pas récupérer les fichiers non suivis ou non poussés.

Le dépôt et la branche ci-dessus sont ceux de la campagne courante. Les
anciennes procédures pointaient vers `mickaelbegon/cocofest` et la branche
`codex/acados-pr-refresh`; ce couple ne contient ni les scripts d'endurance
Ryzen ni les correctifs de toolchain décrits en section 2.

`bioptim` n'est volontairement pas une dépendance Git du dépôt Cocofest : le
benchmark l'installe depuis ce clone épinglé. Par conséquent, la commande
`python -m pip install --no-deps -e .benchmark-deps/bioptim` n'est valide
qu'après cette section. Si elle affiche *not a valid editable requirement*, ne
pas la relancer : vérifier d'abord que
`$COCOFEST_ROOT/.benchmark-deps/bioptim/pyproject.toml` existe puis reprendre
le clone ci-dessus.

## 7. Environnement IPOPT, FATROP et ACADOS

### 7.1 Créer l'environnement

Équivalent automatisé : `bash .github/scripts/install_benchmark_stack_linux.sh rho32`
couvre les sections 6, 7.1, 7.2 et 7.3. Procédure détaillée :

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
cd "$COCOFEST_ROOT"

conda env create \
  --name cocofest-rho32 \
  --file .github/cycling-benchmark-linux-environment.yml
conda activate cocofest-rho32

export CMAKE_BUILD_PARALLEL_LEVEL=32
export CASADI_CXX_ABI=0
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m pip install --no-deps "casadi==3.7.2"
bash .github/scripts/install_biorbd_casadi_linux.sh
python -m pip install --no-deps -e .benchmark-deps/bioptim
```

Le script biorbd applique automatiquement, dans son checkout temporaire, le
correctif `#include <cstdint>` requis par `biorbd Release_1.12.2` avec GCC 15
(`SIZE_MAX` n'est plus fourni transitivement). Après avoir mis Cocofest à jour,
il suffit donc de relancer ce script; ni le compilateur ni les sources biorbd
installées ne doivent être modifiés manuellement.

Si l'environnement existe déjà, utiliser :

```bash
conda env update \
  --name cocofest-rho32 \
  --file .github/cycling-benchmark-linux-environment.yml \
  --prune
```

### 7.2 Compiler ACADOS 0.5.5

```bash
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"

export CMAKE_BUILD_PARALLEL_LEVEL=32
bash .github/scripts/install_acados_linux.sh .benchmark-deps/bioptim 32 "$CONDA_PREFIX"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Ce script enveloppe `.benchmark-deps/bioptim/external/acados_install_linux.sh`,
qui reste la source de vérité pour la compilation. Il ajoute deux garanties que
l'installateur amont ne fournit pas :

- il détecte les diagnostics que GCC 14 et 15 transforment en erreurs et les
  ramène à des avertissements, ce qui permet à `qpOASES_e` de compiler comme
  sur l'image CI (voir section 2);
- l'installateur amont ne s'exécute pas sous `set -e`. Un `make install` en
  échec y est donc suivi d'un `pip install .` réussi, et le script sort en `0`
  alors que `$CONDA_PREFIX/lib` ne contient aucun `libacados.so`. Le wrapper
  vérifie les artefacts et échoue explicitement.

Ne jamais conclure au succès sur le seul code de sortie de l'installateur amont :
contrôler `libacados.so`, `libhpipm.so` et `libblasfeo.so` comme en section 7.3.

Le script installe `acados_template` dans l'environnement et configure son
préfixe sur `$CONDA_PREFIX`. Il modifie temporairement le sous-module ACADOS,
puis restaure les fichiers suivis à la fin; ne conserver aucun développement
non commité dans ce sous-module pendant l'installation.

### 7.3 Valider l'environnement

```bash
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -f "$CONDA_PREFIX/lib/libacados.so"
test -f "$CONDA_PREFIX/lib/libhpipm.so"
test -f "$CONDA_PREFIX/lib/libblasfeo.so"

python - <<'PY'
import acados_template
import biorbd_casadi
import casadi as cas

print("CasADi", cas.__version__)
print("biorbd", biorbd_casadi.__version__)
print("acados_template", acados_template.__file__)
assert cas.has_nlpsol("ipopt")
assert cas.has_nlpsol("fatrop")
PY

python -m pytest -q \
  tests/shard1/test_solver_backends.py \
  tests/shard1/test_reduced_cycling.py \
  tests/test_benchmark_readme.py
```

## 8. Environnement MadNLP/MUMPS

### 8.1 Installer Julia 1.12.6

```bash
curl -fsSL https://install.julialang.org | \
  sh -s -- --yes --default-channel 1.12.6

export PATH="${HOME}/.juliaup/bin:${HOME}/.julia/bin:$PATH"
juliaup status
julia --version
```

### 8.2 Vérifier le runtime hôte

```bash
cd "$COCOFEST_ROOT"
bash .github/scripts/check_libmad_host_linux.sh
```

Ne pas continuer si le script ne trouve pas `GCC_13.0.0` dans la bibliothèque
`libgcc_s.so.1` réellement résolue par `cc`.

### 8.3 Créer et compiler la pile MadNLP

Équivalent automatisé, une fois Julia installé et le runtime hôte validé :

```bash
bash .github/scripts/install_benchmark_stack_linux.sh madnlp32
```

La compilation de CasADi depuis les sources est l'étape la plus longue de tout
le protocole. Le script la saute si l'environnement expose déjà
`has_nlpsol("madnlp")`; utiliser `--force-step casadi` pour la refaire.

Procédure détaillée équivalente :

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
cd "$COCOFEST_ROOT"

conda env create \
  --name cocofest-madnlp32 \
  --file .github/cycling-benchmark-linux-environment.yml
conda activate cocofest-madnlp32

export PATH="${HOME}/.juliaup/bin:${HOME}/.julia/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=32
export CASADI_VERSION=3.7.2
export CASADI_MADNLP_COMMIT=973b086f4dcda9f49cd9c1948432ae4b7ee54886
export CASADI_CXX_ABI=1
export JULIAC_COMMIT=73be8587a80bbb65dab7acd71d406f72867a3571
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"

bash .github/scripts/install_libmad_mumps_linux.sh \
  .benchmark-deps/libMad \
  .cache/madnlp-mumps \
  "$JULIAC_COMMIT"

export LD_LIBRARY_PATH="$COCOFEST_ROOT/.cache/madnlp-mumps/lib:$COCOFEST_ROOT/.cache/madnlp-mumps/share/julia/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

bash .github/scripts/install_casadi_madnlp_linux.sh \
  .cache/madnlp-mumps
bash .github/scripts/install_biorbd_casadi_linux.sh
python -m pip install --no-deps -e .benchmark-deps/bioptim
```

Si l'environnement existe déjà, remplacer `conda env create` par
`conda env update --name cocofest-madnlp32 --file ... --prune`, puis
recompiler CasADi et biorbd seulement si le commit, l'ABI ou une dépendance a
changé.

### 8.4 Valider MadNLP et MUMPS

```bash
conda activate cocofest-madnlp32
cd "$COCOFEST_ROOT"

export PATH="${HOME}/.juliaup/bin:${HOME}/.julia/bin:$PATH"
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$COCOFEST_ROOT/.cache/madnlp-mumps/lib:$COCOFEST_ROOT/.cache/madnlp-mumps/share/julia/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import casadi as cas
import bioptim
import biorbd_casadi

print("CasADi", cas.__version__)
print("flags", cas.CasadiMeta.compiler_flags())
print("Bioptim", bioptim.__version__)
print("biorbd", biorbd_casadi.__version__)
assert "-DCASADI_WITH_THREAD" in cas.CasadiMeta.compiler_flags()
assert cas.has_nlpsol("ipopt")
assert cas.has_nlpsol("madnlp")
assert hasattr(bioptim.Solver, "MADNLP")
PY

python -m pytest -q \
  tests/shard1/test_solver_backends.py \
  tests/shard1/test_reduced_cycling.py \
  tests/test_benchmark_readme.py
```

Vérifier ensuite que le backend linéaire est bien MUMPS. libMad transporte
`linear_solver` comme un **type Julia**, dont le registre est sensible à la
casse : l'alias `mumps` envoyé littéralement est refusé, libMad émet un
avertissement et retombe silencieusement sur son solveur par défaut.
`cocofest/optimization/solver_backends.py` traduit donc `mumps` en
`MumpsSolver` avant l'appel.

Le contrôle décisif est l'**absence** de cet avertissement dans le log du
premier solve MadNLP :

```bash
grep -i "libMAD WARNING: option linear_solver is of unknown type" \
  local-results/madnlp-mumps-reduced/solver.log
```

La commande ne doit rien retourner. S'il apparaît, la traduction n'a pas eu
lieu et le temps mesuré n'est pas celui de MUMPS.

Le nom exact `MumpsSolver` est celui du type interne; il n'est pas imprimé dans
`result.json` ni dans `solver.log`, car le benchmark exécute MadNLP à un niveau
de verbosité réduit. Pour l'observer directement, utiliser le binaire de fumée
construit avec libMad, dont la bannière annonce la version de MUMPS :

```bash
source .github/scripts/benchmark_env.sh madnlp32
.benchmark-deps/libMad/build-cocofest-mumps/no_hsl_example 2>&1 | grep -i "MUMPS v"
# This is MadNLP version v0.9.2, running with MUMPS v5.8.2
```

Deux détails rendent ce contrôle fragile si on l'écrit autrement :

- la bannière part sur **stderr**, d'où le `2>&1`;
- le mot `MadNLP` est **colorisé** caractère par caractère, donc
  `grep "MadNLP version"` ne correspond jamais. Filtrer sur `MUMPS v`, qui
  n'est pas coloré.

Sans `benchmark_env.sh`, le binaire ne trouve pas son `LD_LIBRARY_PATH` et sort
en erreur sur CUDA.

et confirmer la table de traduction utilisée par le benchmark :

```bash
python -c "from cocofest.optimization.solver_backends import \
  MADNLP_LINEAR_SOLVER_RUNTIME_NAMES as m; print(m['mumps'])"
# MumpsSolver
```

## 9. Politique des 32 cœurs

### 9.1 Baseline reproductible

`benchmark_env.sh` applique cette politique, active l'environnement Conda et
exporte `COCOFEST_ROOT`, `GITHUB_WORKSPACE`, `PYTHONPATH` et le
`LD_LIBRARY_PATH` propre à chaque pile. Il remplace les blocs d'export répétés
des sections 7, 8, 10 et 11 :

```bash
cd "$COCOFEST_ROOT"
source .github/scripts/benchmark_env.sh rho32      # IPOPT, FATROP, ACADOS
source .github/scripts/benchmark_env.sh madnlp32   # MadNLP/MUMPS + IPOPT
```

Il doit être sourcé, pas exécuté, et accepte `BENCHMARK_THREADS` et
`NUMERIC_THREADS` en surcharge :

```bash
BENCHMARK_THREADS=16 source .github/scripts/benchmark_env.sh rho32
```

L'équivalent manuel, dans les deux environnements :

```bash
export BENCHMARK_THREADS=32
export CMAKE_BUILD_PARALLEL_LEVEL=32

export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1
```

`BENCHMARK_THREADS=32` autorise la construction parallèle prévue par le code.
Les variables numériques à 1 empêchent une oversubscription cachée et rendent
les temps comparables à la CI historique.

Pour ACADOS seulement, une campagne séparée peut utiliser :

```bash
export OMP_NUM_THREADS=32
export OMP_THREAD_LIMIT=32
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

Ne pas mélanger ces temps avec la baseline mono-thread numérique.

### 9.2 Sweep de parallélisme MUMPS

Après certification de la baseline, tester successivement
`1, 2, 4, 8, 16, 30, 32` threads. Exécuter au moins trois répétitions du même
NLP chaud et ne lancer aucun autre solveur en parallèle. Enregistrer :

- temps total;
- temps d'évaluation des fonctions;
- temps Jacobien/Hessienne;
- factorisation et backsolve;
- itérations;
- fréquence CPU et topologie NUMA.

Si le MUMPS fourni avec IPOPT ou libMad est sériel, modifier
`OMP_NUM_THREADS` ne donnera pas de gain. Il faut le constater dans les temps,
pas supposer un speedup de `32x`.

## 10. Construire la seed commune localement

Effectuer cette étape dans `cocofest-rho32`, avec les threads numériques à 1.
Le script `build_benchmark_seed_linux.sh` enchaîne les deux solves IPOPT
(`reduced` puis `full`), vérifie leur certification avec `jq` et copie les trois
artefacts dans `benchmark-seed/` :

```bash
source .github/scripts/benchmark_env.sh rho32
bash .github/scripts/build_benchmark_seed_linux.sh
```

Il produit :

| Artefact | Rôle |
|---|---|
| `benchmark-seed/common-reduced.npz` | branche initiale commune, mécanique réduite |
| `benchmark-seed/common-full.npz` | branche initiale commune, mécanique complète |
| `benchmark-seed/reduced-cycling-fourier12.npz` | profil de cadence Fourier 12 |

Les JSON de contrôle restent dans `benchmark-seed-result/` :
`seed-check-reduced.json` et `seed-check-full.json` doivent tous deux porter
`success = true` et `attempted_windows = 1`. Le script échoue si ce n'est pas le
cas, donc le seul code de sortie suffit ici.

Le solve `full` réutilise `common-reduced.npz`; l'ordre des deux étapes n'est
pas interchangeable. Les avertissements `RuntimeWarning` sur la projection du
warm-start et sur l'écrêtage des largeurs d'impulsion aux bornes de Ding sont
attendus : ils décrivent l'adaptation de la seed historique
`legacy-resistive-0p22-warmup.npz` au couple courant.

Ne jamais réutiliser une seed provenant d'un autre couple, d'une autre force
passive ou d'une autre transcription sans validation explicite.

## 11. Lancer les premiers cas localement

Variables communes. `benchmark_env.sh` couvre déjà `GITHUB_WORKSPACE`,
`PYTHONPATH`, `BENCHMARK_THREADS`, `LD_LIBRARY_PATH` et la politique de threads;
il ne reste que les paramètres propres à la campagne :

```bash
source .github/scripts/benchmark_env.sh rho32

export BENCHMARK_CYCLES_PER_WINDOW=1
export BENCHMARK_ASSISTANCE=0.00
export BENCHMARK_Q_SLACK=0.002
export BENCHMARK_MAX_ITER=2000
export BENCHMARK_CYCLES=1
```

### 11.1 IPOPT et FATROP

```bash
cd "$COCOFEST_ROOT"
source .github/scripts/benchmark_env.sh rho32

bash .github/scripts/run_cycling_benchmark_case.sh \
  ipopt ipopt reduced mumps collocation local-results \
  "$BENCHMARK_CYCLES" true sx none 3

bash .github/scripts/run_cycling_benchmark_case.sh \
  ipopt-radau5 ipopt reduced mumps collocation local-results \
  "$BENCHMARK_CYCLES" false sx none 5

bash .github/scripts/run_cycling_benchmark_case.sh \
  fatrop-collocation fatrop reduced fatrop collocation local-results \
  "$BENCHMARK_CYCLES" true sx none 3
```

### 11.2 MadNLP/MUMPS

```bash
cd "$COCOFEST_ROOT"
source .github/scripts/benchmark_env.sh madnlp32

bash .github/scripts/run_cycling_benchmark_case.sh \
  madnlp-mumps madnlp reduced mumps collocation local-results \
  "$BENCHMARK_CYCLES" true sx none 3

bash .github/scripts/run_cycling_benchmark_case.sh \
  madnlp-mumps-radau5 madnlp reduced mumps collocation local-results \
  "$BENCHMARK_CYCLES" false sx none 5
```

Le script conserve une non-convergence dans `result.json` et peut retourner
un code shell nul pour permettre la suite de la campagne. Il faut donc lire le
JSON et le log; le code de sortie seul ne certifie pas le RHO.

Passer ensuite manuellement `BENCHMARK_CYCLES` à `5`, puis `30`, puis `100`.
Ne pas automatiser les trois valeurs dans une boucle : inspecter le préfixe
physique et les artefacts avant chaque palier.

### 11.3 Depuis PyCharm ou un autre IDE

`.github/scripts/run_benchmarks.py` lance la matrice des sections 11.1 et 11.2
et imprime un compte rendu à la fin. Il enveloppe `run_cycling_benchmark_case.sh`,
qui reste la référence du protocole; le driver n'ajoute aucun réglage
scientifique.

```bash
python .github/scripts/run_benchmarks.py --list             # les 28 cas et leurs tags
python .github/scripts/run_benchmarks.py                    # toute la matrice, 1 RHO
python .github/scripts/run_benchmarks.py --tags core        # le sous-ensemble des sections 11.1/11.2
python .github/scripts/run_benchmarks.py --tags radau-sweep # Radau 3, 4, 5, 6
python .github/scripts/run_benchmarks.py --tags compiled    # évaluateurs C seulement
python .github/scripts/run_benchmarks.py --tags madnlp full # intersection des tags
python .github/scripts/run_benchmarks.py --cycles 5         # 5 RHO
python .github/scripts/run_benchmarks.py --report-only      # re-rapporter sans resolve
python .github/scripts/run_benchmarks.py --seed-only        # construire la seed
```

La matrice reprend celle du workflow `cycling_solver_benchmark_linux.yml` :

| Groupe | Tag | Contenu |
|---|---|---|
| Matrice de base | `core` | IPOPT, FATROP, MadNLP en `reduced` compilé et interprété, plus les trois `full` |
| Balayage Radau | `radau-sweep` | IPOPT et MadNLP en Radau `4`, `5`, `6`, profils `scientific-radauN`, plus Radau 5 `full` |
| Comparaison | `comparison` | les trois solveurs à transcription figée Radau 3 puis Radau 5, évaluateurs compilés |
| Variantes IPOPT | `variants` | `dual-all`, et le prédicteur KKT paramétrique en modes `reset`, `preserve`, `predict` |
| ACADOS | `acados` | voir section 11.4 |

Les tags `compiled`, `interpreted`, `reduced`, `full`, `ipopt`, `fatrop`,
`madnlp` et `acados` sont dérivés automatiquement de chaque cas; `--tags` en
demande l'intersection.

Deux mises en garde sur les variantes. Le prédicteur KKT et le warm start dual
agissent sur le **transfert d'un RHO au suivant**; à `--cycles 1` ils n'ont rien
à transférer et rendent exactement le résultat du cas de base. Il faut au moins
`--cycles 5` pour les différencier. Le JSON reste la preuve que le réglage a
bien été pris : `results[0].parametric_kkt_audits` est vide pour le cas de base
et non vide pour les variantes KKT.

Il se lance depuis une configuration Python ordinaire, **sans variable
d'environnement à saisir**, parce qu'il pose lui-même ce qu'un IDE ne fournit
pas :

- `CONDA_PREFIX`, déduit de `sys.prefix`. Sans lui, tout cas ACADOS meurt sur
  `KeyError: 'CONDA_PREFIX'` : l'IDE lance l'interpréteur sans `conda activate`,
  or `acados_template/utils.py` lit cette variable sans garde;
- `MPLBACKEND=Agg`, sinon `cocofest/result/plot.py` bascule sur `TkAgg` et
  ouvre des fenêtres pendant la mesure, ou échoue s'il n'y a pas d'affichage;
- `GITHUB_WORKSPACE`, `PYTHONPATH`, `LD_LIBRARY_PATH` et la politique de threads
  de la section 9.1.

Les deux environnements portant des ABI CasADi différentes, chaque cas est
exécuté avec l'interpréteur de l'environnement qui possède son solveur. Un cas
appartenant à l'autre environnement y est dispatché automatiquement : le
résultat est le même quel que soit celui des deux interpréteurs configuré dans
l'IDE.

Le compte rendu final donne, par cas, le statut, les RHO certifiés, les
itérations, l'objectif, le temps solveur et le temps mur-à-mur, puis l'écart
relatif entre solveurs partageant la même transcription. Le code de sortie vaut
`0` si tous les cas passent, `1` sinon, ce qui colore correctement l'exécution
dans l'IDE. La sortie complète de chaque cas est conservée dans
`<output-root>/_logs/<cas>.log`; seuls les jalons sont échoués vers la console,
sauf avec `--verbose`.

Dix-sept configurations prêtes à l'emploi sont versionnées dans
`.idea/runConfigurations/` et apparaissent dans le menu déroulant de PyCharm :
`Benchmark - full matrix (28 cases)`, `- core matrix`, `- Radau sweep 3-6`,
`- solver comparison`, `- IPOPT variants`, `- compiled evaluators`,
`- interpreted evaluators`, `- IPOPT + FATROP`, `- MadNLP/MUMPS`, `- ACADOS`,
`- core matrix, 5 RHO`, `- report only`, `- build seed`, ainsi que les quatre
configurations `Full horizon` de la section 11.5. Elles épinglent le
chemin absolu de l'interpréteur Conda; après une réinstallation sur une autre
machine, corriger `SDK_HOME` dans ces fichiers ou recréer la configuration.

Une exécution lancée depuis l'IDE reste une mesure valable tant que rien d'autre
ne tourne sur la machine, la politique de threads étant identique à celle du
shell. Ne pas lancer deux configurations simultanément.

### 11.4 Le cas ACADOS

ACADOS n'est pas un solveur de `run_cycling_benchmark_case.sh` : ni le workflow
ni le sweep d'endurance ne passent par ce script pour lui, ils appellent
`cycling_fes_solver_comparison.py --solvers acados` directement. Les sections
11.1 et 11.2 ne contiennent donc aucun cas ACADOS, et une matrice limitée à
ces deux sections laisse ACADOS non exercé alors qu'il est installé.

Le driver de la section 11.3 ajoute le cas `acados-irk`, repris des drapeaux du
cas de référence reduced de `run_ryzen5950x_endurance_sweep.sh` :

```bash
python .github/scripts/run_benchmarks.py --cases acados-irk
```

Transcription SQP/IRK Gauss-Legendre, 4 stages, 5 pas, tolérance de
stationnarité `5e-3`. Elle n'est **pas** une collocation Radau : son objectif
n'est pas comparable à celui d'IPOPT, FATROP ou MadNLP, et le compte rendu la
range dans son propre groupe plutôt que de l'inclure dans l'écart inter-solveurs.
Sur un RHO à couple nul, ACADOS converge en `3` itérations SQP contre `90` pour
IPOPT, pour un objectif supérieur d'environ `8e-3` en relatif; comparer ces deux
nombres sans tenir compte de la transcription n'a pas de sens.

Deux prérequis propres à ACADOS :

- `t_renderer` doit être présent dans l'environnement, sinon la génération de
  code échoue. Il n'est pas installé par le script ACADOS principal :

```bash
source .github/scripts/benchmark_env.sh rho32
bash .github/scripts/install_acados_tera_renderer.sh "$CONDA_PREFIX"
python -c "from acados_template import get_tera; print(get_tera())"
```

- ACADOS écrit son C généré dans le répertoire courant. Le driver crée donc un
  sous-répertoire `codegen/` par cas et s'y place, comme le fait le sweep.

### 11.5 Le benchmark full horizon

`run_full_horizon_benchmark.py` fait croître un problème monolithique plein
horizon (FHO) à partir de deux cycles RHO reduced, chaque taille étant
ré-optimisée depuis la dernière solution certifiée. Il n'appartient pas à la
matrice RHO : il a son propre pilote et son propre rapport.

```bash
python .github/scripts/run_full_horizon.py --max-cycles 6      # MadNLP, par défaut
python .github/scripts/run_full_horizon.py --solver ipopt --max-cycles 6
python .github/scripts/run_full_horizon.py --resume --max-cycles 12
python .github/scripts/run_full_horizon.py --report-only
```

Le solveur des FHO monolithiques est MadNLP par défaut, donc le run par défaut
exige `cocofest-madnlp32` et son ABI CasADi `1`, exactement comme le job
`full_horizon` du workflow. Le pilote sélectionne l'environnement d'après
`--solver`, quel que soit l'interpréteur configuré dans l'IDE.

Le benchmark écrit lui-même `full-horizon-results/full-horizon-report.md`; le
pilote l'affiche dans la console à la fin. Le rapport donne la limite RSS
retenue, la chaîne RHO/FHO construite, le plus grand full horizon validé, les
trous de convergence, les extensions RHO depuis l'état terminal FHO, les replis
`+1` après un saut multi-cycles refusé, et le pic RSS et le temps de chaque
passage.

Rappel de la section 3 : le sweep full horizon s'arrête pour des raisons
**numériques** avant d'être limité par la mémoire, et une machine plus grosse ne
le corrige pas sans meilleure seed multi-cycle. Un arrêt sous `--max-cycles` est
donc un résultat, pas une panne d'infrastructure; lire l'étiquette d'arrêt du
rapport avant de conclure. Le pilote sort en `0` dès qu'un rapport a été produit,
même si la continuation s'est arrêtée tôt.

Sur cette machine, une continuation à `--max-cycles 4` atteint
`requested_ceiling_reached` sans trou de convergence, avec un pic RSS inférieur
à `0.4 GiB` : la mémoire n'est pas le facteur limitant à ces tailles.

## 12. Résumer les résultats locaux

```bash
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"

python .github/scripts/summarize_cycling_benchmark.py \
  local-results/*/result.json \
  --output-dir local-summary
```

Contrôler au minimum dans chaque JSON :

- `success` et statut natif;
- premier RHO en échec;
- préfixe physique strict;
- durée de construction, compilation, préparation et solveur;
- fatigue, AUC et capacité par muscle;
- transcription, SX, force passive et backend linéaire;
- hash de la bibliothèque compilée.

## 13. Utiliser la machine comme runner GitHub Actions

Pour reproduire automatiquement la matrice et les artefacts, enregistrer un
seul runner avec le label personnalisé `linux-32core` :

1. ouvrir le dépôt GitHub;
2. aller dans `Settings > Actions > Runners`;
3. choisir `New self-hosted runner`, Linux x64;
4. exécuter sur la machine les commandes de téléchargement affichées par
   GitHub;
5. ajouter le label lors de la configuration :

```bash
./config.sh \
  --url https://github.com/mickaelbegon/cocofest \
  --token TOKEN_TEMPORAIRE_AFFICHÉ_PAR_GITHUB \
  --labels linux-32core
```

Ne jamais écrire le token dans le dépôt ou dans ce document. Il expire
rapidement. L'utilisateur du runner doit pouvoir exécuter les commandes
`sudo apt-get` du workflow sans invite interactive, ou les dépendances système
doivent être préinstallées et le workflow adapté.

Sur un dépôt public, ne pas autoriser du code de pull request non approuvé à
s'exécuter sur ce runner. Utiliser un runner ou un groupe limité à ce dépôt et
aux branches de confiance.

Avec une seule instance de runner sur la machine, les jobs seront séquentiels;
c'est souhaitable pour éviter la contention lors des mesures. Enregistrer
plusieurs instances sur la même machine rendrait les temps solveurs difficiles
à interpréter.

## 14. Campagne GitHub graduelle sur le runner 32 cœurs

Les modifications locales doivent d'abord être commités et poussées sur
`codex/acados-pr-refresh`.

### Gate 5 RHO

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=linux-32core \
  -f cycles=5 \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000 \
  -f compile_nlp_evaluators=true \
  -f acados_smoke_rhos=5 \
  -f acados_option_rhos=5 \
  -f refined_collocation_validation=true \
  -f refined_collocation_rhos=5 \
  -f collocation_diagnostic_rhos=5
```

### Gate 30 RHO

Seulement après certification du gate 5 :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=linux-32core \
  -f cycles=30 \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000 \
  -f compile_nlp_evaluators=true \
  -f acados_smoke_rhos=30 \
  -f acados_option_rhos=5 \
  -f refined_collocation_validation=true \
  -f refined_collocation_rhos=30 \
  -f collocation_diagnostic_rhos=5
```

### Gate 100 RHO

Seulement après certification du gate 30 :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=linux-32core \
  -f cycles=100 \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000 \
  -f compile_nlp_evaluators=true \
  -f acados_smoke_rhos=100 \
  -f acados_option_rhos=5 \
  -f refined_collocation_validation=true \
  -f refined_collocation_rhos=100 \
  -f collocation_diagnostic_rhos=5
```

Après chaque lancement :

```bash
gh run list \
  --repo mickaelbegon/cocofest \
  --workflow cycling_solver_benchmark_linux.yml \
  --limit 5
```

Lire tous les logs et télécharger les artefacts avant d'ouvrir le gate suivant.
Un workflow vert signifie que l'infrastructure a terminé; il ne remplace pas
la lecture du préfixe physique de chaque solveur.

## 15. Ordre recommandé de la première journée

1. Préflight Ubuntu et CPU.
2. Installation de `cocofest-rho32` et smoke tests IPOPT/FATROP/ACADOS.
3. Construction de la seed commune.
4. Installation de `cocofest-madnlp32` et smoke test `MumpsSolver`.
5. Reproduire le gate reduced Radau 4/5/6 avec IPOPT et MadNLP.
6. Reproduire le contrôle full/reduced Radau 5 avec les mêmes seeds.
7. Transférer les PW R5 vers R6 et R6 vers R5, sans réoptimisation initiale.
8. Réintégrer ces mêmes PW avec un intégrateur dense commun.
9. Gate 30 seulement si le gate 5 est scientifiquement certifié.

Le calcium isolé a déjà établi que Radau 4 est trop imprécis (`0.4518 %`),
alors que Radau 5 (`0.01730 %`) et Radau 6 (`0.000415 %`) passent ce test. Le
gate couplé sur cinq RHO montre néanmoins un écart R5--R6 reproductible de
`0.343–0.398 %` sur la fatigue et de `0.580–0.645 %` sur l'AUC. La première
décision expérimentale sur la machine 32 cœurs est donc de séparer erreur de
transcription et changement de bassin par transfert croisé et réintégration
dense. La campagne 30/100 RHO et le sweep de threads viennent ensuite.

## 16. Dépannage

### `check_libmad_host_linux.sh` échoue sur un hôte pourtant compatible

Symptôme : le script annonce que `libgcc_s.so.1` n'exporte pas `GCC_13.0.0`
alors que la bibliothèque l'exporte réellement. Vérification indépendante :

```bash
readelf --version-info "$(cc -print-file-name=libgcc_s.so.1)" |
  grep -oE 'GCC_[0-9]+\.[0-9]+\.[0-9]+' | sort -uV
```

Cause : la version historique testait `strings ... | grep -qx`. Sous
`set -o pipefail`, `grep -q` sort dès la première correspondance, `strings`
reçoit `SIGPIPE` et termine en `141`, ce qui transforme un succès en échec. La
détection était donc intermittente et dépendait de l'ordonnancement. Le script
matérialise désormais la liste des versions dans une variable avant de la
filtrer. Le même motif `producteur | grep -q` est à proscrire dans tout script
du dépôt utilisant `pipefail`.

### ACADOS s'installe « avec succès » mais `libacados.so` est absent

Symptôme : `acados_install_linux.sh` sort en `0`, `acados_template` est
importable, mais `test -f "$CONDA_PREFIX/lib/libacados.so"` échoue et les cas
ACADOS ne démarrent pas.

Cause : l'installateur amont ne s'exécute pas sous `set -e`. Sur GCC 14 ou 15,
la cible `qpOASES_e` casse la compilation avec
`error: passing argument 1 of 'ConstraintsCPY' from incompatible pointer type`;
`make install` échoue, mais le `pip install .` suivant réussit et fixe le code
de sortie du script à `0`.

Correctif : passer par `.github/scripts/install_acados_linux.sh`, qui neutralise
les diagnostics concernés et vérifie les artefacts installés. Pour retrouver la
première erreur réelle, conserver le journal complet — `tail` sur la sortie
masque l'erreur, qui survient tôt :

```bash
bash .github/scripts/install_acados_linux.sh .benchmark-deps/bioptim 32 \
  "$CONDA_PREFIX" > acados-install.log 2>&1
grep -nE 'error:|Error [0-9]' acados-install.log | head
```

### `bioptim` pointe vers un chemin supprimé

Symptôme : `python -c "import bioptim; print(bioptim.__file__)"` renvoie un
chemin hors du dépôt, par exemple sous `~/.local/share/Trash/`. Une
installation éditable survit à la suppression de son répertoire source et
l'environnement paraît complet alors que le code chargé n'existe plus.

```bash
python -m pip uninstall -y bioptim
python -m pip install --no-deps -e .benchmark-deps/bioptim
python -c "import bioptim; print(bioptim.__file__)"
```

Le chemin affiché doit être `$COCOFEST_ROOT/.benchmark-deps/bioptim/bioptim/__init__.py`.

### Messages attendus qui ne sont pas des erreurs

- `Julia version error:` suivi de `Compatible julia version: 1.12.6` pendant la
  configuration CMake de libMad : la variable d'erreur est vide et le contrôle
  a réussi.
- `Error: CUDA.jl could not find an appropriate CUDA runtime to use` pendant la
  précompilation Julia : `MadNLPGPU` est facultatif et la campagne est CPU.
- `RuntimeWarning` sur la projection du warm-start et l'écrêtage des largeurs
  d'impulsion aux bornes de Ding pendant la construction de la seed
  (voir section 10).

## 17. Références d'installation

Pour la campagne d'endurance actuelle sur Ryzen 9 5950X, ne pas réutiliser
directement les anciennes commandes de la section 14, qui pointent vers une
branche et un dépôt historiques. Utiliser le protocole reprenable
[`ryzen5950x_endurance.md`](ryzen5950x_endurance.md). Il conserve les deux
environnements décrits ici, calibre `16` contre `30` threads, puis exécute
séquentiellement IPOPT, MadNLP+IPOPT et ACADOS+IPOPT sur plusieurs couples
résistants.

## 18. Références externes

- [Miniforge — installateurs et installation Linux](https://github.com/conda-forge/miniforge/blob/main/README.md)
- [Juliaup — installation et sélection d'une version Julia](https://github.com/JuliaLang/juliaup)
- [GitHub — ajouter un runner autohébergé](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
- [GitHub — labels des runners autohébergés](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/apply-labels)
