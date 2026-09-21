# Simulation workbench

The setup layer uses the Python standard library. Importing
`cocofest.simulation` does not load Bioptim, CasADi, NumPy, Qt, or Tk. Scientific
engines remain in their existing Conda environments and are run as child
processes.

Open `.github/scripts/run_simulation_gui.py` in PyCharm and run it, or use:

```bash
python -m cocofest.simulation.gui
python -m cocofest.simulation.gui --config simulation.json --prefix /path/to/solver/environment
python -m cocofest.simulation.gui --config simulation.json --dry-run
```

Tkinter and a working graphical display are required only for the window.
The dry-run command, JSON model, capability validation and subprocess tests
work without a display. The GUI uses the repository's existing environment
discovery/HSL/library policy when starting a run. Choose **Détecter** after
switching to a solver in a different environment, or provide a custom prefix.

## Rejeu croisé IPOPT / ACADOS / DOP853

L'onglet **Rejeu croisé / DOP853** charge une ou plusieurs archives NPZ de
solutions. Il conserve les commandes réellement sauvegardées et propage les
états sans réoptimisation, avec DOP853 comme référence. Sélectionner les trois
évaluateurs produit une matrice source × intégrateur. Le modèle musculaire,
le profil mécanique, la durée, l'horizon et les conditions entrantes sont
audités ; des états initiaux ou conditions physiques différents empêchent de
classer les sources comme si elles résolvaient le même problème.

Le bouton **Configurer un RHO réoptimisé** ouvre le formulaire du contrôleur
existant. Choisir son solveur et sa formulation, puis **Lancer** : les décisions
sont alors recalculées à chaque fenêtre. Cette action ne charge pas les
commandes sélectionnées dans le rejeu ouvert.

Un exemple de requête du rejeu est
[`cross_rollout_gui.example.json`](../../docs/cycling_solver_benchmark/cross_rollout_gui.example.json).
Le charger avec **Charger requête…**, vérifier ses sources et le dossier de
résultats, puis **Prévisualiser le rejeu** ou **Lancer le rejeu ouvert**. Les
requêtes se sauvegardent séparément des configurations d'optimisation. Les
chemins relatifs désignent la racine du dépôt, comme les configurations du GUI.
Le champ **Coût commun JSON** accepte les poids de `CommonCost` ; il s'applique
à toutes les cases de la matrice. Un exemple de poids explicites est
[`cross_rollout_cost.example.json`](../../docs/cycling_solver_benchmark/cross_rollout_cost.example.json).

Le sous-processus utilise l'environnement affiché dans le GUI et conserve
`gui-run.log`. La matrice s'affiche dans **Matrice des replays** ; les coûts,
écarts vs DOP853 et violations échantillonnées de phase/vitesse sont visibles.
Les résultats détaillés sont `matrix.json` et les NPZ de trajectoires. Une
intégration réussie ne certifie pas toutes les contraintes du NLP ; le classement
est explicitement un classement de scores à examiner avec les diagnostics.

**Radau-5 signifie cinq stages (ordre neuf) dans le projet.** Les évaluateurs
Radau et Gauss-Legendre 4×5 résolvent les tableaux implicites indépendamment ;
ils n'appellent pas IPOPT ou le code natif ACADOS. Le score physique commun
n'est pas la valeur totale des objectifs NLP. Voir
[les équations, périmètre, métadonnées et validation du moteur](../../docs/cycling_solver_benchmark/solver_cross_rollout_engine.md).

Pour lancer le GUI depuis ce dépôt :

```bash
cd /home/mickaelbegon/Documents/Kevin/cocofest-pedalage
python .github/scripts/run_simulation_gui.py --prefix /home/mickaelbegon/miniforge3/envs/cocofest-rho32
```

Le Python qui ouvre la fenêtre doit disposer de Tkinter. Les calculs sont
exécutés avec le Python de `--prefix`. Ce lancement n'effectue aucun calcul
avant de cliquer sur le bouton correspondant.

## Deux bras indépendants, cadence imposée

L'onglet **Deux bras indépendants** crée deux OCP unilatéraux distincts,
droite et gauche, puis les lance simultanément. Il est volontairement séparé
du modèle `bilateral_reduced` : seuls l'horloge et \(\omega\) sont communs.
Pour chaque bras, l'opérateur choisit un travail positif par tour ou un couple
moyen équivalent, avec \(W=2\pi\bar\tau\). Le couple instantané demeure celui
inféré par le bilan isocinétique; il n'est pas prescrit dans le formulaire.

Les résultats sont `right/result.json`, `left/result.json` et `summary.json`.
Les cibles peuvent être changées entre deux appels en réutilisant les handles
compilés si la fabrique runtime le prend en charge; le GUI n'affirme cette
réutilisation que si le coordinateur la déclare. Le flux complet, son contrat
de fabrique et ses limites sont documentés dans
[independent_isokinetic_arms_gui.md](../../docs/cycling_solver_benchmark/independent_isokinetic_arms_gui.md).

## Workflow

1. Select the strategy, dynamics, mechanics and solver. `dynamic` means free
   crank dynamics; `isokinetic` prescribes angular velocity.
2. Set cycle count, controls per cycle, signed torque, terminal angle slack,
   and optionally the hard ΔPW bound in microseconds between successive
   controls. The RHO boundary is included; there is no artificial periodic
   closure between the last and first planned controls.
3. Select Radau/IRK, linear solver, solver and numerical-library threads.
4. Supply the exact IPOPT cycle-1 seed for ACADOS. Its contents, provenance and
   compatibility with the target OCP are verified by the scientific engine.
5. For weighted RHO, provide the weights JSON with its provenance. Optional
   model JSON uses the existing fingerprinted model adapter.
6. Save the versioned JSON, review the scientific summary and command, then
   launch into a fresh output directory. The GUI refuses existing result or
   log files. **Arrêter**, or closing the window during a run, terminates the
   process group that this window started.

The complete log is `gui-run.log` beside the exact `result.json`. Only the
most recent log lines remain in the widget. The process exit code is shown
separately from the result JSON's declared success and validated cycle count.
Exit code zero alone never means scientific success.

## Connected capabilities

`CapabilityRegistry` is the shared authority used by the form and launchers.
It rejects unsupported combinations before starting a solver:

| Option | Connected restriction |
| --- | --- |
| RHO-Physio / RHO-PACE | IPOPT, reduced dynamic mechanics, one-cycle windows, uncompiled evaluator, positive resistance, weights JSON, at most 2,000 cycles |
| Configured model adapter | IPOPT, reduced dynamic mechanics, positive resistance, at most 2,000 cycles |
| Isokinetic | Reduced mechanics |
| ΔPW | Reduced mechanics and one-cycle windows |
| ACADOS | One-cycle RHO windows, IRK and explicit IPOPT cycle-1 seed; reduced mode uses the existing experimental path |
| Experimental ACADOS Ding local reduction | Explicit opt-in, reduced dynamic RHO, 50 stimulations/cycle, SQP IRK Gauss-Legendre 4×5, active guard, no slew; initial/transfer IRK rollouts refused; fixed initial muscle states checked by the engine |
| FHO | IPOPT/MadNLP; one optimization window spanning the requested executed-cycle count |

The compatibility registry describes connected implementations, not a
scientific certification of every selectable combination. New solver/mode
connections must add engine integration tests before widening these rules.

The **ACADOS : réduction locale Ding (expérimental)** checkbox defaults to
off. Its saved field is `acados_ding_local_reduction`; selecting it forwards
`--acados-ding-local-reduction`. Incompatible choices remain visible as
validation errors. The [French README with equations and references](../../docs/cycling_solver_benchmark/acados_ding_local/README.md)
describes state reconstruction, bound conversion, RHO/dual limitations and a
previewable example configuration. Existing JSON documents keep the option
disabled when the field is absent.

## Python API

```python
from pathlib import Path
from cocofest.simulation import SimulationConfig, CapabilityRegistry, build_launch_plan

config = SimulationConfig(cycles=100, threads=6, pulse_width_max_step_us=100)
issues = CapabilityRegistry.validate(config)  # structured field/code/message tuple
plan = build_launch_plan(config, Path('/path/to/cocofest-rho32'))
print(plan.command)       # quoted preview; execute plan.argv with shell=False
print(config.to_json())   # schema_version=1
```

Unknown JSON fields or versions are refused. Advanced CLI arguments are a
JSON array of individual strings and cannot override options managed by the
form. The historical PyCharm CONFIG dictionary remains supported and delegates
to this layer. Its legacy implicit ACADOS IRK selection remains compatible;
typed configurations state the integration choice explicitly.
