# IPOPT cycle-1 → ACADOS : diagnostic du transfert ΔPW

Revue du 12 septembre 2026. Deux défauts déterministes de l'initialisation
avec ΔPW ont été reproduits puis corrigés. Ils ne s'appliquent pas aux
campagnes où `pulse_width_max_step_us=None` et ne prouvent pas, à eux seuls,
que la campagne ACADOS convergera après correction.

## 1. Le seed IPOPT était partiellement remplacé

Dans `_adapt_warmup_solution_to_periodic_nodes`, l'activation ΔPW côté cible
déclenchait systématiquement le remplacement des carriers, des contrôles
`pw_slew_next_*` **et des PW physiques** `last_pulse_width_*` par les guesses
par défaut du nouvel OCP. Ce remplacement devait servir seulement au warmup
générique sans variables auxiliaires. Il s'appliquait aussi à un common seed
IPOPT déjà complètement levé et à une solution de recovery certifiée.

Les états FES/mécaniques provenaient donc d'une trajectoire optimisée, tandis
que les commandes qui les avaient produits étaient remplacées par des
constantes. Le contrat « initialiser ACADOS avec la solution IPOPT » était
violé avant le premier appel natif.

La reproduction donne `[300,300,300] µs` à la place de `[160,240,320] µs`, soit
140 µs de modification maximale. Pour un simple état `xdot=u`, les défauts
de cette trajectoire initialement exacte deviennent non nuls. Ce défaut
existe aussi lors de la décimation Radau-5 → shooting.

Correction : le fallback constant est réservé aux sources sans aucune clé
auxiliaire. Une source levée complète conserve ses états et commandes ; une
source partiellement levée est refusée au lieu d'être mélangée aux valeurs
par défaut. Voir
[`_adapt_warmup_solution_to_periodic_nodes`](../../examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py#L15027),
notamment `source_has_slew` et `initialize_target_only_slew`.

Les carriers étaient aussi créés avec une guess CONSTANT d'une seule
colonne. Une telle représentation ne peut recevoir un vrai profil variable.
Le constructeur leur donne maintenant la même grille explicite que les
états physiques : `N+1` colonnes en shooting, `N*(d+1)+1` en collocation.
Voir [`add_auxiliary_bounds_and_guesses`](../../cocofest/optimization/pulse_width_slew.py#L95).

## 2. La levée devenait incohérente à l'avancement RHO

Le transfert des états ne classait pas les `pw_slew_carrier_*` parmi les
états transférés. Les PW physiques étaient, elles, transférées et parfois
extrapolées. Les contrôles `pw_slew_next_*` suivaient le transfert générique,
sans la prédiction réservée aux `last_pulse_width_*`.

La levée impose `z[k]=u[k]`, `zdot=(v-u)/dt`, donc
`z[k+1]=v[k]=u[k+1]`. Des carriers anciens et des `v` indépendamment
transférés violent ces égalités même lorsque le prédicteur physique est
acceptable.

Après la prédiction et le bornage des commandes physiques, la correction
reconstruit `z[k]=u[k]`, `v[k]=u[k+1]` et choisit librement
`v[N-1]=u[N-1]`. Les valeurs internes Radau utilisent leurs vraies abscisses,
avec la rampe affine exacte du carrier. Il n'y a pas de fermeture artificielle
entre la dernière et la première commande. Voir
[`_rebuild_pulse_width_slew_initial_guess`](../../examples/fes_multibody/cycling/cycling_pulse_width_mhe.py#L718)
et son appel dans `advance_window_initial_guess_controls`.

La correction conserve le prédicteur physique. Si ce prédicteur viole la
borne ΔPW ou la borne du raccord effectivement exécuté, cette infaisabilité
initiale reste à résoudre par le NLP ; aucun nouvel algorithme de projection
des PW n'a été ajouté. Les contraintes du problème ne sont pas relâchées.

## Vérifications complémentaires

| Domaine | Conclusion de la revue ciblée |
| --- | --- |
| Radau-5 → shooting à fréquence identique | La décimation par stride `d+1` dans `_resample_warmup_data` conserve les nœuds de shooting et le terminal. Le test d'intégration utilise une vraie solution IPOPT/Radau-5. |
| Changement de degré sans ratio entier | Le fallback historique interpole des indices uniformes, pas les temps Radau exacts. C'est une heuristique de warmstart à restaurer par le NLP, pas une conversion préservant exactement les défauts. Aucun changement ici. |
| Ordre et scaling x/u | `_stack_initial_guess_values` et `_acados_variable_scaling` utilisent les indices Bioptim. Le rollout transmet `x/Sx`, `u/Su` et remultiplie par `Sx`. Le test existant avec scalings distincts et données par étage passe. |
| lbx/ubx | `_sync_acados_state_bounds` utilise les mêmes indices et le scaling des bounds. L'interface pousse les colonnes START/PATH/END aux étages correspondants. L'initialisation native du rollout synchronise de nouveau les bounds après la création de capsule. Aucun nouveau défaut déterministe établi ici. |
| Avancement/phase | Le code courant convertit les intervalles en cycles par `divmod(time_idx_to_cycle,cycle_len)` avant de déplacer la référence absolue. Les tests existants de référence absolue et du décalage theta/omega passent. |
| Terminal et chargement du seed | Le chemin common-seed appelle `finalize_absolute_wheel_q_initial_guess` après la copie, pour fixer la séquence terminale sur l'état initial effectivement chargé. Les tests associés passent. |
| Full/reduced | Les branches de lift/projection utilisent le profil cinématique ; une projection full→reduced au-delà de 0,01 rad conserve volontairement la mécanique cible et réutilise les données FES/PW. Cette branche n'est donc pas une identité de trajectoire. Les tests existants de lift et de projection passent. |

## Historique pertinent

- `1ed65fa` (20 août 2026), *Interpolate certified seeds onto recovery grids*,
  introduit le fallback d'interpolation entre grilles de recovery.
- `5c72727` (21 août 2026), *Align terminal angle recovery targets*, traite les
  références d'angle terminal au cours des récupérations.
- `2f29780`, *Preserve global angle references in terminal profiles*, traite
  les conventions de référence globale.
- `c756031`, *Preserve ACADOS-native resistant seed dynamics*, documente un
  précédent où une deuxième projection Ding transformait un seed natif exact
  en seed de défaut `9,63e-2`. La politique actuelle impose un seed IPOPT,
  mais cet historique montre pourquoi chaque transformation du seed doit être
  auditée séparément. Une solution de collocation IPOPT n'est pas automatiquement
  une trajectoire exactement faisable pour la discrétisation IRK.

Les deux défauts ΔPW corrigés ici étaient dans les modifications locales non
commitées au moment de la revue ; ils ne sont pas attribués à ces commits.

## Validation

### Politique du seed IPOPT commun

Pour ACADOS dynamique chargé depuis `--common-initial-solution`, le comportement
par défaut préserve maintenant le primal IPOPT : pas de projection FES,
pas de Phase-I, pas d'homotopie des commandes, pas de raffinement IPOPT
supplémentaire. La translation des bornes de position absolue conserve leurs
largeurs et ne modifie pas le seed.

`--acados-assisted-hot-start` reste une option explicite pour activer la
préparation assistée historique. `--periodic-ipopt-refinement` reste une option
explicite et indépendante, désactivée par défaut dans les deux lanceurs et
dans l'API `main` du comparateur. Ces options ne remplacent pas l'initialisation
IPOPT obligatoire du cycle 1 ; elles concernent uniquement les transformations
supplémentaires avant ACADOS. Les options individuelles explicitement demandées
de Phase-I, d'homotopie ou de raffinement ne sont pas réinitialisées.

Les tests de `tests/test_acados_pristine_seed_policy.py` vérifient ces défauts
et la disponibilité des options explicites, sans lancer ACADOS.

### Transfert ΔPW

Avant correction, quatre tests nouveaux échouent : perte des commandes à
stride 1 et 6, et acceptation silencieuse de deux sources partiellement levées.
Après correction : **27 tests ΔPW/seed réussis**, plus **10 tests existants
ciblés** de transfert, scaling, projection et angle absolu.

Le test `test_ipopt_npz_common_seed_reaches_pre_solve_state_and_control_packing_unchanged`
résout un petit OCP IPOPT/Radau-5, exporte puis recharge réellement le NPZ,
vérifie la provenance cycle 1, applique le même adaptateur que le common-seed
et vérifie les commandes physiques, les carriers et le packing/scaling
d'état pré-solve de l'interface Bioptim ACADOS. Il ne construit ni ne lance
de capsule ACADOS.

```bash
MPLBACKEND=Agg /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest -q tests/test_slew_seed_transfer.py tests/test_pulse_width_slew.py
MPLBACKEND=Agg /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest -q tests/shard1/test_periodic_pulse_width.py -k 'standard_warmup_projects_reduced or reduced_common_seed_is_lifted or warmup_state_resampling or pulse_width_transfer or phase_shifted_warmup_shifts_reduced or acados_irk_transfer_rollout_uses_scaled or absolute_terminal_reference or cycle_boundary_wheel_angle_uses_fixed_absolute'
```

`git diff --check` passe pour les fichiers de cette correction. La convergence
d'une nouvelle campagne ACADOS reste une validation distincte, non exécutée
par ces tests.
