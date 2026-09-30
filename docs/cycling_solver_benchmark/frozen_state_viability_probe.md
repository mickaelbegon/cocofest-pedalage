# Sonde de viabilité après arrêt RHO

`scripts/probe_rho_endurance_viability.py` répond à une question plus stricte
qu'un arrêt IPOPT : à l'état exact qui suit le dernier cycle RHO certifié,
existe-t-il une PW admissible qui ferme le cycle suivant sous la même charge ?

Chaque campagne RHO construite avec `scripts/run_four_model_rho_campaign.py`
conserve maintenant `last-certified-rho-replay.npz` dans chaque condition. Ce
fichier est écrit seulement après l'avancement d'un RHO certifié ; il contient
donc les états Ding, l'état mécanique et le primal/les PW qui initialisent le
cycle suivant. Il n'est jamais remplacé par l'itéré du solveur qui échoue.

Après un arrêt, lancer par exemple :

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=four-model-r022-1500-dynamics-20260923
CASE=$RUN/ding_triceps_alpha_a_x1/rho
"$PY" scripts/probe_rho_endurance_viability.py \
  --python "$PY" \
  --model-config "$RUN/ding_triceps_alpha_a_x1/model.json" \
  --source-result "$CASE/result.json" \
  --source-configuration-audit "$CASE/configuration-audit.json" \
  --checkpoint "$CASE/last-certified-rho-replay.npz" \
  --output-directory "$CASE/frozen-state-viability"
```

Le script reprend automatiquement la transcription, la résistance, Radau-5,
les contraintes, MA57 et les tolérances de la campagne source. Il remplace
uniquement l'horizon par un RHO d'un cycle et recale les bornes du premier
noeud sur le checkpoint. Le `viability-report.json` est valide seulement si
`checkpoint_completed_windows == source_validated_cycles`.

Interprétation :

- `feasible_closed_cycle_found` est une preuve numérique positive, pour le NLP
  configuré, qu'au moins une PW donne le cycle fermé suivant.
- `no_feasible_closed_cycle_found_by_local_probe` n'est **pas** une preuve
  d'épuisement. IPOPT traite un problème non convexe et peut échouer malgré
  l'existence d'une solution admissible.
- `normalized_constraint_deficit` est le résidu primal du dernier itéré divisé
  par la tolérance de faisabilité. Il sert à comparer des relances identiques ;
  ce n'est ni un déficit de moment, ni une marge physiologique, ni un gap
  d'optimalité.

Cette sonde garde l'objectif de fatigue de la campagne mais l'autorise à être
différent du fichier source grâce au mode `feasibility-probe`. C'est un test
d'existence locale d'allocation, pas une nouvelle politique RHO, et son
résultat ne doit pas être utilisé comme score d'endurance ou pour entraîner
PACE. Pour étayer une conclusion négative, il faut ajouter des démarrages
alternatifs, une restauration Phase-I/relâchement explicite avec déficit de
moment normalisé, puis rapporter qu'il s'agit d'une borne numérique plutôt que
d'une preuve globale.
