# Validation isolée de Ding à PW imposées

`scripts/validate_ding_fixed_pulse_width.py` isole les cinq équations de Ding
du problème de cyclage. Il lit un `common-seed.npz` périodique à 50 Hz, reprend
pour chaque muscle son état initial complet et ses **mêmes 50 PW**, puis propage
les 50 intervalles avec Radau-3, Radau-5 et DOP853 (`rtol=1e-11`,
`atol=1e-13`). Aucun OCP n'est résolu; les deux Radau ne peuvent donc ni choisir
des PW différents, ni être affectés par un warm-start d'OCP.

La mécanique est volontairement neutralisée (gain force-longueur-vitesse et
force passive réunis à 1). Le rapport donne, pour `Cn`, `F`, `A`, `Tau1` et
`Km`, l'erreur maximale aux extrémités des 50 intervalles, l'erreur terminale,
l'échelle utilisée pour la normalisation et le résidu des équations de
collocation. Cette sonde répond à la question numérique sur Ding seulement;
elle ne certifie ni le mouvement ni la fermeture du cycle.

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
SEED=radau-frequency-validation-20260912/50hz-x0p5-r010/preparation-radau5/common-seed.npz
"$PY" scripts/validate_ding_fixed_pulse_width.py \
  --common-seed "$SEED" \
  --output-json radau-frequency-validation-20260912/50hz-x0p5-r010/ding-fixed-pw.json
```

Le code reconstitue l'amplitude calcique périodique tronquée à six stimulations,
avec la constante de modèle `r0_km_relationship=1.04`. Les archives v3 ne
stockent pas cette constante; elle est donc explicitement rapportée et peut être
changée avec `--r0-km-relationship`. L'exécution refuse par défaut une archive
qui n'a pas exactement 50 stimulations par cycle ou la formulation
`exact_exponential_periodic_node`.

Interprétation: on compare d'abord Radau-3 et Radau-5 état par état. Si Radau-3
reste dans le budget souhaité ici mais échoue dans l'OCP, la cause est hors des
équations isolées de Ding (par exemple transfert d'initialisation ou couplage
mécanique). Si son erreur de `Cn` ou de `F` dépasse le budget, il faut augmenter
le degré ou raffiner l'intégration **sans changer les 50 décisions PW**.
