# Données synthétiques Ding pour l'identification en pédalage

Le module `cocofest.identification.synthetic_cycling` génère des essais où un
seul des huit muscles (`right/left` × `Delt_ant`, `Delt_post`, `Biceps`,
`Triceps`) est stimulé. Il propage les cinq états de Ding 2007
`(Cn, F, A, Tau1, Km)` avec le propagateur numérique du dépôt. Les impulsions
peuvent avoir des intervalles et des PW différents. Le train commence sans
historique de stimulation; la récurrence du calcium inclut les impulsions déjà
administrées et le terme `R_i` dépendant de l'intervalle précédent.

`CyclingTrialSpec` contient la grille temporelle, l'angle, les impulsions et
les projections mécaniques. Le bras de levier signé transforme la force
musculaire en couple, puis la division par le rayon de manivelle fournit la
force tangentielle. La projection radiale est facultative et doit être fournie
séparément. Les projections constantes de `simple_mono_muscle_spec` ne sont que
des valeurs de démonstration : une étude de pédalage doit fournir les courbes
issues du bioMod réduit ou d'une calibration expérimentale, ainsi que le gain
force-longueur/vitesse dans `mechanical_gain`.

```python
from cocofest.identification.synthetic_cycling import (
    SensorNoise, generate_trial, predict_trial, simple_mono_muscle_spec,
)

spec = simple_mono_muscle_spec(
    "right_Biceps", frequency_hz=30, pulse_width_s=300e-6,
    duration_s=2, radial_gain=0.1,
)
trial = generate_trial(
    spec, noise=SensorNoise(tangential_std_n=0.2, radial_std_n=0.2), seed=7,
)
predicted, ding_states = predict_trial({"a_scale": 3314.7}, spec)
```

`trial.observed` et `predicted` ont la forme `(échantillons, canaux)` :
colonne 0 tangentielle, colonne 1 radiale si elle existe. Les sorties
`trial.noiseless`, `trial.state`, `trial.noise_std`, `trial.spec` conservent la
vérité synthétique, le niveau de bruit et les conditions de l'essai. Pour un
estimateur ou un plan d'expérience par différences finies, utiliser
`lambda params, context: predict_trial(params, context)[0]`.

Les paramètres de départ reproduisent les échelles musculaires du runner de
pédalage, avec les deux côtés égaux comme *prior* initial. L'identification
peut ensuite fournir une valeur indépendante pour chaque côté. Le biais et
l'écart-type gaussien sont configurables par canal, avec une graine aléatoire
explicite. Les données générées servent à vérifier les algorithmes et leur
identifiabilité numérique; elles ne constituent pas une validation clinique.

Un jeu de huit fichiers `.npz` et un manifeste `metadata.json` peuvent être
matérialisés avec :

```bash
python scripts/generate_ding_identification_synthetic.py \
  --output /tmp/ding-mono-muscle-example --two-axis --seed 7
```

La commande refuse un dossier de sortie existant pour éviter l'écrasement
d'une acquisition ou d'un autre jeu synthétique.
