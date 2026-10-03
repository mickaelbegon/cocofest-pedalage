# Calibration locale de la réserve de tâche

`scripts/calibrate_independent_rho_task_reserve.py` organise un benchmark de
sondes complètes d'un cycle, à partir de checkpoints préparés RHO et certifiés
par restauration exacte. Un manifeste traite **un seul bras**. Il faut lancer
un manifeste gauche et un manifeste droit pour une étude bilatérale : leurs
modèles musculaires et leurs contextes ne doivent pas être mélangés.

Le protocole garde la même grille de facteurs de travail pour tous les
checkpoints et exige une partition explicite `train` / `holdout` avant les
sondes. Une charge réalisable donne une borne inférieure observée de réserve.
Un échec numérique reste indéterminé. La grandeur ajustée n'est ni la charge
maximale exacte ni une prédiction certifiée d'endurance.

## Exemple de manifeste

Les chemins sont relatifs au manifeste ou absolus. Les noms de coordonnées et
leur normalisation doivent correspondre au modèle réel. L'exemple ci-dessous
montre le format ; ses valeurs ne constituent pas un jeu de données clinique.

```json
{
  "schema_version": 1,
  "side": "left",
  "build_context": {
    "cycle_period_s": 1.0,
    "cycle_len": 30,
    "formulation": "isokinetic",
    "mechanical_formulation": "reduced",
    "terminal_half_step_guard": false
  },
  "work_scales": [1.0, 1.025, 1.05, 1.075, 1.1],
  "tolerance": 0.00001,
  "coordinates": [
    {"state_key": "A_Biceps", "index": 0, "scale": 1000.0, "offset": 0.0}
  ],
  "fit": {
    "center": [0.8],
    "trust_radius": [0.1],
    "maximum_error": 0.01,
    "maximum_condition_number": 1000000,
    "minimum_train_samples": 2,
    "minimum_holdout_samples": 1
  },
  "samples": [
    {"id": "c20", "receipt": "results/checkpoints/cycle-20/receipt.json", "partition": "train"},
    {"id": "c30", "receipt": "results/checkpoints/cycle-30/receipt.json", "partition": "holdout"},
    {"id": "c40", "receipt": "results/checkpoints/cycle-40/receipt.json", "partition": "train"}
  ]
}
```

`build_context` est optionnel pour une calibration descriptive. Il devient
nécessaire lorsqu'un modèle accepté doit être raccordé comme coût au RHO : ces
valeurs structurelles sont comparées au problème réellement compilé. Une
incohérence (par exemple Radau/phase, formulation ou garde de demi-pas)
bloque le raccordement au lieu de réutiliser silencieusement le modèle.

Une entrée `probe_result` peut désigner un résultat déjà présent ; elle est
strictement en lecture seule. Sinon, les nouvelles sondes sont déposées sous
`OUTPUT/probes/ID/`. L'option `--run-missing` est nécessaire pour lancer des
résolutions ; sans cette option, seuls des résultats existants sont lus.
Les sondes sont séquentielles dans un manifeste. On peut répartir des
manifestes de bras indépendants sur des CPU distincts en dehors du script.

```bash
"/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python" \
  scripts/calibrate_independent_rho_task_reserve.py \
  --manifest /chemin/manifest-left.json \
  --output-directory /chemin/calibration-left \
  --run-missing
```

## Vérifications et limites

Les coordonnées sont extraites de la colonne initiale des **bornes d'état
figées** dans l'archive (`problem__x_bounds:STATE:min/max`). Le warm-start de
la trajectoire n'est pas utilisé comme état réel. Une borne initiale non
égale, une coordonnée absente ou non finie bloque la calibration. Toutes les
autres composantes Ding, la mécanique et l'historique de stimulation restent
dans le checkpoint complet utilisé par la sonde.

Chaque résultat doit pointer vers ce même checkpoint, le même modèle et le
même contexte physique. Les facteurs de travail, la tolérance, les groupes
de contraintes validés et les résidus sont revérifiés. Les résumés de succès
sont recalculés ; les trajectoires témoins doivent exister, être finies et
porter la bonne provenance. Ces vérifications relisent le compte rendu de
l'audit NLP indépendant ; elles ne constituent pas un nouveau replay DOP853.

Une archive préparée ne peut pas être réutilisée pour fabriquer deux points
ni passer dans les deux partitions. Il faut au moins `dimension+1` points
d'apprentissage, un rang complet et un holdout indépendant. Un jeu de points
pris sur une seule trajectoire de fatigue peut rester presque colinéaire :
augmenter le nombre de cycles observés ne remplace pas l'excitation de chaque
direction. Le script rejette un ajustement mal conditionné ou imprécis,
ainsi qu'une réserve observée constante qui ne renseigne aucun choix de PW.

Un `calibration-audit.json` garde les empreintes des entrées, coordonnées,
provenances, exclusions, diagnostics et motifs de rejet. Un
`local-reserve-model.json` est publié **uniquement** pour un ajustement accepté.
Ce dernier contient le `LocalReserveModel`, son contexte et l'empreinte de
l'audit. Le contexte du modèle ajoute `coordinate_layout` au contexte
physique des sondes, conformément au contrat de `TaskReserveObjectiveBinding`.
Les preuves NLP conservent leur contexte d'origine dans l'audit.

Atteindre le facteur le plus élevé de la grille est explicitement signalé :
la frontière peut se trouver bien au-delà. Une grille plus fine améliore la
résolution des différences observées mais augmente le coût de calibration.
L'acceptation du fit ne prouve pas une amélioration de l'endurance : elle
autorise seulement le prochain test, une continuation RHO appariée avec
validation du domaine de confiance et comparaison aux poids unitaires.
