# Pilote causal dose--durée de transfert de couple

## Question et portée

Le premier pilote causal suggère qu'à proximité de l'échec du bras gauche,
transférer 0,05 Nm du gauche vers le droit pendant dix cycles retarde la limite
bilatérale de l'ordre de 1--2 cycles. Ce résultat est directionnel, mais son
amplitude est comparable à la résolution nécessaire pour décider une règle de
supervision. Ce protocole estime donc une réponse locale à la **dose** et à la
**durée**, sans déclarer une politique clinique ni modifier le RHO.

Le solveur de chaque bras reste le RHO de référence : Radau-5, 30 Hz, IPOPT avec
MA57, poids musculaires unitaires, résistance totale 1,92 Nm (0,96 Nm par bras
au départ). Après chaque phase, les mêmes audits discrets complets et le même
critère d'échec gelé (réessais, faisabilité objectif nul, contre-factuel de repos
de fatigue) sont exigés. Un arrêt IPOPT seul est un arrêt numérique, jamais une
limite physiologique.

## Plan préenregistré

Deux checkpoints bilatéraux certifiés, issus de la référence RHO à poids
unitaires, sont requis : cycles 120 et 140. Chaque intervention conserve
exactement le total de couple :

| Facteur | Niveaux |
| --- | --- |
| ancrage | 120, 140 cycles |
| amplitude transférée G vers D | −0,025, −0,05, −0,10, +0,05 Nm |
| durée maintenue | 5, 10 cycles complets |
| contrôles | 5 répétitions à chaque paire ancrage–durée |
| suite | retour explicite à G=D=0,96 Nm jusqu'à 240 cycles |

Le signe négatif soulage le bras gauche; le signe positif le charge. Le plan
complet compte 36 branches : 18 par ancrage, dont 10 contrôles appariés et 8
interventions. Les contrôles sont exécutés en premier. **Toute** branche
d'intervention est bloquée si un contrôle ne fournit pas un endpoint gelé
attribuable. Les deux bras sont résolus séparément, mais seuls leurs préfixes
certifiés communs constituent un résultat bilatéral; les lignes unilatérales
ultérieures sont explicitement contre-factuelles.

La transition de −0,10 Nm ou +0,05 Nm à 0,96/0,96 est une discontinuité prescrite de
simulation. Elle ne représente pas une rampe de confort ou une recommandation
clinique; elle isole l'effet du transfert bref puis du retour à la référence.

## Résolution et décision

Pour chaque paire ancrage–durée, l'écart maximal entre les deux contrôles est la
résolution numérique observée. Un décalage de l'endpoint qui ne dépasse pas cet
écart est rapporté comme égalité. Une décision de superviseur ne sera considérée
que si une direction est répétable entre les deux ancrages, dépasse cette
résolution, n'induit pas une dette au bras droit après retour à 50--50, et garde
la faisabilité mécanique à chaque cycle certifié.

## Exécution

Les deux checkpoints requis (`cycle-120` et `cycle-140`) existent dans l'archive
de référence. Ne remplacez jamais un receipt déjà utilisé dans un plan scellé.

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=slow-layer-sim-20261003/torque-dose-duration-full

"$PY" scripts/slow_layer/run_torque_transfer_dose_duration.py prepare \
  --manifest "$RUN/manifest.json" \
  --receipts \
  asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/unit_fixed/results/checkpoints/cycle-120/receipt.json \
  asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/unit_fixed/results/checkpoints/cycle-140/receipt.json \
  --maximum-cycles 240

"$PY" scripts/slow_layer/run_torque_transfer_dose_duration.py campaign \
  --manifest "$RUN/manifest.json" --cpus 0 1 2 3 --timeout-seconds 7200
```

Le script refuse d'écraser un dossier existant et vérifie les hashes du code,
des receipts et du manifeste avant chaque worker. Toute modification du code
après `prepare` demande un nouveau dossier de campagne.
