# Fidélité décisionnelle PACE-VR : comparaison contrôlée des candidats

Le script `scripts/benchmark_pace_vr_candidates.py` repart d'un snapshot
PACE-VR enregistré à la fin d'un cycle RHO certifié. Il conserve exactement
les états Ding, les PW précédentes, la géométrie échantillonnée et le travail
demandé. Il compare les poids courants, la proposition effectivement appliquée,
un demi-pas logarithmique et le pas opposé. Un JSON optionnel permet d'ajouter
des candidats (`{"bo_fixed": [w1, w2, w3, w4]}`). Chaque candidat est évalué
depuis le **même** snapshot et sur le **même** horizon.

Exemple sur le bras droit au cycle source 20 de la campagne H=100/50/25/20/5 :

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=asymmetric-sides-r192-rho-bo-20260928/pace-vr-ladder-h100-50-25-20-5-20260930
"$PY" scripts/benchmark_pace_vr_candidates.py \
  --source "$RUN/results/pace_vr/right/result.json" \
  --source-cycle 20 --horizon 20 \
  --proposal-journal "$RUN/results/pace_vr/right/pace_vr.jsonl" \
  --output "$RUN/decision-fidelity-right-c20-h20.json"
```

Le rapport donne le préfixe de cycles validés **par le surrogate**, la marge
minimale, le score terminal, les résidus travail/contraintes, les solveurs QP
utilisés et le temps de chaque candidat. Un meilleur préfixe prime sur la marge;
si les préfixes sont égaux, la marge départage les candidats. Le champ
`best_versus_incumbent.margin_difference_exceeds_tolerance` rend visible une
différence de marge supérieure à la tolérance déclarée (par défaut `1e-5`,
à calibrer par analyse de sensibilité numérique). Le classement brut reste
disponible même quand les différences sont sous cette tolérance.

Sur ce snapshot, à H=20, les quatre candidats atteignent seulement 13 cycles
dans le surrogate. La proposition PACE-VR est première par une différence de
marge d'environ `5.7e-6` face au pas opposé; cette différence est inférieure
à la tolérance par défaut. À H=5, les quatre candidats terminent. Ces chiffres
ne prédisent pas le nombre de cycles du RHO réel.

Au cycle source 100 du bras droit, toujours à H=20, les quatre candidats
terminent. Les marges minimales classent `opposite_step` (`0.046972`),
`incumbent` (`0.046692`), `half_step` (`0.046509`), puis `proposal`
(`0.046293`). La proposition effectivement appliquée est donc dernière selon
ce score, avec un écart de `0.000680` face au pas opposé. C'est une alerte
concrète pour la règle actuelle de traduction sensibilité → poids : accepter
une proposition valide n'établit pas qu'elle améliore même le surrogate à
court horizon. L'écart doit encore être comparé à l'erreur du rollout et à de
vraies continuations RHO avant d'en déduire un effet sur l'endurance.

La validation forte nécessite ensuite des continuations RHO bilatérales
appariées depuis un checkpoint complet et certifié, avec gel des poids de
chaque candidat. Le lecteur `scripts/validate_pace_vr_decision_fidelity.py`
calcule concordance et regret, mais exige un reçu de redémarrage exact qui
n'est pas encore émis par le runner bilatéral. Il refuse une comparaison
causale sans ce reçu. Le test de faisabilité gelé reste nécessaire pour
attribuer un arrêt à la fatigue physiologique.
