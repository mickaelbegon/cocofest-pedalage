# Admissibilité du holdout `higher-load-208` rerun1

## Décision

`higher-load-208` n'est **pas admissible pour une comparaison causale de
transfert G→D** dans sa préinscription actuelle. Il peut toutefois servir de
test de sûreté d'**abstention** : la règle ne doit pas transférer la charge à
droite lorsque la réserve Biceps droite est déjà sous son seuil.

Cette conclusion ne change ni le seuil `.62`, ni le seuil receveur `.78`, ni
la dose `0,10 Nm`, ni la durée de 10 cycles.

## Faits lus dans `rerun1`

La source est le RHO unitaire à charge totale `2,08 Nm` (`1,04 Nm` par bras),
avec Radau-5/30 Hz, IPOPT-MA57, et checkpoints prévus 40, 60, 80, 100 et 120.
Les rapports suivants sont lus dans les archives préparées; ils ne sont pas
des interpolations :

| Checkpoint préparé après cycle certifié | Biceps G `A/a_scale` | Biceps D `A/a_scale` | Décision strictement héritée |
|---:|---:|---:|---|
| 40 | 0,8535 | 0,9169 | maintien |
| 60 | 0,8016 | 0,8876 | maintien |
| 80 | 0,7317 | 0,8479 | maintien |
| 100 | 0,6281 | 0,7982 | maintien : G est encore au-dessus de 0,62 |
| 120 | 0,5000 | 0,7478 | maintien : D est déjà sous 0,78 |

Il n'existe donc aucun checkpoint exporté qui satisfait simultanément
`G <= 0,62` et `D >= 0,78`. Le croisement éventuel entre 100 et 120 ne peut
pas être reconstruit a posteriori pour choisir une ancre : ce serait une
sélection post-résultat. Le générateur de holdout sélectionnerait seulement
un checkpoint de maintien, pas une branche de transfert.

La trajectoire commune atteint 124 cycles certifiés. Au cycle 125, G est
infeasible au travail nominal tandis que D certifie encore son cycle. Le
probe objectif nul est infeasible et le probe de repos de fatigue est
faisable. Selon le classificateur déjà implémenté, cela satisfait le label
`physiological_failure_certified` : le rétablissement de `A`, `Tau1`, `Km`
restaure la faisabilité avec tâche et PW conservés.

Ce label n'est toutefois pas une preuve de fatigue **exclusive** : les probes
de légère réduction de travail et de relâchement de PW sont également
faisables. L'interprétation correcte est une frontière de tâche locale,
fatigue-attribuable sous le protocole gelé, avec une interaction charge/PW;
non pas « le muscle fatigué est la seule cause possible ».

## Second blocage : provenance des checkpoints

Les checkpoints 40–120 ne sont actuellement que des exports préparés. Leurs
`prepared-export.json` indique `exact_bilateral_restart=false` et
`fresh_worker_replay_verified=false`; aucun `receipt.json` n'a encore été
publié. Les branches comparatives sont donc interdites avant une restauration
fraîche indépendante des deux bras, même pour un maintien.

La correction minimale, qui ne résout pas l'absence d'ancre de transfert,
est :

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=slow-layer-sim-20261003/holdout-higher-load-208-source-rerun1/results

for C in 40 60 80 100 120; do
  "$PY" scripts/validate_independent_rho_checkpoint.py \
    --run-directory "$RUN" --completed-cycles "$C"
done
```

Cette validation ne résout aucun OCP : elle reconstruit le NLP dans deux
workers frais et ne publie un reçu que si le primal, les bornes, l'historique
et le hash de problème correspondent. Après cela, `higher-load-208` peut être
scellé comme ablation d'abstention, jamais présenté comme test de gain du
transfert.

## Condition alternative préenregistrée : `higher-load-200`

Cette condition est disponible dans le builder. Elle reste une **nouvelle
condition**, et non une modification du run 2,08 Nm :

| Élément | Valeur fixée |
|---|---:|
| Charge totale | 2,00 Nm |
| Référence initiale | 1,00/1,00 Nm |
| Action inchangée | 0,90/1,10 Nm, 10 cycles, puis 1,00/1,00 Nm |
| Seuils inchangés | G `<=0,62`, D `>=0,78` |
| Poids / intégrateur / solveur | unitaires / Radau-5 30 Hz / IPOPT-MA57 |
| Exports proposés | c40, c60, puis tous les 10 cycles c80–c150 |
| Plafond | 240 cycles |

Pourquoi 2,00 Nm est une hypothèse raisonnable, mais pas une preuve : à c120,
la simulation 1,92 Nm donne environ `(G,D)=(0,699,0,849)` et celle à 2,08 Nm
donne `(0,500,0,748)`. Une interpolation purement descriptive place 2,00 Nm
près de `(0,60,0,80)`, soit au voisinage de la fenêtre d'action. Le nouveau
maillage de 10 cycles empêche de rater une fenêtre étroite. La décision reste
fondée uniquement sur les valeurs réellement certifiées, pas sur cette
interpolation.

### Go / no-go avant toute branche

1. Exécuter la source RHO unitaire et certifier chaque checkpoint disponible
   par restauration fraîche G/D.
2. Lire la règle héritée à chaque reçu. Il faut au moins un maintien précédant
   et une première ancre réellement déclenchée; sinon la condition devient un
   holdout d'abstention et aucune branche de transfert n'est lancée.
3. Vérifier que l'ancre laisse les 10 cycles d'action dans le plafond, puis
   lancer d'abord les contrôles appariés jusqu'au protocole d'arrêt gelé.
4. Lancer les deux réplications transfert seulement si tous les contrôles ont
   un endpoint attribuable. Une différence inférieure ou égale à l'étendue
   des contrôles reste une égalité numérique.

Il ne faut ni ajouter rétroactivement c102/c104 à `higher-load-208`, ni
déplacer la fenêtre des seuils, ni réduire le seuil receveur. Ces changements
transformeraient un échec de couverture en recalibrage post hoc.

Préparer la source sans lancer le RHO :

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=slow-layer-sim-20261003/holdout-higher-load-200-source
"$PY" scripts/slow_layer/prepare_causal_transfer_holdout.py baseline \
  --condition higher-load-200 --directory "$RUN"
```

Le manifeste scelle le runtime `cocofest-rho32`, le chemin et SHA-256 de
`libhsl.so` (MA57), les modèles, la grille de checkpoints et le contrat
d'action. Les branches futures exigent des `receipt.json` fraîchement
reconstruits, jamais un simple `prepared-export.json`.

## Sources lues

- [Résumé rerun1](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/slow-layer-sim-20261003/holdout-higher-load-208-source-rerun1/results/summary.json)
- [Export préparé c120](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/slow-layer-sim-20261003/holdout-higher-load-208-source-rerun1/results/checkpoints/cycle-120/prepared-export.json)
- [Configuration source](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/slow-layer-sim-20261003/holdout-higher-load-208-source-rerun1/baseline-input.json)
