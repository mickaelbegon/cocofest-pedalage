# Superviseur causal expérimental de transfert G→D

## Objet

Ce protocole teste un unique levier causal mis en évidence par les branches
auditées : diminuer temporairement la résistance cible du bras gauche de
0,10 Nm et transférer exactement cette charge vers le bras droit. Il ne
change ni les poids musculaires ni le NLP RHO de chaque bras. Chaque OCP reste
donc un RHO unilatéral compilé, à Radau-5, 30 Hz, IPOPT avec MA57.

Le superviseur est **désactivé par défaut**. Il n'est ni un prédicteur de
faisabilité ni une règle clinique.

## Contrat causal

À une frontière RHO certifiée, le runner crée une `TransferObservation` :

- l'index de cycle complété;
- les états lents archivés `A`, `Tau1`, `Km` des deux bras;
- les empreintes SHA-256 du checkpoint, des modèles et de cette observation.

Une politique pure choisit ensuite l'action. La seule politique autorisée par
ce runner est `BicepsReservePolicy` : elle teste le rapport archivé et
normalisé `A_Biceps/a_scale_Biceps`, pas la valeur brute de `A`. Le déclencheur
est `r_B,G <= 0,62`, seulement si `r_B,D >= 0,78` et si une porte d'état rapide
validée est fournie. Entre 0,62 et 0,67, elle s'abstient. Ces quantités restent
des **signaux expérimentaux**, pas une capacité physiologique validée.

Lorsque le seuil est atteint, le bloc suivant reçoit exactement :

| Fenêtre | G | D | Total |
|---|---:|---:|---:|
| 10 cycles d'action | 0,86 Nm | 1,06 Nm | 1,92 Nm |
| Tout cycle ultérieur | 0,96 Nm | 0,96 Nm | 1,92 Nm |

La durée de dix cycles, l'amplitude et les niveaux de référence sont dans
`TransferConfig`; aucun numéro de cycle n'est codé dans le noyau de la
politique. Une action unique est la valeur par défaut : un déclenchement ne
se transforme pas silencieusement en redistribution permanente.

## Benchmark reproductible

Le runner crée, pour chaque checkpoint et réplication, une branche témoin à
50–50 et une branche de politique. Les branches sont des processus neufs et
restaurent le même checkpoint byte-pinné. Elles poursuivent ensuite jusqu'au
plafond ou au protocole d'échec gelé. Les répétitions sont une vérification
numérique, pas des réplications biologiques.

Préparer sans lancer :

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=slow-layer-sim-20261003/causal-transfer-threshold-a-triceps
"$PY" scripts/slow_layer/run_causal_transfer_supervisor.py preflight \
  --fast-state-gate-receipt /chemin/vers/fast-state-gate.json

"$PY" scripts/slow_layer/run_causal_transfer_supervisor.py prepare \
  --manifest "$RUN/manifest.json" \
  --fast-state-gate-receipt /chemin/vers/fast-state-gate.json \
  --maximum-cycles 240 --replicates 2
```

Le reçu `fast-state-gate.json` doit avoir le schéma
`causal-transfer-fast-state-gate-v1`, le statut `validated`, et identifier
explicitement l'empreinte du checkpoint et les deux empreintes de modèles.
Sans ce reçu, le preflight est une abstention. La seule exception est la
dérogation hors ligne explicitement étiquetée ci-dessous; elle ne constitue
pas une validation de la porte rapide. Ensuite seulement :

```bash
"$PY" scripts/slow_layer/run_causal_transfer_supervisor.py campaign \
  --manifest "$RUN/manifest.json" --cpus 0 1 2 3 --timeout-seconds 7200
```

Pour un benchmark hors ligne strictement expérimental, il existe une
dérogation explicite :

```bash
"$PY" scripts/slow_layer/run_causal_transfer_supervisor.py prepare \
  --manifest "$RUN/manifest.json" \
  --experimental-unvalidated-fast-state-override \
  --maximum-cycles 240 --replicates 2
```

Elle est inscrite dans le manifeste comme
`fast_state_status=unvalidated_experimental_override`, avec un avertissement,
et laisse `activation_allowed=false`. Elle n'est jamais appelée « validée » et
ne peut servir qu'à mesurer le comportement end-to-end de la politique : c120
reste en maintien car sa réserve Biceps est au-dessus de la bande, alors que
c140 peut déclencher l'action. Le défaut sans reçu ni dérogation reste une
abstention.

Avant les branches de politique, tous les contrôles répétés doivent atteindre
un endpoint attribuable par la faisabilité gelée. Les cycles certifiés
bilatéraux communs sont les seuls utilisés pour l'endurance et le travail;
une trajectoire unilatérale qui dépasse l'autre reste explicitement
contrefactuelle.

## Interprétation prévue

Le gain pertinent est l'écart de cycles certifiés de la branche politique par
rapport à ses contrôles, comparé à l'étendue des contrôles. Un écart inférieur
ou égal à cette étendue est une égalité numérique. Un succès IPOPT isolé, une
sortie au plafond, ou l'amélioration d'un seul bras ne suffisent pas à établir
un bénéfice d'endurance.
