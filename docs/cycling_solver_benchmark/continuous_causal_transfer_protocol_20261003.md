# RHO continu avec transfert causal G vers D

## Question

Les pilotes redémarrés aux cycles 120 et 140 ont montré qu'une décharge
temporaire de 0,10 Nm du bras gauche vers le bras droit, pendant dix cycles,
peut retarder l'échec gelé du bras gauche. Cette expérience ne démontrait pas
qu'un superviseur déclencherait cette action au bon moment au cours d'une
trajectoire RHO complète. Le protocole `run_continuous_causal_transfer.py`
teste précisément cette boucle continue depuis le repos.

## Exécution

Deux processus persistants, un par bras, construisent chacun un NLP RHO
isocinétique réduit puis le réemploient à chaque cycle. Le calcul utilise
IPOPT/MA57, Radau-5 et 30 stimulations par cycle. Chaque frontière est un
rendez-vous : les deux solveurs retournent d'abord le cycle courant, le parent
contrôle la certification, puis choisit les deux objectifs de travail du cycle
suivant. Les modifications de charge sont des mises à jour numériques de la
borne terminale `E_prod`; elles ne reconstruisent pas le NLP.

La règle observe `A_Biceps/a_scale` des deux bras à toutes les frontières
certifiées. Si le Biceps gauche est inférieur ou égal à 0,62 et le bras droit
supérieur ou égal à 0,78, elle applique une seule séquence de dix cycles à
0,86 Nm (G) et 1,06 Nm (D). Le budget total reste 1,92 Nm. La frontière
suivante est ensuite explicitement remise à 0,96/0,96 Nm. Les seuils sont des
paramètres déclarés de `BicepsReservePolicy`, jamais des numéros de cycle.

Le contrôleur PACE historique limitait tout changement de travail aux blocs
de 20 cycles. Cette règle causale peut franchir son seuil entre deux blocs :
elle ne pourrait donc ni démarrer ni terminer une action de dix cycles avec
ce verrou. Le runner configure uniquement son adaptateur numérique de cible
`E_prod` à une frontière RHO par cycle. Les poids restent unitaires et fixes,
le feedback de capacité est désactivé, et la charge ne change jamais pendant
un cycle résolu. Ainsi, c135 peut commander c136--145, puis c146 revient à
50--50, sans reconstruction du NLP.

## Garde-fous et interprétation

La fermeture de l'état rapide (calcium, force, historique PW, phase) n'est pas
encore démontrée. Le comportement normal est donc une abstention sûre. La
reproduction de la règle issue du pilote exige
`--experimental-unvalidated-fast-state-override`; chaque reçu conserve cette
dérogation et `activation_allowed=false`.

Un échec IPOPT isolé ne compte jamais comme une limite d'endurance. Chaque bras
conserve le retry sur état gelé, le probe zéro-objectif et le contre-factuel de
repos de fatigue. Le libellé `physiological_failure_certified` exige le même
témoin local que les campagnes antérieures. Le résultat bilatéral rapporte
uniquement le préfixe certifié commun.

## Reproduction préparée

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
OUT=slow-layer-sim-20261003/continuous-causal-transfer-experimental
"$PY" scripts/slow_layer/run_continuous_causal_transfer.py prepare \
  --manifest "$OUT/manifest.json" --maximum-cycles 240 --right-cpu 0 --left-cpu 1 \
  --experimental-unvalidated-fast-state-override
"$PY" scripts/slow_layer/run_continuous_causal_transfer.py run \
  --manifest "$OUT/manifest.json" --timeout-seconds 7200
```

La première commande ne lance aucun solveur. La seconde consomme deux cœurs et
produit `decisions.json`, `progress.json` puis `campaign-completion.json`.
Elle doit être exécutée seulement après avoir vérifié que les CPU 0 et 1 sont
libres. Le manifeste est scellé : toute modification du code ou du fichier
d'entrée entre les deux commandes est refusée.
