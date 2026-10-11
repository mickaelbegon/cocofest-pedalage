# Validation par paires des états rapides — porte expérimentale, pas fermeture de Markov

## Objet et limite

Les états lents Ding `A`, `Tau1`, `Km` ne suffisent pas, par hypothèse, à
omettre calcium, force, historique PW et mécanique de `Phi`. Une enveloppe
min/max ou un nombre fini de tests ne peut pas établir une fermeture de
Markov. Cette procédure fournit seulement une porte beaucoup plus étroite :
dans les paires réalisables et lentes-identiques préenregistrées, les états
rapides testés ne changent pas le classement des trois actions au-delà de la
résolution numérique observée.

Même si tous les critères passent, le reçu porte
`closure_established=false`, `activation_allowed=false`, et le superviseur
archive le statut `validated_paired_local_robustness_not_closure`.

## Contrat préenregistré

Le point de départ est
[preregistered-left.json](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/slow-layer-sim-20261003/fast-probes/preregistered-left.json).
Pour chaque checkpoint source, il faut créer, par un matching RHO audité, une
paire de frontières qui possède exactement les mêmes :

- hash de modèle, tâche, règle de retour, règle d'arrêt et discrétisation ;
- hash de frontière lente `A`, `Tau1`, `Km` ;
- action parmi −0,05, 0 ou +0,05 Nm G→D.

Les deux membres doivent au contraire avoir des hashes d'états rapides
distincts. Les groupes `Cn`, `F`, `PW_history` et `mechanics` doivent être
soit différents dans une paire admissible, soit documentés comme invariants
par les contraintes/la cadence réellement archivées. Une paire de trajectoires
simplement « proches » est refusée : elle confond état lent et état rapide.

Chaque résultat doit être audité, posséder un endpoint local gelé, et fournir
l'intégrale de force pondérée par muscle. Une censure à droite, un arrêt
numérique, un audit absent ou une couverture incomplète est **inconclusif**,
jamais une équivalence.

## Contrôles et décision gelée

Trois contrôles indépendants par source et action mesurent l'étendue de cycles
certifiés. Pour chaque paire/action, la différence de cycles doit rester dans
cette étendue; la faisabilité doit être identique. Le classement des trois
actions doit également rester identique, en considérant comme égalité les
écarts dans la résolution des contrôles. Une différence résolue, une inversion
de classement ou un changement de faisabilité rejette la robustesse locale.

Les intégrales de force pondérées sont obligatoires pour l'audit et le futur
diagnostic mécanistique; aucune tolérance physiologique a posteriori n'est
inventée pour les transformer en preuve d'équivalence.

## Exécution

Préparer le contrat, sans simulation :

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=slow-layer-sim-20261003/fast-state-paired-validation

"$PY" scripts/slow_layer/run_fast_state_pair_validation.py prepare \
  --preregistration slow-layer-sim-20261003/fast-probes/preregistered-left.json \
  --output "$RUN"
```

Après production d'un fichier d'évidence `slow-phi-fast-pair-evidence-v1`
par un runner de matching RHO audité :

```bash
"$PY" scripts/slow_layer/run_fast_state_pair_validation.py assess \
  --validation-plan "$RUN/validation-plan.json" \
  --evidence /chemin/vers/evidence.json \
  --output "$RUN/assessment.json" \
  --gate-output "$RUN/fast-state-gate.json"
```

Sans `--evidence`, l'outil produit délibérément un rapport `inconclusive` et
refuse de créer le reçu de porte. Le reçu
`causal-transfer-fast-state-gate-v1` ne peut être écrit qu'après un résultat
`validated_local_robustness`; il est lié aux hashes bilatéraux des checkpoints
et modèles. Il reste un permis de benchmark hors ligne à portée locale, pas
une autorisation clinique ni une démonstration de suffisance d'état.
