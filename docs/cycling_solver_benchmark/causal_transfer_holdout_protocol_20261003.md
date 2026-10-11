# Holdouts sans recalibrage du superviseur causal G→D

## Question

Le gain local observé avec une décharge temporaire de G ne devient une règle
d'endurance que s'il survit à un changement de contexte. Ce protocole teste
la règle scellée, sans modifier ses paramètres : seuil Biceps gauche
`A/a_scale <= 0,62`, réserve minimale à droite `>= 0,78`, décharge G→D de
`0,10 Nm`, pendant exactement 10 cycles, puis retour explicite au partage
initial.

Les holdouts ne sont pas une ré-optimisation du seuil, de la dose, de la
durée, du sens de transfert ou des poids musculaires. Toute tentative de les
modifier est rejetée par le contrat.

## Trois contextes pré-enregistrés

| Identifiant | Changement hors distribution | Référence par bras | Rôle attendu de la politique |
|---|---|---:|---|
| `higher-load-208` | mêmes modèles Ding, charge totale 2,08 Nm au lieu de 1,92 Nm | 1,04 Nm | vérifier si la règle peut déclencher plus tôt sous charge accrue |
| `higher-load-200` | mêmes modèles Ding, charge totale 2,00 Nm et exports denses c80–c150 | 1,00 Nm | capturer sans réajustement une éventuelle fenêtre étroite de seuils |
| `reversed-ding-sides-192` | paramètres Ding favorable/défavorable inversés entre G et D, charge totale 1,92 Nm | 0,96 Nm | vérifier l'abstention : G n'est plus a priori le bras déficitaire |

Les deux premiers sont des holdouts de charge (+8,3 % et +4,2 % de travail)
et le troisième un holdout directionnel de paramètres. L'inversion ne crée pas de nouveau jeu de
paramètres : elle réutilise les deux modèles explicitement déclarés, mais les
attache à l'autre bras. Les empreintes des fichiers sont scellées avant tout
calcul.

## Exécution en deux étapes

La première étape fait tourner seulement le RHO unitaire de référence et
exporte un checkpoint commun à chaque 20 cycles. Elle doit finir avec le
protocole d'échec gelé habituel (retries, faisabilité objectif nul, témoin de
repos pour fatigue). Un statut IPOPT seul n'est jamais un échec.

Le fichier d'entrée généré épingle aussi le `runtime_prefix` RHO32 et le
chemin absolu de `libhsl.so` MA57, avec son SHA-256. La préparation refuse une
bibliothèque absente ou modifiée avant tout lancement. Cela évite qu'un
processus démarré depuis conda `base` retourne immédiatement `Invalid_Option`,
ce qui est une erreur de configuration du solveur et non un échec mécanique.

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
RUN=slow-layer-sim-20261003/holdout-higher-load-200-source

"$PY" scripts/slow_layer/prepare_causal_transfer_holdout.py baseline \
  --condition higher-load-200 --directory "$RUN"

"$PY" scripts/run_independent_isokinetic_arms.py \
  --config "$RUN/baseline-input.json" --output "$RUN/results"
```

Après le run, le préflight lit exclusivement les reçus de checkpoints
certifiés par restauration fraîche G/D. Les exports préparés seuls sont
explicitement refusés. Il sélectionne le dernier checkpoint où la politique s'abstient,
puis le premier où elle déclenche. S'il n'existe aucun déclenchement (résultat
attendu possible pour l'inversion Ding), il conserve un unique checkpoint
d'abstention : c'est une falsification utile, pas un motif pour changer le
seuil.

La porte d'état rapide n'est pas encore validée. L'option suivante est donc
explicitement **hors ligne et expérimentale** ; elle inscrit ce statut dans le
manifeste et interdit toute activation clinique.

```bash
"$PY" scripts/slow_layer/prepare_causal_transfer_holdout.py preflight \
  --condition higher-load-200 --directory "$RUN" \
  --experimental-unvalidated-fast-state-override

BRANCH=slow-layer-sim-20261003/holdout-higher-load-200-branches
"$PY" scripts/slow_layer/prepare_causal_transfer_holdout.py seal-branches \
  --condition higher-load-200 --directory "$RUN" --output "$BRANCH" \
  --experimental-unvalidated-fast-state-override --replicates 2

"$PY" scripts/slow_layer/run_causal_transfer_supervisor.py campaign \
  --manifest "$BRANCH/branches/manifest.json" --cpus 0 1 2 3 \
  --timeout-seconds 7200
```

Pour les autres contextes, remplacer `higher-load-200` dans les commandes par
`higher-load-208` ou `reversed-ding-sides-192`. Une preuve future de fermeture des états rapides
peut remplacer la dérogation par `--fast-state-gate-receipt`; elle doit être
liée à chaque checkpoint et aux deux hashes de modèles de ce holdout.

## Critères d'analyse gelés

1. Le contrôle et la politique partent exactement du même checkpoint bilatéral
   préparé, dans des processus frais. Les répétitions ne sont que des contrôles
   numériques.
2. On ne compare que le préfixe de cycles certifié par **les deux** bras. Les
   lignes qui survivent sur un seul bras sont contrefactuelles.
3. Le gain est la différence de cycles certifiés avec les contrôles appariés.
   Une différence qui ne dépasse pas l'étendue des contrôles est une égalité
   numérique.
4. Une amélioration locale ne vaut pas si le bras receveur acquiert un échec
   plus précoce; l'état final, le travail et le bras limitant restent donc
   rapportés.
5. Absence de déclenchement dans le holdout Ding inversé est un résultat
   favorable de sûreté de la règle, mais pas une preuve d'efficacité.

Le runner générique accepte une référence autre que 0,96/0,96 seulement si
elle est explicitement attachée aux reçus source. Ainsi, le changement de
charge de 2,08 Nm fait revenir l'action à 1,04/1,04 Nm, non pas au niveau
calibré 0,96/0,96 Nm; le transfert reste exactement −0,10/+0,10 Nm.
