# PACE-VR: validation appariée avant branchement au RHO

Le classement d'un rollout de 100 cycles doit être comparé à de vraies
continuations RHO, toutes lancées depuis **le même état complet certifié**.
Les points de départ prévus sont les cycles 40, 100 et 150. À chaque point,
évaluer au minimum les candidats `unit`, `physio`, `bo_fixed` et
`intermediate`; ajouter le candidat sélectionné par PACE-VR s'il diffère.
Chaque intervention gèle son choix de poids pendant la continuation. Il faut
garder la même charge totale, sa répartition G/D, les modèles Ding, les bornes
PW, la discrétisation, le solveur et la tolérance. Les résultats BO historiques
obtenus depuis un autre état ne constituent pas cette comparaison appariée.

## Reçu exigé pour une reprise exacte

Le validateur `scripts/validate_pace_vr_decision_fidelity.py` refuse de calculer
un classement causal sans un reçu JSON vérifiable. Son schéma minimal est :

```json
{
  "schema_version": 1,
  "checkpoint_kind": "certified_bilateral_shifted_primal",
  "exact_bilateral_restart": true,
  "completed_cycles": 40,
  "configuration_path": "configuration.json",
  "configuration_sha256": "<64 caractères hexadécimaux>",
  "arms": {
    "right": {
      "completed_cycles": 40,
      "certified": true,
      "stimulation_history_complete": true,
      "replay_roundtrip_exact": true,
      "primal_path": "right.npz",
      "sha256": "<64 caractères hexadécimaux>",
      "model_path": "right-model.json",
      "model_sha256": "<64 caractères hexadécimaux>",
      "prepared_problem_sha256": "<64 caractères hexadécimaux>",
      "restored_problem_sha256": "<même empreinte>"
    },
    "left": {"...": "mêmes champs que right"}
  }
}
```

Les chemins sont relatifs au reçu. Chaque archive NPZ doit contenir des états,
des contrôles et une métadonnée `producer_mode=rho_replay_checkpoint` avec le
cycle source. Les empreintes protègent contre une substitution accidentelle de
fichiers; elles ne démontrent pas, à elles seules, que la dynamique est
restaurée. Le worker doit calculer les deux empreintes du problème préparé à
partir d'une sérialisation canonique des états Ding et mécaniques, de
l'historique des stimulations, du primal translaté, des bornes, des paramètres
fixes, de la charge et de ses réglages numériques. Après rechargement, il doit
reconstruire la même empreinte **avant** d'appliquer les poids du candidat.
L'égalité et la présence de l'historique doivent être attestées par le worker,
pas déclarées manuellement.

Le runner bilatéral peut maintenant exporter ces données à une frontière
explicitement demandée avec `prepared_restart_checkpoint_cycles`, par exemple
`[20, 40, 60]` dans le JSON de campagne. Chaque worker écrit
`checkpoints/cycle-20/{right,left}.npz` après le décalage natif du RHO et après
la mise à jour numérique de charge/poids. La barrière parent écrit ensuite
`prepared-export.json` seulement si les deux archives sont présentes. L'export
est vérifié immédiatement par désérialisation : signature du primal et digest
du problème préparé doivent être identiques.

Le digest inclut aussi l'horloge absolue de manivelle utilisée par le driver
(indice de cycle, origine et décalage angulaire). Cela évite une reprise qui
serait correcte au premier cycle mais utiliserait une référence angulaire
décalée lors de la fenêtre suivante.

La routine de restauration est également disponible dans
`cocofest.simulation.rho_restart_checkpoint.restore_prepared_checkpoint`. Elle
refuse un nom de variable, une borne, un paramètre fixe ou une forme de tableau
différents; après restauration, elle recalcule le digest. Elle rend donc la
prochaine validation déterministe : lancer un worker neuf à partir de la même
configuration et exiger son digest restauré avant de produire les quatre
continuations contre-factuelles.

Quand un export provisoire existe, la certification dans deux processus neufs
se lance ainsi :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/validate_independent_rho_checkpoint.py \
  --run-directory CHEMIN/results/pace_vr --completed-cycles 20
```

La commande ne résout pas de cycle : elle ne fait que reconstruire puis
restaurer les deux NLP. Si les deux digests correspondent, elle crée
`checkpoints/cycle-20/receipt.json`. C'est ce reçu — et non
`prepared-export.json` — que les continuations appariées doivent fournir à
`validate_pace_vr_decision_fidelity.py`.

Ce fichier est volontairement distinct de `receipt.json`: il porte
`exact_bilateral_restart=false` et `fresh_worker_replay_verified=false`.
L'export ne suffit pas à autoriser une comparaison causale. La prochaine
étape est de reconstruire les deux NLP dans de nouveaux workers, restaurer les
archives, vérifier le même `prepared_problem_sha256`, puis seulement publier
le reçu final avec `replay_roundtrip_exact=true`. Le validateur continue donc
de retourner `missing_exact_bilateral_restart` tant que cette dernière preuve
n'existe pas; il ne reconstruit jamais un état à partir de résumés de fatigue.

On peut vérifier un cycle existant sans supposer qu'un snapshot PACE-VR est
redémarrable :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/audit_independent_rho_restart.py \
  --run-directory CHEMIN/results/pace_vr --completed-cycles 20
```

Sur la campagne H=100/50/25/20/5, le cycle 20 est certifié pour les deux bras
et fournit les snapshots de projection, mais l'audit retourne
`matched_continuation_ready=false` et `missing_exact_bilateral_restart`. Les
snapshots conservent l'état terminal Ding et les PW du cycle source; ils ne
contiennent ni le primal décalé, ni les bornes actives, ni une attestation de
l'historique de stimulation et des paramètres fixes restaurés. Le plus petit
ajout au producteur est un export atomique de ces éléments à la barrière
bilatérale, suivi d'une reprise dans deux nouveaux workers et d'une empreinte
identique du problème préparé avant toute modification des poids.

## Lecture du manifeste et du résultat

Le manifeste contient pour chaque checkpoint `completed_cycles`,
`horizon_cycles`, `restart_receipt`, `selected_candidate_id` et une liste
`candidates`. Chaque candidat fournit `id`, `prediction_file` et
`continuation_summary`. Tous les chemins sont relatifs au manifeste. Une
prédiction acceptée fournit `feasible_prefix_cycles`, `minimum_task_margin`,
`terminal_value`, `work_residual_max` et `constraint_violation_max`. Les seuils
optionnels `maximum_work_residual` et `maximum_constraint_violation` sont
exprimés dans les unités propres au score et doivent être déclarés dans le
manifeste si un filtrage supplémentaire est voulu.

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/validate_pace_vr_decision_fidelity.py \
  --manifest CHEMIN/manifest.json
```

La commande est relançable au fur et à mesure que les résultats arrivent.
Elle calcule la concordance des paires dont l'ordre réel est établi par les
cycles certifiés; Spearman exige des échecs dus à la fatigue certifiés pour
tous les candidats retenus. Une continuation atteignant seulement sa limite
de cycles est censurée. Le regret est exact si tous les endpoints le sont,
sinon seule une borne inférieure peut être calculée. L'échec du petit solveur
de projection et un arrêt technique du RHO conservent leurs statuts distincts
et ne sont jamais rebaptisés « échec physiologique ».
