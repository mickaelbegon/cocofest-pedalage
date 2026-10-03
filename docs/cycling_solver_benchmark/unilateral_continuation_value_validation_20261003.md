# Valeur de continuation unilatérale : analyse préenregistrée

## Cadre et état

Le bras gauche fournit le travail correspondant à 0,96 Nm par cycle. Les ancres sont les checkpoints exacts c120 et c140 du RHO à poids unitaires. Chaque action modifie les poids de fatigue pendant 1 ou 3 cycles; un autre processus restaure l'endpoint et poursuit avec les poids unitaires jusqu'au protocole d'arrêt. La campagne comporte 4 témoins, 24 interventions d'apprentissage et 8 interventions de validation. Les deux durées d'une action et les deux ancres de cette action gardent la même partition.

Les 36 branches sont terminées et donnent une fin physiologique certifiée. Les quatre témoins reproduisent tous le dernier cycle certifié `c169`; les 32 interventions contrastées ont également un avantage exact nul après retour aux poids unitaires. Le rapport final est dans `docs/cycling_solver_benchmark/unilateral_continuation_label_results_20261003.md` et le résultat machine dans `asymmetric-sides-r192-rho-bo-20260928/unilateral-continuation-labels-20261003/value-analysis-final-20261003.json`. Les portes de validation directionnelle échouent donc, et aucun gradient d'endurance appris n'est libéré.

## Définition du label

Pour une action et son témoin unitaire ayant la même ancre et la même durée d'intervention, l'avantage est la différence entre leurs **derniers cycles certifiés** sous la politique de retour unitaire. Cette différence est un label exact dans cette étude seulement si les deux reçus satisfont le critère local de fin : préfixe RHO audité, absence de témoin faisable dans le test à objectif nul avec statut d'infaisabilité locale, et témoin faisable lorsque la fatigue est remise à l'état de repos, sans avancer le RHO réel. Cela ne démontre pas une infaisabilité globale du problème non convexe.

Un arrêt numérique, un délai ou la limite de 240 cycles ne fournissent pas de label exact. Ils sont recensés séparément. Aucun échec d'IPOPT seul n'est transformé en durée physiologique.

## Coordonnées et limites du modèle

La coordonnée primaire est `A/a_scale` pour chacun des quatre muscles, lue au premier nœud du problème préparé après l'intervention. Les équations lentes de Ding impliquent que les combinaisons

`Tau1 - Tau1_repos - (alpha_tau1/alpha_a)(A - A_repos)` et
`Km - Km_repos - (alpha_km/alpha_a)(A - A_repos)`

évoluent sans excitation de force, uniquement par récupération exponentielle. Elles ne sont pas quatre degrés de liberté supplémentaires lorsque les offsets initiaux sont nuls. Les checkpoints c120/c140/c160, à poids unitaires et BO, ont des offsets nuls à la précision inspectée. Le lecteur recalcule ces offsets pour chaque endpoint et refuse le modèle à quatre capacités si l'offset relatif dépasse `1e-7`.

Le calcium `Cn`, la force `F`, les 30 PW précédentes par muscle et les états mécaniques sont conservés dans l'audit. Le modèle primaire à quatre capacités ne prouve pas que ces états rapides n'influencent pas la continuation; même une bonne prédiction des holdouts ne justifiera pas encore une activation clinique. Ces variables doivent être examinées comme facteurs de confusion avant de promouvoir le gradient.

## Ajustement et validation fixés avant les contrastes

Les différences d'état et de cycles sont centrées sur le témoin unitaire **de même ancre et même durée**. L'ajustement principal utilise uniquement les endpoints après 3 cycles. Leurs homologues après 1 cycle servent au contrôle de cohérence temporelle; ils ne sont pas comptés comme échantillons indépendants, car ils proviennent des mêmes familles d'actions.

Pour chaque ancre séparément, une SVD des différences de capacités des six familles d'apprentissage détermine le sous-espace réellement atteint. La régression est contrainte à ce sous-espace, sans prétendre identifier les directions manquantes. Les deux familles mixtes préenregistrées restent entièrement hors entraînement. Les conditions de passage du diagnostic directionnel sont :

1. Quatre témoins c169 avec le même critère de fin, et au moins deux familles holdout exactes par ancre.
2. Rang atteint au moins 1, conditionnement au plus `1e5`.
3. Résidu de projection de chaque holdout hors du sous-espace entraîné au plus `0,1` en unités normalisées; chaque coordonnée projetée reste dans l'étendue observée à l'entraînement.
4. Les variations des états rapides, des PW antérieures et des états mécaniques des holdouts restent dans l'enveloppe composante par composante des interventions d'apprentissage appariées, avec une tolérance numérique de `1e-9` dans les unités des archives. Ce test de support ne prouve pas que ces états n'influencent pas la durée.
5. Au moins deux avantages holdout non nuls, les deux signes correctement prédits (`>=80 %` sur deux), et la paire de holdouts correctement classée (`>=75 %` sur une paire).

Un avantage nul n'apporte pas de preuve de signe. Si un holdout est censuré ou numérique, la porte est indéterminée plutôt que réussie. Ces seuils sont des filtres de recherche, pas une garantie statistique ou physiologique. Un passage aux deux ancres autoriserait seulement des branches causales appariées `+g`, `−g`, demi-amplitude et zéro, suivies de la même continuation. Il n'activerait pas directement le costate dans une campagne longue.

## Traçabilité et exécution

`scripts/analyze_unilateral_continuation_value_20261003.py` vérifie les SHA-256 du manifeste, des ancres, des témoins NPZ, des deux reçus de phase, des archives d'endpoint et du résultat du job. Il vérifie aussi le transfert entre processus, le travail de 0,96 Nm, la politique de retour, la continuité des cycles et leurs résidus NLP indépendants. Les 4 reçus témoins ont été lus avec succès; les tests synthétiques vérifient notamment la séparation des familles, les offsets Ding et le rejet d'un arrêt non certifié.

Exemple de nouvelle analyse, dans un fichier qui n'existe pas encore :

```bash
taskset -c 17 /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/analyze_unilateral_continuation_value_20261003.py \
  --manifest asymmetric-sides-r192-rho-bo-20260928/unilateral-continuation-labels-20261003/manifest.json \
  --output asymmetric-sides-r192-rho-bo-20260928/unilateral-continuation-labels-20261003/value-analysis-next.json
```

L'analyse est en lecture seule sur les résultats de simulation. Elle crée uniquement le nouveau rapport indiqué par `--output` et refuse d'écraser un rapport existant.
