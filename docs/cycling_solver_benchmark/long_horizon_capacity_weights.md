# Recherche des poids par projection longue de capacité

Le superviseur `predictive_moment` propose maintenant une recherche tous les
20 cycles, à partir du dernier état RHO certifié, sur 500 cycles futurs. La
configuration de référence est
`examples/fes_multibody/cycling/two_model_ding_campaign/predictive_moment_policy_x0p5.json`.
Les états calcium, force, A, Tau1 et Km sont propagés à chaque stimulation.
Chaque candidat conserve ses poids durant cette projection; le RHO réel les
réévalue au prochain point de mise à jour. Les paramètres Ding proviennent du
modèle configuré, sans trajectoire FHO.

## Pourquoi changer le critère

Une contrainte de moment exacte arrêtait toutes les projections presque au
même endroit. Demander 500 cycles à cette procédure ne créait donc pas
d'information supplémentaire. Le nouveau mode `projected_capacity` continue
le scénario de capacité: lorsque le moment demandé est hors de l'intervalle
réalisable avec les PW admissibles, il produit le moment réalisable le plus
proche et conserve explicitement l'écart avec la demande d'origine. Les PW
admissibles qui réalisent un moment accessible sont réparties par le petit
QP de recrutement pondéré existant.

Cette continuation reste à la cinématique de référence. Après un déficit,
elle est un scénario musculaire de capacité et de récupération sous demande
répétée; elle n'est plus une prédiction d'un mouvement mécaniquement réussi.
Un déficit d'approximation ne démontre pas l'incapacité du véritable OCP.

## Choix des poids

Le premier critère minimise la moyenne temporelle du déficit de moment au
carré sur les 500 cycles. Le dénominateur est la moyenne quadratique du moment
de référence, identique pour tous les candidats; les phases à moment nul
n'entraînent donc aucune division singulière. Cela pénalise à la fois la
durée et l'importance des déficits. À déficit égal, la réserve de moment
moyenne des 20 derniers cycles départage les poids, puis une égalité complète
conserve les poids en cours. Les scores sont arrondis à 12 décimales seulement
pour que le bruit numérique de l'allocation ne décide pas d'un changement.

Les neuf candidats initiaux sont les poids en cours et leurs perturbations
par muscle dans les deux directions. Les poids restent positifs et bornés.
Le superviseur travaille sur leurs rapports (moyenne géométrique fixée), comme
le coût RHO existant; cela conserve la force globale du coût par rapport aux
autres objectifs. La normalisation physiologique initiale par le maximum est
une étape distincte.

Un candidat doit avoir réellement simulé le même horizon complet que les
poids en cours. Une erreur numérique, un état hors domaine ou un horizon
partiel n'a jamais un coût de queue nul et ne peut gagner. Si les poids en
cours n'ont pas une projection complète, aucun changement n'est proposé. Le
budget est vérifié entre candidats; une projection en cours n'est pas
interrompue. Les journaux incluent chaque candidat, les erreurs éventuelles,
le premier déficit, le nombre de cycles effectivement propagés, les scores
par blocs de 20 cycles et le temps de calcul.

## Validation et portée

Les tests couvrent une véritable propagation de 500 cycles, le calcul du
déficit vis-à-vis de la demande originale, l'identité avec le mode exact
lorsqu'il est réalisable, l'exclusion des échecs numériques et la comparaison
d'horizons égaux. La mesure de coût et le choix des poids se reproduisent avec:

```bash
"$PY" scripts/validate_long_horizon_weight_rollout.py \
  --source two-speed-30hz-pilot-20260912/rho/trajectory.npz \
  --model-config examples/fes_multibody/cycling/two_model_ding_campaign/ding_triceps_alpha_a_x0p5.json \
  --reduced-profile resistance-fho-pilots-20260910/seed-0p10/reduced-cycling-fourier12.npz \
  --horizon 500 --budget-seconds 600 \
  --output long-capacity-validation/rollout500.json
```

Le lien entre poids de recrutement du prédicteur et poids de fatigue du RHO
reste un substitut à valider par des continuations RHO appariées. Le gain
d'endurance ne découle pas du score seul. Le calcul lent est actuellement
synchrone aux frontières de mise à jour; il faut mesurer son temps avant de
revendiquer une boucle clinique asynchrone.

Mesure du 12 septembre 2026, état terminal du cycle 1 de
`two-speed-30hz-pilot-20260912/rho/trajectory.npz`, modèle x0p5,
30 stimulations par cycle, 16 sous-pas de quadrature: les neuf candidats ont
tous propagé 500 cycles en 64,66 s au total. Le candidat augmentant le poids
relatif du triceps donne `[0.903602, 0.903602, 0.903602, 1.355403]`; son déficit
normalisé moyen vaut 0,00470621, contre 0,00531382 pour les poids uniformes,
soit une réduction de 11,43 % du critère de projection. Le premier déficit
apparaît dans les deux cas au cycle futur 74 (indice 73), sous la borne
inférieure de moment: la force résiduelle empêche ici de réduire assez le
moment à cette phase. Ce résultat n'est donc pas une preuve d'épuisement.
Un autre candidat retarde la première violation mais a un déficit cumulé
plus élevé; le critère long terme ne maximise pas le temps jusqu'à la
première violation exacte. Rapport: `long-capacity-validation-20260912/rollout500.json`.
