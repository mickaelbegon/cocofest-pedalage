# Validation PACE-RT à deux vitesses

## Question

`PACE-RT` teste une façon de donner au RHO une information prédictive sans
construire un FHO clinique : un rollout compact estime localement une réserve
de travail future, puis le RHO suivant reçoit un objectif terminal numérique
mis à jour sans compilation du NLP.

Cette expérience ne prétend pas certifier l'endurance par le rollout. La
certification de fin d'exercice reste le test séparé de faisabilité gelée,
avec les états musculaires, mécaniques et l'historique de stimulation figés.

## Contrôleurs appariés

Les quatre conditions imposent la même tâche isocinétique, 30 Hz, Radau-5,
IPOPT/MA57 et 0,96 + 0,96 Nm.

| Condition | Poids de fatigue | Information future |
| --- | --- | --- |
| `unit` | fixes et unitaires | aucune |
| `physio` | poids physiologiques fixes | aucune |
| `pace` | rétroaction causale de capacité | aucune |
| `pace_rt` | unitaires fixes | rollout compact asynchrone et coût terminal local |

## PACE-RT

Tous les 10 cycles, le superviseur reçoit une copie certifiée de l'état et
des PW. Il simule trois cycles avec la même contrainte de travail à chaque
cycle, centre un modèle local sur l'état terminal prédit du **prochain** cycle
et transmet uniquement ses paramètres numériques au RHO. Les deux bras sont
évalués en parallèle sur deux cœurs qui ne sont pas utilisés par les solveurs
RHO.

Le coût privilégie une réserve au-dessus d'une cible bornée tout en restant
près du centre local. Il ne récompense donc pas indéfiniment une direction
affine. La v1 identifie explicitement les coordonnées `A/a_rest`; `Cn`, `F`,
`Tau1` et `Km` sont propagés dans le rollout mais ne reçoivent pas de gradient
terminal indépendant.

### Mise à l'échelle du manque de réserve

Le score du rollout est arbitrairement petit par rapport au terme proximal
adimensionnel. Lorsqu'il est activé, `normalize_terminal_shortage` divise le
manque de réserve et son lissage par la diminution maximale prédite dans la
boîte de confiance, soit la somme de `abs(gradient) * rayon`. Le poids de
manque compare alors une fraction de l'amélioration localement atteignable à
la proximité, plutôt que deux nombres dont l'échelle dépend de la définition
du score. Un gradient nul utilise le lissage comme borne inférieure afin de
préserver un problème fini. Cette échelle est un paramètre numérique du modèle
local : elle ne reconstruit pas le NLP et ne devient pas une décision du RHO.

Sur le test persistant de 40 cycles à 1,92 Nm, le terme non normalisé était
indiscernable de l'ablation « proximité seule » (écarts de capacité inférieurs
à `1e-6`). La version normalisée change effectivement les solutions : au bras
droit, le triceps terminal passe de `0,84187` à `0,84309` tandis que les
deltoïdes contribuent davantage; au bras gauche le minimum passe de `0,80389`
à `0,80366`. Cette divergence est attendue d'un contrôle actif mais ne permet
pas encore de conclure à un bénéfice d'endurance : le test jusqu'à la
faisabilité gelée et les deux bras reste nécessaire.

## Étape suivante : réserve prioritaire, fatigue secondaire

Le graphe expérimental offre maintenant une contrainte terminale de cible de
réserve, activable par `enforce_terminal_target_constraint`. Elle est
numériquement inactive avant une mise à jour PACE-RT acceptée et se formule
comme `prediction - target <= 0`. La contrainte et le coût possèdent des
activations séparées : cette séparation permet de conserver la cible puis de
retirer le coût PACE-RT lors du second solve, où la fatigue devient le
départage des solutions qui satisfont la réserve.

Cette capacité ne transforme pas encore le rollout compact actuel en valeur
d'endurance : sa cible est construite à partir d'une boîte de coordonnées,
non d'une première résolution RHO. Le protocole scientifique suivant doit
donc, hors ligne, (1) résoudre le RHO avec la réserve comme objectif, (2)
installer la valeur trouvée plus une tolérance numérique comme borne, puis
(3) résoudre le même RHO en minimisant fatigue et régularisation. Les essais
`fatigue × 0,01` et `fatigue × 1e-8` restent des ablations de conflit de
priorité; ils ne constituent pas une hiérarchie lexicographique.

Après chaque solve, l'état terminal réel est contrôlé contre le domaine de
confiance. S'il sort du domaine et qu'aucune proposition fraîche ne le
remplace, le terme PACE-RT est désactivé numériquement avant le RHO suivant.
Le NLP compilé est conservé dans les deux cas.

## Ablations

Après la comparaison principale, trois cas PACE-RT séquentiels utilisent la
même paire de cœurs : sans proximité terminale, sans terme de cible et sans
fatigue physiologique. Ce dernier garde un poids de fatigue `1e-8` uniquement
comme brise-égalité numérique au premier cycle, où PACE-RT est encore inactif;
un objectif exactement nul fait échouer IPOPT par `Invalid_Number_Detected`
bien que la sonde de faisabilité gelée soit réalisable.

## Lancement et analyse

Le script `scripts/run_pace_rt_validation_campaign.py` prépare les
configurations immuables, lance d'abord les quatre conditions en parallèle,
puis les ablations optionnelles. L'analyse extrait le préfixe certifié, les
temps IPOPT, les temps superviseur, les poids effectivement présents dans
l'objectif, les scores locaux aux états terminaux réalisés et le rayon de
confiance observé. `scripts/plot_pace_rt_validation.py` génère les quatre
figures correspondantes.

Les scores locaux mesurés à l'état terminal sont une validation du modèle
local, pas une valeur d'endurance réellement observée sur un horizon long.
