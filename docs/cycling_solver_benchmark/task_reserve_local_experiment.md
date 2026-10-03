# Réserve de tâche : protocole local après la première calibration

## Ce que montrent les résultats du 30 septembre 2026

La grille de travail `1.00, 1.05, 1.10, 1.15, 1.20` ne mesure pas une charge
maximale. À droite, le facteur 1.20 est réalisable à chacun des checkpoints
20–160 : la réserve est **au moins** 20 %. La constance de ce nombre vient
du plafond de grille et ne démontre pas une réserve physiologique constante.
À gauche, les meilleures réserves témoignées sont 15 % au cycle 140 et 5 %
au cycle 160. Les résolutions non validées au-dessus restent indéterminées ;
elles ne fournissent aucune borne supérieure de la capacité.

Le fit gauche global est rejeté : erreurs maximales de 0.119 en apprentissage
et 0.203 en holdout après correction empirique du biais. Son rang algébrique
est complet, mais son conditionnement atteint environ 88 322. Quatre états
`A/a_scale` évoluant sur la même trajectoire peuvent donner un rang complet
par faible courbure, sans identifier ce qui arrivera lorsque le RHO change
la répartition de stimulation. Un gradient obtenu ainsi pourrait extrapoler
dans une direction qui n'a jamais été observée.

Le planificateur `scripts/plan_local_task_reserve_experiment.py` relit les
receipts, archives et témoins des sondes, revérifie leurs empreintes et leurs
résidus rapportés, puis produit un **plan d'expérience**. Il n'exécute pas de
RHO, ne publie pas de modèle et n'active aucun coût. Il n'effectue pas non plus
un nouveau replay DOP853.

## Une localité choisie avant les nouvelles simulations

Commencer par le bras gauche au checkpoint 140, assez proche de l'échec pour
que la réserve renseigne la décision, mais avec plusieurs cycles nominaux
encore réalisables. Utiliser le checkpoint 160 comme seconde situation plus
critique. Le bras droit est ensuite utile pour contrôler que la méthode ne
produit pas artificiellement une direction à partir du plafond de grille.

La boîte pilote est centrée sur les valeurs `A/a_scale` du checkpoint, avec
un rayon de 0.05 par muscle. Ce rayon est une décision expérimentale, pas un
résultat de validation. La boîte n'est jamais élargie automatiquement pour
atteindre le nombre minimal de points. Avec ce choix, chaque checkpoint
gauche 120, 140 et 160 n'a actuellement qu'un point local. Il faut donc créer
des observations locales atteignables.

Les résultats 20–160 sont désormais des données exploratoires : ils ont servi
à choisir ce protocole. Leur ancien holdout ne peut pas devenir un nouveau
test confirmatoire après ces choix. Le script conserve les anciennes
partitions pour documenter le diagnostic, puis préattribue les **nouvelles**
branches d'apprentissage et de holdout.

```bash
"/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python" \
  scripts/plan_local_task_reserve_experiment.py \
  --calibration-audit asymmetric-sides-r192-rho-bo-20260928/task-reserve-calibration-20260930/left/calibration-audit.json \
  --anchor c140 \
  --coordinate-radius 0.05 \
  --output /tmp/task-reserve-local-left-c140-plan.json
```

Le fichier de sortie doit être nouveau ; le script refuse de l'écraser.
Sans `--anchor`, les trois derniers checkpoints sont proposés. La commande
ci-dessus limite volontairement le premier pilote à un seul ancrage.

## Générer des états différents avec de vrais cycles RHO

Restaurer exactement le même checkpoint pour chaque branche. Garder le
modèle, le travail nominal, la cadence, la grille d'intégration, les bornes
PW et la contrainte de demi-pas. Faire varier uniquement les pondérations
de fatigue pendant quelques cycles complets validés. Le prochain checkpoint
contient alors `Cn`, `F`, `A`, `Tau1`, `Km`, les états mécaniques et l'historique
de stimulation effectivement atteints.

Pour quatre muscles, le plan utilise :

- Une branche unitaire, avec checkpoints après 1 et 3 cycles.
- Trois directions indépendantes de contraste entre muscles, chacune avec
  perturbation positive et négative : six branches, checkpoints après 1 et
  3 cycles. Les contrastes sont orthogonaux dans les log-poids ; l'amplitude
  pilote est 0.25 et chaque vecteur de poids conserve une moyenne de 1.
- Deux combinaisons de ces directions, amplitude 0.15, réservées au holdout,
  avec checkpoints après 2 et 4 cycles. Ces branches ne servent pas au fit.

Cela représente au maximum 29 cycles RHO et 18 endpoints par ancrage avant
les sondes. Deux endpoints d'une même branche partagent un préfixe : ils
restent dans la même partition et ne comptent pas comme deux trajectoires
indépendantes. Un holdout est ici une nouvelle politique atteignable ; la
répétition du protocole à un autre ancrage constitue une validation de
transport distincte.

Les trois directions de poids ne garantissent pas quatre directions d'état
indépendantes. La dérive de fatigue entre horizons de 1 et 3 cycles peut en
apporter une supplémentaire, mais cela doit être mesuré. **Auditer rang et
conditionnement des endpoints avant de lancer la grille de charge.** Si le
rang ou l'excitation est insuffisant, ne pas ajuster les quatre composantes
d'un gradient. Réviser les politiques expérimentales, ou tester une
description de dimension plus faible limitée au sous-espace réellement
atteint, avec un nouveau protocole et un nouveau holdout.

L'adaptateur de continuation doit injecter les poids paramétriques après
restauration exacte, puis exporter et certifier les endpoints. Le
planificateur ne fournit pas cet exécuteur. Il ne modifie jamais directement
`A`, et ne compte pas différents facteurs de travail résolus depuis un même
checkpoint comme des échantillons d'état distincts.

## Raffiner la mesure de réserve sans fabriquer de frontière

Une fois les endpoints atteignables vérifiés, appliquer la même grille à
tous les endpoints retenus d'une expérience. La grille initiale est raffinée
uniformément à un pas maximal de 0.025. Si un plafond exploratoire était
atteint, ajouter des sondes plus élevées, par défaut 1.30, 1.40 et 1.50. Ce
plafond 1.50 est un budget d'exploration choisi, jamais une borne supérieure
physiologique. Si la borne inférieure y est encore censurée, le constat doit
rester explicite.

Le facteur nominal 1.00 est toujours résolu. Tous les facteurs prévus sont
évalués, même après un échec numérique, sans hypothèse de monotonie et sans
dichotomie basée sur le statut IPOPT. Une grille à 0.025 n'implique pas que
la frontière réelle soit connue à ±0.025. Elle fournit seulement des
témoins à cette résolution. Des initialisations alternatives aux facteurs
ambigus servent à quantifier la dépendance numérique de la borne observée.

À quatre muscles et douze facteurs, le budget maximal est de 216 sondes
d'un cycle par ancrage, avant exclusions d'endpoints hors boîte ou invalides.
Ce budget est celui d'une validation scientifique initiale, pas celui d'une
mise à jour clinique. Réduire ensuite le nombre de sondes du superviseur
exige un benchmark séparé de latence, de stabilité et de qualité décisionnelle.

## Conditions avant d'activer un coût terminal

L'ensemble d'apprentissage doit renseigner l'intercept et chaque direction
d'état conservée dans la boîte déclarée. Évaluer la sensibilité des
coefficients aux branches plutôt qu'aux seuls points, et vérifier leur
stabilité quand la grille de travail est raffinée. Préfixer la tolérance
admissible en réserve, les seuils de conditionnement et le nombre minimal
de branches avant d'examiner le nouveau holdout. Un fit de bornes inférieures
peut être utile empiriquement ; son acceptation ne devient pas un certificat
de frontière physiologique.

L'état complet du modèle de Ding reste une variable de confusion possible
pour un fit limité à `A/a_scale`. Si deux branches atteignent des `A` très
proches mais des réserves différentes au-delà de la reproductibilité
numérique, ajouter des coordonnées normalisées `Tau1`/`Km` ou abandonner ce
fit au profit d'une comparaison directe de politiques courtes. Ajouter des
variables augmente le besoin de données et ne se fait pas après avoir
consommé le holdout sans créer un nouveau test.

Enfin, depuis le même checkpoint exact, comparer le coût accepté, le coût
unitaire et une direction opposée sur de vraies continuations RHO, avec
limite d'âge et domaine de confiance du modèle. La question décisive est
double : une marge prédite plus grande devient-elle une marge effectivement
témoignée plus grande, et cette amélioration conserve-t-elle davantage de
cycles nominaux ? Seule cette étape justifie ensuite un essai d'endurance.
