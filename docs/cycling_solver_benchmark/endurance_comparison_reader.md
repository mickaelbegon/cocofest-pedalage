# Lecteur comparatif des campagnes d’endurance

`scripts/analyze_endurance_comparison.py` produit un JSON et un Markdown à partir des archives déjà présentes. Il est conçu pour être relancé pendant l’exécution d’une campagne : une source absente, un manifeste sans progression et une progression non terminée restent explicitement tels quels.

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/analyze_endurance_comparison.py
```

Les sources par défaut sont les quatre essais factoriels RHO (poids unitaires et BO, avec ou sans répartition adaptative), le pilote dose--durée, le superviseur par reprise et les répertoires de RHO continu présents. Une archive additionnelle, par exemple un holdout, se passe sans modifier le code :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/analyze_endurance_comparison.py \
  --source RHO_continu_holdout=/chemin/vers/holdout \
  --output /chemin/vers/comparaison.json \
  --markdown /chemin/vers/comparaison.md
```

## Convention d’évidence

- Un essai **cold start** commence au repos. Son travail cumulé utilise exclusivement les cycles partagés dont les deux bras sont marqués `certified`.
- Un essai **checkpoint continuation** ne constitue pas une estimation directe de l’endurance depuis le repos. Il peut démontrer l’effet local d’une action, mais il reste dans une table distincte.
- Un endpoint est appelé `frozen_feasibility_exhaustion` seulement si l’archive le déclare explicitement avec `exact_label_allowed=true` et le statut `physiological_failure_certified`.
- Un `summary.json` historique peut documenter un préfixe certifié et du temps de calcul, mais ne suffit pas à lui seul pour transformer l’arrêt du solveur en endpoint physiologique.
- Les temps `pair_cycle_wall` sont préférés. Lorsqu’ils manquent dans une reprise, le lecteur étiquette les temps disponibles comme temps solveur par bras; ils ne sont pas assimilés au temps mur d’un RHO bilatéral.
- Le statut des états rapides est recopié depuis le manifeste/résultat. `unvalidated_experimental_override` signifie que le résultat ne justifie pas une activation clinique.

Cette séparation est importante pour comparer honnêtement RHO unitaire, physio/PACE ou BO, les pilotes causaux, le RHO continu, puis le holdout : l’analyse décrit les archives; elle ne remplit ni les données manquantes ni les certificats absents.

Le lecteur ne suppose pas non plus que deux archives ont les mêmes paramètres de Ding, charge mécanique, tolérances solveur ou protocole d’endpoint. Cette équivalence doit être vérifiée dans les manifestes scellés avant toute conclusion confirmatoire inter-condition.
