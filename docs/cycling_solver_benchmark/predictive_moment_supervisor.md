# Superviseur prédictif lent relié au RHO

Le mode `adaptation_strategy: predictive_moment` de RHO-PACE applique maintenant
au vrai objectif RHO le résultat du superviseur lent. À chaque frontière lente
(ici toutes les deux révolutions), il part du **dernier RHO certifié**, recrée
les paramètres Ding de la variante active, et évalue les neuf candidats de
poids relatifs sur un rollout musculaire de trois cycles. Le candidat retenu
est transmis à `bioptim.update_objectives` avant le RHO suivant.

Le mécanisme est volontairement conservateur : aucun FHO, aucune trajectoire
future RHO, ni valeur terminale apprise ne sont employés. Une projection
invalide ou incomplète maintient le dernier coût; elle ne déclenche jamais le
retour causal historique `capacity_feedback`.

## Première campagne longue appariée

Le 12 septembre 2026, deux RHO IPOPT/MA57, Radau degré 5, 30 Hz, mécanique
réduite, variant Ding x0p5 et résistance constante 0,225 N.m ont été lancés
depuis le même seed certifié. La limite était 60 cycles.

| Bras | Cycles certifiés | Arrêt | Médiane solveur chaude |
|---|---:|---|---:|
| RHO uniforme | 43 | cycle 44 non faisable | 0,855 s |
| RHO + superviseur prédictif | 43 | cycle 44 non faisable | 0,863 s |

Le journal du bras prédictif confirme 22 applications au coût. Toutefois, les
neuf candidats avaient tous seulement un préfixe de rollout incomplet, de même
durée; la règle sûre `longest_explicit_prefix` conserve alors le candidat
incumbent. Les poids sont donc restés uniformes, et les deux trajectoires sont
logiquement identiques. Ce résultat valide le raccordement logiciel et montre
que cet horizon/critère ne fournit pas encore d'information discriminante à
0,225 N.m; **il ne démontre aucun gain d'endurance**.

Les artefacts sont dans `predictive-moment-long-20260912/`. Le journal
`predictive/weights.jsonl` est la preuve de la proposition, de sa sélection et
de l'écriture effective de l'objectif.

## Prochaine amélioration scientifique

Lorsque tous les candidats échouent sur la même durée explicitement simulée,
leur marge au même préfixe est une information disponible mais n'est pas
encore utilisée : la règle actuelle refuse volontairement de transformer un
rollout incomplet en pénalité arbitraire. La prochaine expérience doit tester
un critère de **marge de moment sur préfixe commun**, séparé et validé contre
des continuations RHO réelles, avant de l'autoriser dans le contrôleur. Cela
peut rendre le poids prédictif discriminant tout en gardant la règle de refus
pour des préfixes de durées différentes.
