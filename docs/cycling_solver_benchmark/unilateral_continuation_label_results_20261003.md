# Résultat final — labels de continuation unilatérale

## Question et décision

Ce protocole teste si de brèves redistributions des poids de fatigue au bras
gauche fournissent un signal causal exploitable pour apprendre une valeur de
continuation. La tâche est isocinétique, avec un travail fixe équivalent à
`0,96 Nm` par cycle et les mêmes paramètres musculaires, solveur et contraintes
que la référence `RHO_unitaire`.

**Décision : aucun gradient de valeur ni costate n'est activé.** Les données
ne montrent pas encore qu'une direction locale des capacités augmente
l'endurance après retour à une politique RHO commune.

## Protocole réalisé

Depuis les ancres exactes `c120` et `c140`, 36 branches indépendantes ont été
exécutées :

- 4 témoins `RHO_unitaire` ;
- 24 contrastes d'apprentissage, sur trois directions de redistribution
  orthogonales et leurs signes ;
- 8 actions mixtes tenues hors apprentissage ;
- une intervention de 1 ou 3 cycles, suivie d'une reprise fraîche sous les
  poids unitaires jusqu'au test de faisabilité gelée.

Chaque cycle publié a une solution RHO auditée, et chaque fin est attribuée à
la fatigue uniquement si le test à objectif nul est explicitement infaisable
avec les états figés, tandis que le contre-factuel « fatigue au repos » est
faisable. Un échec IPOPT isolé n'est pas compté comme une fin physiologique.
Le manifeste, les archives et les deux phases de chaque branche sont protégés
par des SHA-256.

## Résultats

Les **36/36** branches ont une fin physiologique certifiée ; aucune n'est
censurée, arrêtée numériquement ou sortie de la variété lente de Ding. Les
quatre témoins reproduisent exactement le dernier cycle certifié `c169`.

Les contrastes déplacent effectivement les PW et les capacités, mais leur
avantage observé après retour à `RHO_unitaire` est de **0 cycle** dans les
36 observations. Ce n'est donc pas une démonstration que les poids n'ont
aucun effet : c'est une démonstration que cette intervention brève, puis
effacée par la politique de retour, ne crée pas un label d'endurance mesurable
à cette résolution.

| Ancre | Familles apprentissage | Familles holdout | Rang atteignable | Conditionnement | Porte directionnelle |
| --- | ---: | ---: | ---: | ---: | --- |
| c120 | 6 | 2 | 4 | 4,88 | rejetée |
| c140 | 6 | 2 | 4 | 5,59 | rejetée |

Le rang local est suffisant et les offsets lents de Ding restent négligeables
(maximum relatif `5,72e-12`). Le rejet vient donc de la validation : les deux
holdouts ne fournissent pas de contraste non nul, et certains états rapides
(`Cn`, force, historique de PW, phase/état mécanique) sortent de l'enveloppe
des actions d'apprentissage. Un modèle limité à `A/a_scale` confondrait alors
des effets qui ne sont pas identifiés.

## Interprétation

Le résultat ne soutient pas une promotion de `PACE-RT` ou d'un costate comme
objectif principal. En particulier, il serait incorrect d'interpréter le
gradient nul ajusté ici comme « la fatigue instantanée est optimale » : il
résulte de labels d'avantage tous nuls sous une politique de retour qui efface
l'action expérimentale.

Cela est cohérent avec le contraste entre le RHO unitaire (169 cycles
communs certifiés) et le BO à poids fixes, qui peut atteindre davantage de
cycles en modifiant l'allocation tôt dans l'exercice. Une correction locale et
tardive ne peut pas, à elle seule, reconstituer les capacités déjà consommées.

## Suite préenregistrée

La prochaine expérience n'utilisera pas de pseudo-gradient appris. Lorsque
une direction candidate indépendante sera disponible, elle devra comparer des
branches RHO appariées `+g`, `0` et `-g` **sans retour immédiat aux poids
unitaires**, avec :

1. mêmes ancre, travail, solveur, contraintes et historique PW ;
2. application causale de l'action pendant une fenêtre explicitement fixée ;
3. audit avant chaque transfert, puis continuation sous une même politique de
   référence choisie avant l'expérience ;
4. comparaison des derniers cycles certifiés et des états rapides, plutôt
   qu'une simple valeur prédite ;
5. abandon du modèle si une branche est numérique, censurée ou hors de son
   domaine de support.

Avant de l'exécuter, il faut définir `g` sans employer les labels nuls
ci-dessus : par exemple par une hypothèse mécanique déclarée à l'avance, ou
par une nouvelle campagne d'actions persistantes permettant de révéler des
différences d'endurance. Cette étape sépare clairement l'étude de la priorité
fatigue/réserve de la validation clinique d'un superviseur en temps réel.

## Artefacts reproductibles

- Résultat machine :
  `asymmetric-sides-r192-rho-bo-20260928/unilateral-continuation-labels-20261003/value-analysis-final-20261003.json`
- Manifeste :
  `asymmetric-sides-r192-rho-bo-20260928/unilateral-continuation-labels-20261003/manifest.json`
- Exécuteur : `scripts/run_unilateral_continuation_labels.py`
- Analyseur : `scripts/analyze_unilateral_continuation_value_20261003.py`
- Protocole et seuils de validation :
  `docs/cycling_solver_benchmark/unilateral_continuation_value_validation_20261003.md`
