# Réponse technique au prompt d'accélération Ding

Ce document répond au contenu de `PROMPT_acceleration_Ding_Cocofest.md` comme
à une proposition technique externe. Ses chiffres et ses injonctions ne sont
pas repris comme des résultats de ce dépôt sans vérification locale.

## Résumé exécutif

Le prompt contient une bonne intuition : exploiter les sous-systèmes linéaires
de Ding et mesurer le coût du graphe avant de changer de solveur. Sa promesse
d'un gain d'un ordre de grandeur ne se transpose cependant pas directement à
notre baseline **IPOPT/MA57, SX, mécanique reduced, periodic_node et
collocation Radau-5**.

Son levier principal — éliminer une somme historique de stimulations dans le
calcium — est déjà largement présent dans `periodic_node`. Le calendrier est
fixe, le forçage calcique est préparé comme donnée numérique et le PW agit sur
le recrutement de force, non sur la dynamique de calcium. Il n'y a donc pas
une convolution symbolique de vingt impulsions à supprimer dans le graphe de
la référence actuelle.

La condensation symbolique maximale déjà testée, qui conserve strictement
Radau-5, a supprimé des variables mais a densifié la Hessienne et ralenti
IPOPT/MA57. La voie qui reste raisonnable à examiner est plus locale : garder
`F,A` aux nœuds et stages, pré-calculer `Cn`, puis reconstruire `Tau1,Km` par
des offsets exponentiels. Aucun gain n'est encore démontré pour cette variante.

## Vérification des propositions du prompt

| Proposition | Verdict pour le cas actuel | Éléments vérifiés |
|---|---|---|
| Supprimer la convolution historique de `Cn` | Déjà largement réalisé dans `periodic_node`; pas de gain ×10 restant démontré | `ding2007_with_fatigue_periodic_node.py:75`, `:83`, `:140`, `:249` |
| `Cn` est exogène | Vrai seulement avec calendrier, paramètres et état calcique initial fixés; le PW courant ne modifie pas sa dynamique | `periodic_node.py:176-194`; `ding2003.py:320` |
| Réduire `A,Tau1,Km` à une convolution de `F` | Mathématiquement valide car ces équations partagent le même opérateur linéaire | `ding2007_with_fatigue.py:204-247`; [réduction analytique Radau-5](ding_analytic_reduction_radau5.md) |
| Radau est inutile car Ding ne serait pas raide | Non démontré pour notre OCP. Le prompt emploie une constante différente : Ding 2007 local a `tauc=11 ms`, donnant un seuil RK4 linéaire d'environ `30.64 ms`, inférieur à l'intervalle de `33.33 ms` à 30 Hz | `ding2007.py:50-66` |
| La carte complète proposée serait exacte | Faux en général : calcium exact, mais gel d'états lents, quadrature et propagation de `F` deviennent approximatifs si mécanique et fatigue restent couplées | Équations du modèle et validation locale Radau/DOP853 |
| Une carte ou une élimination rend nécessairement la Hessienne plus creuse | Faux : les substitutions peuvent créer des dépendances inter-stages | [A/B symbolique](ding_radau5_symbolic_ab.md) |

## Résultat local déterminant : condensation stricte Radau-5

La condensation maximale a éliminé `Cn,A,Tau1,Km` aux stages, tout en
reconstruisant objectifs, contraintes et bornes. Elle est algébriquement
équivalente au premier RHO : écart de décision échelonnée maximal `1.35e-10`
et écart d'objectif `1.58e-14`.

Elle n'accélère pourtant pas IPOPT/MA57 sur dix RHO :

| Mesure | Baseline | Condensé |
|---|---:|---:|
| Médiane chaude | `0.947 s` | `1.022 s` |
| Première résolution | `10.797 s` | `26.675 s` |
| Itérations première résolution | `554` | `1 133` |
| Hessienne par évaluation | `5.931 ms` | `7.611 ms` |
| Temps solveur total | `19.183 s` | `35.839 s` |

Les bornes des variables supprimées avaient dû rester des contraintes et le
graphe substitué était plus couplé. La réduction du nombre de variables ne
réduisait donc pas le coût KKT autant qu'attendu. Les artefacts sont dans
[`ding-radau5-symbolic-ab-20260912`](../../ding-radau5-symbolic-ab-20260912/).

## Piste encore distincte et testable

La variante suivante n'est pas la condensation négative précédente :

1. Conserver `F,A` aux nœuds et aux stages Radau-5.
2. Pré-calculer `Cn` aux mêmes stages Radau lorsque ses données entrantes sont
   fixées.
3. Reconstruire localement `Tau1,Km` depuis `A` et deux offsets exponentiels
   transmis au début du RHO.
4. Convertir les bornes de `A,Tau1,Km` en intersection de bornes locales sur
   `A`, avec traitement explicite du signe des coefficients.

Cette forme préserve davantage la localité par stage que la substitution de
tous les états par une combinaison des cinq forces de stage. Son potentiel est
plausible mais non mesuré; elle doit être comparée à la baseline dans une
fenêtre figée avant tout RHO fermé.

## Affirmations à ne pas appliquer sans étude propre

- Une moyenne ordinaire de force ne rend pas l'intégration de fatigue exacte;
  l'intégrale pertinente est exponentiellement pondérée, sauf force constante.
- Le fait que `F` soit affine à états et mécanique prescrits ne rend pas le
  modèle cycling complet linéaire en `F`.
- La réalisation locale du filtre calcique à pôle double nécessite en général
  mémoire et `Cn`; compter systématiquement seulement `Sigma,F,Phi` omet un
  état ou introduit une dépendance non locale.
- La convolution commune ne prouve pas à elle seule une non-identifiabilité des
  coefficients de fatigue.
- Les benchmarks SciPy d'une EDO isolée ne prédisent pas le coût des dérivées,
  de MA57 ni les itérations d'un NLP collocation.

## Plan de validation si la piste locale est retenue

1. Prototype hors production sur une fenêtre et seed figés, avec stages et
   coefficients Radau-5 identiques.
2. Vérification aux stages des états reconstruits, objectifs, bornes,
   contraintes, Jacobienne et Hessienne contre le graphe original.
3. Mesure séparée des dimensions/non-zéros, évaluations, Hessienne, MA57,
   itérations et temps solveur sur un RHO figé.
4. A/B sur dix fenêtres **rejouées** à état initial commun; seulement ensuite
   un RHO fermé, dont les bifurcations de trajectoire doivent être rapportées.
5. En cas de résultat négatif, ne pas remplacer la baseline IPOPT/MA57
   Radau-5 par une carte semi-analytique approximative sans validation
   indépendante DOP853/Radau-5 de la mécanique et de la physiologie.

## Conclusion

Pour le cas actuel, IPOPT/MA57 + Radau-5 reduced demeure la meilleure
référence scientifique. Le prompt apporte surtout une hypothèse ciblée à
tester — élimination locale de `Cn,Tau1,Km` en gardant `F,A` — et non une
justification de remplacer Radau-5, IPOPT/MA57 ou la formulation actuelle.
