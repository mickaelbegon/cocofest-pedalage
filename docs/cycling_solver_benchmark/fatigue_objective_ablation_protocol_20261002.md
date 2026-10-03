# Ablation appariée de la fonction de fatigue (2 octobre 2026)

## Question et portée

Le RHO actuel minimise une intégrale de déficits de capacité musculaire au carré. Deux mécanismes sont confondus : la somme sur toute la phase favorise certaines parties du cycle, et le carré rend la perte supplémentaire plus coûteuse lorsque le muscle est déjà fatigué. Cette ablation les sépare sans modifier la tâche mécanique. Elle teste la décision locale d'un RHO à un cycle; elle ne démontre à elle seule aucun gain d'endurance.

La configuration expérimentale `simulation_conditions["fatigue_objective_variant"]` accepte `integral_quadratic`, `integral_linear`, `terminal_quadratic` et `terminal_linear`. La valeur par défaut `legacy` conserve exactement le choix de `objective_shape` déjà utilisé. Chaque branche dispose de son propre NLP compilé; aucun changement de forme d'objectif n'est injecté dans un NLP déjà compilé. Les poids musculaires fixes ou paramétriques, la pénalisation du contrôle et les contraintes restent identiques entre branches.

Le pilote périodique expose le même choix avec `--fatigue-objective-variant MODE`. Ce choix fait partie de la signature du graphe compilé; les différents modes ne doivent donc pas réutiliser le même callback IPOPT. Pour la comparaison, sélectionner `--objective fatigue`, `--objective-shape quadratic` et un mode explicite parmi les quatre ci-dessous. La restauration d'un checkpoint complet exige le pilote de branches appariées; l'option CLI seule lance un RHO depuis sa procédure normale d'initialisation.

| Variation temporelle | Déficit linéaire | Déficit quadratique |
|---|---|---|
| Intégrale sur le cycle | `integral_linear` | `integral_quadratic` = référence explicite |
| État à la fin du cycle | `terminal_linear` | `terminal_quadratic` |

Le déficit normalisé du muscle *i* est `f_i = 1 - A_i/a_scale_i`. Le coût temporel est `10 000 c_f ∫ Σ_i w_i f_i^p dt`; le coût terminal est `10 000 c_f T Σ_i w_i f_i(T)^p`, où `T` est la durée physique de la fenêtre et `p` vaut 1 ou 2. Le facteur `T` garde l'échelle nominale en secondes du coût intégré. Il ne garantit pas des coûts totaux égaux si la fatigue évolue fortement dans le cycle. Pour le coût linéaire, `Σ_i w_i f_i(T)` est équivalent, **pour la décision à état initial figé**, à `Σ_i w_i [f_i(T)-f_i(0)]` : le terme initial est alors constant. Les résultats ne doivent pas être interprétés comme une estimation d'endurance restante en cycles.

Comparaisons causales principales :

1. `integral_quadratic` face à `terminal_quadratic` : effet du placement temporel lorsque le carré est conservé.
2. `integral_linear` face à `terminal_linear` : second test du placement temporel, à prix marginal constant.
3. `integral_quadratic` face à `integral_linear` : effet du carré dans le coût intégré.
4. `terminal_quadratic` face à `terminal_linear` : effet du carré dans le coût terminal.

La dérivée par rapport à `A_i` du terme linéaire vaut `-w_i/a_scale_i`; celle du terme quadratique vaut `-2 w_i f_i/a_scale_i`, avant le facteur global et la propagation par la dynamique. Cette différence formalise l'hypothèse selon laquelle la fatigue déjà présente domine la décision. Une comparaison de valeurs numériques brutes des quatre objectifs n'est pas informative. Il faut comparer les décisions, les contraintes et les états obtenus.

## Préparation des checkpoints et des branches

Utiliser le même modèle asymétrique bilatéral et une résistance totale de 1,92 Nm, partagée strictement en 0,96/0,96 Nm. Choisir trois checkpoints entièrement restaurables (début, milieu et proche de l'échec), par exemple autour des cycles 1, 80 et 150 dans une archive certifiée de 169 cycles. Enregistrer les états Ding complets, les états mécaniques, l'historique de stimulation, les contrôles décalés, les poids, les bornes, la configuration numérique et l'empreinte du modèle. L'essai BO historique qui adapte le partage G/D n'est pas une branche appariée.

À chacun des trois checkpoints, lancer les quatre variantes à partir d'un **même état initial figé** avec les mêmes poids unitaires. Dans une deuxième couche, refaire les quatre variantes avec les meilleurs poids BO à partage 50/50 lorsqu'ils sont disponibles. Garder exactement la même résistance, le même travail imposé, les mêmes bornes de PW et de variation, les mêmes objectifs autres que fatigue, les mêmes tolérances et le même solveur IPOPT/MA57. Employer le même maillage Radau-5 à 30 Hz; rejouer indépendamment la solution avec DOP853. Si des profils de PW diffèrent, comparer aussi la réalisation mécanique et les états Ding recalculés, dont le calcium et la force.

Avant d'interpréter une différence, vérifier que la variante `integral_quadratic` reproduit le coût et la décision du mode `legacy` lorsque `objective_shape="quadratic"`. Consigner les composantes non-fatigue de l'objectif. Si le temps d'exécution change, rapporter séparément compilation, itérations et temps du solveur pour ne pas attribuer un coût de compilation unique au fonctionnement par cycle.

## Mesures et décisions

Pour chaque branche, enregistrer : convergence et stationnarité, contraintes de travail/couple, PW et leurs bornes, `A`, `Tau1`, `Km`, calcium et force en fin de cycle, intégrales et gradients des quatre coûts **évalués a posteriori sur le même trajet**, capacité de travail résiduelle et temps RHO. Les gradients locaux quantifient si le terme de fatigue a une influence effective face aux régularisations et aux contraintes actives. Évaluer chaque coût sur chacun des quatre trajets est essentiel pour séparer l'effet de la règle de décision de l'effet de sa seule échelle numérique.

Si les décisions diffèrent et restent faisables, prolonger les branches appariées de 5 puis 20 cycles avec leur propre loi RHO. Définir *avant* la comparaison une marge de progrès supérieure à la dispersion observée lors de replays numériques indépendants. Poursuivre ensuite uniquement les variantes qui améliorent cette marge sur au moins deux checkpoints, sans sacrifier la tâche, jusqu'à l'échec évalué par le test de cycle gelé. Rapporter les cycles certifiés et la cause de l'arrêt; un échec IPOPT seul ne prouve pas l'impossibilité physiologique.

Abandonner ou réviser l'hypothèse si :

- la référence explicite ne reproduit pas `legacy`;
- les trajectoires diffèrent seulement parce qu'un coût a été mal mis à l'échelle ou qu'une contrainte mécanique change;
- le replay indépendant invalide les états ou le travail produit;
- les gradients de fatigue sont négligeables devant les autres objectifs aux checkpoints choisis;
- une variante gagne sur un cycle mais perd de façon systématique sur les branches de 5 et 20 cycles;
- le changement de forme rend le temps médian du RHO incompatible avec la cible clinique d'environ 1 s/cycle sans gain d'endurance robuste.

Les tests unitaires `tests/test_fatigue_objective_ablation.py` vérifient l'enregistrement des quatre formes, l'échelle temporelle des coûts terminaux, les poids paramétriques et les gradients symboliques. Ils ne constituent pas une validation d'endurance.
