# Costate unilatéral pour l’endurance FES

Ce document prépare une analyse externe du costate appliqué à un seul bras, à tâche mécanique fixée. Il distingue les résultats mesurés, les limites actuelles et les expériences nécessaires avant de revendiquer un bénéfice d’endurance. Le partage adaptatif droite gauche est volontairement hors périmètre : chaque problème étudié impose 0,96 Nm équivalents par cycle au bras considéré.

## Décision actuelle

**Décision : ne pas activer un costate appris dans une campagne d’endurance longue.**

Le RHO sait maintenant recevoir un terme terminal linéaire paramétrique et le désactiver avant le cycle suivant lorsque sa confiance n’est plus établie. Cette disponibilité numérique n’est pas une validation scientifique. Les données actuelles ne permettent pas encore d’estimer de façon fiable une valeur de cycles restants, ni son gradient, hors des très petites zones observées.

Un contrôle court avec un gradient artificiel, son opposé et une valeur nulle reste justifié pour vérifier le câblage. Il ne doit pas être présenté comme un test d’endurance du costate.

## Question unilatérale

Pour l’état courant d’un bras, avec l’historique de stimulation, la dynamique Ding, les bornes de PW et le travail par cycle fixés, on cherche une fonction de valeur :

`V(x) = nombre de cycles encore certifiables sous une politique RHO de référence définie`.

La politique de référence doit être explicitement nommée. Une intervention terminale pendant quelques cycles peut ensuite être évaluée par son effet sur les cycles restants après retour à cette même politique. Cette définition ne requiert aucune trajectoire FHO clinique.

Si `V` est une endurance restante à maximiser, le RHO qui minimise un coût doit recevoir le terme terminal opposé à son gradient. La fatigue devient un critère secondaire, soit par pondération très faible pour une ablation exploratoire, soit préférablement par une seconde résolution lexicographique qui conserve la valeur obtenue par la première.

## Faits mesurés qui motivent l étude

Dans les quatre campagnes bilatérales indépendantes, à split fixe 50–50 ou avec le même cadre de deux bras, le bras gauche est le premier bras qui bloque. À split strictement fixe, il fournit donc une ancre unilatérale utile :

| Politique de poids | Cycles gauche certifiés | Cycles droite certifiés |
|---|---:|---:|
| Unitaires | 169 | 170 |
| Poids BO fixes | 179 | 180 |

Les poids fixes ont donc amélioré le préfixe commun de 10 cycles, sans modifier le travail du bras gauche. Cela montre qu’une fonction objectif unilatérale peut modifier l’endurance. Ce résultat ne démontre pas que les poids BO sont optimaux, ni qu’un costate les reproduira.

Le coût historique de fatigue est une intégrale quadratique de la perte relative de capacité. Son coût marginal augmente fortement avec la fatigue accumulée : dans le bras gauche unitaire, la somme des pentes locales absolues passe approximativement de 290 au cycle 1 à 38 710 au cycle 169. Ce fait suggère que le coût courant peut décourager certains échanges musculaires tardifs; il ne prouve pas que ces échanges prolongeraient l’exercice.

Quatre ablations strictement unilatérales ont ensuite été comparées depuis des reprises exactes c120 et c140 : intégral ou terminal, quadratique ou linéaire. Elles ont toutes exécuté 20 cycles à travail identique, puis une marge mécanique a été réoptimisée dans un processus frais. Aucune ne satisfait le critère préfixé de gain de marge supérieur à 0,01 aux deux checkpoints sans régression du bras droit. Le terminal quadratique aide légèrement la gauche au checkpoint tardif, mais détériore le checkpoint précoce et le bras droit. Ainsi, changer uniquement la forme temporelle ou la puissance du coût de fatigue ne résout pas le problème observé.

## Ce que le modèle lent permet déjà

La transition lente Ding par cycle a été validée sous les profils de force archivés. L’intégrale exponentiellement pondérée de la force reproduit les états lents `A`, `Tau1` et `Km` avec une précision proche de la machine sur les témoins disponibles, pour un coût d’environ 0,53 ms par cycle et par bras. C’est suffisamment rapide pour un superviseur.

Cette validation est conditionnelle : elle rejoue des forces déjà connues. Elle ne prédit pas les forces futures produites par le RHO après changement de coût, ni la faisabilité du travail futur. La propagation lente est donc un composant de modèle, pas encore une valeur d’endurance.

## Pourquoi la valeur de marge actuelle ne suffit pas

Une marge de travail réalisable a été reoptimisée à des checkpoints exacts du bras gauche. Les marges observées diminuent avec la fatigue, mais l’extrapolation temporelle simple échoue sur c160 : l’erreur hors échantillon dépasse le seuil préfixé de 0,01.

Des branches courtes avec contrastes de poids fournissent une précision locale apparente dans une boîte d’état très petite. Cette précision ne se généralise pas : les holdouts sortent de la boîte de confiance, et le modèle complet est sous-identifié. Le diagnostic de rang ne doit pas être interprété naïvement comme un besoin de vingt variables indépendantes. Les états `Tau1` et `Km` possèdent des relations structurelles avec `A` si les offsets sont connus; l’état `Cn` dépend aussi fortement de la phase et de l’historique. Le bon espace d’apprentissage est donc un sous-espace terminal atteignable, pas l’espace ambiant de tous les états exportés.

Conséquence : une marge instantanée, même physiquement réalisable, n’est pas encore une estimation de cycles restants. Son gradient ne doit pas être injecté dans le RHO comme costate d’endurance.

## Limites du costate actuel

### Cible mal alignée

Le prototype PACE et les premières marges classent une réserve locale ou un déficit de tâche sur un horizon court. Quand tous les rollouts compacts sont réalisables, ils ne distinguent pas nécessairement des durées d’exercice différentes. Une valeur utile doit classer les continuations sous une politique explicitement fixée.

### Données de valeur insuffisantes

Les séries temporelles d’un seul RHO sont corrélées et ne permettent pas d’identifier une dérivée causale. Les petites branches à poids modifiés sont informatives seulement si leur endpoint est restauré, vérifié dans un worker frais, puis suivi par une continuation comparable. Des exports préparés seuls ne sont pas des ancres exactes.

### Domaine de confiance trop petit

Le modèle de marge étudié quitte sa boîte de confiance rapidement. L’activation d’un coût terminal au-delà de ce domaine confond une extrapolation du surrogate avec une décision physiologique. Le repli vers le RHO de référence est obligatoire; une désactivation du proxy ne signifie pas une incapacité physique.

### Interaction avec la fatigue

Les expériences montrent que changer uniquement la pénalité de fatigue ne produit pas un gain de marge robuste. Cela ne réfute pas une priorité de valeur; cela signifie que la valeur devra identifier un échange spécifique que la fatigue quadratique ne voit pas. Une très petite pondération de fatigue n’est pas une vraie priorité lexicographique et les poids nuls sont à éviter dans le résidu racine carré.

### Câblage numérique, maintenant sécurisé mais non validé scientifiquement

Le binding costate est inactif par défaut. Son audit post-solve est distinct de PACE-RT; un domaine de confiance invalide désactive numériquement le terme avant le RHO suivant. Les paramètres du coût terminal ne reconstruisent pas le graphe compilé. Ces propriétés sont testées, mais ne remplacent ni une validation directionnelle ni une mesure d’endurance.

## Direction de recherche recommandée

### 1. Définir une politique de retour unilatérale

Choisir une politique RHO de référence unique pour le bras gauche : d’abord unitaire pour quantifier un gain au-dessus de 169 cycles, puis les meilleurs poids fixes comme référence forte à 179 cycles. Conserver même travail, mêmes bornes, même intégrateur, même solveur et même protocole de faisabilité gelée.

### 2. Construire des labels de continuation

Depuis des checkpoints exacts c120 et c140, appliquer de petites interventions admissibles pendant 1 et 3 cycles. Les actions peuvent être des contrastes centrés de poids relatifs ou une base réduite de PW. Revenir ensuite à la politique de référence et mesurer les cycles restants certifiés. Les interventions d’apprentissage et de validation doivent être disjointes.

Le label est l’avantage ou la perte de cycles de la continuation, pas une simple somme de capacités ni une marge locale. Les cycles voisins d’une même trajectoire doivent rester dans la même partition pour éviter une validation artificiellement optimiste.

### 3. Identifier des coordonnées atteignables

Réduire l’état terminal à des coordonnées qui varient réellement : capacités `A`, offsets lents indépendants de `Tau1` et `Km`, et une description compacte de la phase et de l’historique de stimulation. La réduction doit être justifiée par les équations Ding et contrôlée par son rang et son conditionnement, pas choisie seulement parce qu’elle est petite.

### 4. Tester une dérivée directionnelle avant un gradient complet

Apprendre d’abord le classement entre quelques directions terminales ou actions. Une direction est acceptable seulement si elle prédit, sur un checkpoint ou une action laissés de côté, le bon signe de l’avantage de continuation. Le test `+g`, `−g`, demi-amplitude et `g=0` est plus informatif qu’une norme faible d’erreur de régression.

### 5. Injecter le costate de façon contrôlée

Lorsque les critères précédents passent, exécuter d’abord des branches RHO de 1, 5 et 20 cycles depuis les mêmes ancres. Comparer le coût terminal appris, son opposé, sa demi-amplitude et la référence. Contrôler le déplacement réel des PW, la valeur recalculée, la confiance, la marge réalisable, les temps de résolution et les contraintes.

Seulement un gain directionnel répliqué peut justifier une continuation jusqu’au protocole d’arrêt. Le superviseur asynchrone, l’âge maximal d’un modèle et la cadence de mise à jour sont des étapes d’ingénierie ultérieures : ils ne doivent pas masquer un échec de valeur hors ligne.

## Portes de décision

| Porte | Condition de passage | Décision en cas d’échec |
|---|---|---|
| Fidélité de reprise | Reproduction d’un témoin depuis une ancre exacte | Corriger la reprise, aucune conclusion de coût |
| Valeur locale | Classement correct d’actions holdout dans une zone explicitement bornée | Réduire le domaine ou collecter des continuations |
| Direction | `g` bat `−g` et `0` sur des branches indépendantes | Abandonner ou redéfinir la valeur |
| Robustesse | Même signe sur c120 et c140, sans violation ni repli systématique | Ne pas lancer de campagne longue |
| Endurance | Gain de cycles certifiés face à la politique de référence | Étudier ensuite cadence et asynchronisme |

## Sources locales à examiner

- `asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/rapport_fr.md` : matrice poids/split et reçus exacts aux checkpoints.
- `asymmetric-sides-r192-rho-bo-20260928/fatigue-cost-checkpoint-ablation-20261003/rapport_fr.md` : ablation des quatre coûts de fatigue.
- `asymmetric-sides-r192-rho-bo-20260928/task-load-margin-value-validation-20261003/rapport_fr.md` : limites de la valeur de marge.
- `docs/cycling_solver_benchmark/terminal_costate_release_gate_20261003.md` : décision actuelle de non-libération.
- `cocofest/optimization/ding_slow_cycle.py` : propagation lente par convolution exacte.
- `cocofest/optimization/terminal_costate_ocp.py` : canal terminal paramétrique expérimental.
- `cocofest/simulation/independent_arms_process.py` : audit et désactivation sûre avant transfert du cycle suivant.

## Demande proposée à un autre LLM

Analyser ce protocole comme un problème de contrôle prédictif unilatéral sans FHO clinique. Distinguer les résultats mesurés des hypothèses. Évaluer si une valeur de continuation sous politique de retour est le meilleur label pour apprendre une dérivée terminale, ou proposer une alternative moins coûteuse mais testable. Ne pas proposer d’utiliser les sorties FHO comme données cliniques. Exiger des expériences causales à checkpoints exacts et considérer un échec IPOPT comme un résultat numérique, sauf lorsqu’un contre-factuel gelé certifie explicitement une limite dans le protocole défini.
