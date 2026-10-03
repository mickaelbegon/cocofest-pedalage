# Décision de validation du costate terminal — 3 octobre 2026

## Décision

**Ne pas activer actuellement le costate d’endurance dans une campagne jusqu’à l’échec.** Les étapes 1–4 ont produit des résultats utiles et restaurables, mais aucun modèle n’a encore validé un gradient de cycles restants. Cette décision scientifique ne signifie pas que l’utilisateur doit redonner son autorisation : les expériences complémentaires et les contrôles de câblage restent dans le mandat reçu.

Un **contrôle numérique court à gradient artificiel**, explicitement sans interprétation d’endurance, peut être informatif dès maintenant. Le test causal du gradient appris et sa continuation longue restent conditionnés aux critères ci-dessous. L’absence de promotion ne prouve pas que l’idée de coût terminal est mauvaise; elle indique précisément quelles hypothèses n’ont pas encore reçu de soutien expérimental.

Périmètre : modèles bilatéraux Ding asymétriques, tâche isocinétique, couple moyen équivalent total 1,92 Nm, 30 Hz, RHO réduit d’un cycle, Radau-5, IPOPT/MA57. Les conclusions ne sont pas généralisées à d’autres muscles, résistances ou modes mécaniques.

## Ce qui est acquis

### Références et effet du partage de charge

Le [factoriel poids × répartition](../../asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003/rapport_fr.md) donne :

| Poids | Partage 50–50 | Partage adaptatif |
|---|---:|---:|
| Unitaires | 169 | 211 |
| BO optimisé sous partage 50–50 | 179 | 228 |

Les deux références fixes reproduisent exactement les archives indépendantes : 169 et 179 cycles communs certifiés. Les gains observés sont +10/+17 cycles pour les poids, +42/+49 cycles pour le partage, et une interaction de +7 cycles. Les quatre arrêts ont reçu le même classement opérationnel `infeasible`, associé à un contre-factuel de restauration de fatigue faisable et à une saturation PW documentée. Ce protocole est plus probant qu’un échec IPOPT isolé, sans devenir une preuve globale d’infaisabilité du NLP non convexe.

Les 12 checkpoints c120/c140/c160 des quatre cellules disposent de reçus de restauration exacte, vérifiés dans de nouveaux processus solveurs. Le partage fixe reste strictement 0,96/0,96 Nm; le total du partage adaptatif reste 1,92 Nm. Une seule exécution nouvelle par cellule ne mesure pas une variance biologique ou numérique générale. Le résultat historique 234 utilise d’autres poids, optimisés avec partage adaptatif; il ne remplace pas la cellule appariée 228.

**Conséquence :** le problème initial n’est plus un écart démontré de 65 cycles imputable aux seuls poids. Le gain reproductible des poids BO sous la tâche fixe est de 10 cycles, et l’adaptation de travail constitue un levier distinct, plus fort dans ce cas.

### Ablation du coût de fatigue

L’[évaluation des quatre coûts](../../asymmetric-sides-r192-rho-bo-20260928/fatigue-cost-checkpoint-ablation-20261003/rapport_fr.md) contient 16 branches de 20 cycles, soit 320 cycles unilatéraux audités. Les observations à 1/5/20 cycles sont des préfixes de ces branches. Toutes les branches sont complètes; la violation normalisée maximale parmi les observations de préfixe est `8,60e-7`, sous le seuil `1e-6`.

La référence c120→140 reproduit `A/a_scale` à `3,14e-14` à gauche et `6,74e-10` à droite. À gauche, bras limitant, les variations de marge finale relativement au coût intégral quadratique sont :

| Coût | Depuis c120 | Depuis c140 |
|---|---:|---:|
| Terminal quadratique | −0,00671 | +0,00541 |
| Intégral linéaire | −0,01789 | −0,00874 |
| Terminal linéaire | −0,03310 | −0,00950 |

**Aucune variante** ne passe le critère annoncé : amélioration de marge supérieure à 0,01 à gauche aux deux ancrages, sans régression droite. Le petit gain terminal quadratique à c140 ne suffit pas. Le changement linéaire/quadratique change aussi l’amplitude marginale : on n’a pas isolé une propriété pure de courbure à pente égale. Une mauvaise performance de ces trois variantes n’invalide donc ni toute repondération ni toute hiérarchie de coûts.

Les temps moyens hors première résolution vont d’environ 1,11 à 2,09 s par résolution de bras, sous charge parallèle. Cette campagne ne valide pas l’objectif clinique de moins d’une seconde par cycle. L’audit est celui du NLP discret; le rapport de campagne ne documente pas un nouveau rejeu DOP853 de chacune des 16 branches.

### Propagation lente et témoins de marge

La [convolution lente Ding](ding_slow_cycle_validation_20261002.md) reproduit 29 cycles sous les forces archivées à environ `1e-12` relatif au repos, pour environ 0,53 ms par cycle de quatre muscles. C’est un outil validé de propagation conditionnelle, pas une validation des forces futures admissibles.

Le [jeu de validation de marge](../../asymmetric-sides-r192-rho-bo-20260928/task-load-margin-value-validation-20261003/rapport_fr.md) contient cinq états temporels et douze endpoints de branches, soit 17 états. Chaque état a un témoin nominal et un témoin de charge dans un processus frais. La violation maximale rapportée sur ces témoins est `7,13e-7`. La provenance et les contraintes sont donc suffisamment auditées pour utiliser ces observations comme données de charge **localement réalisable**. Elles ne certifient ni le maximum global de charge ni un nombre de cycles restants.

## Ce qui interdit actuellement de libérer le gradient appris

| Critère | Résultat observé | Verdict |
|---|---|---|
| Extrapolation temporelle vers c160 | Erreur 0,02718 à 0,03462 selon quatre ajustements, seuil 0,01 | Échec |
| Erreur locale A seul, politiques tenues à l’écart | Max 0,002382; retrait d’une paire 0,003102 | Réussite du seul seuil d’erreur 0,01 |
| Identification A seul + temps | Rang 6/6, condition 104,3 pour plafond exploratoire 100 | Échec du critère retenu |
| Identification 20 états + temps | Rang 9/22 malgré erreur max `7,00e-5` | Gradient ambiant non identifié |
| Confiance locale des politiques de validation | Déplacements A normalisés 0,013–0,030, boîte ±0,01 | Échec |
| Contraste utile pour classer les politiques mixtes | 0,001481 à c142; 0,006278 à c144, seuil exploratoire 0,01 | Non démontré à ce seuil |
| Validation externe par ancrage et par bras | Un seul ancrage de branches c140, bras gauche | Absente |
| Lien marge → durée restante sous politique définie | Aucune cible de durée ni classement externe validé | Absent |
| Effet d’un vrai costate sur un NLP RHO | Tests symboliques/paramétriques uniquement | À faire |

Seul le premier test linéaire c120/c140→c160 avait été annoncé avant la mesure c160. Les autres variantes temporelles sont exploratoires. Les seuils locaux 100 et 0,01 doivent également être pré-déclarés avant une nouvelle expérience : leur succès ou échec rétrospectif ne constitue pas une validation confirmatoire.

Une petite erreur sur la valeur ne suffit pas à identifier sa dérivée. Plusieurs gradients peuvent interpoler presque les mêmes valeurs sur un sous-espace étroit, puis produire des décisions différentes lorsque le solveur explore d’autres directions. Il faut vérifier le gradient **dans les directions terminales que le RHO peut effectivement atteindre**.

### Corriger le problème d’identification avant de collecter davantage

Le rang 9/22 ne signifie pas simplement qu’il manque treize points. La dynamique lente a la même constante de récupération et le même forçage par la force pour `A`, `Tau1` et `Km`. Les offsets de `coupled_slow_offsets` suivent une récupération homogène; lorsqu’ils sont nuls, `Tau1` et `Km` sont déterminés par `A`. À phase et histoire de stimulation fixées, `Cn` possède également des dépendances fortes. Ces relations peuvent rendre le rang plein des 22 colonnes physiquement inaccessible.

La prochaine régression doit donc employer des coordonnées indépendantes sur l’ensemble des états atteignables. Une possibilité à vérifier est de conserver A, la force terminale et les offsets lents non nuls, avec la phase et l’histoire explicites. Il faut démontrer cette réduction sur les archives et sur les perturbations testées. Ni la suppression automatique de `Tau1/Km`, ni leur variation indépendante hors de la dynamique réelle ne seraient justifiées.

L’évaluateur appelle `fast` toutes les coordonnées autres que A. Dans l’interprétation physique, `Tau1` et `Km` restent des états lents; les contributions 0,00044–0,00419 relevées dans le diagnostic KKT sont des **contributions de variables omises**, pas exclusivement d’états rapides. Elles peuvent dépasser le contraste c142 et méritent d’être prises en compte, sans valider le gradient KKT lui-même.

## État exact du canal logiciel

Le [canal terminal](../../cocofest/optimization/terminal_costate_ocp.py) minimise `−activation × V_terminal`, où `V` est annoncé en cycles faisables restants. L’enregistrement Mayer utilise `quadratic=False`, poids externe 1 : le signe n’est pas inversé par une élévation au carré. Les trois tests de `tests/test_terminal_costate_ocp.py` ont été relancés dans `cocofest-rho32` pendant cet audit : **3 passed**, deux avertissements SWIG.

Ces tests vérifient le signe et l’échelle de la dérivée, l’inactivité par défaut, le refus des entrées périmées/non finies/hors domaine, ainsi que la conservation du canal de fatigue et de l’identité des objets du graphe lors des mises à jour. Ils utilisent un programme simplifié; ils ne mesurent pas une résolution compilée de RHO réel.

Trois points doivent être complétés avant une activation expérimentale :

1. Le constructeur du pilote crée actuellement les quatre coordonnées `A/a_scale`. La classe accepte un autre jeu de coordonnées, mais un gradient sur force et offsets ne peut pas être transféré tel quel dans ce graphe à quatre A. La réduction et le graphe doivent correspondre exactement.
2. `update_from_remaining_cycles_value` vérifie forme, âge et boîte; il ne reçoit pas de certificat externe d’apprentissage. Son résumé annonce volontairement `gradient_scientifically_validated: false`. Il faut transporter une provenance vérifiée de modèle, tâche, politique de continuation, données, unités et validation avant toute interprétation scientifique.
3. Le garde post-solve du worker `_observe_pace_rt_terminal` est réservé au type `PaceRtObjectiveBinding`; la priorité à deux résolutions l’est également. La présence de `validate_terminal_point` sur le costate ne raccorde pas automatiquement ces mécanismes. Un pilote de branches costate doit vérifier le terminal **avant transfert physique**, rejeter l’extrapolation, restaurer le checkpoint puis résoudre avec le témoin, et auditer effectivement la réutilisation du solveur compilé.

Le canal ne garantit pas que la fatigue soit secondaire : conserver son poids historique peut noyer le nouveau terme. La grandeur utile est la variation décisionnelle `gradient · déplacement terminal`, pas l’ordonnée constante `V0`, qui ne change pas l’optimum. Une hiérarchie lexicographique doit avoir son propre contrat, avec le signe correct pour une valeur à maximiser, et ne peut être déduite du seul paramètre `activation`.

## Contrôle négatif utile dès maintenant

Un test de câblage sur un checkpoint exact c120, éventuellement confirmé à c140, est autorisé scientifiquement **comme test d’ingénierie** : quatre branches identiques avec activation nulle, gradient artificiel g, demi-gradient et −g. Les unités sont alors conventionnelles et aucun résultat n’est nommé prédiction d’endurance.

Définir g dans les coordonnées réellement compilées et fixer son amplitude avant le test. La fatigue et les autres objectifs restent identiques dans ces quatre bras; une seconde série éventuelle sans fatigue doit être étiquetée comme une ablation différente. Observer les PW, la variation terminale projetée sur g, les contraintes, les itérations, le temps et les identifiants de graphe/solveur. Rejouer un même cas pour estimer le plancher numérique, et vérifier le transfert et le rejet hors boîte.

Si g et −g produisent une décision identique, cela peut indiquer un terme trop faible, une contrainte active ou un sous-espace de décision insensible; ce n’est pas automatiquement un défaut de signe. Si une direction améliore une marge locale, cela demeure exploratoire tant qu’un témoin externe et une continuation commune ne lient pas cette amélioration à l’endurance. Cette étape peut déboguer le canal, mais ne satisfait pas à elle seule la phase 5 scientifique.

## Expérience minimale pour libérer la phase 5 scientifique

1. **Définir la cible.** Fixer une politique de continuation π et le partage 50–50. Appeler `Vπ` le nombre de cycles réalisables restants sous cette politique, avec le même protocole de faisabilité gelée. Une borne de charge d’un cycle ne sera pas renommée `Vπ`. Les cibles de cette étude peuvent provenir de continuations RHO, sans aucun FHO; elles servent à valider la méthode hors ligne, pas à exiger ces continuations en clinique.
2. **Identifier l’espace local.** Sur les archives existantes, mesurer les invariants et le rang des déplacements terminaux atteignables, avec une normalisation physique définie avant l’ajustement. Définir une base de dimension d et son domaine. Prévoir au moins d+1 points d’entraînement réellement indépendants, plus des points supplémentaires de validation; compter des préfixes communs comme des observations corrélées. Augmenter le nombre de points ne corrige pas une base redondante.
3. **Créer des contrastes atteignables.** Depuis c120 et c140, utiliser des perturbations de poids ou de PW positives/négatives définies à l’avance, puis une continuation π commune. Les branches doivent produire des contrastes dépassant la dispersion de re-résolution de la marge. Conserver tous les endpoints, y compris défavorables. Les branches déjà calculées peuvent servir de pilote, mais pas de validation « jamais vue » d’un modèle adapté à leurs résultats. Réserver au minimum un nouvel ancrage et des directions complètes de politique à la validation externe.
4. **Valider valeur et direction.** Réoptimiser indépendamment la marge aux endpoints et faire les continuations jusqu’au même arrêt pour obtenir `Vπ`. Pré-déclarer une tolérance de durée liée au gain recherché et une tolérance directionnelle; les erreurs de marge 0,01 ne se convertissent pas automatiquement en cycles. Répéter les contrastes décisifs avec initialisations numériques indépendantes. Exiger une erreur supérieurement bornée dans le domaine, un classement externe correct pour les contrastes clairement séparés, et une dérivée directionnelle stable aux pas h et h/2, dans les directions atteignables. Si le NLP change de régime actif ou que le nombre entier de cycles crée un plateau, employer une valeur lissée explicitement définie et vérifier son classement; ne pas prétendre avoir une dérivée classique d’une cible discontinue.
5. **Libérer un test causal limité.** Une fois ces points acquis et le raccordement logiciel validé, comparer témoin, BO fixe, gradient, demi-gradient, gradient opposé et version fatigue secondaire depuis un checkpoint exact tenu à l’écart. Faire 1 puis 5/20 cycles sous continuation comparable, avec audit du domaine avant transfert. Prolonger jusqu’à l’arrêt uniquement les variantes passant les critères annoncés. Le gain final sera mesuré en cycles communs certifiés, avec temps RHO et coût du calcul de valeur rapportés séparément.

Le résultat minimal attendu avant cette libération est donc **un gradient directionnel de valeur sous politique définie, identifié sur l’ensemble atteignable et validé sur un ancrage externe**, ainsi qu’un pilote garantissant le contrôle de confiance avant transfert. Aucun seuil d’erreur arbitrairement abaissé après inspection des données ne remplace ces conditions.

## Provenance de l’audit

Audit indépendant de lecture, sans modification de code de simulation, sans lancement OCP long et sans commit. Fichiers de décision figés par SHA-256 au moment de la lecture :

| Fichier | SHA-256 |
|---|---|
| `factorial-weights-split-20261003/final.json` | `7bb17f442f92c24ee5bd4466218985f6082fd088a4f70976b5f4db73c574db53` |
| `fatigue-cost-checkpoint-ablation-20261003/assessment.json` | `a63931228991e7fb114f835277ccb0f5c4bbf8fd7b336e4f67aefe10ff4538f9` |
| `task-load-margin-value-validation-20261003/assessment-v2.json` | `7282c84254c9e4a8f123be060ec2d2803751e8ffd8382ad98e89435e751e2971` |
| `cocofest/optimization/terminal_costate_ocp.py` | `96b80fb11815debb31463a6617501d12603e401a4b8cfb72aa1ee9f053a3df46` |

Les trois premiers chemins sont relatifs au dossier `asymmetric-sides-r192-rho-bo-20260928`. Le [protocole costate](terminal_costate_checkpoint_protocol.md) reste la base des branches appariées; le présent rapport précise pourquoi son étape de validation du prédicteur n’est pas encore satisfaite.
