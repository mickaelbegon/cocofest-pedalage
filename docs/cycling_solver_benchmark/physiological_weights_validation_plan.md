# Poids physiologiques recalculés pour chaque cas

## Sources et décision

L'annexe est désormais disponible : Supplementary Material 1.docx, tableaux
S3/S4 et S6, figure S4. La référence exécutable est le
[code public gelé au commit 31e064f4](https://github.com/pyomeca/cocofest/blob/31e064f4d80741c8b5444e6ce4d91bfe4eb78c3d/examples/fes_multibody/cycling/physiological_weight_calculation.py).

Les poids initiaux sont recalculés pour chaque jeu de paramètres, sans FHO.
Le noyau est physiological_muscle_weights.py; le catalogue de cas est
physiological_weight_cases.py; le script reproductible est
scripts/validate_physiological_weights.py. Les
[résultats numériques et limites](physiological_weights_validation.md)
accompagnent ce protocole.

## Calcul reproduit

1. Extraire les moments signés avec un muscle activé à la fois. Comme dans le
   code, les contributions passives des muscles inactifs restent présentes.
2. Mesurer la fraction d'échantillons angulaires où chaque profil est positif.
3. Imposer une force de 80 % de Fmax pendant cette fraction d'une seconde,
   puis un repos pendant le reste de la seconde, sur 1500 cycles.
4. Mesurer la pente entre les capacités relatives après les cycles 1 et 1500,
   divisée par 1500. Son signe est opposé dans S6, sans effet après élévation
   au carré.
5. À chaque cycle, multiplier les profils mécaniques par la capacité restante.
   Intégrer la contribution positive exclusive et celle des 90° avant une
   entrée dans la zone à risque. Moyenner ces contributions sur les cycles.
6. Multiplier cette criticité mécanique par la fatigabilité au carré, puis
   appliquer la normalisation min–max du code.

Cette calibration n'est ni 80 % de PW ni le modèle Ding complet. Le risque du
code est une somme de moments positifs inférieure à 0,20 N·m; il ne signifie
pas exclusivement que tous les moments sont négatifs. Ce seuil ne remplace
pas automatiquement l'assistance ou la résistance réelle du RHO.

## Comprendre le poids nul

La normalisation min–max impose un zéro au score brut minimal. Un triceps
devenu non minimal reçoit donc un poids positif, mais un autre muscle reçoit
le zéro. Un plancher positif ou une division par le maximum serait une
variante différente, pas la reproduction publiée.

Dans la référence nominale reproduite, la criticité du triceps vaut aussi
zéro. Changer seulement sa fatigabilité ne suffit donc pas. Il faut examiner
les changements de contribution mécanique, notamment quand les capacités
d'autres muscles changent. Conserver séparément fatigabilité, criticité,
scores bruts, poids et écart du score triceps au minimum des autres muscles.

## Références à ne pas confondre

| Paramètre | Code publié | Dépôt actuel |
| --- | ---: | ---: |
| alpha A du deltoïde antérieur | −0,14 | −1,4 |
| alpha A du deltoïde postérieur | −0,11 | −1,1 |
| alpha A du biceps | −0,056 | −0,56 |
| alpha A du triceps | −0,034 | −0,24 |
| A au repos du triceps, N/s | 4915,5 | 7036,3 |
| Fmax du triceps, N | 262 | 617 |
| tau de récupération du triceps, s | 109,1 | 76,2 |

Les autres valeurs sont conservées telles qu'elles apparaissent dans chaque
référence. Un test compare les 16 paramètres à ceux extraits par AST du
dictionnaire local de set_fes_model. Il vérifie notamment la notation
Python 10e-2, qui vaut 0,1 et non 0,01.

S3 donne 7036,3 et 617 pour le triceps, alors que S4 et le code donnent 4915,5
et 262. S3 donne également −0,034 et 109,1. Les trois sources ne sont pas
fusionnées artificiellement; aucun paramètre du RHO public n'est modifié.

## Panel élargi

Chaque référence contient 69 cas déterministes : le nominal, 64 variations
indépendantes et quatre interactions. Les variations portent sur quatre
muscles, quatre paramètres (alpha A, tau de récupération, A au repos, Fmax)
et les facteurs 0,25; 0,5; 2; 4. Les interactions croisent la fatigabilité du
deltoïde antérieur et du triceps avec les facteurs 0,5 et 2.

Ce sont des perturbations numériques, pas des profils cliniques. Changer A
et Fmax indépendamment isole leurs effets mais ne représente pas forcément
un changement physiologique cohérent de taille musculaire. Un panel couplé
PCSA/fibres viendra après réconciliation de ses relations et paramètres.

Chaque variation de Fmax recalcule les moments avec les contributions
passives. L'adaptateur CasADi évite de redimensionner indépendamment une
colonne de moment alors que les autres muscles peuvent aussi être affectés.

## Contrôles scientifiques

Les capacités sont contrôlées avant et après le repos de chaque cycle. Une
capacité non positive rend les poids inutilisables pour le contrôleur. Le
résultat historique reste disponible pour audit seulement. L'échec de cette
calibration à force prescrite ne démontre pas une impossibilité du RHO.

La cinématique conserve le problème de moindres carrés du code source, ses
bornes et conventions. La lecture native des bornes provoquant un arrêt du
processus avec le binding disponible, l'adaptateur les lit dans le bioMod et
résout le même problème avec des marqueurs et jacobiennes CasADi. Aucune
dépendance n'est modifiée.

Les écarts aux marqueurs sont enregistrés. Retrouver les poids publiés ne
valide pas le contact du véritable RHO. Le sens temporel du « pré-risque »
et les signes du travail doivent également être confrontés à la cinématique
exécutée. La grille 120 intervalles reproduit le code; une grille 240 mesure
la sensibilité des trois cas présélectionnés, sans certificat de convergence.

## Comparaison avec les contrôleurs

| Approche | Commande | Poids |
| --- | --- | --- |
| RHO normal | Un cycle | Uniformes |
| RHO physiologique fixe | Un cycle | Recalculés au départ pour ce cas |
| Approche à deux vitesses | Un cycle et superviseur lent | Même initialisation puis corrections lentes |
| FHO sur quelques cas | Horizon fini déclaré | Politique de coût explicitement déclarée |

L'approche fixe intermédiaire isole l'effet de l'initialisation de celui du
superviseur. Les trois RHO doivent partager modèle, état initial, charge,
cadence cible, limites de PW/force, intégrateur et critères de validation.

Le coût de l'article intègre la RMS de perte de capacité, pas le recrutement
au carré du premier prototype. Passer aux pertes relatives exige de
transformer les poids. La gestion des poids nuls reste à ajouter au
superviseur, actuellement limité aux poids positifs.

IPOPT/MA57 reste la référence demandée. Le calcul initial des poids n'ajoute
aucun OCP ni compilation NLP. Le délai inférieur à une seconde doit encore
être mesuré sur la boucle RHO complète.

Le FHO est exclusivement évaluatif. Avant ses résultats, les règles retiennent
le nominal publié, le premier cas admissible avec triceps positif dans
l'ordre déclaré, et le cas admissible dont la calibration garde la plus
petite capacité positive. Ces cas doivent d'abord passer les contrôles du
RHO. Aucune sortie FHO ne règle les poids ou ne sélectionne les paramètres.

Comparer le temps exécuté, le suivi, les PW, les forces et réserves. Les
coûts d'optimisation pondérés différemment exigent aussi une métrique commune
d'évaluation. Un FHO fini ne prouve pas une endurance globalement optimale.
Rapporter latences RHO médiane/p95/maximum, dépassements de délai, durée du
superviseur et âge des propositions acceptées.
