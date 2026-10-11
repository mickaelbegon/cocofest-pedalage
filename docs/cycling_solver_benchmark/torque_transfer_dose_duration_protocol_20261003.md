# Protocole préenregistré : dose–durée du transfert de couple G→D

## Question et point de départ

Le pilote T3 scellé (`slow-layer-sim-20261003/causal-torque-pilot`) part des
mêmes états certifiés, avec poids unitaires, Radau-5 à 30 Hz et IPOPT/MA57. À
120 et 140 cycles, transférer 0,05 Nm de G vers D pendant dix cycles puis
revenir immédiatement à 0,96/0,96 Nm donne respectivement +1 et +2 cycles
certifiés. Le transfert opposé coûte trois cycles. Au cycle 160, l'action
positive ne peut pas terminer ses dix cycles; ce point est donc
**ininterprétable comme effet dose–durée**. La limite bilatérale est G, alors
que D est censuré à 240 cycles.

Le but du présent protocole est de déterminer si le gain observé est durable
après le retour à 50/50, ou seulement une dette déplacée de G vers D. Il ne
valide pas encore une règle en boucle fermée.

## Design figé avant calcul

Les ancrages sont les checkpoints physiques certifiés 120 et 140 du même RHO
unitaire. Ils encadrent la fenêtre informative sans utiliser 160, trop proche
de l'échec pour une exposition de dix cycles. Tous les bras gardent : même
archive et historique PW, même graphe RHO, même solveur, même tolérance,
même travail imposé par cycle et même couple total de 1,92 Nm.

Après l'ancrage, appliquer l'une des actions suivantes, puis revenir à
**0,96/0,96 Nm** jusqu'à l'échec physiologique certifié :

| Classe | Transfert `delta` (Nm) | Couple (G, D) pendant l'action | Durées |
|---|---:|---:|---:|
| témoin | 0 | (0,96 ; 0,96) | 5 et 10 cycles, mêmes données regroupées comme témoins |
| décharge G | −0,025 | (0,935 ; 0,985) | 5, 10 |
| décharge G | −0,050 | (0,910 ; 1,010) | 5, 10 |
| décharge G | −0,100 | (0,860 ; 1,060) | 5, 10 |
| contrôle de signe | +0,050 | (1,010 ; 0,910) | 5, 10 |

`delta` est ajouté au couple de G et soustrait à celui de D : le travail
mécanique de la tâche ne change donc pas. Chaque combinaison non nulle est
résolue deux fois dans des processus frais; deux témoins indépendants sont
aussi résolus par ancrage. Cela fait 18 branches bilatérales par ancrage,
soit **36 branches**. Ces répétitions quantifient la reproductibilité
numérique, pas une variabilité biologique.

La borne administrative est 300 cycles. Une branche censurée ou non résolue
numériquement n'est jamais traitée comme une fin de tâche, ni comparée comme
un effet exact. L'exposition doit être intégralement certifiée et suivie d'au
moins cinq cycles bilatéraux après le retour à 50/50 pour pouvoir revendiquer
un bénéfice durable.

## Critères et mesures

Le critère primaire est le dernier cycle du préfixe **bilatéral commun**
certifié, avec échec attribué par le test de faisabilité gelée; un échec IPOPT
isolé est inadmissible. Les trajectoires unilatérales après ce préfixe restent
des contrefactuels et ne sont pas sommées comme travail bilatéral.

Pour chaque ancrage, le contraste primaire est `N(action) − N(témoin)`.
La résolution numérique est l'étendue des deux témoins. Les critères
secondaires, audités sur un préfixe commun, sont :

- travail positif certifié total et travail jusqu'au préfixe commun ;
- bras limitant, cycle d'échec gelé et résultats de faisabilité gelée de G et
  D ;
- états Ding et fatigue (`Cn`, force, `A`, `Tau1`, `Km`) à la fin de l'action
  et cinq cycles après le retour ;
- marge de faisabilité à 0,96 Nm de chaque bras à la fin de l'action ;
- capacité contrefactuelle de D après la fin du préfixe commun, explicitement
  étiquetée comme telle.

La dette vers D est établie si D devient limitant, si sa faisabilité gelée à
0,96 Nm échoue avant celle du témoin apparié, ou si sa marge post-action se
dégrade sans gain bilatéral durable. Une simple diminution d'un état de
fatigue ne suffit pas : le test doit rester mécaniquement faisable.

## Règles de décision, avant lecture des résultats

Une action « décharge G » est *résolue* à un ancrage seulement si ses deux
répétitions, les deux témoins, l'exposition et le retour sont exacts, et si le
pire contraste contre un témoin est strictement supérieur à la résolution.
Une action est *candidate au superviseur* uniquement si, aux deux ancrages :

1. elle produit au moins **deux cycles** additionnels après retour, au-delà de
   la résolution témoin ;
2. D ne satisfait aucun critère de dette ci-dessus ;
3. l'effet est non négatif pour les deux répétitions ;
4. le contrôle de signe `+0,050` est non bénéfique (test de direction).

Parmi les candidates, choisir celle qui maximise le gain minimal sur les deux
ancrages et toutes les répétitions. En cas d'égalité, choisir le plus petit
impulsion de transfert `abs(delta) × durée`, puis la plus petite amplitude.
Si aucune action ne satisfait ces conditions, la sortie préenregistrée est
« pas de transfert automatique » : on conserve 50/50 et on poursuit le
travail sur la fonction coût intra-bras.

Cette règle rend explicite que le +1 observé au cycle 120 est une indication,
mais pas encore un niveau d'effet suffisant pour activer un superviseur.

## Contrôles d'intégrité et rapport

Chaque branche doit porter les hachages du manifeste scellé, du checkpoint,
du modèle, du problème préparé et des témoins de cycle. Les contraintes de
travail sont ré-auditées cycle par cycle; les poids restent unitaires. Le
rapport publie toutes les 36 branches, y compris échecs incomplets, censures
et diagnostics de D. Il ne fusionne pas des cycles post-échec d'un bras avec
la trajectoire bilatérale.

L'étape suivante, seulement si une candidate existe, est un test tenu hors
échantillon à un checkpoint distinct et une mise à jour lente bornée (tous les
10–20 cycles), jamais une modification directe et non auditée de la commande
RHO rapide.
