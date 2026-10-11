# Déclencheur causal de décharge G→D : règle expérimentale bornée

## Décision proposée

La campagne dose–durée fournit une seule action candidate robuste dans le cas
asymétrique courant : transférer `0,10 Nm` de G vers D pendant **10 cycles**,
puis imposer explicitement le retour à `(0,96 ; 0,96) Nm`. Le total reste
`1,92 Nm`; les poids musculaires restent unitaires. La règle est à usage de
simulation et ne prétend ni estimer un temps avant échec, ni certifier la
faisabilité du cycle suivant.

La décision ne lit pas le numéro absolu du cycle. À une frontière RHO
certifiée, elle lit l'état lent actuel de Biceps des deux bras :

`r_B = A_Biceps / a_scale_Biceps`.

`A` est l'amplitude de la dynamique de Ding et `a_scale` sa valeur de
référence déclarée dans le modèle. `r_B` est donc un **proxy observé**, sans
interprétation abusive de réserve mécanique. `Tau1` et `Km` sont archivés avec
l'observation mais ne sont pas ajoutés à la règle : avec seulement deux
snapshots, les intégrer donnerait une frontière post-hoc à trois dimensions.

| État observé à la frontière | Décision |
| --- | --- |
| `r_B,G <= 0,62`, `r_B,D >= 0,78`, porte d'état rapide validée | une seule impulsion : `(0,86 ; 1,06) Nm` pour 10 cycles |
| `0,62 < r_B,G < 0,67` | abstention : la bande est non identifiée |
| `r_B,G >= 0,67` | maintien 50–50 |
| `r_B,D < 0,78`, état incomplet, histoire rapide non validée ou bras déjà actionné | abstention / maintien 50–50 |

La fenêtre `0,62–0,67` est volontairement une **zone sans action**. Elle ne
se comporte ni comme une interpolation, ni comme une valeur seuil clinique.
Le superviseur ne peut déclencher qu'une fois par tâche; il ne transforme donc
pas une décharge ponctuelle en répartition chronique. Après les dix cycles, il
revient à 50–50 sans re-déclenchement.

L'implémentation est dans
`cocofest.cycling.slow.causal_transfer.BicepsReservePolicy` et
`biceps_reserve_observation`. Elle vérifie que les deux états proviennent de
la même frontière certifiée, vérifie les artefacts hachés, inclut ces
provenances dans le reçu de décision et refuse par défaut la décision lorsque
la porte d'état rapide est absente.

## Calibration explicite, et ce qu'elle ne démontre pas

Les deux points de calibration sont les checkpoints RHO unitaires certifiés
utilisés pour la campagne dose–durée, pas des mesures prospectives :

| Checkpoint | `r_B,G` | `Tau1_B,G` | `Km_B,G` | `r_B,D` | `Tau1_B,D` | `Km_B,D` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c120 | 0,698744 | 0,091401 | 0,164867 | 0,849196 | 0,079436 | 0,154041 |
| c140 | 0,590457 | 0,102472 | 0,174884 | 0,813229 | 0,083928 | 0,158105 |

Ils sont séparés par une marge : c120 est au-dessus de `0,67`, c140 au-dessous
de `0,62`, et D reste au-dessus de `0,78`. Le seuil actif a donc été choisi à
l'intérieur de l'intervalle séparant les deux états, avec une bande
d'abstention de `0,05`. C'est une calibration de faisabilité de la règle,
pas une validation prédictive. Les résultats associés sont : témoin 169 cycles
dans toutes les répétitions; décharge `-0,10 Nm` pendant 10 cycles, suivie du
retour 50–50, 172 cycles depuis c120 et 174 depuis c140; la surcharge G
`+0,05 Nm` est défavorable. D est censuré à 240 cycles, donc la règle ne peut
pas conclure à l'absence de dette sur une durée indéfinie.

Les valeurs `a_scale` sont spécifiques à ces deux modèles Ding
asymétriques. Une autre personne, un autre côté, un autre paramétrage Ding ou
une autre cinématique exige une nouvelle calibration; réutiliser ces nombres
serait une fuite de distribution.

## Fuites temporelles à éviter

La politique autorisée ne reçoit que l'état **à la fin du dernier cycle RHO
certifié**. Elle ne peut pas utiliser :

- le dernier cycle connu de la simulation entière, l'échec ultérieur, ou une
  régression ajustée sur le futur;
- les PW, forces ou états résolus pendant le bloc de dix cycles à venir;
- un warm start décalé comme si c'était une observation physique;
- le statut IPOPT comme indice de fatigue ou de capacité.

Le numéro de cycle est présent dans le reçu uniquement pour l'audit et le
calendrier d'appel; il ne participe pas au test de seuil. De même, les deux
checkpoints c120/c140 doivent être exclus de la validation prospective de la
règle, car ils ont servi à fixer la bande.

## Conditions d'abstention et test de robustesse

Avant tout raccordement au RHO rapide, la règle doit rester désactivée tant que
la porte suivante n'est pas verte : les observations de calcium, PW/historique
de stimulation et états mécaniques rapides doivent avoir démontré qu'elles ne
renversent pas la décision à état lent identique. Le drapeau
`fast_state_gate` est donc `False` par défaut. Il ne doit jamais être remplacé
par une hypothèse implicite de fermeture de Markov.

Le protocole de robustesse minimal est :

1. prendre au moins trois frontières **non employées** pour calibrer, couvrant
   l'intérieur et les deux côtés de la bande;
2. pour chaque frontière, comparer contrôle 50–50, action candidate et
   contrôle de signe, avec la même faisabilité gelée et des répétitions;
3. exiger le maintien de la tâche sur les dix cycles, un retour certifié et un
   préfixe bilatéral commun; jamais une conclusion tirée d'un simple échec de
   solveur;
4. répéter sur la variante de muscle asymétrique tenue hors échantillon, puis
   sur une condition mécanique différente;
5. publier les déclenchements, abstentions, état du bras receveur, échec
   gauche/droit, travail par cycle et coût de calcul. Toute détérioration de D
   ou effet inférieur à la résolution numérique invalide l'activation.

Même après ces tests, cette règle ne remplace pas la fonction coût intra-bras.
Elle est un levier ponctuel de répartition du travail lorsque la faiblesse
relative G–D est mesurée, alors que la sélection des PW à l'intérieur d'un
bras reste le problème principal d'endurance.
