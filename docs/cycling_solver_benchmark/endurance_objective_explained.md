# Comprendre le coût d'endurance, sans équations

## Ce que l'on cherche réellement

Le terme futur que nous construisons cherche à **garder la possibilité de
réaliser le mouvement plus tard**, et pas seulement à obtenir une faible
fatigue à la fin du cycle actuel.

Deux états peuvent présenter une fatigue moyenne similaire tout en ayant des
perspectives très différentes. Dans le premier, un muscle indispensable à la
prochaine phase est presque à sa limite. Dans le second, la charge a été
répartie autrement et ce muscle garde de la réserve. Le coût futur doit
préférer le second état, même si un autre muscle a un peu plus travaillé.

Ce terme est encore expérimental. Il n'est pas automatiquement ajouté aux
campagnes RHO existantes et ne constitue pas une mesure validée d'endurance.

## Comment cela changerait les PW du cycle actuel

Le mouvement demandé et les bornes de stimulation restent imposés. Parmi
les commandes qui les respectent, le coût actuel favorise certains choix,
par exemple une fatigue plus faible. Le terme futur ajoute une préférence :
ne pas laisser les muscles dans un état qui rendra la suite difficile.

Le solveur pourra donc accepter un peu plus de fatigue immédiate si cela
préserve une ressource utile plus tard. L'importance donnée à ce compromis
reste un réglage global à valider; ce n'est pas une permission de suivre
moins précisément le mouvement.

## La question posée à chaque état musculaire candidat

On imagine partir de cet état et poursuivre la même tâche pendant plusieurs
cycles. Un simulateur léger adapte les durées d'impulsion (PW) au fur et à
mesure de l'évolution des forces et de la fatigue. Il utilise les données RHO et le modèle
musculaire, sans avoir besoin d'une solution FHO.

À chaque phase, il examine deux difficultés possibles :

- **Ne plus pouvoir produire assez de moment.** Même avec la meilleure
  répartition des stimulations autorisées, les muscles risquent de ne plus
  fournir ce que demande la tâche.
- **Ne plus pouvoir réduire suffisamment le moment.** Une stimulation passée
  peut laisser une force résiduelle lorsque la demande baisse ou que la
  géométrie change. Même en diminuant les PW, la commande suivante peut alors
  être difficile à réaliser.

La réserve est la distance entre la demande et ces limites. Les contributions
des muscles qui s'opposent au mouvement sont conservées avec leur signe :
stimuler davantage tous les muscles n'augmente pas nécessairement le moment
total utile.

Le coût devient élevé lorsque l'une de ces réserves devient petite sur les
phases futures. Il accorde donc beaucoup d'importance aux passages les plus
fragiles, avec une transition progressive plutôt qu'un interrupteur brutal
« possible / impossible » dans la petite approximation destinée au solveur.

## Deux coûts différents, avec deux rôles différents

### Le petit QP choisit comment poursuivre le mouvement

Le QP auxiliaire cherche une répartition des stimulations qui respecte le
moment total demandé **maintenant et à la phase suivante**. Il reste proche
des contributions musculaires de référence, en permettant davantage de
redistribution aux muscles qui disposent de flexibilité dans le modèle.
Une très faible pénalité stabilise le choix entre des recrutements presque
équivalents.

Le moment total demandé et les bornes de PW sont des contraintes : le QP n'a
pas le droit de les sacrifier simplement pour réduire son coût. Il ne cherche
ni à rendre les forces de tous les muscles égales, ni à minimiser directement
le nombre de stimulations, ni à maximiser à lui seul l'endurance.

### Le coût terminal juge l'état laissé à la fin du RHO

Ce deuxième coût répond à une autre question : **« Si je termine ce cycle
dans cet état, combien de marge me restera-t-il pour la suite ? »**

Il peut ainsi encourager le RHO à changer ses PW actuelles pour préserver un
muscle utile plus tard ou pour éviter de laisser trop de force résiduelle.
Le petit QP sert à construire cette estimation; son propre coût de répartition
n'est pas utilisé comme un nombre de cycles d'endurance.

## Pourquoi déclencher le preview seulement parfois ?

Résoudre un QP à chaque phase coûte du temps. La politique hybride commence
donc par essayer la répartition rapide à une phase. Elle prédit ensuite
l'état que cette commande laisserait et regarde si la phase suivante garde
assez de réserve.

Si la réserve est suffisante, elle conserve cette commande rapide. Si la
phase suivante paraît difficile, elle sollicite le QP à deux phases pour
revoir la répartition dès maintenant. Seule la première commande est ensuite
appliquée, puis l'analyse est refaite.

Le seuil de déclenchement est un réglage de **vigilance du calcul**, pas une
permission de moins bien suivre la tâche. Par exemple, un seuil de 1 % de
l'échelle de moment de la tâche ne permet pas une erreur de suivi de 1 %.
Les contraintes de moment et les bornes de PW restent inchangées.

## Ce que voit réellement IPOPT

Les simulations futures, les décisions de déclenchement et les petits QP
sont calculés **en dehors du problème d'optimisation RHO (NLP)**. Ils peuvent
changer de comportement à certaines frontières et ne sont pas introduits tels
quels dans la fonction objective différentiable.

On évalue plutôt plusieurs états proches de l'état terminal attendu, puis on
ajuste une petite fonction lisse qui reproduit localement leur coût futur.
IPOPT voit cette approximation et ses dérivées, pas toutes les simulations.
Ses coefficients sont mis à jour entre deux résolutions; le NLP compilé reste
réutilisable avec MA57.

Des états supplémentaires servent à vérifier si l'approximation prédit les
valeurs et classe correctement les candidats. Si elle se trompe trop, si une
simulation échoue ou si le solveur sort du voisinage validé, il faut refuser
ou recalculer l'approximation. Le fait de l'avoir rendue lisse ne suffit pas
à garantir sa justesse, notamment près d'un changement de déclenchement.

## Ce qui reste à régler et à démontrer

On ne demande plus de fixer à la main un poids constant pour chaque muscle :
l'effet de l'état de chaque muscle sur la réserve future est estimé depuis la
tâche. Il reste toutefois des choix globaux : durée de projection, niveau de
réserve souhaité, importance du terme futur dans le coût total, lissage et
seuil de déclenchement du preview. Ces choix ne sont pas automatiquement
calibrés pour un usage clinique.

Un rollout incomplet ne reçoit pas arbitrairement un « bon » ou un « mauvais »
score de remplacement dans l'implémentation actuelle : l'estimation est
refusée. Cela signifie que cette politique de prédiction n'a pas permis de
conclure, pas que toute autre stratégie musculaire serait impossible.

Enfin, ce score n'est pas un compte à rebours en cycles. Il dépend de la
politique simulée, de l'horizon et des réglages du lissage. La preuve utile
restera une comparaison de RHO réellement exécutés : davantage de cycles
réalisables à tâche identique, avec un suivi correct et un temps de calcul
compatible avec l'utilisation visée.

Pour les détails d'implémentation et les étapes de validation, voir
[le plan technique](compact_endurance_approaches.md).
