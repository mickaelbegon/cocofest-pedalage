# Pilote T3 de transfert de couple : analyse préenregistrée

## Données à exporter

L'analyseur `scripts/analyze_torque_transfer_pilot.py` lit directement le
`campaign-completion.json` du pilote causal et vérifie le manifeste scellé,
les reçus des deux bras et les témoins d'audit. Il accepte aussi un JSON
normalisé `torque_transfer_t3_pilot` de version 1. Chaque checkpoint (`anchors`) donne un
`id`, son cycle certifié et, si une règle de décision est évaluée, l'action
`chosen_action`. Chaque branche (`jobs`) donne `anchor_id`, `replicate_id`,
`action` (`minus`, `hold`, `plus`), `status` (`exact`, `censored`, `numerical`),
`terminal_classification`, `audit_certified`, `last_certified_cycle` et la liste
contiguë `certified_work_j_by_cycle` depuis le cycle suivant le checkpoint.
Le champ racine `intervention_cycles` vaut 10.

Le producteur doit vérifier le même checkpoint physique, les états rapides,
l'historique PW, les modèles, le graphe RHO, la politique de continuation et
les contraintes pour les trois actions appariées. Chaque cycle exporté doit
être passé par l'audit discret indépendant. `audit_certified: true` atteste
ce contrôle dans un export déjà normalisé. Pour une campagne native, l'analyseur
vérifie les hachages des reçus, témoins de cycle et checkpoint source ainsi que
la continuité des cycles. Il ne re-soutient pas les NLP ni ne relit les archives
RHO. Il faut donc conserver ces pièces avec le manifeste source.

`exact` requiert la classe `frozen_feasibility_exhaustion` établie par le
test de faisabilité gelée. `censored` requiert `administrative_horizon`.
`numerical` requiert `numerical_unresolved`. Une difficulté IPOPT seule ne
produit jamais une fin exacte. Les trois actions maintiennent les poids
unitaires et, après 10 cycles d'intervention, suivent la même politique de
continuation certifiée. Un arrêt avant le dixième cycle ne teste pas la dose
préenregistrée.

## Résolution et décisions

Pour chaque checkpoint, au moins deux répétitions `hold` avec fin exacte
définissent la **résolution en cycles** : maximum moins minimum des derniers
cycles certifiés. Un contraste de durée doit être *strictement supérieur* à
cette résolution. À égalité, l'effet est indiscernable. Il n'y a ni tolérance
clinique cachée ni seuil numérique universel en cycles.

L'analyse calcule les avantages `minus−hold` et `plus−hold` pour chaque
combinaison de branches issues du **même checkpoint**. Le pilote courant répète
les témoins, mais ne répète pas encore les deux traitements : ses comparaisons
ne sont donc pas des réplications indépendantes du traitement. Elle conclut une
direction `plus` seulement si tous les contrastes `plus−hold` et `plus−minus`
dépassent la résolution. La règle symétrique s'applique à `minus`. Il faut les
trois actions, leurs interventions complètes et toutes leurs fins exactes. Une
branche censurée, numérique ou manquante rend la direction inconclusive. Cette
règle évite de transformer une borne inférieure en durée de vie exacte.

Le travail est sommé sur le **même préfixe de cycles certifiés**, borné par la
branche la plus courte du checkpoint. Le rapport donne aussi le travail de
chaque préfixe propre à sa branche. La résolution de travail est l'étendue des
témoins `hold` à ce préfixe commun. Le travail commun est un diagnostic de
comparabilité; il ne remplace pas la fin de tâche certifiée dans T3.

Si `chosen_action` était fixé avant lecture des résultats, le regret est, pour
chaque combinaison témoin/moins/plus, la meilleure fin exacte moins celle de
l'action choisie. Il est indisponible si une action manque ou si une fin est
censurée ou numérique. Le rapport indique séparément si ce regret dépasse la
résolution témoin. Le pilote natif ne préenregistre aucune action choisie :
son regret est donc `not_predeclared`. Il renseigne la direction locale, pas
encore l'utilité d'une règle de décision tenue hors entraînement.

## Exécution

`python scripts/analyze_torque_transfer_pilot.py campagne/campaign-completion.json --output rapport-t3.json`

Le champ `activation_allowed` reste faux : un effet T3 observé à ces
checkpoints ne valide ni une carte `Phi` ni un superviseur en boucle fermée.
