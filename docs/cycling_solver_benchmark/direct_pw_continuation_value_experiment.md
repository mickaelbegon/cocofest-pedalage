# Valeur de continuation par intervention PW directe

## Question

Les essais précédents de valeur de continuation modifiaient les pondérations de
fatigue pendant une courte fenêtre, puis revenaient au RHO unitaire. Les 36
branches ont toutes donné le même dernier cycle certifié. Elles ne permettent
donc pas d'apprendre un gradient d'endurance ni d'activer le costate terminal.

Cette expérience change l'action expérimentale, et **ne change pas les poids
de fatigue**. Pour un cycle seulement, elle impose une légère translation
constante d'une ou plusieurs traces de PW directes. Le travail nominal, les
contraintes mécaniques et musculaires, l'historique et les bornes de PW restent
ceux du problème RHO restauré. Le solveur doit donc soit produire un cycle
admissible sous cette action, soit livrer un résultat expérimental
indéterminé.

Le premier prototype de verrouillage est dans
`cocofest/optimization/continuation_value_pw_action.py`. Le runtime IPOPT
actuel encode toutefois les bornes de PW par trois colonnes
début/intérieur/fin, alors que la trace contient 30 stimulations. Il refuse
donc explicitement de prétendre verrouiller une trace non constante. La voie
retenue pour la prochaine itération est un coût de suivi PW paramétrique dans
le graphe NLP, et non un changement silencieux de ces bornes compactes.

## Pourquoi une contrainte PW plutôt qu'une simple initialisation

Une variation de l'initialisation des PW n'est pas une action : IPOPT peut
revenir immédiatement à la solution nominale. L'action devra donc être une
cible PW paramétrée dans le coût, avec une intensité et une trace de référence
archivées. Les autres variables conservent leur liberté, notamment pour
maintenir le travail exigé. Un verrouillage dur ne sera autorisé que si le
backend expose réellement une borne par stimulation.

Les formulations avec contrôle auxiliaire de pente ou de dérivée PW sont
refusées explicitement. Il serait incorrect de déplacer leurs PW directs sans
respecter leur égalité de représentation.

## Protocole pré-enregistré

On démarre des checkpoints complets c120, c140 et c160 du même bras et de la
même tâche. À chaque ancre :

1. un témoin sans contrainte PW additionnelle ;
2. deux directions PW indépendantes, chacune avec un signe positif et négatif,
   pour l'apprentissage ;
3. une direction mixte indépendante, avec les deux signes, tenue hors de
   l'apprentissage.

Cela donne sept branches par checkpoint, donc 21 branches. Les directions ne
peuvent pas être choisies après lecture des labels ; elles sont normalisées
dans l'espace des muscles et archivées avec le manifeste. L'amplitude est une
fraction déclarée du plus petit intervalle PW actif. Chaque endpoint est
exporté, restauré dans un processus frais, puis poursuivi sous une politique
de référence unique déclarée avant l'exécution.

Chaque continuation est poursuivie jusqu'au plafond fixé ou au protocole gelé
de faisabilité : plusieurs essais à objectif nul et un contre-factuel de repos
de fatigue doivent être distingués des simples arrêts IPOPT. Une branche
numérique, hors délai ou censurée ne devient pas une durée d'endurance.

## Apprentissage et activation

Le label est le nombre de cycles restants certifiés sous la politique de
continuation. Le premier modèle ne peut utiliser que deux coordonnées
terminales atteignables et normalisées : tenter d'ajuster tous les états de
Ding, le calcium, la force et l'historique PW avec ces quelques branches serait
sous-identifié.

Avant toute injection dans `terminal_costate_ocp.py`, il faut vérifier :

- au moins deux contrastes d'apprentissage non nuls et reproductibles ;
- une erreur de validation hors échantillon d'au plus un cycle ;
- le bon signe pour les deux perturbations mixtes ;
- le support des états rapides et de l'historique PW ;
- une région de confiance imposée **dans** le NLP, pas seulement contrôlée
  après la résolution.

Le coût fatigue reste alors une statistique de sécurité ou un départage faible,
mais n'est plus la cible que l'on ajuste. Si les labels restent tous identiques,
il ne faut pas inventer un gradient : on élargit une seule fois la durée
d'intervention, pré-déclarée, puis on abandonne cette valeur locale si le
contraste est toujours absent.

## État d'implémentation

Le plan à 21 branches et l'exécuteur multiprocessus sont prêts, mais le pilote
IPOPT a volontairement bloqué avant solve sur la représentation compacte des
bornes. La prochaine étape est d'ajouter le coût de suivi PW paramétrique au
graphe compilé, avec activation nulle par défaut, puis de valider son gradient
et son absence de reconstruction avant de relancer le pilote. Aucun coût
terminal n'est activé par ce fichier.
