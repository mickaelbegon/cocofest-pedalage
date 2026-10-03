# Protocole expérimental du gradient terminal d’endurance

Le module `cocofest/optimization/terminal_costate_ocp.py` transporte un gradient numérique dans le RHO isocinétique à un cycle. Il exprime une **valeur estimée en cycles faisables restants**, à maximiser. Le solveur minimise donc son opposé. L’option interne `experimental_terminal_costate_config: {"maximum_age_cycles": 10}` crée le canal compilé, initialement inactif. Une mise à jour par `update_from_remaining_cycles_value` modifie les paramètres numériques du NLP sans toucher au graphe ni aux poids de fatigue.

Ce canal n’identifie aucun gradient. Les gradients analytiques jouets employés dans les tests ne sont pas des prédictions validées de l’endurance. En particulier, un gradient de réserve mécanique locale ne peut être présenté comme un gradient de durée restante.

## Expérience par branches appariées

1. Sélectionner trois checkpoints RHO entièrement restaurables : début, milieu, proche de l’échec. Conserver les états Ding complets, les états mécaniques, les historiques de stimulation, les contrôles, la configuration des modèles et l’empreinte du problème. Le travail demandé et le partage G/D restent identiques entre branches.
2. À chaque checkpoint, calculer un gradient lent candidat sans résultat FHO. Consigner la politique de continuation qui définit les « cycles restants », les données utilisées, les unités, la boîte de confiance, l’âge du gradient et une validation indépendante sur des états terminaux atteignables.
3. Depuis exactement le même checkpoint, résoudre un cycle avec : RHO de référence, meilleurs poids BO à partage fixe, gradient candidat, demi-gradient, gradient de signe inversé et réserve seule. Garder les mêmes bornes, tolérances, solveur et graine numérique. Un gradient opposé qui donne le même résultat indique que le signal est inactif ou noyé.
4. Mesurer la différence de PW et d’état terminal sur ce cycle. Vérifier que chaque solution satisfait le travail prescrit et que son état terminal reste dans la boîte de confiance. Rejouer ensuite des branches de 5 puis 20 cycles sous une **même politique de continuation**. Les branches qui sortent de la boîte de confiance sont signalées comme extrapolations, pas comme succès du modèle.
5. Pour les branches prometteuses, poursuivre jusqu’au certificat de faisabilité gelée et comparer les cycles certifiés, le temps RHO par cycle, le coût de calcul du gradient et l’activation effective du terme terminal. Les échecs du solveur ne valent pas automatiquement preuve de fatigue physiologique.

Le test de signe et la mise à jour du NLP compilé sont couverts par `tests/test_terminal_costate_ocp.py`. La réalisation des branches exige encore un prédicteur lent validé ; ce document ne rapporte donc aucun gain d’endurance attribué au costate.
