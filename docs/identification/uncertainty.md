# Incertitude et choix des essais de fatigue

`cocofest.identification.uncertainty.bootstrap_force_then_fatigue` rééchantillonne
des acquisitions complètes puis réidentifie la force et la fatigue dans chaque
répétition. Les paramètres de force changent ainsi entre répétitions et leur
incertitude se propage aux paramètres de fatigue. Chaque acquisition conserve
tous ses échantillons, canaux, poids et contexte physiologique. Un bloc
stimulation–récupération–sonde continu doit constituer une seule acquisition.

Pour un plan imposé PW × fréquence × phase, les répétitions indépendantes d'une
même condition forment une strate. Le bootstrap conserve le nombre d'essais
par condition : il faut donc acquérir plusieurs répétitions, idéalement lors
de récupérations indépendantes. Un unique essai par condition ne suffit pas
à estimer empiriquement cette incertitude. Sans strates, la méthode suppose
les acquisitions échangeables au sein de chaque étape.

```python
from cocofest.identification.uncertainty import (
    bootstrap_force_then_fatigue, fatigue_design_ensemble,
)
from cocofest.identification.cycling_design import design_trial_catalogue

# predictor, trials et ParameterSpec sont ceux de l'identification séquentielle.
# Ces noms désignent les objets de l'expérience, pas des données de démonstration.
report = bootstrap_force_then_fatigue(
    predictor, force_trials, fatigue_trials, force_specs, fatigue_specs,
    force_strata=force_condition_by_trial,
    fatigue_strata=fatigue_condition_by_trial,
    n_resamples=500, seed=7,
)

# Pour planifier une campagne de fatigue avec des scénarios initiaux explicites :
# chaque scénario doit fournir exactement les noms des fatigue_specs.
ensemble = fatigue_design_ensemble(report, fatigue_scenarios)
design = design_trial_catalogue(
    fatigue_catalogue, [spec.name for spec in fatigue_specs], ensemble,
    [spec.scale for spec in fatigue_specs], count=3,
)

# Après la campagne fatigue, garder les corrélations inter-étapes identifiées :
joint_ensemble = [draw["parameters"] for draw in report["replicates"]]
```

L'interface ci-dessus attend déjà des essais pilotes pour les deux étapes;
elle convient donc à la replanification après un premier bloc de fatigue.
Pour une toute première sélection sans aucun essai fatigue, fournir au design
un ensemble de paramètres de force obtenu indépendamment et des scénarios
fatigue a priori; ne pas fabriquer des observations fatigue pour satisfaire
l'interface du bootstrap séquentiel.

Le premier ensemble croise les tirages de force avec des scénarios fatigue
supposés indépendants; le second conserve la dépendance force–fatigue issue
des réidentifications. Le score de design est la moyenne du gain de log-déterminant
sur ces membres. Il prend en compte le déplacement des sensibilités lorsque
la force fraîche est incertaine; il ne marginalise pas les paramètres de
force comme nuisances dans l'information de Fisher. La limite de fatigue
prédite est évaluée sur le membre le plus défavorable de cet ensemble fini.

Les intervalles retournés sont des percentiles point par point de l'estimateur
MAP borné. Les priors restent fixes; ce ne sont pas des intervalles crédibles
bayésiens. Le rapport conserve la provenance des tirages, les échecs de
convergence, les bornes actives et les défauts de rang. Les intervalles sont
supprimés si moins de 80 % des répétitions convergent (seuil configurable).
Un taux élevé de convergence ne prouve pas l'identifiabilité, et exclure les
échecs peut biaiser les intervalles. Moins de 200 répétitions déclenche une
mise en garde sur les quantiles; 500 ou plus est préférable pour l'analyse
finale, avec vérification de leur stabilité.

Cette procédure suppose des acquisitions indépendantes entre les étapes force
et fatigue. Pour des sessions appariées, une dérive commune d'électrodes ou
plusieurs sujets, il faut rééchantillonner les sessions/sujets conjointement.
Elle ne couvre pas l'incertitude de géométrie, une erreur de modèle Ding ou
la variabilité d'état initial non représentées dans le contexte. Les résidus
de validation et les tests de rang restent nécessaires avant emploi expérimental.
