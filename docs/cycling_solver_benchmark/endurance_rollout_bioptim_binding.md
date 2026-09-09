# Raccordement expérimental du rollout à Bioptim

Le raccordement de `endurance_rollout_objective` est structurel et désactivé
par défaut. Il ne constitue pas une validation prospective d'endurance.
Le chemin public passe par `cycling_fes_solver_comparison.main`,
`cycling_pulse_width_mhe_acados_periodic.solve_case`, `simulation_conditions`,
`prepare_nmpc`, `set_objective_functions` et `CustomObjective`.

Les deux CLI exposent les mêmes options :

- `--experimental-endurance-rollout` : opt-in obligatoire si le poids est positif ;
- `--endurance-rollout-weight` : multiplicateur sans dimension nul par défaut ;
- `--endurance-rollout-profile` : rapport JSON de l'adaptateur avec son
  `rollout_objective_profile` déjà packé ;
- `--endurance-rollout-horizon-cycles` : horizon symbolique fixe, 5 par défaut ;
- `--endurance-rollout-temperature` : température du smooth-max, 0.02 par défaut ;
- `--endurance-rollout-domain-epsilon` : borne des seules quantités strictement
  positives, `1e-8` par défaut ; les domaines fermés gardent une borne nulle.

Le rapport doit porter `status=complete`,
`adapted_policy_fidelity.passed=true` et un profil de schéma
`cocofest-rollout-objective-profile-v1` dont `source_policy_gate_passed=true`.
L'ingestion conserve le vecteur numérique existant ; elle ne reconstruit
ni ne synthétise une politique de remplacement. Noms et ordre des muscles,
paramètres Ding et borne maximale PW doivent correspondre au modèle OCP.
La borne maximale est lue dans les bornes de contrôle effectivement construites,
sans constante `0.0006` dans le raccordement. Une borne dépendant de la phase
(par exemple un masque d'activation) est refusée par cette première version,
dont le noyau suppose une borne maximale constante pour chaque muscle.
La représentation peut être Fourier ou polynomiale par intervalles.
Le CLI accepte uniquement le rapport adaptateur complet ; il applique toujours
l'horizon et la température demandés. L'API `from_payload` vérifie aussi que
le vecteur, les paramètres musculaires, la méthode et les dimensions sont
exactement ceux du rapport embarqué. Un payload indépendant ou divergent est
refusé, même si une copie du gate y indique `passed=true`.

Le coût est un `ObjectiveFcn.Mayer`, `Node.END`, scalaire et non quadratique,
de poids effectif `10000 * endurance_rollout_weight`. La contrainte terminale
utilise un vecteur de bornes dans l'ordre exact des 15 marges par échantillon :
`A`, `Tau1`, `Km+Cn`, `Cn`, gain mécanique, temps de relaxation et recrutement
maximal ont une borne epsilon ; `A_rest-A`, `Tau1-Tau1_rest`, `Km-Km_rest` et
recrutement requis ont une borne zéro. Les marges d'état sont présentes au
midpoint et en fin d'intervalle. Le repos et une force nulle restent donc
admissibles. Le poids zéro n'ajoute ni objectif,
ni contrainte, ni paramètre ; même un chemin de profil inexistant n'est pas lu.

## Paramètres fixes et réutilisation

La version locale épinglée de Bioptim substitue les
`numerical_data_timeseries` dans les fonctions NLP. Changer ce tableau change
donc le graphe compilé. Le prototype emploie à la place un `ParameterList`
de dimension fixe, verrouillé par des bornes inférieure et supérieure égales.
Ce ne sont pas des paramètres libres à optimiser : leurs valeurs sont imposées
à chaque solve par `lbx/ubx`.

Ce choix ajoute **1080 paramètres fixes au vecteur NLP générique** pour quatre
muscles et 30 intervalles (`9 * 4 * 30`). Il peut coûter en taille et en
factorisation ; aucun gain de performance n'est revendiqué. Les dimensions
spécifiques d'autres backends, notamment ACADOS qui représente les paramètres
autrement, ne sont pas validées par ce smoke IPOPT.

`nmpc.endurance_rollout_binding.update(nmpc, new_profile)` accepte une nouvelle
instance `CertifiedRolloutParameters` ayant la même structure. Cette méthode
met à jour les bornes et l'initialisation, sans modifier les pénalités. Elle
préserve aussi les tableaux de paramètres des solutions déjà certifiées :
l'initialisation RHO peut partager leurs vues et ne doit pas être écrasée.
Un changement de dimensions, d'ordre musculaire, de modèle ou de méthode de
reconstruction est refusé. Le suivi du solveur échoue explicitement si l'objet
NLP ou son solveur compilé est reconstruit entre fenêtres.

Le CLI charge une politique initiale figée. La reconstruction et la
certification automatiques de la politique après chaque nouveau RHO ne sont
pas activées par ce raccordement. L'API de mise à jour est exercée par les tests.
Les sorties RHO actives utilisent le format compact afin de préserver ces
paramètres fixes, que l'ancien agrégateur multibody traitait comme des séries
de stimulation.

## Audit et tests

Le JSON `endurance_rollout` indique la méthode, les dimensions, le nombre de
paramètres fixes, le poids, epsilon, le statut expérimental et la provenance
SHA-256 du profil, du rapport d'adaptation et de sa source. Les signatures de
codegen comprennent la structure, la méthode et les options. Les caches de
solutions/initialisations comprennent en plus les valeurs numériques via leur
empreinte. Changer uniquement le profil conserve donc la signature de codegen,
mais invalide une initialisation dépendante de l'ancienne politique.
Les nouveaux rapports exportent également `source_ocp_context` depuis les
métadonnées du NPZ. Période, couple signé, formulation mécanique et formulation
dynamique/isocinétique sont comparés au problème demandé lorsqu'ils sont
disponibles. Une divergence explicite est refusée ; un champ absent, notamment
dans un ancien rapport, apparaît comme `not_verifiable` dans la provenance et
l'audit de compatibilité. Cela ne lève pas le NO-GO scientifique du prototype.

`tests/test_endurance_rollout_ocp.py` vérifie l'absence complète à poids zéro,
y compris l'égalité des fonctions objectifs Bioptim sérialisées, le refus des
gates incomplets, les signatures et la provenance. Deux tests exécutent IPOPT
avec génération/compilation C sur une fixture synthétique : deux solves avec
des profils distincts, puis une vraie boucle `RecedingHorizonOptimization` de
deux fenêtres. Les coûts changent, les solves convergent et le même objet NLP
compilé est conservé (`compiled_solver_build_count=1`). Cette fixture vérifie
l'interface, pas les effets physiologiques du rollout sur le pédalage.
Les tests couvrent aussi le rejet de payloads divergents, l'application des
options CLI, les incompatibilités explicites ou non vérifiables et un solve
IPOPT réussi à force nulle et à l'état de repos avec les bornes mixtes.
