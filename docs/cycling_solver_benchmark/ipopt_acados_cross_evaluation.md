# Comparer les solutions IPOPT et ACADOS avec une référence DOP853

Audit du code disponible le 19 septembre 2026. Ce document distingue les
différences de problème, les différences d'algorithme et le protocole de
réévaluation. Les réglages effectifs enregistrés dans chaque résultat restent
prioritaires : les anciens profils et les configurations du GUI peuvent
différer des valeurs par défaut du lanceur de comparaison.

## Question scientifique

Une solution optimisée avec IPOPT est-elle toujours préférable à une solution
ACADOS lorsqu'on applique exactement ses impulsions au même modèle physique,
depuis le même état initial, et qu'on calcule un coût commun ?

Le solveur ne définit pas un intégrateur universel. « Évaluateur IPOPT »
signifie ici la carte implicite du tableau Radau utilisé par sa transcription ;
« évaluateur ACADOS » signifie le tableau Gauss-Legendre de la capsule IRK.
Le calcul d'un rollout à commandes fixées n'exige aucune optimisation IPOPT
ou ACADOS, mais il exige de résoudre correctement les équations implicites
du tableau choisi. Une réalisation indépendante doit être nommée comme telle,
et sa correspondance à la carte native doit être testée.

**DOP853 est la référence physique de cette comparaison.** Il réintègre les
équations complètes de Ding et la mécanique. L'élimination locale fournit des
cartes exactes pour un tableau fixé ; ses polynômes reconstruits aux stages
ne doivent pas être réutilisés comme une nouvelle dynamique continue DOP853.

## Différences effectives dans le dépôt

| Élément | IPOPT, profil `scientific_radau5` | ACADOS, référence SQP/IRK |
|---|---|---|
| Formulation musculaire | `periodic_node` ; calendrier numérique fixé | Même famille `periodic_node` |
| Transcription | Collocation directe Radau, degré 5, un élément par intervalle de stimulation | Tir multiple ; IRK Gauss-Legendre 4 stages, 5 sous-pas par intervalle |
| Variables de dynamique | États aux nœuds et aux stages du NLP | États aux nœuds ; stages résolus à l'intérieur de l'intégrateur |
| Mécanique | `full` ou `reduced` suivant configuration | Même choix ; les campagnes de réduction Ding considérées sont `reduced`, dynamique |
| Réduction Ding locale | Élimine `Cn,Tau1,Km`, conserve `F,A` et la mécanique aux nœuds/stages ; certificat discret Radau | Même idée, mais offsets et calcium issus du tableau GL ; 22→10 états natifs pour 4 muscles, hors porteurs de slew |
| Algorithme | Points intérieurs, Hessienne du Lagrangien selon options, solveur linéaire MA57 pour la référence concernée | SQP, QP HPIPM selon options, approximation Gauss-Newton par défaut du lanceur |
| Coût | Fonctions Bioptim pondérées ; Lagrange par défaut rectangle gauche | Export `NONLINEAR_LS`, coût natif avec facteur 1/2, matrices `W,W_0,W_e` et scaling temporel ACADOS |
| Bornes | Les états décisionnels de collocation peuvent porter les bornes aux stages ; les contraintes personnalisées ont leur propre sélection de nœuds | Bornes d'état aux nœuds ; les stages IRK internes ne sont pas automatiquement contraints |
| Garde de vitesse reduced | `auto` résout à faux pour IPOPT ; imposer `on` pour une comparaison qui la requiert | `auto` résout à vrai en mécanique reduced/dynamique ; garde `euler_half_step_v1`, distincte d'une borne exacte sur les stages GL |
| Contraintes de départ | `enforce_start_constraints=true` dans le profil scientifique ; inspecter les contraintes réellement construites | Valeur initiale du parseur faux, avec seed/projection et bornes d'état initial spécifiques ; ne pas supposer la même liste de contraintes |
| Scaling | Échelles explicites états/commandes, scaling NLP et options MA57 selon configuration | Échelles états/commandes et scaling QP distincts ; le défaut `OBJECTIVE_GERSHGORIN`/`INF_NORM` n'est pas un changement de coût physique |
| Seed et duales | Raffinement sur la transcription cible ; mode de duales `bounds` par défaut du lanceur | Seed certifiée/projection et mémoire native ; mode dual `reset` par défaut du lanceur |
| Réduction et duales | L'adaptateur expérimental IPOPT reconstruit des duales complètes | La façade ACADOS reconstruit les états complets ; `pi/lam` restent natifs réduits, sans équivalence duale complète |
| RHO | Avancement d'un cycle après certification, horizon configurable | Même intention ; retries, Phase-I ou fallback éventuels modifient le parcours numérique et doivent être identifiés |

Attention au nom **Radau-5** : dans ce dépôt il désigne cinq stages/degré 5,
pas le schéma Radau IIA usuellement nommé « ordre 5 » à trois stages.
Les cartes à comparer doivent être construites à partir des abscisses et du
nombre de sous-pas enregistrés, plutôt qu'à partir du seul nom « Radau ».

`full/reduced` mécanique et réduction locale Ding sont deux transformations
différentes. La mécanique reduced impose sa géométrie par construction ; la
mécanique full possède des coordonnées et contraintes de contact dont la
dérive est auditée. Une comparaison initiale propre utilise la même mécanique
reduced dans les deux sources.

## Les coûts natifs ne sont pas directement interchangeables

Les résidus physiques du constructeur commun sont

\[
r_{A,m}=1-A_m/A_{r,m},\qquad
r_{F,m}=F_m/F_{\max,m},\qquad
r_{u,m}=PW_m/PW_{\max,m}.
\]

En mode quadratique, le coût de fatigue seul prend comme référence commune

\[
J_{\mathrm{phys}}=10^4\int_{t_0}^{t_f}
\sum_m\left(1-\frac{A_m(t)}{A_{r,m}}\right)^2dt.
\]

Les termes force et stimulation sont ajoutés avec les poids déclarés. Les
termes terminaux, de suivi mécanique, de slew ou de réserve sont rapportés
séparément et ne sont inclus dans un total commun que lorsque leur définition
et leur poids ont été explicitement alignés. Le normalisateur est `a_scale`
du modèle, pas la capacité à l'entrée d'une fenêtre déjà fatiguée.

Trois détails du code justifient cette réévaluation :

1. `set_objective_functions` crée les coûts principaux Lagrange avec
   `Node.ALL`, sans quadrature explicite. Bioptim choisit
   `RECTANGLE_LEFT`. L'intégration de la dynamique aux stages Radau n'implique
   donc pas une intégration du coût par les poids Radau.
2. L'export ACADOS transmet les résidus non pondérés et les poids dans `W`.
   Le coût `NONLINEAR_LS` natif est \(\tfrac12 r^TWr\). Il transmet aussi les
   Lagrange `Node.ALL` au chemin Mayer, donc à `W_e`. Le scaling ACADOS par
   défaut est `[dt_0,...,dt_{N-1},1]`. Un résidu terminal principal peut donc
   être pondéré autrement que dans la fonction Lagrange Bioptim, qui porte
   son facteur `dt`. **Multiplier simplement le coût ACADOS par deux ne
   corrige pas nécessairement l'écart.** Exporter séparément les contributions
   initiales, courantes et terminales de la capsule effective permet de le
   quantifier pour une campagne précise.
3. Les scores historiques `_executed_fatigue_objective_by_muscle` utilisent
   une quadrature trapézoïdale et une durée en cycles, en supposant les colonnes
   uniformes. Ce score n'est ni le coût natif ni une référence DOP853. Des
   colonnes de collocation Radau ne sont pas uniformément espacées ; leur
   calendrier doit être reconstruit avant toute quadrature.

Le rapport doit donc conserver trois notions : coût annoncé par le solveur,
coût selon une recette discrète explicitement nommée, et coût physique continu
commun. DOP853 peut intégrer une variable supplémentaire
\(\dot z=L(x,u,t)\), avec \(z(t_0)=0\), pour obtenir ce dernier. Une vérification
avec tolérances plus serrées mesure son incertitude numérique.

## Matrice de rollout ouvert

| Commandes source, figées | Carte Radau IIA 5 stages | Carte IRK GL4×5 | DOP853 de référence |
|---|---|---|---|
| Solution IPOPT | Trajectoire et coûts communs | Trajectoire et coûts communs | Trajectoire et coûts communs |
| Solution ACADOS | Trajectoire et coûts communs | Trajectoire et coûts communs | Trajectoire et coûts communs |

Pour chaque ligne, les trois rollouts partent du **même état physique complet**,
avec les mêmes paramètres, géométrie, charge, durée, impulsions et PW. Ils
propagent leur propre état jusqu'à la fin. Recaler l'état sur chaque nœud
optimisé mesure un défaut local de transcription, mais masque la dérive du
rollout ouvert : ce diagnostic doit être nommé séparément.

Pour comparer les lignes entre elles, les sources doivent aussi partager
l'état initial et l'horizon évalué. Deux fenêtres tirées de RHO fermés
différents ont généralement des fatigues entrantes différentes ; elles ne
permettent pas d'attribuer un classement au seul solveur.

Chaque impulsion et changement de commande découpe un intervalle
d'intégration. Le forçage `periodic_node` de l'intervalle est conservé jusqu'à
sa fin, puis mis à jour au suivant. Cela évite d'activer l'impulsion future
au dernier stage Radau ou à une évaluation DOP853 sur la borne droite. Le
calcium et la fatigue à l'origine, ainsi que l'historique numérique nécessaire,
font partie de la solution exportée ; on ne les remplace pas par le repos.

Le résultat par cellule contient au minimum les contributions de coût,
les écarts à DOP853, la capacité finale et minimale, les violations de bornes
de PW/états, la vitesse et la phase mécaniques, la validité et le motif
d'échec. Une solution moins coûteuse mais mécaniquement infaisable ne doit
pas être classée meilleure. Les contraintes entre nœuds demandent un
échantillonnage dense et un test de raffinement ; une grille finie ne prouve
pas à elle seule une borne continue exacte.

Un classement robuste exige que le signe de
\(\Delta J=J(A)-J(B)\) reste identique pour les évaluateurs, que les deux
solutions satisfassent le même contrat physique et que l'écart dépasse
l'incertitude de réintégration/quadrature. À défaut, conclure « classement
non établi », avec les écarts mesurés.

## RHO réoptimisé : expérience complémentaire

Ce mode lance une nouvelle optimisation à chaque fenêtre. Une comparaison
contrôleur de référence applique le premier cycle de la solution à une
simulation DOP853, utilise son état terminal comme nouvelle mesure, puis
réoptimise. Elle fixe entre variantes la durée de fenêtre, l'avancement,
le modèle, la charge, les coûts et contraintes communs ainsi que les règles
de certification/reprise.

Les entrées divergeront naturellement parce que les décisions divergent.
Ce mode mesure performance et stabilité des contrôleurs, tandis que les
fenêtres gelées du mode précédent isolent une comparaison locale de solutions.
Le simple lancement des RHO historiques, où le transfert utilise les états
de la transcription, constitue une autre expérience et doit l'annoncer.
La disponibilité d'une option de lancement GUI ne certifie pas à elle seule
que la propagation fermée a été remplacée par DOP853.

La somme des coûts de fenêtres recouvrantes compte plusieurs fois la même
portion temporelle. Pour l'endurance, le coût commun doit porter seulement
sur les cycles effectivement appliqués. Comparer également un préfixe commun
lorsqu'une chaîne s'arrête tôt, avec le nombre de cycles validés : un total
plus petit sur une chaîne plus courte n'est pas une meilleure solution.

## Ce que les mesures antérieures établissent

La réduction locale IPOPT a montré une médiane plus basse sur fenêtres gelées
et sur 50 RHO propagés ; cela ne prouve pas un meilleur contrôle qu'ACADOS.
Le benchmark ACADOS local/complet de 100 RHO a validé 76 cycles puis s'est
arrêté au 77 dans les deux cas, avec médianes chaudes natives 0,167/0,455 s.
Ces résultats comparent deux représentations d'un même backend. Ils ne
valident ni l'équivalence IPOPT/ACADOS des coûts terminaux, ni un classement
physiologique entre leurs solutions. Les anciens tests IPOPT à 30 stimulations
et ACADOS à 50 ne forment pas davantage une paire contrôlée.

## Sources primaires et points d'entrée

Les références ci-dessous sont le code local audité, sans dépendre de la
documentation d'une autre version installée :

- [Profils et statistiques de comparaison](../../examples/fes_multibody/cycling/cycling_fes_solver_comparison.py) : `IPOPT_PROFILE_DEFAULTS`, `BENCHMARK_CONFIGURATION_FIELDS`, `_executed_fatigue_objective_by_muscle`.
- [Construction des objectifs et contraintes](../../examples/fes_multibody/cycling/cycling_pulse_width_mhe.py) : `set_objective_functions`, `set_constraints`, garde mécanique reduced.
- [Résidus physiques](../../cocofest/custom_objectives.py) : `minimize_overall_muscle_fatigue`, `minimize_overall_muscle_force_production`, `minimize_overall_stimulation_charge`.
- [Runner RHO](../../examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py) : `build_ode_solver`, `resolve_reduced_internal_crank_velocity_guard`, `acados_objective_weight_layout`, `solve_case`, audits et transferts certifiés.
- [Quadrature Bioptim](../../.benchmark-deps/bioptim/bioptim/limits/objective_functions.py) et [fonctions pondérées](../../.benchmark-deps/bioptim/bioptim/limits/penalty_option.py).
- [Export Bioptim/ACADOS](../../.benchmark-deps/bioptim/bioptim/interfaces/acados_interface.py) : `add_nonlinear_ls_lagrange`, `add_nonlinear_ls_mayer`, `__set_costs`.
- [Définition NLS ACADOS](../../.benchmark-deps/bioptim/external/acados/interfaces/acados_template/acados_template/acados_ocp_cost.py) et [scaling temporel par défaut](../../.benchmark-deps/bioptim/external/acados/interfaces/acados_template/acados_template/acados_ocp.py).
- [Réduction IPOPT et résultats](ding_radau5_local_ab.md), [réduction ACADOS et résultats](acados_ding_local/README.md), [audit de l'ordre Radau](ding_radau_observed_order.md).

Lancer le GUI depuis la racine du dépôt, avec l'environnement Python du projet
activé :

```bash
python -m cocofest.simulation.gui
```

## Validation contrôlée du 19 septembre 2026

Une première paire réellement homogène a été produite dans
[`cross-ipopt-acados-contract-20260919`](../../cross-ipopt-acados-contract-20260919).
Les deux optimisations utilisent le même seed entrant, quatre muscles Ding avec
les mêmes paramètres déclarés, mécanique réduite dynamique, `periodic_node`,
charge constante de `0,1 Nm`, 50 stimulations, un cycle de 1 s et la garde de
vitesse activée. IPOPT emploie la collocation Radau IIA à cinq stages et MA57 ;
ACADOS emploie le multiple shooting IRK Gauss-Legendre 4x5, **sans** réduction
Ding locale. Les deux retours natifs sont `success=True` et physiquement
certifiés selon leur propre vérification de nœuds.

| Source de commandes figées | DOP853, coût physique commun | Radau IIA 5 stages | GL4x5 | Écart ACADOS vs IPOPT (DOP853) |
|---|---:|---:|---:|---:|
| IPOPT/MA57 | 0,012954771176 | 0,012954822380 | 0,012954771209 | — |
| ACADOS complet | 0,012961935777 | 0,012961987121 | 0,012961935808 | +0,055305 % |

L'ordre IPOPT puis ACADOS est donc invariant sous les trois évaluateurs. Le
score DOP853 est la référence ; les temps du rejeu ne comparent pas les
solveurs. Les objectifs natifs des fenêtres (`0,04520115` IPOPT,
`0,04523909` ACADOS) restent volontairement séparés de ce coût commun.

Cette conclusion est **préliminaire**, et non une victoire de solveur : le
rejeu dense met en évidence une violation de la garde de vitesse entre nœuds
pour les deux candidats (`0,00527 rad/s` IPOPT, `0,01956 rad/s` ACADOS sous
DOP853). La prochaine validation doit donc rendre cette contrainte continue
ou la vérifier/raffiner identiquement pour les deux transcriptions, puis
répéter la matrice. Les fichiers `matrix.json`, trajectoires réintégrées et
provenance se trouvent dans
[`cross-rollout`](../../cross-ipopt-acados-contract-20260919/cross-rollout).

Les archives du 19 septembre ne contenaient pas encore les paramètres Ding
effectifs. La configuration
[`model-config-baseline-ding.json`](../../cross-ipopt-acados-contract-20260919/model-config-baseline-ding.json)
les déclare explicitement et le rapport l'étiquette correctement comme une
déclaration utilisateur, non une vérification rétrospective des archives.

### Essai de garde rapide partagée

Le 19 septembre, les nouvelles options
`--wheel-qdot-fast-bound-margin` et `--wheel-qdot-slow-bound-margin` ont été
ajoutées au runner de comparaison. Elles alimentent désormais IPOPT et ACADOS
avec la même marge asymétrique; une éventuelle option `--acados-*` reste une
surcharge explicitement différente. Les valeurs par défaut restent inchangées.
La compatibilité des seeds vérifie ces marges : un seed créé à `3,0` est refusé
pour une marge rapide de `2,95`, ce qui a permis de détecter une omission de
câblage ACADOS avant tout benchmark.

Un nouveau seed IPOPT, puis les deux fenêtres, ont été produits avec les
marges communes `(rapide, lente)=(2,95, 3,00) rad/s`. ACADOS atteint cette
borne par une homotopie de `3,00` à `2,95` et garde une convergence native
(`status=0`, une itération pour la fenêtre mesurée). La matrice est dans
[`fast-guard-cross-rollout`](../../cross-ipopt-acados-contract-20260919/fast-guard-cross-rollout).

| Évaluation | Score IPOPT | Score ACADOS | Dépassement rapide IPOPT | Dépassement rapide ACADOS |
|---|---:|---:|---:|---:|
| DOP853 | 0,012641431600 | 0,012655279302 | 0,00692385 rad/s | 0,00747090 rad/s |
| Radau IIA 5 stages | 0,012641478811 | 0,012655326833 | 0,00677059 rad/s | 0,00730011 rad/s |
| GL4x5 | 0,012641431629 | 0,012655279330 | 0,00692379 rad/s | 0,00747080 rad/s |

L'ordre reste IPOPT puis ACADOS, mais cet essai **ne certifie pas** la garde :
réduire empiriquement la marge du prédicteur Euler au demi-pas déplace le pic
sans le borner. La bonne étape suivante est une contrainte/certification aux
points internes obtenus par une carte de dynamique cohérente (ou une borne
conservatrice prouvée), ajoutée progressivement et validée sur ACADOS avant
de l'imposer à IPOPT. Ce résultat évite de transformer une correction de
transcription en réglage numérique arbitraire.

Un contrôle DOP853 à 65 échantillons par intervalle localise le minimum au
45e intervalle, vers `0,906 h` pour IPOPT et `0,922 h` pour ACADOS : il ne se
situe donc pas au demi-pas. Un essai analytique de prédicteurs Euler aux mêmes
états de shooting donne, à `0,9 h`, `-9,32635` (IPOPT) et `-9,33083 rad/s`
(ACADOS), contre des minima continus `-9,24021` et `-9,24095 rad/s`. Ce serait
un excès conservateur d'environ `0,09 rad/s`, supérieur à dix fois le
dépassement restant. Ajouter des gardes Euler à `0,75 h`/`0,9 h` est donc
écarté : cela resserrerait le NLP pour corriger l'erreur du prédicteur, pas la
physique. Les données denses sont dans
[`fast-guard-dop853-dense`](../../cross-ipopt-acados-contract-20260919/fast-guard-dop853-dense).

Enfin, le moteur de rollout ne publie désormais un `ranking` que si les sources
sont compatibles **et** si DOP853 ne détecte aucune violation échantillonnée
de PW, phase, vitesse ou charge. La paire présente produit ainsi correctement
`rankings: {}` avec les deux dépassements comme motifs de refus, dans
[`fast-guard-dop853-certification`](../../cross-ipopt-acados-contract-20260919/fast-guard-dop853-certification).

### Prototype de carte interne complète : non retenu en l'état

Une carte symbolique RK4 qui réintègre les 22 états réduits (Ding et mécanique)
à une fraction interne a été prototypée avec cinq sous-pas. Elle est beaucoup
plus fidèle localement que l'Euler gelant la force : à `0,9 h`, son erreur de
vitesse sur le rollout local est de l'ordre de `1e-5 rad/s`. Un unique RK4 est
instable pour `tau_c=11 ms`; cinq sous-pas sont le minimum observé pour rester
dans le domaine Ding. Les tests unitaires de la carte et du moteur de rollout
passent.

En revanche, l'ajout direct de cette expression (20 évaluations de dynamique
par nœud) à toutes les contraintes a laissé la construction IPOPT sans sortie
pendant plus de 90 s, contre environ 10 s pour le cas de référence; le test a
été interrompu avant résolution. La voie « dérouler RK4 dans chaque contrainte
symbolique » est donc rejetée pour les benchmarks de production. L'option
expérimentale reste désactivée par défaut. Une suite crédible doit soit
factoriser cette carte dans une fonction CasADi/codegen réutilisable, soit la
garder comme certificat post-solve et employer une méthode de raffinement des
commandes seulement sur les intervalles fautifs.

### Régularisation quadratique de la vitesse ACADOS : essai négatif

Une alternative beaucoup moins coûteuse a été mesurée : ajouter à ACADOS le
terme existant de régularisation de vitesse autour de `-2π rad/s`, avec un
poids `0,01`, tout en gardant exactement le même seed, le même profil réduit
et les mêmes contraintes aux nœuds. La résolution ACADOS converge en sept
itérations, mais elle est moins bonne au rejeu de référence DOP853 : le coût
physique commun passe de `0,012961935777` à `0,013142031097` (+1,39 %), et le
dépassement de la borne rapide échantillonné augmente de `0,01955960` à
`0,02058434 rad/s`. Le coût natif ne doit pas être comparé ici car il inclut
ce nouveau terme de régularisation.

Ce réglage est donc écarté : il n'améliore ni la faisabilité continue ni le
critère physique commun. Les résultats reproductibles sont dans
[`acados-speed-regularization-0p01-dop853`](../../cross-ipopt-acados-contract-20260919/acados-speed-regularization-0p01-dop853).

### Contrôle continu réutilisable lors d'un benchmark

L'option existante `--validate-integrator-maps` produit maintenant aussi un
audit DOP853 dense de la vitesse de manivelle pour toute formulation contenant
`theta` et `qdot`. Il évalue 65 points par intervalle et exporte les minima,
maxima, bornes déduites des marges effectivement demandées, et le dépassement
maximal dans `high_accuracy_trace_rollout.dense_crank_velocity_audit`. C'est
un diagnostic post-solve : il ne change ni le problème ACADOS/IPOPT ni le
temps rapporté par le solveur. Il rend néanmoins immédiatement visible la
différence entre faisabilité aux nœuds et faisabilité de la trajectoire
continue, avant de publier une comparaison de solveurs.

La validation sur une fenêtre ACADOS dynamique réduite (50 intervalles, seed
commun, IRK Gauss--Legendre 4x5) confirme le fonctionnement : le solveur
converge toujours (`status=0`, cinq itérations), tandis que l'audit imprime
`omega=[-9,30397364, -3,30024620] rad/s` pour les bornes
`[-9,28318531, -3,28318531]`, soit un dépassement continu de
`0,020788337 rad/s`. `physical_success=True` conserve volontairement ici son
sens historique de certificat aux nœuds ; l'audit ne doit donc pas être lu
comme une certification NLP, mais comme le signal nécessaire pour refuser un
classement physique non continu.

### Calibration externe de la marge rapide : rapprochement, pas certification

Un essai commun à marge rapide `2,90 rad/s` a été résolu par IPOPT/MA57. Son
audit local DOP853 donne seulement `3,71e-05 rad/s` de dépassement, contre
plusieurs milliradians auparavant. Une marge `2,899 rad/s` donne le même
résidu : le prédicteur Euler demi-pas reste actif et son erreur locale ne peut
pas être éliminée en déplaçant seulement sa borne. ACADOS consomme le seed
`2,899` et converge en une itération (`0,0958 s` solveur), mais son audit local
reste à `0,0025163 rad/s`.

Le rejeu **commun** DOP853 est plus important que ces diagnostics locaux : il
mesure encore `0,0051462 rad/s` pour la commande IPOPT et `0,0077159 rad/s`
pour ACADOS. Il refuse donc correctement `ranking={}`. Les trajectoires et le
rapport sont dans
[`fast-guard-2p899-dop853`](../../cross-ipopt-acados-contract-20260919/fast-guard-2p899-dop853).
La calibration externe resserre utilement les itérés, mais n'est pas une voie
de certification. La priorité reste une carte interne fidèle et factorisée,
ou un raffinement de maillage qui expose les points internes à l'OCP.

### Comparaison numérique qualifiée

Le rapport de rollout sépare désormais deux conclusions. `rankings` demeure
un classement strict et exige un dépassement DOP853 nul. `qualified_rankings`
autorise une comparaison de solveurs, sans la présenter comme une certification
continue, lorsque les sources sont compatibles, qu'il n'y a aucun dépassement
PW/phase/charge, que chaque dépassement de vitesse est inférieur à
`0,01 rad/s`, et que leur ratio est inférieur à `2`. Ces seuils sont explicites
et surchargeables avec `--qualified-max-velocity-violation-rad-s` et
`--qualified-max-velocity-ratio`.

Sous cette règle, la paire `2,899` est comparable : DOP853 ordonne IPOPT
(`0,013834354327`) devant ACADOS (`0,013845973214`), soit +`0,084 %` pour
ACADOS. Les écarts de vitesse sont respectivement `0,0051462` et `0,0077159
rad/s` (ratio `1,50`). Le rapport
[`fast-guard-2p899-qualified-dop853`](../../cross-ipopt-acados-contract-20260919/fast-guard-2p899-qualified-dop853)
conserve simultanément `rankings: {}` et ce classement qualifié : il ne masque
donc jamais la différence entre une approximation numérique comparable et une
trajectoire continûment certifiée.

### Pourquoi les commandes IPOPT et ACADOS ne sont pas identiques

Sur cette paire qualifiée, l'écart de coût commun est faible (+`0,084 %` pour
ACADOS), mais il ne signifie pas que les commandes doivent être identiques.
IPOPT optimise une transcription par collocation Radau IIA à cinq stages,
alors qu'ACADOS optimise un multiple shooting IRK Gauss--Legendre 4x5. Ils
évaluent donc les défauts, les sensibilités et les contraintes de vitesse à
des points internes différents. Dans un problème non convexe et faiblement
régularisé, ces petites différences peuvent sélectionner deux redistributions
musculaires voisines dans une vallée d'objectif presque plate.

La mesure directe des 50 PW montre précisément ce mécanisme : les deltoïdes
sont pratiquement superposés (RMS IPOPT--ACADOS `0,79 µs` pour Delt_ant et
`1,13 µs` pour Delt_post). Les écarts sont concentrés sur quelques commandes
de compensation : RMS `33,56 µs` pour Biceps et `13,39 µs` pour Triceps. Les
plus grands écarts sont Biceps à l'intervalle 21 (`173,14` contre `376,02 µs`)
et Triceps à l'intervalle 49 (`213,10` contre `131,41 µs`). La plupart des
autres commandes sont à leur PW minimale (`131,41 µs`). C'est le profil attendu
d'une non-unicité locale de la répartition des efforts, non la signature d'un
écart global de dynamique.

Pour réduire ces différences sans imposer artificiellement une solution, la
prochaine expérience doit ajouter une régularisation commune, très faible, de
variation des PW ou de proximité à une référence, puis comparer à nouveau le
coût DOP853 et la répartition musculaire. Elle doit être identique dans les
deux NLP : une régularisation de vitesse appliquée à ACADOS seul a déjà été
testée et dégrade le critère commun.
