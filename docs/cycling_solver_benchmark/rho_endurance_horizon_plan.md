# Donner un horizon d’endurance au RHO sans données FHO

## Objet et contrainte clinique

Ce plan vise à prolonger l’endurance d’un pédalage FES contrôlé par des RHO
d’un cycle. Le contrôleur clinique ne doit dépendre ni d’une solution FHO, ni
d’une base de données produite par des FHO. Les seules informations admises
sont :

- le modèle biomécanique et le modèle de Ding identifiés pour le patient ;
- le couple résistant, la cadence et la géométrie de la tâche ;
- les RHO déjà exécutés pendant la séance ;
- de courtes données de calibration ou des rollouts directs du modèle.

Le coût historique est une somme quadratique de fatigue normalisée :

\[
J_{\mathrm{fatigue}} =
\sum_n\sum_i\left(1-\frac{A_i(n)}{A_{i,\mathrm{rest}}}\right)^2.
\]

Ce coût mesure une fatigue moyenne locale. Il ne mesure ni le rôle mécanique
d’un muscle, ni la proximité de la saturation des PW, ni le nombre de cycles
encore réalisables. Il ignore aussi les états lents `Tau1` et `Km`, qui
modifient la production de force.

L’endurance sera donc définie opérationnellement comme le nombre de cycles
certifiés avant que le couple demandé ne puisse plus être produit avec des PW
admissibles. Un statut d’échec du solveur ne constitue pas, à lui seul, une
preuve d’épuisement physiologique.

## Principes de conception

1. Toutes les métriques sont sans dimension ou normalisées par des paramètres
   physiologiques identifiés.
2. Aucun poids spécifique à un muscle n’est ajusté a priori.
3. Les coûts lisses servent à l’optimisation ; les rapports scientifiques
   conservent aussi le maximum brut, le muscle critique et la phase critique.
4. Le graphe de chaque formulation RHO garde une taille fixe. Le temps et le
   nombre de compilations initiales sont rapportés séparément des réutilisations.
   Le niveau 1 utilise un seul graphe ; le niveau 4 peut employer un graphe de
   démarrage puis un graphe anticipatif, chacun compilé une seule fois.
5. Chaque niveau doit franchir ses tests numériques et scientifiques avant que
   le niveau suivant soit activé dans le contrôleur.

## Organisation des revues et état courant

L’implémentation est séparée de ses validations afin d’éviter qu’un même agent
définisse seul la méthode et son critère de succès :

- propagation Ding et audit numérique : agent `Terra`, raisonnement `high` ;
- inversion recrutement/PW et adaptateur RHO : agent `Sol`, raisonnement
  `xhigh` ;
- production MA57, reprise et provenance : agent `Sol`, raisonnement `high` ;
- revue scientifique croisée des primitives : agent `Terra`, raisonnement
  `high`.

Au 4 septembre 2026, deux audits indépendants donnent **GO pour la primitive
expérimentale du niveau 0**. Le câblage structurel du niveau 1 est validé : vrai
objectif Mayer terminal, absence effective lorsque le poids vaut zéro,
propagation CLI/JSON et signatures de cache. Le smoke test réel passe maintenant
sur trois RHO avec IPOPT/MA57, Radau-5 et une résistance signée de `+0.2 N.m`.
Le niveau 1 reçoit donc un **GO d’infrastructure**, mais demeure **NO-GO pour
activation scientifique** : dans l’ablation appariée disponible,
\(\lambda_R=0.1\) dégrade légèrement les indicateurs de fatigue à trois cycles.
Aucun niveau n’est encore validé comme prédicteur clinique de l’endurance.

## Progression des méthodes

| Niveau | Méthode | Information future | Complexité | État |
|---|---|---:|---:|---|
| 0 | Métrique de réserve terminale `A/A_scale` | Aucune | Très faible | Implémentée comme primitive |
| 1 | Coût terminal sur un minimum lisse des réserves | Implicite | Très faible | Infrastructure validée ; gain scientifique non démontré |
| 2 | Dommage marginal issu des équations de Ding | Cycle courant | Faible | Primitive et rapport hors ligne implémentés |
| 3 | Marge instantanée de recrutement/PW | Cycle courant | Faible à moyenne | Inversion et gate du cycle source implémentés ; gate réel non franchi |
| 4 | Rollout affine `A, Tau1, Km` sous profil de force gelé | 5–20 cycles | Moyenne | Noyau numérique validé ; interprétation réelle suspendue au niveau 3 |
| 5 | Rollout Ding complet sous politique fixe | 5–20 cycles | Moyenne à forte | Après validation du niveau 4 |
| 6 | Allocation future adaptative ou fonction de valeur | Long horizon | Forte | Recherche avancée |

### Niveau 0 — Primitive de réserve terminale

Pour les ratios \(r_i=A_i/A_{i,\mathrm{rest}}\), la réserve agrégée est le
minimum lisse normalisé :

\[
R_\tau(r) = -\tau\log\left(
\frac{1}{n_m}\sum_i\exp(-r_i/\tau)
\right).
\]

La pénalité \(1-R_\tau\) est symétrique, différentiable et converge vers la
capacité du muscle le plus dégradé lorsque \(\tau\to0\). À température finie,
elle est toutefois optimiste :

\[
0 \le R_\tau-\min_i r_i \le \tau\log(n_m).
\]

Ainsi, \(\tau\) doit être choisi depuis un biais maximal accepté
\(\varepsilon_R\), avec \(\tau\le\varepsilon_R/\log(n_m)\). La valeur
expérimentale par défaut est `0.005` : pour quatre muscles, le biais garanti est
inférieur à 0,7 point de capacité normalisée. Une analyse de sensibilité à
`temperature` demeure obligatoire.

Ce niveau supprime les poids musculaires arbitraires, mais reste seulement un
proxy. Il traite les muscles symétriquement sans connaître leur importance
mécanique et ne prédit donc pas l’endurance. Toute sortie scientifique doit
conserver simultanément le minimum brut, le minimum lisse, leur écart et sa
borne théorique.

Implémentation actuelle :

- `cocofest/optimization/muscle_reserve.py` : versions NumPy et CasADi ;
- `tests/test_muscle_reserve.py` : propriétés numériques et gradient.

### Niveau 1 — Coût de Mayer sur la réserve

Ajouter au dernier nœud du RHO :

\[
J_T = 10^4\lambda_R(1-R_\tau).
\]

Quand le poids vaut zéro, l’objectif ne doit pas être ajouté au graphe, afin de
reproduire strictement le problème historique jusque dans sa structure. Les options
prévues sont `--terminal-reserve-weight` et
`--terminal-reserve-temperature`. Ce coût dépend uniquement des états
symboliques terminaux et de constantes patient ; il est donc compatible avec
SX, MX et la compilation C unique.

L’option `terminal-reserve-weight` représente le multiplicateur sans dimension
\(\lambda_R\), pas le poids Bioptim brut. Le facteur `10000` reprend l’échelle
de base du coût de fatigue existant. Le JSON doit conserver à la fois
\(\lambda_R\) et le poids effectif `10000 * lambda_R`. Le premier balayage est
prédéfini à `0`, `0.1` et `1.0`; un poids négatif ou non fini est invalide.

Critère go/no-go : le coût est monotone, ses gradients sont finis, les résidus
mécaniques ne se dégradent pas et le suivi sur plusieurs fenêtres conserve :

```text
compiled_library_build_count == 1
compiled_library_reused == true
graph_rebuild_detected == false
```

#### Validation scientifique minimale du niveau 1

Le test automatisé `tests/test_terminal_reserve_rho_science.py` emploie un RHO
synthétique à deux muscles résolu globalement sur une grille. Il vérifie la
symétrie, la permutation, la conservation de la tâche, l’effet des paramètres
de dommage et la décroissance de la pénalité optimale lorsque \(\lambda_R\)
augmente.

Un contrôle négatif est conservé comme résultat scientifique, et non masqué :
avec un muscle faible mais dix fois moins efficace mécaniquement, le cas
synthétique complète 35 cycles au baseline contre 33 avec
\(\lambda_R=10\). Cela démontre que protéger le minimum de capacité ne suffit
pas à prédire l’endurance et justifie les niveaux recrutement/rollout.

Le gate suivant est un smoke test réel de trois RHO IPOPT, mécanique réduite et
profil `scientific-radau5`, avec seed, résistance et tolérances identiques. Il
doit certifier trois cycles, une violation primale au plus `1e-5`, une seule
compilation réutilisée et aucun rebuild. Ce test valide le câblage, pas un gain
d’endurance.

Résultat courant : une ablation appariée a utilisé la même primale initiale
(SHA-256 `407c39765b85c726aab7d6a7aafb260c799f5dd43c2eba79c33957ca19316da8`),
IPOPT/MA57, aucun warm-start dual, Radau-5, la même résistance signée de
`+0.2 N.m` et un budget de 5000 itérations. Les deux bras certifient 3/3 RHO.
Chaque bras observe `compiled_library_build_count=1`,
`compiled_library_reused=true`, trois vecteurs de bornes distincts et
`graph_rebuild_detected=false`.

Le baseline \(\lambda_R=0\) termine avec `minimum_ratio=0.991799`, une fatigue
intégrée normalisée de `0.029125` cycle et 876/71/97 itérations. Le bras
\(\lambda_R=0.1\) termine avec `minimum_ratio=0.990928`, une fatigue intégrée
de `0.030186` cycle et 891/70/72 itérations. L’écart de lissage reste dans sa
borne analytique et le domaine physiologique est valide dans les deux cas.
Ainsi, ce smoke test valide le câblage et la compilation unique, mais rejette
\(\lambda_R=0.1\) comme amélioration sur ce cas. Trois cycles ne constituent de
toute façon pas un test d’endurance.

Le choix `--ipopt-dual-warm-start-mode off` est intentionnel. Le mode `bounds`
réinjectait uniquement les multiplicateurs de bornes alors que les bornes
initiales et terminales se déplacent entre RHO ; le second RHO pouvait alors
rester primalement faisable tout en atteignant la limite d’itérations sur les
conditions KKT. Le warm-start primal reste actif.

Sur la machine de validation, `libhsl.so` dépend encore de `libgfortran.so.4`.
Le runtime est chargé depuis le paquet Conda local `libgfortran4-7.5.0`; il doit
être présent dans `LD_LIBRARY_PATH` avec le répertoire `lib` de l’environnement
RHO. Ce montage est suffisant pour les essais reproductibles, mais une CoinHSL
recompilée avec le compilateur Fortran courant est requise avant déploiement
clinique afin de supprimer cette dépendance ancienne et l’avertissement ABI
METIS de la bibliothèque HSL disponible.

L’ablation prospective utilisera deux états initiaux, deux charges fixées avant
calcul, \(\lambda_R\in\{0,0.1,1\}\) et une sensibilité
\(\tau\in\{0.0025,0.005,0.01\}\). Une campagne qui atteint partout le nombre
maximal demandé est censurée et reste inconclusive sur l’endurance.

Le premier écran numérique est automatisé par
`.github/scripts/run_terminal_reserve_sweep.py`. Il teste par défaut
\(\lambda_R\in\{0,0.01,0.03,0.1,1\}\), ne calcule le baseline qu’une fois et
exécute les cas séquentiellement avec MA57 pour préserver les mesures de temps
et de mémoire. Un cas n’est classé `pareto_nondominated` que s’il ne dégrade ni
la capacité minimale, ni l’AUC de fatigue, ni la saturation PW, et améliore au
moins une de ces métriques au-delà de tolérances propres à chaque indicateur.
Ce classement est un filtre de Pareto numérique, jamais une preuve de gain
d’endurance.

Le runner conserve un manifeste de campagne immuable contenant les SHA-256 du
seed, de HSL, de Python et des sources critiques. Une reprise n’est acceptée
que si ce contrat complet est identique. Avant une vraie relance, les anciens
artefacts du cas sont déplacés dans `previous-attempts`; un code retour nul, un
NPZ structuré, tous les cycles physiques, les métriques finies, les diagnostics
KKT et la compilation unique sont ensuite requis. Un préflight `dlopen` et
symbole MA57 arrête la campagne avant le premier NLP si HSL n’est pas
chargeable.

Chaque paire doit recevoir exactement le même fichier
`--common-initial-solution`; sa signature numérique et sa provenance sont
auditées. La validation de ce seed autorise explicitement un changement de
\(\lambda_R\) ou de \(\tau\), car ceux-ci modifient l’objectif mais pas le
domaine physique. Les caches de *standard warmup* restent au contraire séparés
par objectif pour éviter une réutilisation opaque ; ils ne doivent pas être le
seed final de l’ablation appariée.

Lorsqu’un `common-initial-solution` certifié est fourni, le code saute désormais
le warmup standard redondant et applique directement cette primale. Cette règle
évite qu’un cache de warmup différent influence l’ablation. La stabilisation
MA57 ci-dessus remplace le premier essai exploratoire avec MUMPS qui avait
atteint `maxiter=2000`. Elle n’autorise pas encore de conclusion causale sur
l’endurance, mais fournit désormais un protocole numérique apparié pour le
balayage prospectif.

### Niveau 2 — Dommage musculaire marginal

Les équations lentes du modèle de Ding permettent de normaliser la contribution
de la force à la perte de capacité :

\[
d_i(t) = -\frac{\alpha_{A,i}F_i(t)}{A_{i,\mathrm{rest}}}.
\]

Le coût protège le pire dommage instantané avec un `smoothmax`, sans poids par
muscle. Cette formulation tient compte des paramètres physiologiques, mais pas
encore du caractère indispensable d’un muscle pour produire le couple.

La primitive `marginal_capacity_damage_rate` est implémentée dans
`cocofest/optimization/recruitment_margin.py`. Elle retourne uniquement le
terme de dommage forcé, en `s^-1`, et ne le confond pas avec la dérivée totale
qui contient également la récupération vers le repos.

### Niveau 3 — Réserve instantanée de recrutement

Pour chaque muscle et chaque phase, calculer le recrutement demandé relativement
au recrutement maximal disponible sous `PW_max`. La métrique

\[
u_i(\phi)=\frac{a_{i,\mathrm{requis}}(\phi)}
{a_{i,\max}(A_i,\mathrm{PW}_{i,\max})}
\]

a une interprétation directe : `u < 1` est réalisable, `u = 1` atteint la
saturation et `u > 1` est irréalisable. Le coût protège le pire couple
muscle–phase. Toute impossibilité doit être rapportée explicitement, jamais
masquée par un clipping à 1.

L’inversion implémentée reconstruit exactement le recrutement puis le PW depuis
`F`, `F_dot`, `Cn`, `A`, `Tau1`, `Km` et les gains mécaniques. `F_dot` doit être
la dérivée analytique du polynôme de collocation ou d’un interpolant périodique
audité ; les seuls échantillons nodaux de `F` ne suffisent pas. Aucun
`np.gradient` implicite n’est autorisé. Les singularités et dépassements de PW
sont conservés sous forme de statuts explicites.

### Niveau 4 — Rollout affine sous profil de force gelé

Le dernier cycle RHO certifié fournit un profil de force périodique réalisable
\(F_i^{\mathrm{ref}}(\phi)\). Ce n’est pas l’unique force nécessaire, mais une
politique musculaire observable que l’on peut tester : combien de temps
resterait-elle réalisable si elle était répétée ?

Quand le profil de force est fixé, chaque état lent
\(z\in\{A,\mathrm{Tau1},\mathrm{Km}\}\) suit une dynamique affine :

\[
\dot z = -\frac{z-z_{\mathrm{rest}}}{\tau_{\mathrm{fat}}}+\alpha_z F(t).
\]

Pour une force constante sur \(\Delta t\) :

\[
z_{k+1}=z_{\mathrm{rest}}
+e^{-\Delta t/\tau_{\mathrm{fat}}}(z_k-z_{\mathrm{rest}})
+\alpha_zF_k\tau_{\mathrm{fat}}
\left(1-e^{-\Delta t/\tau_{\mathrm{fat}}}\right).
\]

Pour le profil Fourier continu retenu, l’implémentation ne remplace plus la
force par sa seule valeur au milieu de l’intervalle. Elle calcule analytiquement
la convolution exponentielle de chaque harmonique avec la récupération de
Ding. La propagation de `A`, `Tau1` et `Km` est donc exacte pour cet interpolant
continu. La reconstruction depuis les nœuds du RHO reste une approximation qui
doit être auditée. La carte peut ensuite être répétée sur 5 à 20 cycles sans
nouvelles variables de décision. À chaque cycle projeté, on reconstruit le
recrutement et le PW requis pour reproduire le profil de force, puis on calcule
la pire utilisation \(u_i^{(h)}(\phi)\).

Le premier coût anticipatif sera :

\[
J_{\mathrm{rollout}}=
\operatorname{smoothmax}_{h,i,\phi}u_i^{(h)}(\phi).
\]

Le profil issu du cycle précédent devra être transmis comme un vecteur
numérique de dimension fixe. Il ne doit jamais être capturé comme une constante
Python reconstruisant le graphe entre deux RHO. Une première version acceptable
peut résoudre le cycle initial, construire puis compiler une seule fois le RHO
anticipatif pour tous les cycles suivants : deux compilations par séance, puis
aucune recompilation.

La carte exacte NumPy/CasADi est disponible dans
`cocofest/optimization/ding_fatigue_rollout.py`. Une représentation de Fourier
dans `cocofest/optimization/periodic_force_profile.py` rend explicites la
périodicité et la dérivée du profil de force ; ses extrema continus et son audit
rapportent toute oscillation négative sans la masquer par clipping. Pour les
exports réels, `cocofest/optimization/rho_rollout_adapter.py` ajuste une série
sur `sqrt(F)`, puis la met exactement au carré par convolution des coefficients.
Le profil de force obtenu est non négatif par construction et sa dérivée reste
analytique.

Le CLI `scripts/analyze_rho_endurance_rollout.py` sélectionne explicitement le
dernier cycle certifié, reconstruit les temps Radau déclarés, aligne `F`, `Cn`,
`theta`, `omega` et les gains musculaires aux mêmes phases, puis produit les
horizons 5, 10 et 20. Il vérifie d’abord que l’interpolant peut reproduire la
dynamique du cycle RHO source avec ses propres états `A`, `Tau1` et `Km`. Un
échec de ce gate donne `reference_policy_not_reproducible` et un code retour
non nul ; les projections sont conservées comme diagnostic, mais ne doivent
pas être interprétées comme une endurance prédite.

Sur l’export certifié IPOPT/MA57 de 150 RHO à `+0.10 N.m`, le fit positif passe
les audits d’entrée (`RMSE` relative maximale `0.0350`, erreur relative maximale
`0.1901`). Les projections donnent respectivement, pour H=5/10/20, une pire
utilisation de `1.20879/1.21562/1.22894` et un minimum final `A/A_rest` de
`0.809515/0.805371/0.797420`. Elles restent toutefois **non interprétables pour
l’endurance** : sur les 120 couples muscle–phase du cycle source, 72 inversions
sont valides, 46 demandent un recrutement négatif et 2 dépassent la capacité.
Le prochain travail scientifique porte donc sur une reconstruction de `F_dot`
cohérente avec la collocation/dynamique, pas sur l’augmentation de l’horizon.

### Niveaux 5 et 6 — Méthodes avancées

Après validation du rollout affine :

- propager les cinq états de Ding avec une intégration fixe et différentiable ;
- alterner les deux ou trois derniers profils RHO plutôt que d’en geler un seul ;
- employer une allocation analytique lissée minimisant la pire utilisation ;
- ajouter quelques variables de dose par blocs de cycles (`move blocking`) ;
- construire une fonction de valeur monotone à partir de rollouts du modèle,
  jamais à partir de FHO ;
- propager un ensemble de paramètres patient et optimiser le pire cas ou une
  CVaR pour tenir compte de l’incertitude clinique.

Une QP imbriquée dans le NLP n’est pas la première option recommandée. Une
politique analytique lissée ou une table paramétrique produite par le modèle est
plus simple à différencier et à compiler.

## Plan d’implémentation et gates

### Étape A — Primitives indépendantes du solveur

- [x] ratios de capacité normalisés ;
- [x] minimum lisse stable NumPy/CasADi et borne de son biais optimiste ;
- [x] pénalité terminale sans poids musculaires ;
- [x] minimum brut et écart de lissage disponibles pour le rapport ;
- [x] tests de valeur, domaine, compilation C et gradient ;
- [x] runner d’ablation appariée MA57, reprise et filtre non dominé ;
- [x] carte affine exacte pour `A`, `Tau1`, `Km` ;
- [x] interpolation périodique de `F` avec dérivée analytique et audit ;
- [x] inversion recrutement–PW avec statuts de domaine explicites ;
- [x] diagnostic composite répété sur 5–20 cycles depuis un export RHO ;
- [x] gate séparé de reproductibilité dynamique du cycle RHO source ;
- [ ] reconstruction `F/F_dot` franchissant ce gate sur un export réel.

Gate du sous-niveau réserve : borne analytique, domaines NumPy/CasADi,
compilation C et gradients CasADi vérifiés. Gate du rollout : égalité avec des
solutions analytiques et une intégration DOP853 sur 1 et 100 cycles, extrema
continus non négatifs, absence de `NaN` et gradients CasADi vérifiés. Ce gate
numérique est franchi ; le gate de reproduction du cycle RHO réel ne l’est pas.

### Étape B — Diagnostic hors ligne sur RHO uniquement

Pour chaque frontière RHO certifiée, sauvegarder le pire `u`, le muscle et la
phase critiques, le premier horizon saturé et les trajectoires projetées des
états lents. Le diagnostic doit comparer la carte rapide à une intégration
directe du modèle de Ding, pas à un FHO.

Gate : erreur relative d’état inférieure à `1e-8` sur un cycle et `1e-6` sur
100 cycles ; tendances physiques monotones ; résultats déterministes.

### Étape C — Intégration Bioptim désactivée par défaut

Le chemin d’intégration est :

```text
cycling_fes_solver_comparison.py
  -> cycling_pulse_width_mhe_acados_periodic.solve_case
  -> simulation_conditions
  -> cycling_pulse_width_mhe.prepare_nmpc
  -> set_objective_functions
  -> CustomObjective
```

Le coût terminal sera un `ObjectiveFcn.Mayer`, `Node.END`, scalaire et non
quadratique. Les options et la méthode devront apparaître dans le JSON final et
dans la signature de codegen.

Gate : objectif entièrement absent si son poids est nul et problème strictement
identique au baseline ; une seule compilation sur
un petit RHO multi-fenêtres ; mêmes paramètres pour tous les solveurs comparés.

### Étape D — Validation scientifique prospective sans FHO

Comparer :

1. fatigue intégrale historique ;
2. fatigue intégrale avec réserve terminale ;
3. rollout de recrutement/PW.

La matrice est définie avant les calculs : plusieurs résistances, cadences,
états initiaux et perturbations plausibles des paramètres Ding. Toutes les
variantes emploient les mêmes contraintes, le même solveur, la même
initialisation et les mêmes critères de certification.

Métrique primaire : nombre de cycles certifiés avant perte de faisabilité
physiologique confirmée. Métriques secondaires : saturation PW, marge de couple,
capacité minimale, travail mécanique, résidus, temps, mémoire et nombre de
compilations.

Gate scientifique du rollout :

- médiane du gain en cycles strictement positive ;
- dixième percentile du gain non négatif ;
- aucune perte supérieure à un cycle sur les cas de référence ;
- aucune dégradation des contraintes mécaniques ;
- arrêt confirmé par un OCP de faisabilité avec plusieurs initialisations, pas
  par le seul statut IPOPT.

## Tests numériques requis

1. Récupération analytique vers le repos lorsque `F = 0`.
2. Force constante contre DOP853 sur 1 puis 100 cycles.
3. Reconstruction du PW du cycle de référence à `1e-9 s` près.
4. Égalité du membre droit de la dynamique de force après inversion.
5. Monotonie par rapport à `A`, `Km`, `PW_max`, force et horizon ; `Tau1` est
   testé séparément car son effet n’a pas nécessairement le même signe.
6. Invariance par permutation des muscles et changement cohérent d’unités.
7. Jacobien CasADi contre différences finies centrales, erreur relative
   inférieure à `1e-5`.
8. Cas `Cn` quasi nul, capacité non positive, dénominateur nul et recrutement
   impossible classés explicitement.
9. Aucun changement de dimension du NLP et aucune reconstruction du graphe.
10. Pour le minimum lisse, rapport de la borne `temperature * log(n_m)`, du
    minimum brut et de l’écart observé ; sensibilité à la température.

## Risques et interprétation

- Répéter le dernier profil RHO peut conserver une partie de sa myopie et être
  conservateur face à une stratégie d’alternance musculaire.
- Une saturation PW isolée n’est pas une preuve suffisante d’épuisement.
- La précision dépend de l’identification patient de `A_scale`, `Tau1`, `Km`,
  `alpha` et `tau_fat`.
- Un coût de réserve `A/A_scale` ne doit pas être présenté comme un prédicteur
  d’endurance avant validation des niveaux 3 et 4.
- Les seuils scientifiques doivent être fixés avant l’exécution des campagnes,
  afin d’éviter une sélection a posteriori des cas favorables.
