# Trois pistes d’accélération du RHO réduit : parallélisme, masque PW et interpolation temporelle

Rapport d’analyse du 12 septembre 2026. Périmètre : dynamique réduite **isorésistance +0,1 N·m, 50 contrôles par cycle de 1 s, |ΔPW| ≤100 µs, IPOPT/MA57, collocation Radau de degré 5**. Aucun code applicatif modifié, aucune optimisation ou simulation longue lancée, aucun commit. Les calculs nouveaux se limitent à l’inspection de profils/trajectoires existants et à une minuscule inspection symbolique CasADi sans résolution.

La piste la moins intrusive est le parallélisme, mais augmenter seulement `n_threads` peut rester sans effet avec le graphe **SX** de la référence. Le masque géométrique strict et l’interpolation des PW changent l’ensemble admissible : ils doivent être évalués comme des approximations de commande. Leurs gains de dimension sur le NLP complet sont proches de **1 %**, même lorsqu’ils retirent presque la moitié des degrés de liberté PW. Le masque strict retire déjà des commandes recrutées dans la solution de référence ; il n’est donc pas justifié comme une élimination exacte de variables inutiles.

## 1. Référence inspectée et taille réelle du problème

Le [rapport IPOPT de 100 RHO](../../ipopt-slew-100-reference-20260912-110146/REPORT.md), son [JSON](../../ipopt-slew-100-reference-20260912-110146/result.json) et sa [commande](../../ipopt-slew-100-reference-20260912-110146/run.sh) fournissent une référence proche du périmètre demandé. Elle ajoute **λ=0,1** de régularisation quadratique des incréments PW, normalisée par 100 µs. La seule mention « DeltaPW100 » ne fixe pas λ : les expériences proposées gardent λ=0,1 dans les deux bras pour reprendre cette référence ; si l’objectif voulu est λ=0, il faut une seconde baseline appariée λ=0. Ne pas attribuer un changement de λ aux trois pistes étudiées.

La référence utilise SX, `ipopt_c_compile=false`, état mis à l’échelle `full`, PW mis à l’échelle par 0,0025 s, masque d’activité `none`, les trois relations Hill actives, calcium `exact_exponential_periodic_node`, garde interne de vitesse active, cible terminale absolue ±0,002 rad et 2000 itérations maximales. Elle valide 100/100 RHO, sans récupération. Le seed commun est [common-cycle1.npz](../../ipopt-slew-100-reference-20260912-110146/common-cycle1.npz), SHA-256 `63d4cfd3e54bbf063bc381aa5e1c7d1b9ae0eaa86feecbff43f28c07e67a46b1`.

| Quantité | Taille vérifiée sur le seed |
|---|---:|
| États Ding : 5 × 4 muscles | 20 |
| États mécaniques θ, ω | 2 |
| États auxiliaires de ΔPW, un carrier par muscle | 4 |
| Total d’états | 26 |
| Colonnes d’états Radau5 : 50×6+1 | 301 |
| États discrétisés | 7 826 |
| PW physiques : 4×50 | 200 |
| Incréments auxiliaires ΔPW : 4×50 | 200 |
| Variables états + contrôles | 8 226 |

Un scalaire de durée fixé peut s’ajouter dans le vecteur Bioptim ; les bornes fixes des états initiaux sont également éliminables dans IPOPT. Ce tableau décrit le vecteur explicite, pas un comptage post-élimination du KKT. Il n’inclut pas de paramètres supplémentaires non présents dans cette référence. Sans le lift ΔPW, le même comptage aurait donné 6 822 : les auxiliaires représentent 1 404 scalaires supplémentaires. Les 200 PW physiques ne sont donc que 2,43 % des 8 226 variables.

Le lift actuel garde les PW constants par intervalle. Pour chaque muscle, `z_dot=delta_pw/h`, `z[k]=u[k]` aux nœuds de shooting ; donc `delta_pw[k]=u[k+1]-u[k]` pour k=0…48. Le dernier incrément conduit à un carrier terminal libre et ne représente pas une fermeture périodique. La liaison au dernier PW réellement exécuté est imposée séparément au premier carrier de la fenêtre suivante. Voir [pulse_width_slew.py](../../cocofest/optimization/pulse_width_slew.py:1).

La référence n’exporte pas de NPZ RHO concaténé à cause d’une garde de continuité appliquée aux carriers, alors que les 22 états physiques sont continus. Cette limite d’artefact est documentée dans son rapport. Pour de futurs tests isolés tardifs, sauvegarder les checkpoints préparés par fenêtre ; ne pas traiter les petits extraits JSON comme un primal complet de collocation.

## 2. Piste 1 : paralléliser les évaluations et dérivées

### Ce que fait effectivement le code

`--n-threads` est transmis jusqu’à `MyCyclicNMPC` par [prepare_nmpc](../../examples/fes_multibody/cycling/cycling_pulse_width_mhe.py:1134). Les dynamiques sont partagées pendant la phase, avec `expand_dynamics=True` et `expand_continuity=False` dans [fes_ocp_multibody.py](../../cocofest/optimization/fes_ocp_multibody.py:41). Les contraintes de continuité sont marquées `multi_thread=True` dans le Bioptim installé. La construction des pénalités appelle alors `Function.map(nombre_de_nœuds, "thread", n_threads)` : [.benchmark-deps/bioptim/bioptim/limits/penalty_option.py](../../.benchmark-deps/bioptim/bioptim/limits/penalty_option.py:645).

Cela parallélise des évaluations de défauts/derivées pour des intervalles dont les états sont déjà fournis par l’itéré NLP. Cela ne parallélise ni la séquence des itérations IPOPT, ni les RHO successifs dépendant de l’état exécuté précédent, ni un rollout causal entier. `n_threads` n’est pas un réglage du solveur linéaire MA57.

**Obstacle SX vérifié localement.** Dans le runtime de la référence, la construction suivante a été inspectée sans solve :

```python
x = casadi.SX.sym("x")
f = casadi.Function("stage", [x], [casadi.sin(x) + x*x])
fm = f.map(50, "thread", 4)
# Appliquer fm à des symboles SX ou MX puis créer Function("whole", ...).
```

Avec entrée SX, la fonction finale est `SXFunction`, et `find_functions()` est vide. Avec entrée MX, elle conserve un `ThreadMap` imbriqué (`threadmap4_map13_stage`, 52 slots pour un travail demandé de 50). L’appel à un mapping peut donc être développé dans le graphe scalaire SX. Cela explique pourquoi le seul réglage `n_threads=4/8` n’est pas une preuve de parallélisme utile. Cette petite inspection démontre le mécanisme CasADi ; elle ne remplace pas l’inspection des fonctions `nlp_g`, `nlp_jac_g` et `nlp_hess_l` réellement produites par l’OCP. La documentation [CasADi sur les mappings](https://web.casadi.org/docs/) confirme l’API `map(..., "thread", ...)`.

L’architecture candidate est un **graphe global MX qui conserve les appels de blocs**, avec les petites dynamiques SX expansées localement. Il faut préserver ces appels après différentiation. Les objectifs simples déjà sériels (`multi_thread=False` pour la régularisation slew, notamment) sont minuscules et ne constituent pas la première cible. Les contraintes de continuité et leurs Hessiennes sont prioritaires. Un changement SX→MX doit être mesuré séparément d’un changement 1→4 threads, car le coût du graphe lui-même change.

La compilation C n’est pas synonyme de parallélisation. L’interface actuelle génère `nlp.c` puis utilise l’importateur `shell` sans contrat explicite de backend parallèle, [interface_utils.py](../../.benchmark-deps/bioptim/bioptim/interfaces/interface_utils.py:168). Un `ThreadMap` interprété n’est pas une garantie d’équivalence du code généré. Le premier A/B doit garder `ipopt_c_compile=false`, puis une expérience indépendante peut inspecter le C produit, les options de compilation et les fonctions réellement chargées.

### Limite de gain mesurable dans les données existantes

Sommes recalculées dans les 99 entrées chaudes `nlp_solver_stats` de la référence :

| Temps mural chaud cumulé | Secondes |
|---|---:|
| Total IPOPT | 179,9245 |
| Évaluation des contraintes `nlp_g` | 9,0807 |
| Jacobienne `nlp_jac_g` | 17,6900 |
| Hessienne `nlp_hess_l` | 40,3823 |
| Objectif + gradient | 0,3182 |
| Total des évaluations ci-dessus | 67,4712 |
| Autres travaux IPOPT, par différence | 112,4533 |

Les évaluations occupent **37,50 %** du total. La différence de 112,45 s n’est pas une mesure exclusive de MA57 : elle inclut aussi la gestion IPOPT, les recherches linéaires et autres opérations. `--ipopt-print-timing-statistics` est déjà exposé pour décomposer ce reste lors d’un pilote diagnostique.

Avec fraction parallélisable f=0,375, l’idéal d’Amdahl vaut `S(p)=1/(1-f+f/p)` : environ ×1,39 à 4 threads, ×1,49 à 8 threads et **×1,60 au maximum**, avant surcoût du mapping et avant changement du graphe. C’est une borne conditionnelle, pas une prédiction d’accélération. La médiane chaude actuelle est 1,733 s et le P90 2,194 s. Sur la boucle complète de 309,835 s, supprimer même les 67,47 s mesurées d’évaluations chaudes ne ferait gagner que 21,8 % ; les callbacks et préparations restent à mesurer séparément.

### MA57, BLAS et absence de surallocation

Le `ldd` de la bibliothèque HSL configurée montre **son propre `libopenblas.so`**, ainsi que `libgomp.so.1`. Fixer seulement les threads NumPy ne prouve donc pas que ceux de MA57 sont fixés. MA57 exploite des BLAS, notamment de niveau 3 ; le parallélisme de ces noyaux est distinct du parallélisme des blocs de dynamique. Sources primaires : [spécification MA57](https://www.hsl.rl.ac.uk/specs/ma57.pdf) et [rapport HSL sur MA97 et les BLAS multithreads de MA57](https://epubs.stfc.ac.uk/manifestation/7236/RAL-TR-2011-024.pdf).

La machine expose actuellement 32 CPU logiques pour **16 cœurs physiques**. Le wrapper détecte les cœurs autorisés par l’affinité dans [default_worker_threads](../../.github/scripts/run_benchmarks.py:183), et sépare déjà `BENCHMARK_THREADS` de `NUMERIC_THREADS` dans [run_cycling_benchmark_case.sh](../../.github/scripts/run_cycling_benchmark_case.sh:97). Un processus peut cependant disposer d’un cpuset plus réduit.

Premier balayage : mapping 1, 2, 4, éventuellement 8, avec **tous les threads BLAS/OMP à 1**. Faire tourner les variantes séquentiellement sur les mêmes cœurs physiques disponibles et éviter leurs frères SMT pour l’estimation principale. Variables à consigner avant import : `OMP_NUM_THREADS`, `OMP_THREAD_LIMIT`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `OMP_DYNAMIC`; enregistrer aussi l’affinité et les bibliothèques chargées. `OMP_THREAD_LIMIT=1` ne désactive pas le backend CasADi `thread`, mais peut limiter un backend BLAS OpenMP : pour tester celui-ci à 2/4 threads, ajuster la limite également.

Deuxième balayage, seulement si le temps linéaire est dominant : **mapping=1**, BLAS HSL=1/2/4, autres pools contrôlés. Vérifier leur configuration effective après chargement HSL, pas seulement les variables d’environnement. Ne combiner les niveaux qu’après les avoir mesurés séparément ; imposer de manière conservatrice `processus_concurrents × threads_mapping × threads_BLAS ≤ cœurs_physiques_alloués`. Les deux phases ne sont pas nécessairement imbriquées dans tout le solve, mais cette règle évite de compter sur cette hypothèse sans instrumentation.

Les diagnostics du 3 septembre ne constituent pas un A/B pertinent ici : les JSON inspectés portent sur MUMPS, 30 stimulations, sans DeltaPW ; certaines valeurs d’environnement de threads ne sont pas renseignées. Leurs écarts ne prouvent pas un gain MA57/50 Hz actuel.

### A/B minimal et code nécessaire ultérieurement

1. Sauvegarder un même RHO préparé, avec état initial, bornes, cibles, primal et dual initiaux ; aucun avancement pendant le microbenchmark. Exclure construction et première évaluation de la mesure chaude, tout en rapportant leur coût séparément.
2. Contrôle de câblage SX 1→4 threads sur ce RHO, une paire, avec inspection des graphes et CPU réellement utilisés. Si les appels sont développés en SX, arrêter cette branche.
3. Si nécessaire, comparer SX/1 à MX/1 pour isoler le changement de représentation ; puis faire l’A/B de parallélisme **MX/1 contre MX/4**, trois répétitions par bras en ordre alterné, mêmes bornes et options IPOPT. MX/2 et MX/8 ne sont explorés que si MX/4 présente un gain crédible.
4. Confirmer le gagnant sur huit RHO successifs, même seed commun, mêmes diagnostics dans les deux bras. Vérifier objectif, admissibilité et métriques physiques à précision identique, puis P50/P90 des appels et de la boucle.

Aucun changement de code n’est nécessaire pour le premier balayage CLI. L’instrumentation utile consiste à exporter les classes/fonctions imbriquées des évaluateurs, les temps par appel et la configuration effective des pools. Si SX absorbe les mappings, la modification structurante se situe au niveau d’assemblage Bioptim/CasADi du graphe global ; les équations Ding et la grille ne changent pas. Ne pas activer globalement `expand_continuity` dans cette variante. Arrêt de sélection : baisse médiane <10 % ou hausse P90 >5 % après trois paires, variation d’itérations indiquant un autre optimum, échec physique ou nouveau défaut numérique. En cas de dispersion >5 %, demander une nouvelle série sur machine plus calme plutôt que conclure.

## 3. Piste 2 : PW=pd0 quand le muscle ne propulse pas

### Géométrie et conventions vérifiées

Le profil effectif est [reduced_cycling_fourier12.npz](../../examples/fes_multibody/cycling/result/cache/reduced_cycling_fourier12.npz). Il a été comparé champ par champ au profil archivé [seed-0p10/reduced-cycling-fourier12.npz](../../resistance-fho-pilots-20260910/seed-0p10/reduced-cycling-fourier12.npz) : identité exacte. Son SHA modèle `b05c2c23d7f2c8420a098e9f8c207692afd3ee6a58100685b7a163e0b87ac70c` correspond au modèle Wu courant. Les zéros ci-dessous viennent du profil Fourier d’ordre 12, et ne prétendent pas être des zéros analytiques du modèle multibody original.

Le bras de levier pertinent est la **projection sur la tangente du mouvement complet**, pas un seul bras de levier d’épaule ou de coude :

```math
b_m(\theta)=-\frac{\partial\ell_m}{\partial q}\,q_\theta,
\quad M_{eff}\dot\omega=\sum_m b_m F_m+b_{ext}\tau_{ext}-g_{proj}-c_{proj}\omega^2.
```

Ces définitions sont directement dans [reduced_cycling.py](../../cocofest/dynamics/reduced_cycling.py:758). Le progrès du pédalage est φ=−θ modulo 2π, le profil ayant origine 0 et direction −1. La vitesse est négative. Pour une force positive, la puissance musculaire dans cette direction vaut `b_m F_m ω` : **propulsion si b_m<0**, freinage si b_m>0. Le terme externe vérifié `b_ext∈[0,590876;1,452200]` est positif ; le couple +0,1 N·m résiste donc bien au sens de pédalage courant.

| Muscle | Intervalle(s) propulsif(s) en progrès φ, degrés | Plage de b_m, m | Fraction angulaire propulsive |
|---|---|---:|---:|
| Delt_ant | 131,235→310,547 | −0,011424 à +0,010040 | 49,8 % |
| Delt_post | 0→131,235 et 310,547→360 | −0,009853 à +0,011343 | 50,2 % |
| Biceps | 152,197→298,205 | −0,011735 à +0,009005 | 40,6 % |
| Triceps | 0→96,335 et 241,648→360 | −0,009203 à +0,015343 | 59,6 % |

Ces fractions angulaires ne sont pas des fractions de contrôles : le pas temporel est uniforme mais la vitesse de l’isoresistance ne l’est pas. Le cycle 1 du seed commence à θ=−6,274014 et finit à −12,555219 rad. Utiliser `k×360/50` pour un masque fixe en temps donnerait des fenêtres incorrectes.

### Pourquoi pd0, et pourquoi le signe instantané ne suffit pas

Le code de Ding utilise `a_eff=A*(1-exp(-(PW-pd0)/pdt))`, [ding2007.py](../../cocofest/models/ding2007/ding2007.py:178). Les valeurs par défaut ici sont **pd0=131,405 µs**, pdt=194,138 µs, PWmax=600 µs. À pd0, le recrutement direct s’annule. À zéro absolu, cette loi donne un recrutement négatif : ce n’est pas un arrêt physique acceptable de cette formulation. Un véritable modèle d’impulsion absente requerrait un contrat différent, notamment pour le calcium ; il est hors de cette expérience.

`pd0` n’annule pas la force instantanément. Au plancher, la dynamique conserve la décroissance de F selon `Tau1 + tau2*Cn/(Km+Cn)`, multipliée par les coefficients Hill. Le repos `Tau1≈60,6 ms` correspond déjà à environ trois intervalles de 20 ms, mais n’est pas une durée universelle de mémoire : la fatigue, le calcium et les relations Hill la modifient. La mémoire de force, et davantage celle des états A/Tau1/Km, n’a pas un support temporel exactement fini.

Le calcium de [PeriodicNode](../../cocofest/models/ding2007/ding2007_with_fatigue_periodic_node.py:16) garde une histoire exponentielle de stimulations à fréquence fixe ; le PW n’en modifie pas l’amplitude. Il faut donc conserver les impulsions/calculs de calcium ainsi que les cinq états Ding même aux PW fixés. Dans le code actuel, le terme dit « passif » entre dans le facteur qui multiplie l’ODE de force, [ding2003.py](../../cocofest/models/ding2003/ding2003.py:360) ; il ne faut pas lui substituer une loi de force passive extérieure par interprétation physiologique.

Un PW peut être utile avant l’entrée dans le secteur propulsif parce que la force persiste. Une force de freinage peut aussi satisfaire une borne de vitesse ou la cible d’angle ; l’isorésistance actuelle laisse une vitesse variable. Le bon critère de suppression exacte serait l’absence de contribution utile **sur toute la réponse future et toutes les trajectoires admissibles**, compte tenu des autres contraintes, et non le seul signe b_m au début du pas. Cette propriété n’est pas démontrée ici.

La solution de référence fournit un avertissement concret :

| Muscle, cycle 1 | Pas non propulsifs aux 7 points inspectés du pas | Parmi eux PW>pd0+1 µs | PW maximal dans ces pas |
|---|---:|---:|---:|
| Delt_ant | 18 | 4 | 137,138 µs |
| Delt_post | 30 | 2 | 132,652 µs |
| Biceps | 21 | 7 | 212,669 µs |
| Triceps | 25 | 1 | 148,476 µs |

Les 7 points sont le début de shooting, les points internes exportés et la borne de shooting suivante. Ils ne constituent pas une preuve de signe continu entre les points. Une vérification par racines et plage de θ/dense reste nécessaire. Au cycle 100, le Biceps atteint 281,770 µs à des nœuds de début non propulsifs ; la stimulation anticipée n’a donc pas disparu avec la fatigue. Ces données établissent que le masque changerait la solution ; elles n’établissent pas à elles seules que chaque stimulation est indispensable à l’optimum.

### Réutilisation du masque existant et conception d’un masque prudent

Il existe déjà `periodic_pulse_width_activity_mask` et `set_u_bounds_and_init`, [cycling_pulse_width_mhe.py](../../examples/fes_multibody/cycling/cycling_pulse_width_mhe.py:2149). Les modes CLI `historical`/`warmup` conservent les phases où un cycle de référence dépasse `pd0 + threshold*(600µs-pd0)`, puis dilatent circulairement l’activité. Avec les valeurs par défaut, le seuil est pd0+4,686 µs et la marge est trois indices, soit 60 ms à 50 Hz. Les bornes deviennent exactement `[pd0,pd0]` hors activité. Ce masque est issu de PW, pas de la géométrie, et ne prouve pas la non-contribution.

Pour un test géométrique, reconstruire le signe à partir du θ prévu et de sa marge d’incertitude, garder les pas qui traversent un zéro et libérer une marge anticipatoire avant l’entrée propulsive. Une valeur initiale de trois pas peut être une ablation mais pas une certification de mémoire ; elle doit être confrontée aux réponses Ding, particulièrement au cycle fatigué. Ne pas figer un masque temporel une fois pour toutes en supposant une vitesse constante. Un masque adaptatif doit être gelé pendant chaque solve, archivé par fenêtre, et posséder une procédure explicite de libération après incompatibilité ou échec.

La borne ΔPW impose aussi des rampes. À partir de 600 µs, atteindre 131,405 µs demande au moins `ceil(468,595/100)=5` transitions. À chaque entrée/sortie de masque, propager les bornes avec `u_i ≤ pd0+100µs*distance(i,pas_fixe)` et vérifier la compatibilité avec le dernier PW exécuté. Si u_0 est fixé à pd0 alors que u_précédent>pd0+100 µs, cette fenêtre est immédiatement infaisable, indépendamment du solveur. Élargir le masque est plus cohérent que violer la limite de variation.

### Gain, A/B et modifications nécessaires

Fixer les 94 candidats du tableau retirerait au plus 94 PW libres, soit 47 % des PW physiques mais seulement **1,14 %** du vecteur explicite. Les états Ding et les défauts Radau restent présents. Le réglage par défaut IPOPT `fixed_variable_treatment=make_parameter` élimine les variables aux bornes égales du problème interne ; `make_constraint` ajoute au contraire des égalités. Vérifier la valeur réellement utilisée et le nombre d’inconnues post-traitement. Source : [options IPOPT](https://coin-or.github.io/Ipopt/OPTIONS.html). Les fonctions symboliques peuvent malgré cela garder leur dimension et leurs calculs : le gain de temps sera généralement modeste ou dépendra surtout de la baisse d’itérations. Une perte de faisabilité peut le rendre négatif.

A/B minimal : un même RHO préparé, A sans masque, B avec masque **géométrique prudent et figé**, plus contrôle ponctuel du masque brut seulement s’il passe la vérification des bornes. Même état initial, même terminal, mêmes poids et même solveur. Projeter le seed B vers le domaine bornes+slew avant le solve et comptabiliser séparément cette préparation ; essayer également le même seed brut pour distinguer l’effet du warm-start. Après le solve B, libérer le masque et résoudre A depuis ce résultat : ce contrôle de relâchement peut mesurer la perte de coût local associée à la restriction, sans prétendre connaître l’optimum global. Répéter sur une fenêtre précoce et une fenêtre fatiguée avec checkpoint complet.

Poursuivre sur huit RHO uniquement si les deux fenêtres restent certifiées, si les règles de rampes/seam passent et si le gain de temps est crédible. Critères de sélection proposés, à annoncer avant mesure : coût physique fatigue +0,5 % maximum sur les fenêtres/trajectoires comparables, perte de capacité normalisée ≤0,001 en absolu, pas d’aggravation de l’erreur mécanique au-delà des seuils communs. Ces valeurs sont des tolérances d’expérience proposées, pas une équivalence physiologique établie. Arrêt dès la première infaisabilité structurelle, certificat perdu ou contrainte ΔPW violée ; ne pas interpréter un échec du masque comme une limite d’endurance.

Modifications ultérieures limitées : ajouter un mode géométrique dans le calcul de masque et la CLI/signature de configuration, injecter le profil et θ de référence, mettre à jour les bornes entre fenêtres sans modifier le graphe, ajouter l’analyse de compatibilité slew et la projection de seed, exporter raisons/marges/indices figés et libérations. Préserver toutes les dynamiques et l’audit final. Les tests ciblés porteront sur les zéros et le signe, les tours absolus, les transitions au bord du cycle et les bornes ΔPW ; aucun test de temps unitaire n’est pertinent.

## 4. Piste 3 : u_impair = moyenne des deux u_pairs voisins

### Contrat temporel et extrémité de la fenêtre

L’interprétation retenue est une contrainte temporelle par muscle, avec indexation Python commençant à zéro :

```math
u_{m,2j+1}=\tfrac12(u_{m,2j}+u_{m,2j+2}),\quad j=0,\ldots,23.
```

À 50 contrôles u_0…u_49, seuls les impairs 1…47 ont deux voisins dans la fenêtre. **u_49 reste libre**. On garde donc les 25 PW pairs u_0,u_2,…,u_48 et u_49, soit **26 par muscle, 104 au total**, au lieu de 200. La réduction est 96 degrés de liberté PW, 48 %. Imposer également `u_49=(u_48+u_0)/2` introduirait une fermeture périodique artificielle ; utiliser le PW futur u_50 nécessiterait un autre contrat de bord. Aucune de ces hypothèses n’est incluse dans la demande. L’option `u_49=u_48` ferait 25 PW libres mais ajouterait une restriction terminale distincte.

Les impulsions physiques restent à **50 Hz** ; on ne garde pas seulement une impulsion sur deux. Les 50 valeurs physiques restent maintenues constantes sur chacun des intervalles de 20 ms et leurs recrutements Ding sont évalués avec ces valeurs. Passer à `ControlType.LINEAR_CONTINUOUS` produirait une rampe à l’intérieur des pas, ce qui changerait le modèle. Le Bioptim installé refuse en outre ce type pour l’intégrateur concerné lorsque `n_threads>1`, [ode_solvers.py](../../.benchmark-deps/bioptim/bioptim/dynamics/ode_solvers.py:199).

### Interaction exacte avec DeltaPW

Sur chaque paire interpolée, les deux variations physiques sont égales :

```math
u_{2j+1}-u_{2j}=u_{2j+2}-u_{2j+1}=\tfrac12(u_{2j+2}-u_{2j}).
```

Ainsi |ΔPW|≤100 µs devient |u_2j+2−u_2j|≤200 µs entre ces ancrages. Garder seulement 100 µs entre ancrages doublerait indûment la restriction de pente. Il faut toujours vérifier le pas u_48→u_49 à 100 µs et le raccord u_précédent→u_0 à 100 µs. Les bornes physiques sont préservées par les moyennes si tous les ancrages sont admissibles.

Le lift existant permet un pilote simple **sans nouveau type de dynamique** : pour j=0…23, imposer `delta_pw[2j]=delta_pw[2j+1]`, car ces incréments sont exactement les deux différences ci-dessus. Une contrainte multi-nœuds sur les deux incréments suffit ; une contrainte explicite sur les trois PW est également possible. Conserver la régularisation existante sur les 50 incréments, le dernier auxiliaire libre compris. Ne pas recalculer sa normalisation comme si la fréquence ou le nombre de PW exécutés avait changé. Le raccord réel reste borné mais n’est pas pénalisé par cette régularisation existante.

### Degrés de liberté versus taille de KKT

Ajouter 96 égalités ne retire **aucune variable explicite** : le vecteur reste de 8 226 scalaires, avec 96 multiplicateurs de contrainte supplémentaires. La dimension admissible baisse mais la matrice de Newton peut grossir. On ne doit pas appeler cela une réduction de 50 % du NLP ni promettre une accélération MA57. Les égalités sont linéaires et ajoutent une structure locale ; elles ne nécessitent pas de nouvelles Hessiennes non linéaires.

Une élimination explicite par `u=P*v` retirerait 96 variables physiques, environ **1,17 %** du vecteur, sans changer le nombre de points Radau. Un simple `BiMapping` entre composantes musculaires ne représente pas à lui seul cette dépendance entre nœuds temporels. Il faudrait adapter l’assemblage des variables/contrôles, les appels dynamiques et objectifs, le warm-start, les sorties à 50 PW et la signature de cache. Le pilote par égalités est donc le premier test de qualité ; l’élimination n’a d’intérêt que si la qualité est acceptable et que le profilage identifie un gain possible.

### Perte d’optimalité et observation des contrôles actuels

Le nouvel ensemble admissible est un sous-ensemble de l’ancien. Pour un optimum global du même objectif, le coût minimal ne peut pas s’améliorer ; avec IPOPT local, une meilleure valeur peut simplement révéler un autre bassin ou une meilleure initialisation. La régularisation/interpolation peut réduire les oscillations et le nombre d’itérations, ou supprimer des créneaux de recrutement et rendre une fenêtre difficile/infaisable.

Écart maximal à la relation de moyenne, calculé sur les commandes de référence existantes :

| Muscle | Cycle 1 | Cycle 10 | Cycle 100 |
|---|---:|---:|---:|
| Delt_ant | 2,880 µs | 7,727 µs | 68,893 µs |
| Delt_post | 0,307 µs | 0,618 µs | 0,149 µs |
| Biceps | 13,582 µs | 41,042 µs | 80,184 µs |
| Triceps | 9,175 µs | 26,353 µs | 47,765 µs |

La contrainte est déjà approximativement satisfaite sur certaines courbes précoces, mais elle ne l’est pas sur les commandes tardives étroites de Biceps/Delt_ant. Le test au seul cycle 1 sous-estimerait donc le risque. Le choix de parité dépend de l’origine temporelle ; avec un déplacement de 50 pas à chaque cycle, la parité se conserve, mais une avance impaire demanderait de gérer explicitement son décalage.

A/B minimal : une même fenêtre précoce et une même fenêtre fatiguée, A libre/B avec les 96 égalités, mêmes contraintes physiques, λ, solveur et seed de départ. Préparer un seed B en conservant les pairs puis en remplaçant les impairs par la moyenne ; cette opération préserve la limite de variation si le seed initial la respectait. Reconstruire les carriers/incréments de manière cohérente et garder u_49 libre. Les états physiques peuvent ensuite nécessiter une restauration de faisabilité, qui doit être comptée. Après B, résoudre de nouveau A depuis B pour estimer la perte de coût locale. Trois résolutions appariées si l’on veut décider sur le temps ; huit RHO de confirmation seulement après succès des deux fenêtres.

Arrêt/sélection : mêmes critères de qualité proposés que pour le masque (+0,5 % coût physique, ≤0,001 perte de capacité normalisée, aucune perte de certificat), et gain chaud médian ≥10 % sans régression P90 >5 %. Si les égalités préservent la qualité mais ne gagnent pas de temps, le résultat valide la simplification de commande, pas l’accélération ; ne lancer l’élimination explicite qu’en la justifiant par le coût d’assemblage/factorisation. Sauvegarder le résidu d’interpolation par muscle au même titre que l’audit slew.

Modifications minimales ultérieures : option de paramétrisation temporelle avec contrat de bord/parité, ajout de la liste de contraintes multi-nœuds à `prepare_nmpc`/`MyCyclicNMPC`, transformation réversible du seed et des auxiliaires, export de l’audit et des dimensions avant/après. Ne modifier ni Radau5, ni le nombre de stimulations, ni les lois de recrutement.

## 5. Combinaison des pistes et protocole scientifique commun

Ne pas introduire simultanément le masque et l’interpolation lors des premières expériences. Si un impair est fixé à pd0 alors que ses deux voisins ont pour borne inférieure pd0, l’égalité de moyenne **force également les deux voisins à pd0**. Le masque effectif se dilate donc par la contrainte d’interpolation, parfois jusque dans un secteur propulsif. Pour une combinaison, calculer la fermeture logique des bornes avant le solve et recalculer le nombre exact de PW libres ; additionner les deux gains serait faux.

Les contrôles de base sont communs : même résistance, même durée/fréquence, même grille/abscisses Radau5, même profil mécanique et SHA modèle, mêmes lois de calcium/Hill, mêmes conditions initiales et terminales, mêmes tolérances et budget IPOPT, même scaling et options MA57, même dernière commande exécutée, même protocole primal/dual. Les seeds de chaque variante et leurs projections doivent être enregistrés ; une restriction nouvelle n’autorise pas à présenter deux seeds différents comme un warm-start identique.

Mesures minimales : dimensions explicites et internes après bornes fixes, non-zéros des Jacobienne/Hessienne, itérations, nombre d’évaluations et temps `f/g/grad/jac/hess`, temps total solveur, préparation, transfert, boucle complète, faisabilité NLP maximale, stationnarité, statut, ΔPW maximal et raccords, erreur d’angle/vitesse, coût fatigue exécuté et capacités par muscle. Comparer les coûts physiques sur la même durée réellement exécutée ; ne pas substituer la somme des objectifs de fenêtres, qui contient la régularisation.

La référence certifie la faisabilité avec le seuil 1e−5 et l’angle terminal nominal ±0,002 rad avec sa tolérance physique de certification. Garder ces seuils exacts dans les deux bras, ainsi que les tolérances slew existantes ; consigner les petites violations numériques au lieu de les arrondir à zéro. Pour les deux pistes qui restreignent les commandes, effectuer un replay indépendant à haute précision sur les courtes fenêtres retenues : mêmes contrôles constants et mêmes impulsions, comparaison de F/A/Tau1/Km/θ/ω aux sorties NLP. Le statut IPOPT seul ne mesure pas l’erreur d’intégration entre les nœuds. La référence de 100 RHO ne fournit pas à elle seule une telle validation indépendante nouvelle.

Garde de coût des pilotes proposée : même plafond de 2000 itérations, watchdog de 30 s par résolution **après construction**, et arrêt après la première fenêtre non certifiée ; journaliser un arrêt au watchdog comme arrêt numérique du pilote. Si la restauration d’un nouveau seed coûte plus que l’économie attendue des huit fenêtres, arrêter le développement de cette variante pour la latence en ligne et rapporter le coût total. Aucun de ces tests courts ne permet de conclure à une endurance maximale ; un essai tardif ou une campagne plus longue n’est justifié qu’après les premiers gates.

Ordre recommandé : **(1)** vérifier le graphe réellement conservé par SX et le mapping MX sans changer la physique ; **(2)** tester les 96 égalités temporelles, faciles à retirer et à auditer ; **(3)** évaluer le masque géométrique avec anticipation et rampes, plus risqué pour l’optimalité. Si le seul objectif est une baisse importante de taille du NLP, les 1 404 variables du lift ΔPW montrent qu’une élimination analytique des auxiliaires serait un sujet séparé et potentiellement plus ample ; elle n’a pas été ajoutée aux modifications demandées ici.

## 6. Décision opérationnelle et commandes proposées, non exécutées

| Variante | Verdict pour le prochain essai |
|---|---|
| `n_threads`, SX inchangé | Contrôle de câblage court, pas de promesse de gain ; arrêter si aucun mapping ne subsiste. |
| Global MX, dynamique locale SX, 1 contre 4 threads | Premier A/B de parallélisme réellement informatif ; vérifier le coût MX/1 avant de lui attribuer un gain. |
| Masque de phase géométrique dur, sans anticipation | **À écarter comme optimisation exacte.** Retire des PW anticipatoires présents dans la référence et peut retirer du freinage nécessaire. Seulement une ablation explicitement restrictive après vérification de faisabilité. |
| Active set `warmup`, marge circulaire | **Meilleur premier pilote de masque disponible**, grâce au bridge de référence commun et à la marge. Reste une restriction heuristique ; un warmup précoce ne garantit pas l’activité tardive. |
| Active set `historical` | Moins probant si son fichier historique ne correspond pas au même état, couple et 50 contrôles. Son masque utilise le pickle initial historique, pas automatiquement les PW du seed NPZ commun ; vérifier sa provenance avant un A/B. |
| Interpolation des impairs par égalités | Premier pilote de qualité, facile avec le lift existant ; ne réduit pas la taille du vecteur explicite. |
| Interpolation **reparamétrée** `u=P*v` | Candidate seulement après réussite du pilote de qualité ; 104 PW libres et 96 variables explicites supprimées, pas 100 PW libres sans autre hypothèse de bord. Gain dimensionnel faible, gain de temps à mesurer. |

Les options des commandes suivantes ont été vérifiées dans le parseur. Elles sont un protocole **proposé**, pas des runs exécutés ou validés. Les lignes utilisent un nouveau dossier et les seeds existants ; aucun fichier historique n’est remplacé. Les trois RHO servent au premier écran, puis remplacer 3 par 8 pour la confirmation. La CLI ne répète pas un RHO figé dans une même capsule : la mesure microbenchmark chaude décrite plus haut nécessite une petite instrumentation supplémentaire.

```bash
cd /home/mickaelbegon/Documents/Kevin/cocofest-pedalage
task_root="$PWD"
task_env=/home/mickaelbegon/miniforge3/envs/cocofest-rho32
task_out=$(mktemp -d "$task_root/rho-ab-review-XXXXXXXX")
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$task_root" MPLBACKEND=Agg MPLCONFIGDIR="$task_out/mpl-cache"
export LD_LIBRARY_PATH="$task_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
task_driver="$task_root/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
task_seed="$task_root/ipopt-slew-100-reference-20260912-110146/common-cycle1.npz"
task_warmup="$task_root/examples/fes_multibody/cycling/result/cache/warmup_05df4060d788.npz"
task_args=(
  --solvers ipopt --benchmark-profile scientific-radau5
  --objective fatigue --objective-shape quadratic
  --ipopt-linear-solver ma57 --warmup-ipopt-linear-solver ma57
  --ipopt-hsl-library "$task_env/opt/libhsl/v2025.7.21/lib/libhsl.so"
  --cycles-per-window 1 --n-windows 3 --stimulations-per-cycle 50
  --signed-crank-torque 0.1 --terminal-wheel-q-slack 0.002
  --mechanical-formulation reduced --formulation dynamic
  --reduced-internal-crank-velocity-guard on
  --pulse-width-max-step-us 100 --pulse-width-slew-reference-us 100
  --pulse-width-slew-weight 0.1 --ipopt-max-iter 2000
  --standard-warmup-seed "$task_warmup"
)

# Contrôle SX 1/4, puis contrôle de représentation MX/1 et A/B MX 1/4.
"$task_env/bin/python" "$task_driver" "${task_args[@]}" --common-initial-solution "$task_seed" --ipopt-use-sx --n-threads 1 --output-json "$task_out/sx1.json" > "$task_out/sx1.log" 2>&1
"$task_env/bin/python" "$task_driver" "${task_args[@]}" --common-initial-solution "$task_seed" --ipopt-use-sx --n-threads 4 --output-json "$task_out/sx4.json" > "$task_out/sx4.log" 2>&1
"$task_env/bin/python" "$task_driver" "${task_args[@]}" --common-initial-solution "$task_seed" --ipopt-no-use-sx --n-threads 1 --output-json "$task_out/mx1.json" > "$task_out/mx1.log" 2>&1
"$task_env/bin/python" "$task_driver" "${task_args[@]}" --common-initial-solution "$task_seed" --ipopt-no-use-sx --n-threads 4 --output-json "$task_out/mx4.json" > "$task_out/mx4.log" 2>&1

# A/B de masque prêt dans la CLI, depuis le même bridge standard.
# Ne pas ajouter --common-initial-solution : son contrat exige le même mode de masque.
"$task_env/bin/python" "$task_driver" "${task_args[@]}" --ipopt-use-sx --n-threads 1 --pulse-width-active-set none --output-json "$task_out/active-none.json" > "$task_out/active-none.log" 2>&1
"$task_env/bin/python" "$task_driver" "${task_args[@]}" --ipopt-use-sx --n-threads 1 --pulse-width-active-set warmup --pulse-width-active-threshold 0.01 --pulse-width-active-margin 3 --output-json "$task_out/active-warmup.json" > "$task_out/active-warmup.log" 2>&1
```

Le bridge `warmup_05df4060d788.npz` est celui du run de référence, documenté Radau5, 50 contrôles, +0,1 N·m, fatigue quadratique ; SHA-256 vérifié `4d076e5c66b92cd8c404fee26f12e3db4a65c2a461be2fa0988b68dad57f1ccd`. Il est chargé directement par `run_standard_ipopt_warmup` après validation. Le mode `warmup` exige de conserver ce chemin de warmup actif. `_target_independent_warmup_conditions` retire les restrictions d’activité du bridge ; le même bridge peut donc initialiser les bras `none` et `warmup`. Vérifier ensuite l’identité des états initiaux, cibles et trajectoires de bridge réellement consommés dans les sorties. Les PW du seed cible peuvent être projetés différemment par les restrictions ; le coût de projection/restauration appartient au bras concerné.

La validation de `--common-initial-solution` compare strictement `pulse_width_active_set`, [validation du seed](../../examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py:4709). Le NPZ cible `none` ne peut donc pas être passé tel quel au bras `warmup` ou `historical`. **Ne pas contourner ce contrôle en changeant artificiellement les métadonnées.** Pour un A/B au même checkpoint tardif avec un nouveau masque/interpolation, ajouter ultérieurement une voie d’expérience explicite qui conserve la provenance physique et documente la seule restriction de commande ajoutée. Le flag de probe existant ne permet que des différences d’objectif, pas de masque.

Les masques géométriques et l’interpolation reparamétrée ne disposent d’aucune option CLI actuelle : il n’existe donc pas de commande honnêtement exécutable pour ces variantes sans les modifications énumérées dans les sections 3 et 4. Après implémentation, elles reprendront exactement le profil physique de `task_args`, avec un flag explicite de variante et un audit de dimensions/résidus ; aucun nom de flag inexistant n’est proposé ici.

Les commandes montrent les branches possibles, pas l’ordre de lancer systématiquement les six runs : appliquer les critères d’arrêt après chaque contrôle, puis alterner l’ordre des bras sur trois répétitions. L’affinité n’est volontairement pas codée avec une liste de CPU fixe ; réserver d’abord le même ensemble de cœurs physiques disponibles pour les deux bras. Les autres campagnes de l’utilisateur ne doivent pas être déplacées ou arrêtées pour ce pilote.
