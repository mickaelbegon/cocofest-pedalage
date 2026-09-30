# Raccordement RHO-Réserve : audit et portes d'intégration

Audit indépendant du 28 septembre 2026. Aucun source modifié. **Décision actuelle : no-go pour une campagne RHO-Réserve**, go pour implémenter et tester un raccordement expérimental derrière une option désactivée par défaut. Le microbenchmark valide une projection de trois états lents sous force imposée ; il ne valide ni une réserve mécanique réelle ni son effet dans le NLP.

## Points de raccordement constatés

Les lignes ci-dessous identifient le code inspecté et peuvent évoluer.

| Fichier / point | Rôle et modification minimale proposée |
| --- | --- |
| `cocofest/custom_objectives.py:17–76` | Les helpers terminaux horizon/rollout montrent comment extraire les états et les paramètres du `PenaltyController`. Ajouter un helper réserve retournant le scalaire signé et les états projetés, depuis `(A,Tau1,Km)` terminaux en ordre musculaire explicite. |
| `cocofest/custom_objectives.py:137` | `minimize_terminal_muscle_reserve` ne dépend que de `A/A_scale`. Il ne constitue pas le nouveau coût projeté ; conserver des noms/options distincts. |
| `examples/fes_multibody/cycling/cycling_pulse_width_mhe.py:1281–1285` | Lire une option réserve et son binding dédié lors de `prepare_nmpc`. Valider modèle, cadences, muscle order, horizon, charge et dimensions avant compilation. |
| même fichier, `:1623–1658`, `:1702–1726` | Construire une fois le binding et les contraintes de domaine. Les expressions CasADi symboliques ne vérifient pas seules la positivité des états projetés. Fixer les limites numériques avant la campagne ; ne pas masquer les sorties de domaine par clipping. |
| même fichier, `set_objective_functions`, `:3243–3261` | Ajouter un objectif Mayer à `Node.END`, `quadratic=False`, de poids explicite. Le score signé peut être négatif : le mettre au carré changerait le problème et pourrait pénaliser une bonne réserve. |
| même fichier, `:1822–1846` | Étendre la garde d'un seul binding fixe, ajouter ses options au constructeur puis l'attacher. **Refuser la coexistence avec `ParametricFatigueWeightBinding`**, le rollout et l'horizon tant qu'un assemblage des listes de paramètres n'est pas validé. Ne pas écraser un `ParameterList`. |
| `examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py:19243`, `:20628`, `:20731` | Le worker appelle ce driver, qui construit puis transmet `simulation_conditions` à MHE. Ajouter le passage de la nouvelle option dans cette fabrique est nécessaire ; modifier seulement MHE ne raccorde pas le runner réel. Limiter le premier pilote à IPOPT SX compilé. |
| `cocofest/simulation/independent_arms_process.py:539–565` | Valider la configuration dès la construction : fatigue statique unitaire, adaptation des poids désactivée, absence du binding fatigue, nouveau binding présent. La garde actuelle autorise compilation pour les poids unitaires statiques ou le binding fatigue ; aucune preuve du nouveau binding n'y existe. |
| même fichier, callback `:578–629` | Extraire la solution avant le shift suivant ; utiliser la certification existante et `physical_completed`, pas `cycle_index`, qui peut compter les tentatives. Le seed `solution is None` est marqué certifié mais n'est pas un premier cycle résolu. |
| même fichier, `:671–740` | Après rejet des retries non certifiés et réception d'un `prepare` valide, initialiser/mettre à jour le profil depuis une copie du cycle certifié, puis écrire ses paramètres numériques avant le solve suivant. Réaffirmer la cible `E_prod` comme aujourd'hui. Capturer ensuite le checkpoint retry, afin qu'il contienne exactement le profil et les bornes utilisés. |
| même fichier, `:762–777` | Exporter les reçus de profil, preuves de compilation, coût réserve séparé, échecs/holds et latences dans le résultat, au-delà du seul résumé fatigue existant. |

Le fallback `update_bioptim_fatigue_cost` reconstruit les objectifs : ce chemin doit être exclu de cette configuration. Le premier pilote conserve la répartition de charge entre bras ; une charge modifiée invalide la linéarisation et exige une politique distincte, pas une mise à jour silencieuse hors cadence.

## Contrat minimal du nouveau binding

Créer le graphe avant le premier solve avec une taille fixe : nombre de muscles, nombre d'intervalles, nombre de marges, horizons, durées physiques, paramètres Ding et températures fixés. Le vecteur numérique contient le profil `F_ref`, le centre `z_ref`, les marges `m_ref`, leur Jacobien `G` et un coefficient d'activation initialement nul. Chaque valeur mutable est un paramètre NLP à bornes égales, avec mise à jour simultanée de l'initial guess.

Attention : `LocalMechanicalMarginModel.evaluate_casadi` transforme actuellement `reference_states`, `reference_margins` et `state_jacobian` en **constantes `DM`**. Remplacer l'objet Python après compilation n'actualise pas ces constantes dans le graphe. Il faut un chemin symbolique dédié aux coefficients paramétrés, mathématiquement équivalent et testé contre le noyau numérique. Les températures, durées et horizons restent structurels ; les changer exige une autre expérience construite une seule fois.

Le premier solve utilise des données de bootstrap finies, de bon domaine, et l'activation zéro. Les éventuelles contraintes nouvelles doivent aussi être inactives pendant ce bootstrap sans évaluation invalide ; multiplier une expression NaN par zéro ne la sécurise pas. Le cycle 1 certifié fournit le premier profil réel, le centre et la linéarisation, puis active le coût pour le cycle 2. Ensuite, les seules mises à jour ont lieu aux frontières physiques certifiées **20, 40, 60, ...** : initialisation à 1, puis cadence sur les multiples de 20, pas 21/41. Le profil est gelé sur tous les autres cycles. Une frontière non certifiée n'initialise ni n'actualise rien ; un retry ne consomme pas la cadence et conserve les mêmes paramètres.

Chaque reçu contient index source et application, certification/status/résidu, ordre musculaire, hash modèle et paramètres, charge, omega, durée, méthode d'échantillonnage, hash du profil, taille des buffers et compte de mises à jour. En reprise, restaurer profil, compteur et hash à partir du checkpoint ; ne pas réinterpréter un seed comme nouveau cycle 1.

Le coût minimal est `activation * lambda * penalty(z_terminal, F_ref, m_ref, G)`. Ici `F_ref` vient du dernier cycle certifié et reste numérique ; les **états terminaux candidats** restent symboliques, ce qui donne une influence sur la décision courante. Injecter un score entièrement numérique serait un ajout constant au coût et n'aurait aucun effet sur les commandes. Utiliser le profil de force candidat courant à la place du profil certifié serait une autre variante et nécessiterait un accès multi-nœuds explicite, non un Mayer terminal ordinaire.

Convention : la projection commence à `z_terminal` du cycle candidat ; H=1 désigne **un cycle futur supplémentaire**, sans compter une deuxième fois le cycle courant. Le profil du cycle certifié décrit une force imposée, pas la garantie de sa reproduction avec des PW futures.

## Provenance des forces et linéarisation mécanique

Extraire les `F_<muscle>` du cycle complet certifié, avec les vrais temps/nœuds de collocation et le même ordre que le modèle. Fixer une conversion documentée vers les intervalles constants du noyau ; les valeurs d'extrémité de phase sont une approximation, pas la force entière certifiée. Mesurer l'erreur de la carte lente sous ce profil par rapport au cycle réel et à un raffinement indépendant. Vérifier `sum(dt)=2*pi/abs(omega)` et le tour signé complet. Ne pas recopier `duration=1/stimulations_per_cycle` sauf lorsque cette identité physique a été prouvée.

Le point encore absent est la provenance réelle de `m_ref` et `G`. Les gains synthétiques du microbenchmark ne conviennent pas à une campagne. Définir une marge mécanique sans dimension et calculer sa linéarisation hors NLP, depuis le modèle réel et une sonde déclarée. Un replay PW-max individuel mesure une opportunité, pas un maximum réalisable du cycle couplé. Préserver les termes communs de mécanique une seule fois et la cible de travail déclarée ; `Delta E_prod` vérifie cette cible sans devenir une demande qui dérive. Valider la linéarisation sur des perturbations non utilisées pour la construire et une région de confiance explicite. Hors région, tenir le profil et journaliser le défaut ou arrêter le pilote selon la règle fixée ; aucune re-linéarisation cachée entre frontières dues.

## Bug de signe confirmé : blocage avant réutilisation de Physio-U

`ReducedFesCyclingModel.mechanical_equilibrium_residual` (`:423–429`) et `dynamics` (`:533–546`) donnent exactement :

```text
r = sum(b_i F_i) + e*tau_load - g - v*omega²
E_prod_dot = omega * r(tau_load=0)
           = omega*sum(b_i F_i) - omega*(g+v*omega²)
d(E_prod_dot)/dF_i = omega*b_i
```

Le bridge `physio_update.py:196–205` transmet `muscle_effectiveness` directement comme `moment_coefficients`. Mais `build_isokinetic_max_pw_envelope`, `:92–101`, emploie `-omega*b_i*F_i` et écrête à zéro. Le signe ne correspond donc pas à `E_prod`. Vérification arithmétique indépendante effectuée : `omega=-2*pi`, `b=-0.02`, `F=10`, `g=v=0`, `e=1` donne `tau_load=0.2`, `E_prod_dot=+1.2566370614 W`, `dP/dF=+0.1256637061`, contre une opportunité historique nulle.

L'override par `Delta E_prod` aux lignes `:216–242` **ne corrige pas** la puissance disponible ; le helper peut même lever son erreur « no positive work » avant d'atteindre cet override. En outre, les intervalles du bridge sont artificiellement d'une seconde par cycle (`:205`).

Recommandation : corriger la convention de contribution à `+omega*b_i`, le calcul de référence correspondant et la durée à `T_cycle/n_phase`, puis vérifier les fixtures avec les vraies équations inverses, les termes `g/v/e` et une cadence différente. Ne pas inverser le signe de `E_prod` pour faire passer les anciennes fixtures : il est cohérent avec `-tau_load*e*omega`. D'ici ces validations, bloquer toute utilisation des sorties actuelles de Physio-U pour définir `m_ref/G`. Un adapter indépendant cohérent et testé peut lever le blocage sans dépendre de Physio-U ; il doit conserver une provenance explicite.

## Tests runtime et décision go/no-go

Ces seuils sont des critères proposés avant campagne, pas des résultats déjà obtenus.

| Test / métrique | Gate |
| --- | --- |
| Profil issu d'un cycle réel | Tous les buffers ont une provenance certifiée, un ordre/dimension/temps corrects ; aucun profil synthétique ou warm-start présenté comme cycle certifié. Erreur de représentation mesurée et publiée. |
| Ordonnancement et retry | Exécuter au moins jusqu'au solve 41 : bootstrap nul, applications 1/20/40 seulement ; retries injectés aux frontières 1/20 n'actualisent pas deux fois. Reprise et arrêt après certification couverts. |
| Compilation réelle | Un build du solveur compilé, zéro recompilation aux changements 1/20/40 ; instrumenter les appels compilation et comparer identité solver/callbacks, tailles et sparsités. `id(nlp)` seul est insuffisant. Vérifier que les nouveaux paramètres parviennent effectivement aux callbacks compilés. |
| Coexistence des bindings | La combinaison réserve + `ParametricFatigueWeightBinding` ou autre binding est rejetée avant le solve. Aucun appel `update_objectives` ou fallback de poids pendant le run. |
| Influence dans le coût | Sur le **même vecteur primal**, modifier un profil réel/gradient non dégénéré modifie le terme réserve et son gradient dans le callback compilé conformément à NumPy ; gradient non nul sur les états terminaux. Deux solves appariés à poids nul/positif doivent exposer coût et commandes, sans exiger arbitrairement une différence de commande pour un optimum contraint identique. |
| Valeur et gradient | Tolérances du microbenchmark : erreur absolue valeur/marge `2e-9`, gradient absolu `2e-7`, relatif `2e-5` avec plancher `1e-6`, sur variables correctement normalisées ; contrôler aussi le chemin paramètre -> callback réel. |
| Mécanique et signe | Accord analytique/numérique de `dE_prod_dot/dF_i=omega*b_i`, de la balance inverse et de la quadrature ; fixtures avec signes mixtes, termes communs non nuls et `T!=1 s`. Aucun crédit propulsif inversé. |
| Linéarisation réelle | Sur perturbations tenues à l'écart : erreur absolue marge `<=0.005`, aucun faux signe positif lorsque la marge de la sonde de référence est `<-0.005`. Inclure repos, fatigue intermédiaire, proximité d'arrêt et asymétrie. Cette sonde reste un proxy si sa réalisabilité n'est pas prouvée. |
| Domaine | États projetés `A,Tau1,Km` dans les limites déclarées ; aucun clipping. Tests de déplétion et sortie de région de confiance avec reçu de refus/hold. |
| Faisabilité terminale réelle | Tous les cycles acceptés passent le même audit complet original : état/contrôle/continuité, angle, vitesse, couple et travail. Seuils isocinétiques existants : angle `1e-8 rad`, vitesse `1e-9 rad/s`, équilibre `1e-6`, travail `1e-6 J`, couple `1e-8 Nm`. Le résidu global NLP utilise la tolérance du runner enregistrée, sans la relâcher. Le replay continu conserve son gate distinct de `0.02 J`. |
| Latence | Publier cold compile, solve warm médiane/P90/P95, coût complet de mise à jour (extraction + linéarisation + écriture + audit), latence des frontières dues et amortie. Gate initial déjà proposé : P90 mise à jour complète `<=20 ms` à H=100 sur matériel cible ; un échec nécessite une décision explicite sur le budget, pas le report du seul noyau rapide. |
| Pilote apparié | Mêmes checkpoints, charge/répartition, tolérances, warm-starts et budget solveur. Comparer 1/5/20 cycles puis au moins 41 pour la cadence ; aucune perte de certification imputable à la nouvelle variante. Rapporter tentatives/recoveries et différences de travail séparément des cycles certifiés. |

Le go autorise un pilote expérimental seulement après satisfaction des gates de mécanique, provenance, compilation, coût et faisabilité. Le gain d'endurance demande ensuite une campagne prospective appariée : nombre de cycles physiques certifiés et travail cumulé à demande identique, cause d'arrêt, échecs de solveur, coût de calcul et marges proxy séparés. Un échec local du solveur ou une marge proxy négative ne démontre pas l'épuisement physiologique ; un score positif ne certifie pas les futurs cycles.

Références locales : `mechanical_reserve_projection_benchmark.md` pour les gates du noyau, `mechanical_reserve_projection_protocol.md` pour l'audit mécanique général. Ce document traite la **variante de coût direct paramétré** ; la pondération fatigue par sensibilités décrite dans le protocole constitue une autre intervention.

## Variante locale PW — état au 28 septembre 2026

La tentative « PW candidat global » a été arrêtée : son coût Mayer lisait les
PW des 30 nœuds de contrôle depuis le nœud terminal. Cette dépendance
multi-nœuds n'est pas une expression Bioptim sûre et les workers natifs
quittaient avant le premier solve. L'option CLI historique est donc refusée
avant construction.

La variante remplaçante, activée seulement par
`--experimental-mechanical-reserve-local-pw-cost`, procède à chaque frontière
certifiée 1, 20, 40, … comme suit :

1. Rejouer le cycle certifié et calculer le tangent causal Ding PW→force.
2. Différencier la réserve projetée par rapport aux forces imposées, puis
   composer les deux dérivées hors NLP.
3. Installer, comme paramètres à bornes égales, une pente, une PW de
   référence et une courbure diagonale. La courbure borne le pas non contraint
   à `--experimental-mechanical-reserve-local-pw-trust-us` (25 µs par défaut).
4. Ajouter à chaque phase seulement le coût de son propre PW. Aucun callback
   ne lit un autre nœud de contrôle et le NLP compilé conserve donc sa
   structure.

Cette courbure est un garde-fou numérique et non un modèle physiologique de
variation de stimulation. Les bornes de PW, la dynamique Ding et les
contraintes du cycle restent les garantes physiques. La quantité est un proxy
prédictif de réserve ; elle ne certifie ni faisabilité future ni endurance.

Validation effectuée : 23 tests ciblés (chaîne de dérivées, dimensions,
rollback, CLI et coût mono-nœud) passent. Le smoke bilatéral réel 3 cycles,
30 Hz, IPOPT/MA57, deux CPU, a terminé les 3 cycles ; la mise à jour du cycle
1 a modifié les six buffers sans reconstruire le graphe (`build_count=1`).
Les temps de paire après amorçage sont 1.12–1.30 s. Une exécution de 21 cycles
a maintenu les résolutions jusqu'au cycle 17 autour de 0.84–1.68 s, mais le
lanceur interactif a imposé son plafond de 30 s avant le cycle 20 ; ce n'est
pas encore la preuve runtime de la seconde mise à jour compilée. Le test
unitaire vérifie toutefois qu'une seconde écriture conserve la même identité
de solveur lorsque le backend la publie.
