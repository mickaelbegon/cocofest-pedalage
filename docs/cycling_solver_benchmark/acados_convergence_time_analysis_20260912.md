# Réduire le temps jusqu'à convergence ACADOS — analyse du 12 septembre 2026

Périmètre : RHO dynamique réduit, isorésistance +0,1 N·m, fenêtre de 1 s, 50 commandes par cycle, quatre muscles, borne dure |ΔPW| ≤ 100 µs, guard interne de vitesse, angle terminal ±0,002 rad. Analyse des trois pilotes demandés et du code local non commité. Aucun changement de code applicatif ni commit dans cette analyse. Les pistes ci-dessous sont des hypothèses à tester lorsqu'elles ne sont pas explicitement associées à une mesure.

Le levier immédiat démontré est l'arrêt anticipé des tentatives qui stagnent, suivi d'une récupération certifiée. Pour faire converger ACADOS lui-même plus vite, la priorité scientifique est la couture du primal F/ω et le comportement SQP après cette couture. Le coût de construction est important au démarrage ; il n'explique pas les fenêtres chaudes à 100 SQP.

## 1. Ce que démontrent les pilotes

| Cas | RHO validés | SQP ACADOS successifs | Temps natif ACADOS | Conclusion limitée aux données |
| --- | ---: | --- | ---: | --- |
| ΔPW explicite, λ=0, sans récupération | 1/5 demandés ; 2 tentés | 4, 100 échec | 1,925 + 47,687 s | Premier seed bon ; transfert vers RHO2 difficile |
| Régularisation, λ=0 | 1/3 | 4, 100 échec | 2,127 + 47,343 s | Échec reproduit avec un thread et exécution séquentielle |
| Régularisation, λ=0,1 | 3/3 | 7, 10, 5 | 3,491 + 4,562 + 2,441 s | Stabilisation prometteuse, objectif modifié |
| Hybride λ=0, budget 100 | 8/8, dont 3 IPOPT | 4, 100, 8, 100, 8, 100, 9, 11 | 158,957 s | RHO2/4/6 récupérés par IPOPT/Radau-5 |
| Hybride λ=0, budget initial 30 | 8/8, dont 3 IPOPT | 4, 30, 8, 30, 8, 30, 9, 11 | 56,406 s | Mêmes récupérations et indicateurs physiques exportés |

Sources : [pilote ΔPW](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/delta-pw-pilot-50hz-r010-20260912T143703Z/REPORT.md), [pilote λ](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/slew-regularization-pilot-20260912/REPORT.md), [pilote budget](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/acados-budget30-delta-pilot-20260912T1452/REPORT.md), et JSON/logs voisins.

Pour λ=0, le coût observé est environ 0,47–0,48 s/SQP dans le premier pilote. Sur les huit fenêtres nominales, 300 des 340 SQP et 141,208 des 158,957 s natives concernent les trois échecs : **88,8 % du calcul ACADOS est dépensé dans des tentatives non certifiées**. Le budget 30 retire 210 SQP. La boucle passe de 190,274 à 84,073 s, soit −55,8 %. Cette économie de temps jusqu'à une solution hybride certifiée n'est pas une amélioration du taux de convergence ACADOS : il reste 5/8 dans les deux cas.

Le gain λ=0,1 ne permet pas de conclure que le même problème λ=0 a été accéléré. Il ajoute une courbure sur les incréments et change les commandes, les états terminaux et donc les RHO suivants. Le coût physique ACADOS du cycle 1 passe de 0,0436673 à 0,0438044, environ +0,31 %. Les deux coûts cumulés ACADOS à trois cycles ne sont pas comparables, car λ=0 échoue au deuxième. Les seeds des deux poids ont les mêmes 22 états physiques initiaux, mais ce ne sont pas les mêmes trajectoires optimales de départ.

## 2. Décomposition du temps : utiliser des rubriques additives

Les temps suivants sont reconstruits directement depuis les JSON du pilote budget, et non depuis les seuls totaux des fenêtres certifiées.

| Temps mural, secondes | Nominal 100 | Budget30 | Interprétation |
| --- | ---: | ---: | --- |
| Préparation du premier primal | 12,693 | 8,858 | `initial_guess_preparation_time_s` |
| Installation avant boucle | 40,739 | 32,394 | `pre_solve_setup_wall_time_s`, construction du problème de récupération comprise |
| Natif ACADOS, toutes tentatives | 158,957 | 56,406 | Somme `solver_attempt_accounting.attempts[].solver_time_s` |
| Construction/chargement/interface ACADOS dans les appels | 4,500 | 3,941 | Somme des temps muraux des tentatives moins leurs temps natifs |
| Récupération complète, toutes tentatives | 21,072 | 18,607 | Somme `acados_ipopt_recovery_summaries[].timing.total_wall_time_s` |
| Reste de la boucle après les trois lignes précédentes | 5,745 | 5,119 | Transfert, callbacks, copies, audits ; résidu de comptabilité |
| **Boucle RHO complète** | **190,274** | **84,073** | Somme des quatre composantes internes précédentes |
| Traitement après boucle | 0,173 | 0,195 | Résumés et audits finaux |
| **Bout en bout, seed IPOPT exclu** | **243,884** | **125,525** | Quelques millisecondes résiduelles d'enveloppe |

Ces lignes ne doivent pas être additionnées une deuxième fois à la boucle : ce sont ses composantes. De même, les sous-chronomètres de `update_functions` sont inclus dans son temps total.

Deux pièges de nomenclature ressortent :

1. `execution_timing.rho_orchestration_wall_time_s` vaut 159,599/55,162 s. Le comparateur le calcule en retranchant les temps des fenêtres retenues après fallback : il inclut donc les tentatives ACADOS abandonnées et une partie des récupérations. Ce n'est pas un temps Python pur. La comptabilité additive ci-dessus donne 5,745/5,119 s de reste après tous les solveurs et récupérations.
2. `acados_ipopt_recovery.recovery_wall_time_s` ne vaut que 8,517/7,748 s. Il additionne les temps de résolution internes IPOPT. La première récupération dure en réalité 13,553/11,611 s, dont 13,103/11,222 s dans l'appel `solve`, alors que son temps interne n'est que 3,322/2,905 s. L'installation du backend et la fabrication des fonctions avant la résolution sont dans cet appel. Les deux récupérations chaudes coûtent encore 4,092/3,427 s nominal et 3,704/3,292 s budget30, audits compris.

Le problème de récupération est construit avant la boucle, une fois par degré demandé, dans [le runner](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py:20890). Cela explique la différence de périmètre entre les 32–41 s d'installation du pilote hybride et environ 0,005 s dans les pilotes sans recovery ; la part exacte de chaque construction symbolique n'est pas encore chronométrée séparément. Déplacer une compilation avant la boucle améliore la latence du premier échec, mais ne supprime pas son coût bout en bout.

### Construction native et coût chaud

| Cas | Fonctions | Templates | Compilation | Total natif construit |
| --- | ---: | ---: | ---: | ---: |
| ΔPW λ0 | 0,079 | 1,454 | 2,864 | 4,397 s |
| Pilote régularisation λ0 | 0,080 | 1,465 | 2,813 | 4,359 s |
| Pilote régularisation λ0,1 | 0,082 | 1,475 | 3,005 | 4,562 s |
| Budget nominal | 0,079 | 1,375 | 2,784 | 4,238 s |
| Budget30 | 0,069 | 1,238 | 2,393 | 3,701 s |

Le total de construction est inclus dans la première fenêtre, pas dans `solver_time_s`. Dans le pilote ΔPW, la différence fenêtre−solveur du RHO1 est 4,464 s, dont 4,397 s de codegen/build ; au RHO2 elle tombe à 0,028 s. Réutiliser les bibliothèques apporte donc un gain au démarrage et aux campagnes répétées, sans enlever 47 s à une fenêtre chaude.

Le runner lit déjà `time_sim`, `time_qp`, `time_qp_solver_call`, `time_glob`, `time_reg` et `time_qpscaling` dans [collect_acados_diagnostics](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py:6238). Ils ne sont pas conservés dans les JSON compacts examinés. Il n'est donc pas possible d'attribuer honnêtement les 0,47 s/SQP à l'intégrateur, au QP ou à la globalisation. Ajouter `time_lin`, `time_sim_ad`, `time_sim_la`, `time_qp_xcond` permettrait de distinguer calcul des dérivées, algèbre de l'IRK et condensing. Les sous-temps ACADOS se recouvrent : `time_sim` participe à la linéarisation, `time_sim_ad/la` sont des sous-composantes ; ne pas tous les additionner. La [documentation des statistiques ACADOS](https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_solver.AcadosOcpSolver.get_stats) décrit ces champs ; le runtime local Python 3.11 les confirme.

## 3. Les diagnostics gonflent la boucle, sans expliquer les 100 SQP

`update_functions` représente 5,420 s nominal et 4,808 s budget30 sur huit RHO, avec une médiane de 0,658 s par appel dans le second. Les copies des états et commandes, projections sur bornes et vérifications de faisabilité explicitement chronométrées restent petites : projection des sept transferts 0,031 s, `advance_window` 0,040 s, faisabilité 0,003 s. Les instantanés ACADOS aux deux endroits suivis coûtent environ 0,122 s en tout. Les diagnostics détaillés du primal ne sont pas inclus dans le petit chronomètre `initial_guess_audit`, limité à la signature et à la finitude.

Avec `echo=true` et `--acados-diagnostics`, le callback :

- appelle `print_initial_guess_diagnostics(_nmpc)` vers la ligne 22607, ce qui recalcule les défauts ;
- appelle ensuite `collect_initial_guess_diagnostics(_nmpc)` vers la ligne 22778, puis imprime les mêmes rubriques.

Chaque collecte évalue les défauts Ding et les défauts de dynamique complète RK4. C'est un doublon dans les pilotes sans transformation intermédiaire entre ces points ; lorsque d'autres options modifient le primal entre les deux, les audits portent en revanche sur des états de préparation différents. Une suppression générale du premier audit sans tenir compte de cette distinction serait incorrecte.

`--acados-diagnostics` suffit à activer `initial_guess_diagnostics_requested`; enlever seulement `--initial-guess-diagnostics` ne retire donc pas ce coût. Inversement, les snapshots des résidus nécessaires à la certification restent collectés même sans affichage détaillé. Le mode production doit préserver ces résidus, la certification de chaque RHO et les audits physiques/slew finaux.

`--exact-initial-nlp-audit` est un autre mécanisme : le comparateur le désactive pour ACADOS, car il est réservé aux solveurs NLP CasADi. Les quelques microsecondes `exact_initial_audits_wall_time_s` de ces JSON correspondent à la copie finale d'un audit éventuellement présent, pas à un audit exact ACADOS. Ne pas lui attribuer le coût du pilote ACADOS. Son coût doit être évalué séparément sur la production du seed IPOPT si cette étape compte dans le besoin utilisateur.

**A/B de production exécuté** : λ=0,1, même seed IPOPT certifié, huit RHO, mêmes cap30/recovery, un thread, variantes séquentielles avec puis sans les trois flags demandés. Les deux valident 8/8 par ACADOS, sans recovery, avec les mêmes 56 SQP `[7,10,5,6,6,7,7,8]`. La boucle passe de **37,612 à 31,767 s (−15,54 %)** ; `update_functions` de **5,380 à 0,122 s (−97,73 %)**. La baisse de callbacks explique 89,95 % des 5,845 s gagnées dans la boucle. Le natif ne change que de 27,221 à 26,850 s (−1,37 %, bruit plausible). Les audits finaux, indicateurs physiques exportés, fenêtres sans leurs temps, et sept signatures des initial guesses sont exactement identiques. Voir [le rapport apparié](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/diagnostics-overhead-w01-rho8-20260912/REPORT.md). Une seule paire, ordre diagnostic puis production, avec deux autres campagnes actives : ce résultat démontre le câblage et localise le surcoût, sans établir un intervalle statistique du gain. Les variations de préparation/codegen ne sont pas imputées aux seuls flags.

## 4. Le défaut inter-RHO est surtout une couture du primal physique

| État/mesure avant ACADOS | Seed RHO1 λ0 | Seed transféré RHO2 λ0 | Seed RHO2 λ0,1 |
| --- | ---: | ---: | ---: |
| Défaut RK4 force maximal | 0,004604 N | 38,0911 N | 22,4431 N |
| Défaut RK4 ω maximal | 0,000101 rad/s | 1,31417 rad/s | 0,726519 rad/s |
| Défaut RK4 θ maximal | proche du microradian | 0,0251099 rad | 0,0146274 rad |
| Résidu dynamique ACADOS initial, échelonné | 2,805e−6 | 0,209 | 0,116 |
| SQP | 4, succès | 100, échec | 10, succès |

Les résidus RK4 physiques et les résidus natifs ACADOS ne sont pas dans les mêmes unités/scalings ni issus de la même carte d'intégration. Leur ordre de grandeur relatif identifie le transfert, pas une erreur de transcription quantifiée par soustraction.

Dans [advance_window_initial_guess_states](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/examples/fes_multibody/cycling/cycling_pulse_width_mhe.py:611), Cn, F et ω sont classés cycliques. Pour un horizon d'un cycle, `set_init_cyclical` répète le cycle précédent ; A, Tau1 et Km sont translatés selon leur dérive, θ reçoit le décalage de tour. Les bornes du premier nœud imposent ensuite le véritable état terminal précédent. Les nœuds 1…N du profil répété n'ont pas été réintégrés à partir de ce nouvel état initial.

L'exemple RHO2 est explicite : F_Triceps(0)=23,0589 N et ω(0)=−8,33125 rad/s ; le prochain ω du profil répété vaut −9,02593, alors que l'intégration du premier intervalle donne −7,71176. Le défaut de −1,31417 est localisé au premier intervalle. La projection **supplémentaire** `transfer_bound_projection` ne change rien à ce stade. Attention : cela ne signifie pas qu'aucune mise à jour de borne/écrêtage n'a eu lieu auparavant ; l'imposition normale du nouvel état initial fait partie de l'avancement dans `cycling_pulse_width_mhe.py`.

La reconstruction actuelle des carriers et des incréments est cohérente : `z[k]=PW[k]`, `ΔPW[k]=PW[k+1]−PW[k]`, dernier delta initialisé à zéro. Les défauts déterministes antérieurs de remplacement du seed et de lift périmé sont documentés comme corrigés ; ils ne doivent pas être rediagnostiqués comme causes encore présentes. La borne dure n'impose pas une trajectoire dynamique faisable et ne contraint pas un raccord périodique artificiel dernier→premier PW du même cycle.

### Pourquoi le SQP ne termine pas

Sur RHO2 λ0, tous les QP retournent `qp_stat=0`, `alpha=1` à chaque itération et les itérations QP alternent longtemps entre 22 et 23. La configuration effective est déjà `FUNNEL_L1PEN_LINESEARCH`, `GAUSS_NEWTON`, `GERSHGORIN_LEVENBERG_MARQUARDT` et `ADAPTIVE_QPSCALING` : recommander simplement « activer une line search » ou « une régularisation » manquerait ce point.

Les résidus finaux λ0 RHO2 sont environ [8,2565e−3, 1,3248e−3, 7,57e−13, 1,96e−9] : les inégalités sont satisfaites, les équations dynamiques et la stationnarité stagnent. Dans le pilote budget, les valeurs retournées à 30 et 100 SQP pour RHO2 sont identiques aux chiffres enregistrés. Le même schéma existe aux RHO4/6. Cela est compatible avec une oscillation/attraction du schéma SQP ; la preuve d'un cycle de période deux demande de conserver les iterates x/u et les résidus par itération, ce que ces JSON n'offrent pas.

La fatigue quadratique porte essentiellement sur les états de réserve. Sans poids ΔPW, elle ne fournit pas de courbure directe du coût en incrément. Le lift ajoute aussi quatre directions terminales libres sans effet physique lorsque λ=0. Ce sont des sources plausibles de mauvais conditionnement ; elles ne prouvent pas que le QP est la partie dominante du temps. La régularisation λ=0,1 améliore à la fois la courbure et l'état terminal qui sera transféré : une expérience sur **le même RHO2 figé** doit séparer ces effets.

## 5. Plan priorisé en préservant la physique cible

### P0 — Comptabilité, mode production et récupération disponible

Conserver les statistiques natives par tentative, en particulier les tentatives remplacées par IPOPT. Remplacer ou compléter le faux « temps orchestration » par les sommes additives décrites plus haut. Chronométrer les deux collectes détaillées du primal et la construction du backend recovery. Ce travail ne modifie ni problème ni solution et rend les prochaines décisions mesurables.

Utiliser le budget30 déjà validé comme référence hybride provisoire ; il ne suffit pas de multiplier l'économie par 100 cycles. La campagne historique d'une autre version de lift avait 16 succès chauds dépassant 30 SQP : un cap peut remplacer des succès lents par des récupérations supplémentaires. Comparer aussi 50 et un arrêt fondé sur la stagnation. Un contrôle de stagnation devrait observer les résidus échelonnés et la taille du pas sur plusieurs itérations, pas un seul seuil de défaut du seed. Par exemple : après 12 itérations, absence de réduction d'au moins 5 % de la meilleure faisabilité sur huit itérations, stationnarité encore au-dessus de la tolérance et répétition des résidus/pas. Ce seuil est un candidat d'expérience, pas une politique déjà validée.

Éviter d'avancer une solution MAXITER sous prétexte qu'elle respecte ΔPW. Le fallback doit toujours résoudre la même fenêtre, avec le même état initial et les mêmes bornes/objectif, et être certifié. Les données actuelles montrent qu'il reprend le primal préparé certifié plutôt que le primal ACADOS échoué ; cela explique pourquoi raccourcir la tentative ACADOS conserve exactement les récupérations de cette paire.

### P1 — Meilleur primal de transfert, avec sélection conservatrice

Tester d'abord les options existantes de rollout IRK sur RHO2 figé. La fonction [rollout_transferred_cycle_acados_irk](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py:12851) réutilise la carte générée, l'échelle de x/u et les paramètres par étage ; sur une fenêtre d'un cycle elle repart au nœud 0. Cela préserve le PW proposé et restaure la cohérence dynamique à partir du vrai x0.

Un rollout pur peut violer la vitesse, la borne terminale θ ou des états musculaires. Une projection composante par composante sur les bornes réintroduit ensuite des défauts. Il faut mesurer les deux objets : rollout brut, puis candidat réellement envoyé au SQP. Le sélecteur existant compare seulement un score RK4 de q/qdot après projection : il ne contrôle pas directement tous les défauts F/A/Cn ni les résidus IRK exacts. Pour une activation robuste, étendre la sélection à tous les blocs échelonnés, aux égalités du lift et au raccord ΔPW ; restaurer exactement le primal d'origine si le candidat échoue. Ne pas choisir un candidat parce que son maximum agrégé mélange N, rad et rad/s.

Si le rollout améliore F/ω mais échoue sur θ terminal, essayer une courte correction de shooting/Phase-I avec x0 fixé, bornes physiques et slew durs, et coût de proximité au PW prévu. Inclure son temps dans la latence. Éviter une projection FES seule qui laisse ω incompatible avec la nouvelle force. Une correction locale au début de l'horizon puis une correction terminale peut être moins coûteuse qu'une restauration complète ; cela reste à vérifier.

Tester ensuite `repeat` contre `lag2` seulement aux fenêtres où deux cycles certifiés sont disponibles. Au RHO2, les deux se réduisent au même prédicteur faute d'historique ; `lag2` ne peut donc pas résoudre ce premier échec par lui-même. Les RHO2/4/6 difficiles suggèrent un comportement alternant, mais la séquence seule n'établit pas une périodicité physique.

### P2 — Courbure SQP et homotopies qui terminent au problème demandé

Distinguer trois interventions :

- **Damping numérique du Hessien/pas SQP** : LM positif, adaptation du damping, ou variante de direction/globalisation, avec le coût physique inchangé et la certification finale habituelle. Tester de petites valeurs de LM adaptées au scaling, par exemple 1e−8, 1e−6, 1e−4, en conservant les autres options. Examiner `alpha`, QP et résidus avant de passer à Hessien exact, plus coûteux et potentiellement indéfini.
- **Homotopie de coût ΔPW** : résoudre λ=0,1 puis 0,01 puis 0 sur la même fenêtre, en réutilisant primal et duals entre étapes ; seul le dernier problème λ=0 certifie le résultat. Un poids permanent λ=0,1 est une modification de l'objectif et doit être évalué comme telle. Comparer le temps cumulé de toutes les étapes, pas uniquement la dernière. Le graphe courant omet le résidu quand λ=0 ; une continuation par W exige un support explicite cohérent des lignes de résidus, des facteurs LS et des métadonnées.
- **Homotopie de bornes ou de proximité** : n'assouplir les bornes que durant les étapes préparatoires, restaurer exactement la cible ±0,002 rad et les contraintes physiques avant certification. Aucune translation opportuniste de la cible finale vers le rollout. Une homotopie d'état initial peut également interpoler vers le vrai x0, mais aucun état intermédiaire ne doit être exécuté.

Les continuations proximales de commandes et de bornes existent déjà dans le runner. Elles doivent être comparées à la récupération IPOPT chaude d'environ 3–4 s : quatre étapes ACADOS de cinq SQP à 0,47 s chacune coûtent déjà environ 9,4 s. Une homotopie doit économiser plus qu'elle ne dépense.

La fixation `ΔPW[N−1]=0` est une expérience structurelle intéressante : ce dernier incrément ne commande aucun successeur physique et le terminal carrier peut valoir le dernier PW, toujours dans ses bornes. À λ=0, cela retire uniquement une liberté auxiliaire et préserve la projection du problème sur ses variables physiques ; à λ>0, cela choisit déjà le minimiseur de ce terme terminal. Il faut cependant tester la gestion d'une borne de contrôle fixe au dernier étage par Bioptim/HPIPM, la reconstruction du lift et le seed/recovery, avant de qualifier le changement de sûr numériquement. Ne pas le remplacer par une fermeture périodique.

### P3 — Réduire le coût d'une SQP sans dégrader l'intégration

La configuration actuelle IRK Gauss-Legendre 4 stages × 5 sous-pas utilise cinq Newton, sans tolérance d'arrêt Newton explicite et sans réutilisation de Jacobien (`acados_jac_reuse=0`). Les deux réglages disponibles sont `--acados-jac-reuse 1` et `--acados-newton-tol`, à tester séparément avant de changer la transcription. Réutiliser un Jacobien ou terminer Newton tôt peut conserver la même carte implicite à la tolérance demandée ; réduire arbitrairement Newton à une itération ne donne pas cette garantie. La [documentation IRK ACADOS](https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_options.AcadosOcpOptions.sim_method_newton_tol) précise que zéro désactive ce critère.

Commencer par une grille 4 stages × {5,3,2,1} sous-pas, puis 3×2 et 2×2 si les cartes et sensibilités restent précises. Conserver 50 contrôles à 50 Hz : modifier la fréquence changerait aussi le problème de commande. Comparer sur les mêmes contrôles à une intégration indépendante très précise, surtout aux premiers intervalles, aux changements de stimulation et près des bornes F=0/ω. Vérifier Jacobiennes par différences finies directionnelles aux mêmes états, car un défaut de sensibilité peut expliquer un SQP qui oscille alors que les intégrations semblent acceptables.

L'étape suivante est une transcription structurée des auxiliaires linéaires du slew hors du gros système IRK, ou une carte discrète avec leurs mises à jour analytiques exactes. Elle peut réduire le coût sans changer la physique, mais demande davantage de code et de tests. ERK/RK4-DISCRETE et Radau implicite doivent être traités comme changements numériques, avec précision validée ; ne pas comparer leur seul statut NLP et supposer une équivalence scientifique. La validation haute précision n'était pas activée sur les traces des trois pilotes.

Tester FULL_CONDENSING_HPIPM contre PARTIAL_CONDENSING_HPIPM avec Ncond={50,25,10}, séparément de l'intégrateur. Ici nx=26, nu=8, N=50 : le choix optimal dépend de la structure, des inégalités et du coût de linéarisation. Sans `time_qp_xcond/time_qp/time_sim`, aucune preuve ne permet de classer le condensing devant l'IRK.

### P4 — Warm starts primal, dual et QP : trois mécanismes différents

Le primal x/u est déjà transféré. Le mode dual `reset` remet pi/lam à zéro à chaque fenêtre. Un test `preserve` après un succès ACADOS certifié conserve les multiplicateurs au même indice de phase et peut réduire le travail QP ; c'est une approximation de la fenêtre suivante, pas une identité des multiplicateurs. Réinitialiser après fallback, changement de topologie, projection importante ou échec. Les multiplicateurs Radau IPOPT n'ont pas le même contrat que pi/lam du multiple shooting : ne pas les recopier sans transformation démontrée.

Le mode `shift` actuel décale de 50 stages dans un horizon de 50. Il annule tous les pi, et presque tous les lam (le bloc terminal n'est éventuellement copié que si les formes coïncident). Pour une fenêtre d'un cycle entièrement avancée, il n'y a aucun horizon temporel retenu : ce mode ne constitue pas le warm-start dual usuel d'un MPC avançant d'un seul échantillon.

Le QP est encore un niveau distinct : `qp_solver_warm_start=0`, premier QP non warm-starté. Comparer les niveaux 0/1/2, puis l'option premier QP, sur un primal identique, en enregistrant la somme `qp_iter`. Dans le runtime local, `nlp_solver_warm_start_first_qp_from_nlp` exige HPIPM avec partial condensing sans réduction d'horizon ; ne pas l'activer directement avec le FULL_CONDENSING actuel. La [documentation des options de premier QP](https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_options.AcadosOcpOptions.nlp_solver_warm_start_first_qp_from_nlp) et le code installé explicitent cette restriction. Réduire des itérations QP n'élimine pas nécessairement les défauts du NLP.

### P5 — Cache et parallélisme

Réutiliser les capsules ACADOS et l'objet recovery à structure identique. Un cache persistant doit inclure source/modèle, nx/nu/N, intégrateur, options structurantes, dimensions de contraintes/coût, version ACADOS/CasADi et ABI ; les tags uniques des pilotes privilégient actuellement l'isolation. Une même structure peut partager une bibliothèque chargée, mais chaque exécution concurrente doit avoir ses propres données/capsules. Préparer le recovery hors de la boucle est pertinent si le critère est la latence d'échec, pas une économie prétendue de coût total.

Le log confirme un ACADOS compilé OpenMP. Il ne démontre pas huit threads actifs pendant la résolution : les pilotes fixent OMP_NUM_THREADS et OMP_THREAD_LIMIT à 1, et `n_threads=8` est fourni aux fonctions de mapping Bioptim. Tester explicitement 1/2/4 threads natifs, avec BLAS à 1 et sans surallocation, en vérifiant l'option native et l'affinité effectives. Les linéarisations des étages d'un multiple shooting peuvent être parallélisées ; un rollout causal complet et deux RHO successifs dépendent, eux, de leur état précédent.

Des variantes indépendantes sur un checkpoint peuvent être exécutées en parallèle pour le débit de recherche, mais les chronométrages A/B doivent être isolés ou randomisés. Une course ACADOS/IPOPT sur la même fenêtre pourrait réduire la latence sous budget de CPU explicite, avec deux objets séparés et une sélection certifiée ; elle augmente le travail total et la complexité, donc passe après les pistes précédentes. La [documentation des batch solvers](https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_batch_solver.AcadosOcpBatchSolver) concerne des problèmes indépendants ; elle n'autorise pas à paralléliser naïvement l'enchaînement physique des RHO.

## 6. Expériences A/B concrètes et critères d'acceptation

Tronc commun : sauvegarder les vrais checkpoints préparés avant RHO2/4/6 difficiles et RHO3/5/7/8 faciles depuis la même trajectoire de référence. Les comparaisons d'algorithmes utilisent exactement le même x0, les mêmes bornes/paramètres/coût et le même primal de départ, sauf transformation spécifiquement étudiée. Une campagne fermée libre suit seulement les candidats retenus. Ordre alterné AB/BA, trois paires au minimum pour le screening et cinq pour les candidats finaux ; afficher médiane et dispersion, pas un p95 prétendument robuste issu de deux fenêtres.

| Priorité | A → B | Données déterminantes | Critère proposé de passage |
| --- | --- | --- | --- |
| 0 | Diagnostics actuels → production sans les trois flags | Boucle, callbacks, natif, copies, identité des traces exportées | Même certification/itérations/physique ; diminution reproductible de l'orchestration ; pas de promesse sur le natif |
| 0 | Export actuel → statistiques natives et temps additifs | Toutes tentatives, build, recovery total, sous-temps non additifs explicités | Somme additive égale à la boucle à <1 % ; aucun changement de solution/seed |
| 1 | Primal répété → IRK complet contrôles fixes et candidat sélectionné | Défauts natifs par état avant/après, bornes, temps préparation+solve | Amélioration ≥10× du défaut mécanique/F dominant sur les cas ciblés, aucune dégradation cachée d'autre bloc ; ≥2 des 3 cas difficiles convergent en ≤30 SQP et latence certifiée meilleure |
| 1 | Cap30 → cap50 ou stagnation | Temps total vers certification, recoveries supplémentaires, SQP/résidus | 100/100 RHO certifiés sur replay retenu ; gain mural net ≥15 % répété ; récupération/physique explicitement comparées |
| 2 | LM actuel → LM adapté ; un seul paramètre changé | Résidus itératifs, alpha, périodicité x/u, temps total | Rupture de la stagnation sur le même checkpoint, solution λ0 certifiée, coût canonique non dégradé de >1 % sans explication |
| 2 | λ0 direct → λ0,1→0,01→0 | Somme de tous les SQP/temps, coût physique final λ0 | Certification du dernier problème λ0 ; temps total inférieur au cap+recovery ; test fermé ensuite |
| 2 | Dernier delta libre → fixé à zéro | Lift et faisabilité physique projetée, singularités QP, source/recovery | Équivalence mathématique et tests de mapping démontrés ; succès natif sans nouvel échec HPIPM |
| 3 | IRK jac_reuse0 → 1, puis Newton tol 1e−10/1e−12 | `time_sim_ad/la`, cartes et sensibilités | Gain ≥15 % sur le natif à nombre de SQP comparable ; écarts échelonnés de carte ≤1e−7 sur le jeu critique, certification identique |
| 3 | IRK4×5 → IRK4×3 puis 4×2 | Carte sur horizon et audit indépendant sous commandes fixes | Erreur θ <2e−4 rad, ω <1e−3 rad/s, force <0,1 N, réserves relatives <1e−5 sur les cas de validation ; puis même certification et gain total ≥20 % |
| 4 | Reset → preserve certifié, puis warm QP0→1/2 | Somme QP iter, temps QP, résidus NLP et erreurs | Aucun échec ajouté, gain chaud ≥10 % répété ; ne pas confondre gain QP et gain SQP |
| 4 | FULL → PARTIAL Ncond50/25/10 | `time_qp_xcond`, QP, linéarisation et temps total | Gain total ≥15 %, mêmes seuils de certification et coût physique comparable |
| 5 | Threads natifs 1 → 2/4 | CPU effectifs, mur/CPU, surallocation, natif par SQP | Gain mural ≥15 % sur charge contrôlée, robustesse inchangée |

Les seuils d'erreur numérique du tableau sont des **seuils de screening proposés**, pas des tolérances déjà justifiées par une étude scientifique du modèle. La garde finale reste : statut solveur de succès, résidus aux tolérances cibles dont faisabilité ≤1e−5, x0 réel conservé, borne ΔPW et raccord exécuté, guard ω, angle terminal ±0,002 rad. Les variantes λ0 doivent rester λ0 au solve final. Pour les solveurs/transcriptions qui trouvent une autre solution locale, présenter les coûts physiques canoniques et la fatigue au même horizon, plutôt que demander une identité bit à bit impossible à garantir.

Ne pas mélanger le succès d'une petite carte d'intégration, la faisabilité d'un seed, la convergence du NLP d'une fenêtre, et la tenue de 100 fenêtres. Chaque niveau valide une propriété différente.

## 7. Deux changements sûrs à implémenter ensuite

**Premier choix : conserver les chronomètres déjà collectés et corriger leur export par tentative.** Ajouter les quelques sous-temps manquants, une décomposition additive indépendante des substitutions de fallback, et une rubrique séparée pour les diagnostics détaillés du primal. Les fonctions concernées sont `collect_acados_diagnostics`, la fabrication de `solver_attempt_accounting`, `run_periodic_nlp_recovery` et [le résumé du comparateur](/home/mickaelbegon/Documents/Kevin/cocofest-pedalage/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py:3307). C'est une modification d'observabilité sans modification de la physique ou de l'algorithme.

**Deuxième choix : supprimer les recalculs diagnostiques redondants sur un même primal inchangé, ou rendre le mode production explicitement accessible au lanceur.** Conserver une collecte détaillée par étape de préparation pertinente, réutiliser le dictionnaire pour l'affichage et la sérialisation, et invalider ce résultat lorsqu'une préparation modifie x/u/bornes/paramètres. Préserver les checks légers et audits finaux. L'A/B court lancé avec les flags retirés fixe le gain maximal que ce changement de reporting peut apporter sans retoucher le solveur.

Ces deux changements sont plus sûrs à intégrer immédiatement qu'un nouveau défaut de condensing, de dual warm-start ou d'intégrateur. Pour le prochain **changement visant directement la convergence**, choisir ensuite le candidat IRK avec sélection par tous les défauts après projection, après son A/B sur RHO2 figé. Garder cette voie opt-in tant que les huit fenêtres et le replay long ne sont pas validés.
