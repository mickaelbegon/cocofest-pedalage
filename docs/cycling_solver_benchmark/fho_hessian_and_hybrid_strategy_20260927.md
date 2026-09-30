# FHO MX : accélération de Hessienne et protocole hybride — audit du 27 septembre 2026

## Décision

Le FHO reste un NLP **MX interprété, non compilé**. Le meilleur levier déjà
validé est le `ThreadMap` natif de Bioptim/CasADi avec `n_threads=12` et
`OMP/BLAS=1`. Le protocole hybride actuel « 25 itérations L-BFGS dans un
nouveau processus, puis redémarrage » ne doit pas être utilisé pour accélérer
une campagne : il perd la mémoire L-BFGS à chaque bloc et a été plus lent que
la résolution exacte directe sur FHO65 et FHO88.

Une Hessienne exacte par paquets reste une expérience prometteuse, mais elle
n'est pas encore intégrable au FHO réel : la partition doit être extraite
*après* le `shake` Bioptim et auditée contre le NLP final. La compilation de
petits noyaux est techniquement possible sans compiler le graphe FHO global,
mais seulement si les dérivées exactes sont elles-mêmes fournies par les
noyaux compilés.

## Mesures de référence

| Cas, MA57 | Itérations | IPOPT mur | `nlp_hess_l` | Part Hessienne |
|---|---:|---:|---:|---:|
| FHO65 exact direct | 131 | 755.56 s | 582.33 s | 77.1 % |
| FHO88 exact direct | 207 | 1632.44 s | 1234.48 s | 75.6 % |
| FHO91 exact direct | 188 | 1577.86 s | 1177.48 s | 74.6 % |

FHO91 confirme que le coût dominant est l'AD de la Hessienne, non la
factorisation KKT. Un gain de 30 % sur `nlp_hess_l` aurait donc un plafond
arithmétique d'environ 22.4 % sur le temps IPOPT FHO91, avant surcoûts de
construction et d'assemblage.

Le benchmark FHO65 contrôlé (même seed, même MX, même MA57, 30 itérations)
montre que Bioptim/CasADi parallélise déjà une partie de la Hessienne native :

| Threads Bioptim | H moyenne/appel | Solve IPOPT | Préparation + nlpsol |
|---:|---:|---:|---:|
| 1 | 9.376 s | 396.49 s | 174.62 s |
| 8 | 5.135 s | 208.12 s | 357.45 s |
| 12 | 4.823 s | 193.79 s | 319.60 s |

Les sorties 1/8/12 ont exactement les mêmes vecteurs initiaux et finaux dans
ce test, ainsi que la même sparsité (265223 variables, 259414 contraintes,
1072504 non-zéros Hessienne triangulaire). Le passage 8 -> 12 apporte encore
6.1 % sur la Hessienne, mais le rendement est décroissant. La voie
`ThreadMap=12`, bibliothèques numériques à un thread et affinité dédiée reste
donc la base de comparaison de toute nouvelle méthode.

Sources : `fho65_threaded_solve_12cores_20260925.md` et
`fho-rho90-terminal-halfstep-0p30-20260926/full-horizon-0091/chance-1/result.json`.

## Diagnostic du protocole L-BFGS -> exact existant

Le script `scripts/run_hybrid_fho_ipopt.py` exécute un nouveau processus
IPOPT par bloc de 25 itérations. Le point primal est exporté, mais la mémoire
L-BFGS interne est nécessairement recréée à chaque processus. Ce n'est donc
pas une phase L-BFGS de 3*N itérations : c'est une suite de redémarrages
quasi-Newton, avec en plus la construction répétée du FHO.

| Cas | Exact direct | L-BFGS redémarré | Exact après L-BFGS | Total hybride | Résultat |
|---|---:|---:|---:|---:|---|
| FHO65 | 755.56 s | 670.51 s (195 it.) | 958.77 s (137 it.) | 1629.28 s | certifié, 2.16x plus lent |
| FHO88 | 1632.44 s | 802.47 s (264 it.) | 1571.93 s (203 it.) | 2374.41 s | certifié, 1.45x plus lent |

Sur FHO65, la violation auditée du dernier L-BFGS atteint 496.66; sur FHO88,
elle atteint 1297.69. Ces valeurs sont très éloignées du seuil de transition
`1e-4`. Elles montrent que, sur ces graines, L-BFGS ne restaure pas la
faisabilité et ne doit jamais être assimilé à un pré-certificat.

Les objectifs finaux exacts sont néanmoins cohérents (FHO65 : 25734.0591785
direct, 25734.0591776 après hybride) : l'exact restaure bien la solution,
mais sans avantage de durée.

## Protocole hybride recommandé

1. **Pré-audit de la graine.** Mesurer le défaut de dynamique, continuité et
   bornes avant de choisir une phase rapide. Si le défaut est supérieur à un
   seuil de restauration à calibrer (initialement `1e-2`), démarrer directement
   en Hessienne exacte : le cas FHO65/88 indique que L-BFGS ne corrige pas ce
   type de défaut.
2. **Une seule invocation L-BFGS**, si et seulement si la graine est déjà
   proche de faisable. Budget initial : `min(3*N, 300)` itérations, historique
   10--20. Ne pas la découper en sous-processus ; la mémoire L-BFGS doit vivre
   pendant toute la phase.
3. **Basculer vers un nouveau solve exact** depuis la dernière itération
   finie. Le primale est transmis; les duaux ne sont transmis que si l'ordre
   des contraintes du NLP est identique et que leur archive est auditée.
   L'unique certificat est le solve exact et son audit indépendant.
4. **Arrêt anticipé optionnel, mais interne à IPOPT.** Pour exploiter le
   plateau objectif 20--30 itérations, il faut un callback `intermediate`
   dans *la même* invocation IPOPT. Le découpage CLI actuel ne le réalise pas.
   Tant que Bioptim ne l'expose pas proprement, retenir le budget fixe et
   conserver la trace des itérations.
5. **Garde-fou.** Si la violation auditée augmente sur deux fenêtres ou reste
   > `1e-2` après 50 itérations, arrêter la phase L-BFGS, étiqueter la graine
   non certifiée et lancer l'exact immédiatement. Ce n'est pas un échec
   physiologique.

Cette variante doit être comparée d'abord sur FHO65 avec la même graine et
les mêmes CPU. Elle n'est retenue que si le temps *L-BFGS + exact* baisse,
avec objectif et audit final identiques. Les résultats existants imposent de
désactiver par défaut le mode hybride redémarré.

## Compilation locale sans compiler le FHO global

### Correct et exact

On peut laisser le graphe FHO agrégé en MX interprété et compiler seulement
un noyau local répété, par exemple un stage ou une contribution de Lagrangien
locale. La voie exacte est :

1. construire le noyau local symbolique (SX si le stage le permet) ;
2. dériver **avant** codegen ses sorties nécessaires : `f_local`, `g_local`,
   `grad_f_local`, `jac_g_local`, ou directement
   `hess(sigma*f_local + lambda' g_local)` ;
3. générer/compiler ces fonctions, les charger en `casadi.external`, puis les
   appeler numériquement dans un callback `hess_lag` qui assemble les blocs
   creux ;
4. vérifier valeurs, sparsité, produits Hessienne-vecteur et certificat
   contre `nlp_hess_l` natif à plusieurs `(x, lambda, sigma)`.

Un test local CasADi 3.7.2 a généré et chargé un noyau de Hessienne exacte ;
la différence entre interprété et `.so` est exactement `0.0`. IPOPT n'a pas à
différencier une seconde fois ce noyau si celui-ci est consommé comme sortie
numérique du callback `hess_lag`.

### Incorrect ou insuffisant

- Compiler uniquement le **primal** puis demander à CasADi de différencier
  l'externe ne préserve pas automatiquement l'AD exacte. Il faudrait fournir
  les fonctions forward/reverse compatibles attendues par CasADi; activer les
  différences finies (`enable_fd`) rendrait la Hessienne inexacte et est exclu
  pour un FHO certifié.
- Un bloc-diagonal sans tous les termes de frontière/globalement couplés n'est
  pas la Hessienne du Lagrangien. Les continuités affines ont Hessienne nulle,
  mais les objectifs multi-noeuds, paramètres globaux et termes non linéaires
  traversant une frontière doivent être attribués une fois à un paquet.
- `--ipopt-c-compile-callback nlp_hess_l` compile la callback globale entière.
  C'est un benchmark utile mais ce n'est pas « FHO MX non compilé » et il
  risque un coût/une taille de compilation prohibitifs à 90 cycles. Ne pas le
  présenter comme la voie locale.

La libérée partagée doit être re-entrante avant un `Function.map(..., "thread")`;
la première mesure fixe `ThreadMap=12`, `OMP/BLAS=1`, sans threads HSL
supplémentaires. Si le stage compilé ne donne pas au moins 20 % de baisse de
`nlp_hess_l` *après assemblage*, cette voie est abandonnée.

## Hessienne par paquets : ordre expérimental

1. Partir de la Hessienne native threadée à 12 cœurs, déjà exacte et auditée.
2. Extraire un registre post-`shake` Bioptim pour construire les paquets ; ne
   jamais partitionner une expression pré-transformée et l'injecter dans un
   NLP transformé sans preuve d'équivalence.
3. Tester d'abord une évaluation, sans solve : valeurs et sparsité pour seed,
   solution et points perturbés; `lam_f=0`, multiplicateurs aléatoires et
   produits Hessienne-vecteur.
4. Essayer des tailles 1 puis 5 cycles. La taille 10 diminue le parallélisme
   (7 paquets seulement sur FHO65) et s'est révélée plus lente sur le
   microbenchmark synthétique.
5. Faire seulement alors un solve FHO65, MA57, seed/affinité/tolérances
   inchangés. Retenir la méthode si certificat identique, `nlp_hess_l` baisse
   >=20 %, temps IPOPT total baisse et pic RSS reste acceptable.

Le prototype mathématique se trouve dans
`cocofest/optimization/packet_hessian.py`; il est correct sur les tests
synthétiques mais n'est volontairement pas branché à une campagne.

### Prototype compilé, contrat IPOPT vérifié

Le prototype a maintenant une interface minimale pour le chemin compilé :
`HessianPacket.local_hessian_kernel` accepte une fonction CasADi externe déjà
différentiée, de signature
`(z, p, sigma, lambda) -> valeurs(triu(H_local))`. Le constructeur conserve
le noyau symbolique seulement pour établir les coordonnées creuses de référence
et ne différentie jamais la fonction externe. Il est donc possible de garder
le NLP global sous forme **MX interprétée**, tandis que les évaluations locales
répétées sont chargées depuis une bibliothèque `.so`.

Le test `test_ipopt_accepts_compiled_local_exact_hessian_without_compiling_global_mx`
génère ce noyau local en C (`gcc -O3`), le recharge avec `casadi.external`,
l'assemble dans `hess_lag` et le passe à IPOPT. Sur le petit NLP MX de
référence, solution et objectif coïncident avec la Hessienne monolithique à
`1e-10` et `1e-12`, respectivement. Ce test valide le **contrat** callback,
l'ordre des valeurs triangulaires et l'absence de différentiation secondaire
de l'externe ; il ne mesure pas encore un FHO.

Le bloqueur restant est volontairement explicite : Bioptim ne publie pas un
registre post-`shake` de termes locaux avec les indices du vecteur NLP final.
Sans cette partition auditable, un callback de paquets pourrait omettre ou
dupliquer un terme. La prochaine étape sûre est donc une instrumentation sans
solve de FHO65 qui exporte ce registre, puis compare à `nlp_hess_l` les
valeurs, la sparsité et les produits Hessienne-vecteur avant tout A/B IPOPT.

## Solveurs linéaires

MA86 n'est pas un remède à une Hessienne lente : dans le test hybride FHO65,
sa restauration a échoué après 1562.91 s (1141.56 s de Hessienne). MA97 n'a
pas produit de résultat, seulement la commande préparée. Il n'existe donc pas
encore de benchmark MA57/MA86/MA97 comparable permettant de définir un seuil
de bascule. Tant que la Hessienne coûte ~75 % du solve, prioriser son
évaluation; ensuite seulement refaire un A/B de factorisation exacte à NLP,
seed, itérations, CPU et HSL identiques.
