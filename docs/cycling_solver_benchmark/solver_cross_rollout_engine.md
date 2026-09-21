# Noyau d'évaluation croisée des solutions

Cette fonctionnalité rejoue les commandes sauvegardées depuis leur état entrant
avec un modèle physique commun. Elle permet d'examiner si l'ordre des coûts de
deux solutions change avec l'intégrateur. DOP853 constitue la référence continue.
Le mode n'effectue aucune réoptimisation : les nouvelles fenêtres RHO utilisent
la suite des commandes déjà enregistrées et les états restent continus.

## Entrées et provenance

L'API est `cocofest.optimization.solver_cross_rollout.run_matrix`. La CLI est
`scripts/evaluate_solver_cross_rollout.py`. Elle accepte un ou plusieurs NPZ du
projet contenant `states__*`, `controls__last_pulse_width_*` et `metadata__json`.
Les archives IPOPT avec colonnes de collocation et les archives ACADOS limitées
aux nœuds de shooting sont acceptées. Une formulation Ding réduite localement
doit exporter les cinq états physiques reconstruits par muscle.

Les paramètres Ding complets sont lus dans l'archive. Pour une archive ancienne
sans paramètres, `--model-config` est obligatoire : un JSON de paramètres
effectifs avec `provenance`, ou un modèle configuré version 1 du dépôt. Ce dernier
est résolu avec les défauts Ding installés et tous les paramètres effectifs sont
enregistrés dans le résultat. Cette déclaration utilisateur n'est pas présentée
comme une vérification rétrospective des paramètres du run historique. Une
empreinte embarquée est vérifiée lorsqu'elle existe. Le profil mécanique fourni
explicitement est identifié par SHA-256 ; son identité avec un profil absent des
métadonnées historiques ne peut pas être prouvée automatiquement.

Le support initial couvre `periodic_node`, mécanique réduite unilatérale,
commande PW physique constante par intervalle, couple extérieur constant,
formulation dynamique ou isocinétique. Les contrôles auxiliaires, les modèles
bilatéraux et les données incomplètes sont refusés explicitement. La durée du
cycle doit être indiquée par `--cycle-duration` si elle n'est pas enregistrée
dans une archive dynamique. Aucun état de physiologie n'est remis à sa valeur
optimisée entre intervalles. Les états optimisés servent à mesurer l'écart de
rejeu ; les changements du compteur de travail `E_prod` à chaque fenêtre ne
créent pas de discontinuité physique : le travail est accumulé séparément.

## Équations et conventions

Pour chaque intervalle de stimulation de durée \(h\), on utilise le temps local
\(s\in[0,h]\), le PW constant \(u\), et le passé périodique tronqué du modèle :

\[
d=e^{-h/\tau_c},\quad
H_0=d^{L-1}+[1+(K_{m,r}+0.04)d]\sum_{j=0}^{L-2}d^j,
\qquad H(s)=H_0e^{-s/\tau_c}.
\]

La somme vide vaut zéro pour \(L=1\). L'impulsion suivante ne devient active
qu'au début de l'intervalle suivant. Les équations répliquent celles de
`DingModelPulseWidthFrequencyWithFatiguePeriodicNode` et du modèle mécanique :

\[
\dot C_n=(H-C_n)/\tau_c,\qquad q=C_n/(K_m+C_n),
\]
\[
\dot F=\left[A\left(1-e^{-(u-pd_0)/pdt}\right)q
-\frac{F}{\tau_1+\tau_2q}\right](f_lf_v+f_p),
\]
\[
\dot A=-(A-A_r)/\tau_{fat}+\alpha_AF,\quad
\dot\tau_1=-(\tau_1-\tau_{1,r})/\tau_{fat}+\alpha_{\tau_1}F,\quad
\dot K_m=-(K_m-K_{m,r})/\tau_{fat}+\alpha_{K_m}F.
\]

Les relations longueur/vitesse/passives sont réévaluées à l'état mécanique
courant. Le facteur passif multiplie ici toute l'équation de force conformément
à l'implémentation du dépôt ; il ne s'agit pas d'un terme additif de force.
La mécanique dynamique utilise \(\dot\theta=\omega\) et l'accélération du profil
réduit. En isocinétique, \(\omega\) est imposé et le couple résistant est obtenu
par équilibre inverse, avec \(P=-\tau_{load}b_{ext}\omega\).

Les cartes implicites satisfont
\[
X_i=x_k+h\sum_j a_{ij}f(t_k+c_jh,X_j,u_k),\qquad
x_{k+1}=x_k+h\sum_i b_if(t_k+c_ih,X_i,u_k).
\]

`radau5` signifie **Radau IIA à cinq stages, d'ordre neuf** dans ce dépôt ; ce
n'est pas le solveur classique RADAU5 à trois stages, d'ordre cinq.
`gauss-legendre4x5` utilise quatre stages de Gauss-Legendre et cinq sous-pas.
Les équations implicites sont résolues jusqu'à un résidu normalisé inférieur à
\(10^{-9}\). Ces cartes sont des implémentations numériques indépendantes des
tableaux ; elles ne constituent pas des appels à IPOPT ou au code natif ACADOS
et ne reproduisent pas ses itérations Newton tronquées. Leurs temps d'exécution
ne doivent pas être utilisés pour classer les performances des solveurs.

## Coût physique commun

Les composantes sont intégrées comme des états supplémentaires. Avec une durée
de cycle \(T\), \(r_m=A_m/A_{r,m}\) et \(v_m=(u_m-pd_{0,m})/(u_{max}-pd_{0,m})\),

\[
J_f=\frac{w_f}{T}\int\sum_m(1-r_m)^pdt,\quad
J_\omega=\frac{w_\omega}{T}\int(\omega-\omega_*)^2dt,\quad
J_u=\frac{w_u}{T}\int\sum_mv_m^2dt,
\]
\[
J_T=w_T\sum_m(1-r_m(t_f))^2,\qquad J=J_f+J_\omega+J_u+J_T.
\]

Par défaut, \(p=2\), \(w_f=10000\) et les autres poids valent zéro, conformément
au critère physique de fatigue déjà utilisé par le diagnostic DOP853 du dépôt.
`--cost-config` permet de déclarer les champs de `CommonCost` pour toute la
matrice. Le terme terminal commun ci-dessus est explicite : il ne réimplémente
pas le proxy soft-min de réserve de certains NLP historiques. Les objectifs
natifs, pénalités, régularisations et valeurs terminales spécifiques des NLP ne
sont donc pas confondus avec ce score physique.

Les métriques rapportent la capacité, les forces, la vitesse, le travail par
cycle, l'écart de phase et, en isocinétique, les violations de couple sur une
grille explicite. Cette grille n'est pas une preuve de respect continu de toutes
les contraintes. Le classement est un ordre de scores, à interpréter avec ces
métriques. Il est supprimé si les modèles, calendriers, conditions physiques ou
états entrants diffèrent entre les sources. Rejouer depuis des états entrants
différents reste utile, mais ne mesure pas la supériorité d'une commande à
conditions identiques. Une étude de convergence de DOP853 et du maillage de
contraintes reste nécessaire si les écarts entre candidats sont très petits.

## Utilisation

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/evaluate_solver_cross_rollout.py \
  --source chemin/solution-ipopt.npz \
  --source chemin/solution-acados.npz \
  --reduced-profile chemin/reduced-cycling-fourier12.npz \
  --model-config chemin/modele.json \
  --cycle-duration 1 --cycle-start 0 --cycles 1 \
  --evaluators dop853 radau5 gauss-legendre4x5 \
  --samples-per-interval 65 --output-dir comparaison-nouvelle
```

Le dossier contient `matrix.json` et un NPZ par source/intégrateur. Un dossier
déjà utilisé est refusé pour préserver les résultats. `--rtol` et `--atol`
permettent une étude de convergence ; la tolérance absolue est multipliée par
une échelle physique enregistrée implicitement dans le choix des états
(Fmax, A-rest, Tau1-rest, Km-rest, 2π pour la mécanique).

## Vérifications effectuées

Douze tests ciblés couvrent la quadrature et l'intégration implicites d'un système
linéaire, l'intégrale analytique de fatigue, l'absence de remise à zéro, la
solution exacte du calcium, la provenance, l'incompatibilité des états entrants
et le refus de commandes hors bornes. Le champ de vecteurs est aussi comparé
directement à `system_dynamics` du modèle CasADi periodic-node natif ; les
violations de garde vitesse et de borne de phase sont testées explicitement.

Un smoke sur le seed réel `two-model-ding-100/x2/seed.npz` (un cycle de 30
stimulations, variante Triceps alpha-A ×2, mécanique dynamique) donne :

| Évaluation | Coût commun | Écart vs DOP853 | Écart de phase terminal |
|---|---:|---:|---:|
| DOP853 | 0,04470459228 | 0 | 0,001308317 rad |
| Radau à 5 stages | 0,04473633178 | +0,00003173950 | 0,001999964 rad |
| Gauss-Legendre 4×5 | 0,04470476796 | +0,00000017568 | 0,001311906 rad |

Il s'agit de **la même commande** sous trois évaluations. Ce test démontre une
différence de transcription, pas qu'un solveur fournit une meilleure commande.
Les tests et le smoke utilisent l'environnement `cocofest-rho32`.

Les références d'implémentation à auditer avec ce document sont
`cocofest/models/ding2007/ding2007_with_fatigue_periodic_node.py`,
`cocofest/models/ding2003/ding2003.py`,
`cocofest/dynamics/reduced_cycling.py`,
`cocofest/optimization/isokinetic_cycling.py` et
`high_accuracy_trace_rollout_diagnostics` dans
`examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py`.
