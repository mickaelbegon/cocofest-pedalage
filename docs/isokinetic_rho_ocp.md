# RHO isocinétique à travail mécanique imposé

## Statut et périmètre

Ce document fixe le contrat scientifique et logiciel de la nouvelle simulation
RHO isocinétique. La première implémentation cible la mécanique réduite, qui
dispose déjà d'une coordonnée physique de manivelle `theta`, de sa vitesse
`omega` et d'une projection validée des forces musculaires sur la manivelle.
Le même OCP devra être transmis sans changement de modèle à IPOPT, MadNLP et
acados.

Une fenêtre RHO nominale représente un tour en une seconde :

\[
\omega^\star=-2\pi\ \mathrm{rad\,s^{-1}},\qquad
T=1\ \mathrm{s},\qquad
\theta(T)-\theta(0)=-2\pi.
\]

Les fenêtres de plusieurs tours restent possibles. Pour `H` tours, `T=H` et
le travail cible est multiplié par `H`.

## Convention de signe et travail demandé

La manivelle tourne dans le sens négatif. Le couple externe nodal
`tau_load` suit la convention actuelle du projet :

- `tau_load > 0` : couple résistant, donc puissance mécanique externe
  `tau_load * b_ext(theta) * omega < 0`;
- `tau_load < 0` : couple assistif, donc puissance mécanique externe
  `tau_load * b_ext(theta) * omega > 0`.

Le couple est appliqué sur le DDL de manivelle du modèle complet. Sa vitesse
est `qdot_crank = b_ext(theta) * omega`, où `b_ext` est le coefficient de
projection déjà présent dans le profil réduit. Le travail net produit contre la
charge est donc défini exactement par

\[
E_{\mathrm{prod}}(T)
=-\int_0^T \tau_{\mathrm{load}}(t)
\,b_{\mathrm{ext}}(\theta(t))\,\omega^\star\,\mathrm dt.
\]

Comme le DDL de manivelle accomplit exactement un tour, une résistance
constante de `0.2 N.m` impose

\[
E^\star=0.2\times 2\pi=1.2566370614\ \mathrm J.
\]

Le paramètre utilisateur sera le couple équivalent
`energy_equivalent_torque_nm`, de valeur nominale `0.2`. Le code calculera
`E_target = energy_equivalent_torque_nm * 2*pi * H`; la constante en joules ne
sera pas dupliquée dans les scripts. Cette égalité porte sur le travail **net** :
une phase assistive retranche donc de l'énergie à une phase résistante. C'est
le sens retenu pour « production globale ».

## Variables de décision

Pour les quatre muscles actuels, le vecteur d'état est

\[
x=\left[
\{C_{n,m},F_m,A_m,\tau_{1,m},K_{m}\}_{m=1}^{4},
\theta,\omega,E_{\mathrm{prod}}
\right]^\mathsf T\in\mathbb R^{23}.
\]

Les vingt premiers états et leurs lois de Ding restent inchangés. Le vecteur
de commande est

\[
u=\left[pw_1,pw_2,pw_3,pw_4,\tau_{\mathrm{load}}\right]^\mathsf T
\in\mathbb R^5.
\]

Les largeurs d'impulsion et le couple de charge sont constants par intervalle
de tir. Les limites du couple seront explicites et communes aux trois solveurs,
par exemple `[-1, +1] N.m` au départ; elles ne devront jamais être déduites du
travail cible. Autoriser les deux signes est important, mais des bornes physiques
finies sont nécessaires pour empêcher une alternance assistance/résistance
arbitrairement grande.

## Dynamique différentielle et équilibre isocinétique

Pour chaque muscle, les dynamiques FES/fatigue existantes s'écrivent

\[
\dot z_m=f_{\mathrm{Ding},m}
(z_m,pw_m,\ell_m(\theta),v_m(\theta,\omega^\star)).
\]

La cinématique et l'accumulateur de travail sont

\[
\dot\theta=\omega^\star,\qquad
\dot\omega=0,\qquad
\dot E_{\mathrm{prod}}=-\tau_{\mathrm{load}}
b_{\mathrm{ext}}(\theta)\omega^\star.
\]

La vitesse ne sera donc pas obtenue en intégrant la dynamique directe : elle
est imposée par construction, y compris entre les nœuds de tir d'un intégrateur
IRK. La dynamique mécanique réduite actuelle devient une égalité d'équilibre
inverse à chaque nœud :

\[
g_{\tau,k}=
\sum_m b_m(\theta_k)F_{m,k}
+b_{\mathrm{ext}}(\theta_k)\tau_{\mathrm{load},k}
-g(\theta_k)
-c(\theta_k)(\omega^\star)^2=0.
\]

Il s'agit exactement du numérateur de
`ReducedCyclingDynamics.casadi_acceleration`; aucune seconde approximation
mécanique ne doit être introduite. L'inertie effective n'apparaît plus dans
l'égalité puisque l'accélération prescrite est nulle. Une petite valeur de
`b_ext` devra provoquer un rejet explicite du profil plutôt qu'un couple mal
conditionné.

## Contraintes

Les contraintes du NLP sont les suivantes.

1. Défauts de transcription des vingt dynamiques de Ding, de `theta`, de
   `omega` et de `E_prod`.
2. `omega_k = -2*pi rad/s` à tous les nœuds d'état. La dynamique
   `dot(omega)=0` garantit également cette vitesse à l'intérieur des
   intervalles.
3. `theta_0` fixé à la fin du RHO certifié précédent et
   `theta_N = theta_0 - 2*pi*H`. La trajectoire isocinétique rend les anciennes
   marges de vitesse et d'angle terminal inutiles.
4. Équilibre `g_tau,k = 0` à chaque nœud portant un couple. Pour une
   collocation, l'équilibre sera aussi évalué aux points de collocation si le
   couple y est disponible; sinon l'audit haute précision bornera le résidu
   interpolé entre deux tirs.
5. Bornes physiologiques existantes de tous les états de Ding et bornes
   `pd0_m <= pw_m,k <= 600 us`.
6. Bornes nodales communes
   `tau_assist_min <= tau_load,k <= tau_resist_max`.
7. `E_prod(0)=0` et
   `E_prod(T)=energy_equivalent_torque_nm*2*pi*H`. L'état d'énergie est remis
   à zéro à chaque nouvelle fenêtre RHO; il n'est pas transféré comme un
   état de fatigue.

L'accumulateur transforme la contrainte intégrale dense en dynamique locale et
borne terminale. Cette forme conserve la structure creuse pour IPOPT/MadNLP et
est directement représentable par acados.

## Fonction objectif

L'objectif scientifique par défaut demeure la fatigue existante :

\[
J_{\mathrm{fatigue}}=
10^4\int_0^T\sum_m
\left(1-\frac{A_m(t)}{A_{m,0}}\right)^2\mathrm dt.
\]

Les variantes force et charge de stimulation existantes restent sélectionnables.
Un faible terme optionnel de régularisation du couple peut être ajouté pour
stabiliser une solution non unique,

\[
J_\tau=w_\tau\int_0^T
(\tau_{\mathrm{load}}-\bar\tau)^2\mathrm dt,
\]

mais `w_tau=0` sera le cas scientifique de référence. Toute valeur non nulle
devra apparaître dans les métadonnées et dans les tableaux de benchmark.

## Transfert entre RHO et amorçage

Les états physiologiques conservent la politique actuelle : les états rapides
sont décalés cycliquement, les états de fatigue sont prolongés continûment et
le premier nœud est raccordé au dernier état certifié. `theta` est décalé de
`-2*pi`, `omega` reste exactement à `-2*pi`, et `E_prod` est recréé sur
`[0,E_target]` à chaque RHO.

Le nouveau couple est initialisé à partir de l'équilibre inverse du seed :

\[
\tau_{\mathrm{load},k}^{(0)}=
\frac{g(\theta_k)+c(\theta_k)(\omega^\star)^2
-\sum_m b_m(\theta_k)F_{m,k}}
{b_{\mathrm{ext}}(\theta_k)}.
\]

Cette trajectoire est projetée dans les bornes de couple. L'état d'énergie
initial est obtenu par quadrature de ce couple. Si son terminal diffère de la
cible, le solveur adapte les stimulations et le profil de charge; les trois
solveurs partent toutefois du même primal. La préparation IPOPT optionnelle
d'acados devra elle aussi résoudre la formulation isocinétique, pas l'ancien
OCP à couple constant.

## Architecture d'implémentation

La logique scientifique sera extraite dans un petit module réutilisable, au
lieu d'ajouter des branches spécifiques aux solveurs dans le grand script de
comparaison. Ce module contiendra :

- la configuration validée (`omega`, couple équivalent, bornes de couple);
- le calcul unique de `E_target` et des conventions de signe;
- l'expression d'équilibre inverse et le calcul du seed de couple;
- les audits indépendants de vitesse, d'équilibre et de travail.

`ReducedFesCyclingModel` recevra un mode isocinétique qui ajoute le contrôle de
couple et l'accumulateur, tout en conservant le mode dynamique actuel par
défaut. `prepare_nmpc`, les bornes et le transfert RHO appelleront des
sous-fonctions dédiées. Le script de comparaison ne fera que traduire les
arguments CLI en cette configuration.

## Interface de benchmark visée

Une seule commande doit lancer exactement la même formulation :

```bash
python .github/scripts/run_benchmarks.py \
  --formulation isokinetic \
  --cases ipopt madnlp-mumps acados-irk \
  --cycles 5 \
  --energy-equivalent-torque 0.2 \
  --isokinetic-omega=-6.283185307179586 \
  --load-torque-min -1.0 \
  --load-torque-max 1.0
```

Le pilote transmettra ces options au script shell et au chemin acados. Le nom
du répertoire de sortie inclura la formulation et le couple équivalent afin de
ne jamais écraser les campagnes dynamiques actuelles. Les scripts existants de
résumé resteront la source unique des temps, itérations et objectifs.

Chaque résultat JSON et chaque seed NPZ enregistrera au minimum : formulation,
vitesse cible, travail cible, bornes de couple, trace `tau_load`, trace
`E_prod`, erreur maximale de vitesse, résidu maximal d'équilibre, travail
recalculé par quadrature et erreur terminale d'énergie.

## Critères d'acceptation

Un RHO n'est certifié que si le solveur annonce le succès et si un audit
indépendant confirme simultanément :

- `max(abs(omega-omega_target)) <= 1e-9 rad/s`;
- erreur angulaire terminale `<= 1e-8 rad`;
- résidu nodal d'équilibre `<= 1e-6` dans les unités mises à l'échelle;
- erreur de travail terminale et erreur de quadrature `<= 1e-6 J`;
- respect des bornes de couple, des PW et des états physiologiques;
- aucune valeur non finie.

Les seuils seront des constantes partagées par les trois solveurs. Les
benchmarks rapporteront aussi l'accord inter-solveur sur l'objectif, le travail
et le profil de couple, sans exiger que les couples nodaux soient identiques si
l'optimum n'est pas unique.

## Plan de réalisation

1. Ajouter et tester le module de configuration, les conventions de signe, la
   cible de travail et l'équilibre inverse.
2. Étendre le modèle réduit avec `tau_load`, `E_prod`, la dynamique
   isocinétique et la contrainte d'équilibre, sans modifier le mode existant.
3. Extraire les constructeurs de bornes/initialisations isocinétiques et adapter
   le transfert RHO pour remettre l'énergie à zéro.
4. Ajouter l'interface CLI et les métadonnées, puis brancher les trois chemins
   IPOPT, MadNLP et acados sur cette interface unique.
5. Ajouter les audits post-solve et les tests unitaires, puis un smoke test à
   un RHO pour chaque backend.
6. Exécuter un benchmark apparié de cinq RHO avant toute campagne longue et
   comparer faisabilité, objectif, temps, itérations et trajectoires.
