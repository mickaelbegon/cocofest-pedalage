# RHO isocinétique à travail mécanique imposé

## Contrat scientifique

Cette formulation cible la mécanique réduite du pédalage. Elle utilise les
mêmes profils cinématiques, projections musculaires et modèles de Ding que la
formulation dynamique, mais impose la vitesse de la manivelle par construction.
Le même OCP continu est transmis à IPOPT, MadNLP et acados; leur transcription
reste propre au backend (Radau direct ou IRK multiple-shooting).

Pour `H` tours et une vitesse négative prescrite `omega_star`, la durée et le
déplacement sont

\[
T=\frac{2\pi H}{|\omega^\star|},\qquad
\theta(T)-\theta(0)=\omega^\star T=-2\pi H.
\]

Les valeurs nominales sont `omega_star = -2*pi rad/s`, `H = 1` et donc
`T = 1 s`. Le paramètre modifiable `energy_equivalent_torque_nm` vaut `0.2`
par défaut et définit

\[
E^\star=\tau_{eq}\,2\pi H
=0.2\times2\pi=1.2566370614359172\ \mathrm J.
\]

La convention est la suivante : `tau_load > 0` est résistant,
`tau_load < 0` est assistif et la rotation est négative. Si
`qdot_crank = b_ext(theta) omega_star`, le travail net produit contre la charge
est

\[
E_{prod}(T)=-\int_0^T\tau_{load}(t)b_{ext}(\theta(t))
\omega^\star\,dt.
\]

Une phase assistive retranche donc bien du travail à une phase résistante.

## OCP résolu

### États et commandes

Pour quatre muscles de Ding avec fatigue, le vecteur d'état est

\[
x=[\{C_{n,m},F_m,A_m,\tau_{1,m},K_m\}_{m=1}^{4},
\theta,\omega,E_{prod}]^T\in\mathbb R^{23}.
\]

Les seules commandes libres sont les quatre largeurs d'impulsion

\[
u=[pw_1,pw_2,pw_3,pw_4]^T\in\mathbb R^4.
\]

Le couple de charge n'est volontairement pas une cinquième commande. Il est
la sortie d'effort de l'expérience isocinétique et est éliminé analytiquement.
Cela supprime une commande et une égalité algébrique par nœud, améliore le
conditionnement et force exactement le même équilibre dans les trois backends.

### Dynamique musculaire et cinématique

Pour chaque muscle,

\[
\dot z_m=f_{Ding,m}(z_m,pw_m,
\ell_m(\theta),v_m(\theta,\omega^\star)).
\]

La cinématique imposée vaut

\[
\dot\theta=\omega^\star,\qquad \dot\omega=0.
\]

`omega(0)=omega_star` suffit alors à imposer cette vitesse aux nœuds et aux
stages de l'intégrateur. L'angle initial est fixé et la dynamique impose
l'angle terminal exact.

### Équilibre inverse et couple résultant

Le numérateur exact de l'accélération réduite est

\[
r(\theta,\omega,F,\tau)=
\sum_m b_m(\theta)F_m+b_{ext}(\theta)\tau
-g(\theta)-c(\theta)\omega^2.
\]

L'accélération prescrite étant nulle, le couple requis est

\[
\tau_{load}^{req}=
\frac{g(\theta)+c(\theta)(\omega^\star)^2
-\sum_m b_m(\theta)F_m}{b_{ext}(\theta)}.
\]

Le profil est rejeté si `b_ext` n'est pas strictement positif ou devient
inférieur à `1e-8`. Sur le profil nominal audité :
`b_ext in [0.590876, 1.452200]`, moyenne `1.0`, rapport max/min `2.458`, et
l'erreur de cohérence de projection tangentielle vaut `1.22e-15`.

La dynamique de l'accumulateur est évaluée sans division suivie d'une
remultiplication :

\[
\dot E_{prod}
=-\tau_{load}^{req}b_{ext}\omega^\star
=\left(\sum_m b_mF_m-g-c(\omega^\star)^2\right)\omega^\star.
\]

### Bornes et contraintes

Le NLP impose :

1. les défauts de transcription des 23 états;
2. `theta(0)=theta_previous`, `omega(0)=omega_star` et `E_prod(0)=0`;
3. `E_prod(T)=E_star`;
4. les bornes physiologiques existantes des états de Ding;
5. `pd0_m <= pw_m,k <= 600 us`;
6. `tau_min <= tau_load_req(x_k) <= tau_max` aux nœuds de tir et au nœud
   terminal.

Les bornes par défaut `[-3, 3] Nm` sont des bornes d'exploration, pas encore
une calibration ergométrique universelle. Elles autorisent les phases
assistives et résistantes. Un garde intérieur de `min(0.05 Nm, 1 % de la
plage)` est appliqué aux nœuds du NLP, puis le couple est réévalué sur une
intégration DOP853 dense (65 échantillons par intervalle).

Bioptim n'expose pas aujourd'hui les états SX intermédiaires de collocation à
cette contrainte personnalisée. La borne analytique est donc imposée aux nœuds
de tir et au terminal, puis vérifiée après résolution sur les stages de la
transcription et sur le replay dense. Le replay est une vérification pratique,
pas une preuve d'extremum continu entre ses échantillons.

Avec `N` intervalles, la séquence de commande libre contient `4N` largeurs
d'impulsion et le travail ajoute une contrainte terminale scalaire. `theta`,
`omega`, `E_prod` et `tau_load_req` ne rajoutent aucune commande libre.

### Objectif

L'objectif sélectionné par le benchmark reste celui du projet : fatigue,
force, charge de stimulation, ou combinaison pondérée. Le cas nominal fatigue
minimise notamment la fonction de fatigue existante avec son poids `1e4`.
Il n'y a pas de pénalité artificielle sur le couple : le profil de couple est
déterminé par les stimulations, l'équilibre inverse et le travail global.

## RHO, initialisation et caches

À la transition entre deux fenêtres, les états physiologiques suivent la
politique de transfert existante. `theta` continue d'un tour, `omega` demeure
`omega_star`, tandis que `E_prod` est remis à zéro et doit atteindre `E_star`
dans chaque nouvelle fenêtre. Ce saut volontaire d'énergie ne représente pas
une discontinuité physique : `E_prod` mesure le travail local de la fenêtre.

Le seed isocinétique construit `theta` linéaire, `omega` constant et
`E_prod` linéaire. Le raffinement IPOPT optionnel d'acados résout la formulation
isocinétique cible, jamais l'ancien OCP à couple constant. Les signatures des
caches de raffinement, des seeds acados et des seeds communs incluent la
formulation, `omega_star`, `tau_eq`, `tau_min` et `tau_max`; un seed physique
incompatible est rejeté.

## Certification numérique

Deux niveaux sont distingués explicitement.

Le certificat du NLP exige simultanément : succès du solveur, valeurs finies,
erreur de vitesse `<= 1e-9 rad/s`, erreur d'angle `<= 1e-8 rad`, résidu réduit
`<= 1e-6`, erreur terminale et quadrature de transcription du travail
`<= 1e-6 J`, et violation des bornes de couple `<= 1e-8 Nm`.

Le replay physique indépendant DOP853 (`rtol=1e-11`, `atol=1e-13`) exige en
plus les bornes de couple sur la grille dense et une erreur de travail
`<= 0.02 J`. Cette tolérance de smoke test vaut 1.59 % du travail nominal. Elle
ne doit pas être présentée comme une précision continue de `1e-6 J` : avec la
discrétisation nominale, les erreurs mesurées sont `7.64e-3 J` au degré Radau 3
et `1.39e-3 J` au degré 5. Une étude de convergence temporelle est nécessaire
avant d'abaisser ce seuil.

L'équilibre inverse a aussi été comparé à la dynamique mécanique complète sur
le profil nominal : erreur maximale `1.17e-5 Nm`. Le seuil de `1e-6` certifie
donc l'équilibre du modèle réduit; une revendication d'équivalence au modèle
complet doit employer au moins `2e-5 Nm` ou raffiner le profil réduit.

## Lancement des benchmarks

La référence isocinétique est Radau-5 pour IPOPT et MadNLP. acados conserve
son IRK natif, mais son raffinement IPOPT par fenêtre utilise aussi Radau-5 et
MA57. Le pilote commun lance les mêmes paramètres physiques sur les trois
backends :

```bash
python .github/scripts/run_benchmarks.py \
  --formulation isokinetic \
  --cases ipopt-radau5 madnlp-mumps-radau5 acados-irk \
  --cycles 5 \
  --ipopt-hsl-library /chemin/vers/libhsl.so \
  --energy-equivalent-torque 0.2 \
  --isokinetic-omega=-6.283185307179586 \
  --load-torque-min -3 \
  --load-torque-max 3
```

L'option explicite est prioritaire. Exporter une seule fois
`IPOPT_HSL_LIBRARY=/chemin/vers/libhsl.so` rend aussi cette bibliothèque le
défaut de toutes les CLI IPOPT : benchmark RHO, raffinement IPOPT d'acados et
workflows full-horizon.

Pour un diagnostic rapide, le même jeu d'options est disponible dans
`cycling_fes_solver_comparison.py`. Les JSON contiennent les paramètres, les
traces `theta/omega/E_prod/tau_load`, les diagnostics solveur et les audits de
certification. Les répertoires isocinétiques portent un suffixe comprenant le
couple équivalent afin de ne pas écraser les campagnes dynamiques.

Le benchmark de validation à cinq RHO, 30 stimulations/tour et bornes
`[-3,3] Nm` donne actuellement :

| Backend | RHO certifiés | Itérations | Temps solveur total | Objectif cumulé | Erreur DOP853 max |
|---|---:|---|---:|---:|---:|
| IPOPT/MA57, Radau-5 | 5/5 | 59, 61, 48, 49, 48 | 5.055 s | 0.361551 | 1.307e-3 J |
| MadNLP/MUMPS, Radau-5, tol. 1e-8 | 5/5 | 55, 44, 49, 48, 50 | 14.863 s | 137.399646 | 2.149e-3 J |
| acados/IRK + raffinement IPOPT | 5/5 | 2, 3, 3, 3, 3 | 0.753 s | 0.352908 | 8.379e-6 J |

Les temps acados ci-dessus sont les temps du solveur cible; les raffinements
IPOPT de préparation ne sont pas inclus et cette ligne reste une référence
Radau-3 de transition jusqu'à la fin de sa campagne Radau-5/MA57. Les deux
campagnes NLP Radau-5 satisfont le certificat de transcription (`E <= 1.47e-13
J`, résidu réduit `< 5e-14`, erreur de vitesse `< 4e-13 rad/s`) et le replay
DOP853. L'objectif MadNLP beaucoup plus élevé
malgré la faisabilité signale un bassin local différent et interdit de conclure
à l'équivalence des optima sur cette seule campagne.

## Plan de match en sept étapes et portes de validation

1. **Contrat et signes.** Centraliser les paramètres et formules. Porte : tests
   analytiques de signe, durée, angle et `E_star`; vérification indépendante
   qu'un couple constant de `0.2 Nm` donne exactement `0.2*2*pi J`.
2. **Profil réduit.** Vérifier `b_ext`, les projections et l'équivalence au
   modèle complet. Porte : positivité/marge de `b_ext`, cohérence tangentielle
   `1.22e-15`, écart complet/réduit `1.17e-5 Nm` documenté.
3. **OCP isocinétique.** Ajouter `E_prod`, imposer la cinématique et éliminer
   analytiquement le couple sans altérer le mode dynamique. Porte : tests de
   dimensions, invariance à l'état `omega`, résidu d'équilibre et dérivée
   d'énergie symbolique/numérique.
4. **RHO et seeds.** Construire bornes/initialisations et remettre l'énergie à
   zéro à chaque fenêtre. Porte : test de raccord des états physiologiques,
   angle/vitesse, saut volontaire d'énergie et résolution de deux RHO.
5. **Interface solveurs.** Exposer une CLI commune, isoler les sorties et
   signer les caches avec tous les paramètres physiques. Porte : tests du
   pilote et rejet d'un seed incompatible.
6. **Certification backend.** Lancer un smoke test IPOPT, MadNLP et acados sur
   le même primal; appliquer le certificat NLP et le replay DOP853. Une étape
   n'est verte que si le statut solveur et les deux audits sont verts; un
   statut acados `MINSTEP` reste un échec même avec un seed physiquement
   faisable.
7. **Benchmark apparié et sensibilité.** Lancer au moins cinq RHO et comparer
   faisabilité, temps, itérations, objectif, travail et extrema du couple.
   Répéter avec les bornes de couple resserrées (notamment `[-1,1]`, `[-2,2]`,
   `[-3,3]`) en conservant Radau-5 avant toute conclusion scientifique
   ou campagne longue.
