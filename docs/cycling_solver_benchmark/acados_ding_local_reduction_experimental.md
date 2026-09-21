# Réduction locale Ding pour ACADOS : faisabilité et prototype isolé

Audit du 13 septembre 2026. Conclusion : **la réduction est mathématiquement
transposable à la carte IRK d’ACADOS, mais l’adaptateur IPOPT actuel ne peut pas
être branché sur ACADOS**. Un prototype de carte native d’un muscle a été écrit
dans `scripts/experimental_acados_ding_local_map.py` et validé par un smoke
natif GL4×5. Aucun gain ACADOS nouveau n’est mesuré. Aucune source de
production existante n’a été modifiée.

## Ce qu’expose effectivement la pile locale

Les sources inspectées sont celles de `.benchmark-deps/bioptim` (package
éditable déclaré 3.5.0) et du Python ACADOS installé dans l’environnement
`cocofest-rho32`. Il faut se fier à ces sources : des documents historiques
décrivent encore Bioptim 3.4, et le nom de version du paquet ACADOS ne résume
pas tous les ajouts présents dans le code local.

- `bioptim/interfaces/acados_interface.py:334` exporte `x`, `xdot`, `u`,
  `f_impl_expr = xdot - f_expl_expr` et les paramètres numériques. L’interface
  refuse explicitement les états algébriques Bioptim. Déplacer les trois états
  éliminés vers `z` n’est donc pas une option disponible par simple réglage.
- À la ligne 419, `nx` est réassigné à `parameters.shape + states.shape`.
  Les bornes utilisent une identité sur tous les états (lignes 587–605), les
  coûts LS gardent les dimensions originales, et l’extraction à partir de
  la ligne 1103 réassemble le vecteur primal attendu par Bioptim. Retirer des
  composantes dans `f_impl_expr` seul violerait ce contrat dimensionnel.
- Le patch existant `patch_bioptim_acados_interface()` dans
  `examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py`
  expose déjà le temps local IRK et `periodic_calcium` via les paramètres
  d’étage. Il corrige aussi le scaling des dérivées et des commandes. Ces
  transformations doivent être conservées dans un futur export réduit.
- ACADOS accepte `GAUSS_LEGENDRE` et `GAUSS_RADAU_IIA`. Le noyau local
  `sim_irk_integrator.c` gère les tableaux et la boucle de Newton; les stages
  IRK sont internes au simulateur, déjà éliminés du NLP de multiple shooting.
  Ce ne sont pas les milliers de variables de collocation explicites que
  l’adaptateur IPOPT supprime.
- Le runner possède déjà une carte `DISCRETE` RK4 et un rollout avec
  `AcadosSimSolver`. Ce sont des points d’accès réels, mais aucun ne remplace
  automatiquement le contrat états/bornes/coûts/seed/extraction.

## Conserver le bon opérateur discret

La référence ACADOS inspectée utilise **Gauss-Legendre 4 stages × 5 sous-pas**,
alors que le pilote IPOPT emploie Radau à cinq stages. Réutiliser les offsets
Radau dans GL4×5 changerait le problème discret. Passer ACADOS à Radau-5
changerait également son intégration de référence; cela ne rendrait pas à lui
seul identiques les coûts, contraintes et transcriptions IPOPT/ACADOS.

Pour un sous-pas de longueur h, de tableau B et poids b, poser
`d_y = y - c_y A`, `c_y = alpha_y/alpha_A`, pour y=Tau1,Km. On a
`d_y' = -(d_y-d_rest)/tau_fat`, indépendamment de F, de la mécanique et de PW.
Le calcium est également indépendant des décisions dans le modèle
`periodic_node` à calendrier, intensité et paramètres fixes. Pour chacune de
ces trois variables v, écrire `v' = -v/tau + r(t)` puis calculer

```
(I + h/tau B) V = 1 v0 + h B r(t_stages)
v_next = v0 + h bᵀ (r(t_stages) - V/tau).
```

Le second calcul est essentiel en Gauss-Legendre : **le dernier stage n’est
pas le nœud final**. En Radau IIA, la propriété stiffly accurate rend les deux
équivalents. On propage les offsets sous-pas par sous-pas, puis intervalle par
intervalle depuis le véritable état initial de la fenêtre. On ne remplace
pas cet état entrant par un point fixe périodique.

La reconstruction locale `Tau1=c_tau A+d_tau`, `Km=c_km A+d_km` conserve F et A,
donc leur couplage non linéaire avec les états mécaniques. Avec x0 fixé, ces
trois profils sont calculables avant le solve. Cette équivalence concerne
les solutions des équations implicites à convergence; un nombre fini de
Newton, des Jacobiennes réutilisées et des tolérances différentes peuvent
produire des cartes numériques différentes. Elle n’implique pas une identité
des itérés SQP hors de la variété faisable.

## Prototype ajouté et portée

Le script crée deux modèles natifs `AcadosSim` indépendants : Ding complet
avec cinq états et Ding réduit avec F,A. Il emploie les vraies fonctions du
modèle `DingModelPulseWidthFrequencyWithFatiguePeriodicNode`, un état initial
fatigué et un PW fixé, sans mécanique.

Les trois profils éliminés sont transmis comme paramètres et interpolés par
des polynômes de Lagrange propres à chaque sous-pas. Le champ réduit prend
donc exactement les mêmes valeurs aux abscisses utilisées par l’IRK. Ce champ
est un dispositif de transcription, **pas un nouveau modèle continu à
intégrer avec un autre pas ou un autre tableau**. Les limites de sous-pas
servent uniquement à choisir le polynôme; elles ne constituent pas des
événements physiologiques supplémentaires.

Le smoke prévu compare les cinq états reconstruits et les sensibilités
natives en F0,A0,PW, sur la variété entrante où
`dTau1=c_tau dA` et `dKm=c_km dA`. Les seuils programmés sont 1e−9 sur les
valeurs échelonnées et 1e−8 sur ces sensibilités. Les sensibilités par rapport
aux états supprimés, aux paramètres physiologiques ou au calendrier ne sont
pas couvertes. Le calcul utilise jusqu’à 12 Newton et une tolérance 1e−12
pour viser la carte convergée, distincte du budget courant de cinq Newton.

Le script supporte `--family radau --stages 5 --substeps 1`, mais cette variante
n’a pas été exécutée. Ses valeurs par défaut sont GL4×5 et h_total=0,02 s,
comme l’intégrateur de la référence à 50 Hz.

**Première tentative** : import réussi puis
`FileNotFoundError: /home/mickaelbegon/miniforge3/lib/link_libs.json`.
Le `utils.get_acados_path()` installé lit directement `CONDA_PREFIX` et ignore
ici `ACADOS_SOURCE_DIR`. Le lancement a sélectionné le bon exécutable Python,
mais conservé le préfixe Conda de base. L’environnement cible possède bien
`lib/libacados.so`, les headers et `bin/t_renderer`. Aucun solveur n’a été
construit et aucun résultat d’équivalence ou de performance n’a été obtenu.
Pour respecter la limite d’un smoke test, aucune relance n’a été faite.

La relance avec l'environnement activé a ensuite construit les deux
simulateurs natifs et validé la carte GL4×5 : erreur relative maximale de
reconstruction d'état **1,332e−16** et erreur maximale de sensibilité avant
`F0,A0,PW` sur la variété affine **3,242e−12**, sous les seuils respectifs
`1e−9` et `1e−8`. Le JSON de sortie a été obtenu sans erreur de compilation.
Cela valide seulement la carte à un muscle et PW fixé; pas le NLP ACADOS,
la mécanique couplée, le RHO ni une performance.

La branche générique Radau IIA a aussi été exécutée (`5` stages, `1` sous-pas)
comme contrôle du traitement des abscisses : erreurs maximales `3,609e−15`
sur l'état et `9,478e−12` sur les sensibilités. Elle ne recommande pas de
changer l'intégrateur ACADOS de référence; elle confirme seulement que la
formule des offsets dépend bien du tableau effectivement choisi.

Commande de reproduction exécutée :

```bash
source .github/scripts/benchmark_env.sh rho32
export ACADOS_SOURCE_DIR="$CONDA_PREFIX"
python scripts/experimental_acados_ding_local_map.py
```

Contrôle générique Radau :

```bash
source .github/scripts/benchmark_env.sh rho32
export ACADOS_SOURCE_DIR="$CONDA_PREFIX"
python scripts/experimental_acados_ding_local_map.py --family radau --stages 5 --substeps 1
```

Chaque lancement place ses fichiers générés dans un nouveau répertoire
`/tmp/acados-ding-local-map-*`; il ne réutilise ni ne remplace une capsule du
benchmark. Le prototype n’est appelé par aucun chemin applicatif.

## Travail restant avant un vrai A/B ACADOS

1. Valider la carte native puis ses sensibilités, avec la mécanique couplée,
   plusieurs PW, états proches des bornes, et tous les muscles/paramètres
   réellement employés. Figer le tableau, les sous-pas, le temps et le
   calendrier dans la signature de génération. Les offsets doivent être
   rafraîchis à chaque RHO et lors de tout changement d’état entrant.
2. Construire un export `acados_template` expérimental réduit explicite, ou
   un adaptateur complet prenant possession de la préparation, des coûts,
   des contraintes, des bornes, du seed et de la reconstruction. Une seule
   substitution dans le modèle Bioptim ne suffit pas.
3. Réduire 22→10 états physiques pour quatre muscles et mécanique réduite,
   **26→14 si les quatre carriers de PW du slew sont présents**. Les carriers
   et contrôles ΔPW restent dans l’OCP, de même que leur raccord au PW exécuté.
   Les paramètres optimisés éventuels ne sont pas supprimés.
4. Transformer les bornes Tau1/Km en intersections sur A avec le scaling et
   le signe correct; vérifier numériquement les bornes Cn. Substituer les
   états reconstruits dans tous les résidus de coût et contraintes, y compris
   au nœud terminal. Ne pas ajouter implicitement des bornes aux stages IRK
   si le NLP ACADOS original n’en impose qu’aux nœuds. Inversement, préserver
   tout guard réellement exporté et auditer le problème original reconstruit.
5. Comparer sur une même fenêtre figée, mêmes x0, paramètres, objectifs,
   bornes et primal correspondant. Mesurer `time_sim`, ses sous-temps AD/LA,
   `time_qp`, `time_qp_xcond`, SQP et résidus; compter préparation/reconstruction.
   La préservation du primal ne fournit pas automatiquement un mapping exact
   des multiplicateurs ACADOS `pi/lam`.

L’interpolation du prototype ajoute 60 paramètres par muscle pour GL4×5;
quatre muscles donneraient 240 constantes par intervalle, plus les autres
paramètres du modèle. C’est un premier véhicule de preuve, pas nécessairement
la représentation la plus rapide. Une API de carte discrète avec un solve
implicite local et des dérivées certifiées serait une autre voie plus
spécialisée; son support codegen natif doit être validé avant de la promettre.

## Gain plausible face aux mesures existantes

Le pilote IPOPT local documente une médiane chaude de 1,0207→0,8415 s
(−17,6 %, deux trajectoires RHO divergentes). Ce chiffre porte en partie sur
la taille du KKT de collocation et les itérations IPOPT; il ne constitue pas
une prévision ACADOS.

Le fichier historique
`local-results-acados-native-irk-20260909/acados-irk-reduced-isokinetic-torque-0.2-omega--6.28318530718-load--3-to-3/result.json`
montre un seul RHO validé à 0,05835 s natif et aucune fenêtre chaude :
`hot_window_count=0`, `first_failed_rho=2`, `success=false`. Ce résultat
isorégime ne certifie pas un ACADOS à 58 ms durable et n’est pas comparable
au problème dynamique de septembre 12.

Sur ce dernier problème (50 Hz, +0,1 N·m, slew dur),
`acados_convergence_time_analysis_20260912.md` rapporte 0,47–0,48 s par SQP,
avec 88,8 % du temps natif nominal dépensé dans trois tentatives non certifiées.
Le budget 30 réduit la boucle de 190,274 à 84,073 s via récupération IPOPT;
le succès ACADOS reste 5/8. Les modes sans diagnostics ont aussi réduit une
boucle λ=0,1 de 37,612 à 31,767 s sans modifier ses 56 SQP; cela ne mesure
aucune accélération de l’intégrateur.

Retirer douze états peut réduire les dimensions des systèmes de Newton IRK,
des sensibilités et des QP. Cela ne garantit pas de résoudre la stagnation
SQP ou le mauvais raccord F/omega. Le fractionnement exact entre IRK et QP
manque dans les JSON compacts historiques; une extrapolation cubique des
dimensions ou une promesse de gain similaire à IPOPT serait injustifiée.

**Recommandation** : conserver cette piste opt-in. Prochaine étape concrète :
réussir la carte native avec l’environnement correctement activé, puis
coupler la mécanique et mesurer sur un checkpoint ACADOS figé. Ne pas
intégrer le prototype dans le runner avant ces validations et la prise en
charge explicite de tous les mappings. Les réglages `jac_reuse` et tolérance
Newton, et la qualité du transfert IRK existant, restent des expériences
moins invasives à comparer au même point de référence.
