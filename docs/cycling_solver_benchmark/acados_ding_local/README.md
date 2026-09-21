# Réduction locale Ding dans ACADOS et son interface graphique

Cette option expérimentale conserve les états musculaires `F,A` dans la
capsule native ACADOS et reconstruit `Cn,Tau1,Km` à partir des états entrants
de chaque fenêtre. Son objectif est de réduire les dimensions des systèmes
IRK, de leurs sensibilités et des QP. **Le gain ACADOS reste à mesurer sur un
A/B couplé** ; les gains déjà observés avec IPOPT ne se transfèrent pas
automatiquement à ACADOS.

## Activation dans le GUI

Lancer depuis la racine du dépôt :

```bash
python -m cocofest.simulation.gui
```

Dans l'onglet **Solveur**, cocher **ACADOS : réduction locale Ding
(expérimental)**. La case est décochée par défaut, y compris au chargement
d'un ancien JSON ne contenant pas ce champ. Le nom enregistré est
`acados_ding_local_reduction` ; la commande produite contient
`--acados-ding-local-reduction` uniquement quand sa valeur vaut `true`.

Le GUI accepte cette option avec `solver=acados`, `mode=rho`, mécanique
`reduced`, formulation `dynamic`, 50 stimulations par cycle (référence à
50 Hz), solveur SQP, intégration `irk`, quatre stages et cinq sous-pas. Le
guard vitesse doit être `on` ou `auto` (résolu en `on` pour ACADOS) ; les
commandes sont libres de borne/coût de slew ΔPW. Les autres combinaisons
produisent une erreur visible et empêchent
le lancement. Changer le solveur ne coche ni ne décoche silencieusement la
réduction. Le résumé rappelle le caractère expérimental, la reconstruction
des états et l'absence de reconstruction des multiplicateurs complets.

Il faut fournir la seed IPOPT certifiée du premier cycle correspondant au
problème cible. Pour tester le formulaire sans solveur ni affichage, utiliser
[l'exemple JSON](gui-config.example.json), puis :

```bash
python -m cocofest.simulation.gui \
  --config docs/cycling_solver_benchmark/acados_ding_local/gui-config.example.json \
  --dry-run
```

L'exemple contient un chemin de seed à remplacer et `dry_run=true`. Pour
lancer réellement, choisir une seed existante compatible, un nouveau dossier
de sortie, puis décocher **Prévisualisation seulement**. L'option du GUI
nécessite la version du runner intégrant le module
`cocofest.optimization.acados_ding_local_reduction` et le flag correspondant ;
elle ne transforme pas à elle seule un ancien moteur ACADOS. Le moteur revérifie
les données physiques et la provenance de la seed au lancement.

## Contrat de l'adaptateur

Le module
[`acados_ding_local_reduction.py`](../../../cocofest/optimization/acados_ding_local_reduction.py)
conserve le contrat complet Bioptim et construit une copie réduite de
`AcadosOcp`. Une façade traduit les entrées/sorties de la capsule native.

| Interface | Transformation |
| --- | --- |
| États et seed primale | Projection sur les états conservés ; reconstruction complète lors de `get(stage, "x")` |
| Dynamique IRK | Substitution de profils calculés aux abscisses internes du tableau effectif |
| Coûts `NONLINEAR_LS` et contraintes non linéaires | Substitution des états reconstruits aux nœuds, y compris initial et terminal |
| Bornes musculaires | Intersection des bornes `A`, `Tau1`, `Km` sur `A` ; contrôle direct des bornes `Cn` |
| Paramètres numériques | Ajout des profils locaux aux données `periodic_calcium` ; rafraîchissement à chaque fenêtre |
| Extraction RHO | Retour de tous les états attendus par Bioptim et les diagnostics existants |
| Multiplicateurs `pi/lam` et itérés plats | Objets natifs de la capsule réduite ; aucune équivalence duale complète annoncée |

Pour quatre muscles et deux états mécaniques, les états physiques passent de
22 à 10. Si quatre états porteurs de PW sont présents, le comptage devient
26 à 14 : les porteurs, commandes et contraintes de raccord doivent être
conservés. Ce comptage dimensionnel ne certifie pas les variantes avec slew ;
elles sont refusées dans la première intégration GUI et demandent un test
intégré spécifique avant extension.

La première intégration exige une phase, aucun paramètre physiologique ou
temporel optimisé, le type exact
`DingModelPulseWidthFrequencyWithFatiguePeriodicNode`, des paramètres
constants par muscle et les données numériques `periodic_calcium` dans le
format attendu. `Cn,A,Tau1,Km` doivent être fixés par leurs bornes au début de
chaque fenêtre. Une seed seule ne suffit pas à rendre un état initial fixé.

Le maillage, la durée d'intervalle et les réglages IRK sont uniformes et fixes
dans une capsule. Il y a une stimulation par intervalle. Un nouveau tableau
ou une nouvelle durée requiert une capsule adaptée. Le noyau mathématique
peut construire GL ou Radau IIA ; le GUI expose uniquement la référence
**Gauss-Legendre 4×5**. L'option ne change pas ACADOS en Radau-5.

Les rollouts `--acados-initial-irk-rollout` et
`--acados-transfer-irk-rollout` sont refusés avec cette option tant que leur
simulation et leur convention temporelle ne sont pas adaptées à la façade.
Le transfert de snapshots natifs d'une capsule complète vers une capsule
réduite n'est pas pris en charge. Les warm-starts duaux internes ne prouvent
pas une correspondance des multiplicateurs du problème original.

## Équations et élimination des états

Pour un muscle, noter \(C=C_n\), \(T=\mathrm{Tau1}\), \(K=K_m\), et
\(A_r=\mathtt{a\_scale}\). Les équations de fatigue effectivement utilisées
par [le modèle du dépôt](../../../cocofest/models/ding2007/ding2007_with_fatigue.py)
sont

\[
\dot A=-\frac{A-A_r}{\tau_f}+\alpha_A F,\qquad
\dot T=-\frac{T-T_r}{\tau_f}+\alpha_T F,\qquad
\dot K=-\frac{K-K_r}{\tau_f}+\alpha_K F.
\]

Avec \(\alpha_A\ne0\), définir pour \(y=T,K\)

\[
r_y=\frac{\alpha_y}{\alpha_A},\qquad
d_y=y-r_y A,\qquad d_{y,r}=y_r-r_y A_r.
\]

En soustrayant \(r_y\dot A\) de \(\dot y\), les termes en force s'annulent :

\[
\boxed{\dot d_y=-\frac{d_y-d_{y,r}}{\tau_f}},\qquad
\boxed{y=r_y A+d_y}.
\]

Dans [la variante `periodic_node`](../../../cocofest/models/ding2007/ding2007_with_fatigue_periodic_node.py),
le forçage calcique de l'intervalle \([t_k,t_{k+1}]\) est fourni par les
données numériques : \(H(t_k+s)=H_k\exp(-s/\tau_c)\), et

\[
\dot C=-C/\tau_c+H(t)/\tau_c.
\]

La largeur d'impulsion demeure une décision dans la dynamique de force.
Elle n'intervient pas dans ce forçage calcique particulier. La réduction
numérique ne s'applique donc pas telle quelle à un modèle dont le calcium
dépendrait de commandes, de paramètres optimisés ou d'un calendrier libre.

Les trois profils \(v=(C,d_T,d_K)\) vérifient chacun
\(\dot v=-v/\tau+r(t)\). Sur un sous-pas de longueur \(h\), avec matrice IRK
\(B\), poids \(b\) et abscisses \(c\), leurs valeurs de stages \(V\) sont
obtenues par le petit système linéaire

\[
\boxed{(I+hB/\tau)V=\mathbf1v_0+hB\,r(t_0+hc)}.
\]

Le nœud de sortie est ensuite

\[
\boxed{v_1=v_0+h\,b^\top\left(r(t_0+hc)-V/\tau\right)}.
\]

En Gauss-Legendre, le dernier stage **n'est pas** le nœud final. Cette
dernière formule est donc nécessaire. Les sous-pas se suivent avec cet état
de sortie, puis les intervalles avec le nouveau forçage \(H_k\). À chaque
RHO, l'initialisation vient de l'état réellement transmis :

\[
v_0=(C_0,\ T_0-r_TA_0,\ K_0-r_KA_0).
\]

Les exponentielles continues de fatigue ne remplacent pas ces profils
discrets. Le code transmet les valeurs de stages comme paramètres et utilise
une interpolation de Lagrange par sous-pas pour les retrouver aux abscisses
IRK. Ce champ interpolé représente le tableau fixé ; il n'est pas un modèle
continu générique à réintégrer avec d'autres abscisses.

Cette élimination conserve les solutions du système IRK à convergence, à
l'arrondi près. Elle ne garantit ni des itérations SQP identiques hors de la
variété faisable, ni l'identité d'intégrations arrêtées après un nombre fini
d'itérations de Newton.

## Bornes et scaling

Pour des états échelonnés \(A=s_A\bar A\) et \(y=s_y\bar y\), la relation
à un nœud vaut

\[
\bar y=\beta\bar A+\delta,\qquad
\beta=r_y s_A/s_y,\quad \delta=d_y/s_y.
\]

Si \(\beta\ne0\), une borne \(\ell_y\le\bar y\le u_y\) impose

\[
\min\!\left(\frac{\ell_y-\delta}{\beta},\frac{u_y-\delta}{\beta}\right)
\le\bar A\le
\max\!\left(\frac{\ell_y-\delta}{\beta},\frac{u_y-\delta}{\beta}\right).
\]

On intersecte cet intervalle avec les bornes originales de \(\bar A\) et
avec l'intervalle provenant de l'autre variable de fatigue. La forme min/max
traite aussi les ratios négatifs. Si \(\beta=0\), on vérifie directement la
constante \(\delta\). Les bornes `Cn` sont également contrôlées directement.
Une intersection vide est une incompatibilité explicite.

Les contraintes sont préservées aux emplacements où elles existaient dans
l'OCP original. Réduire le modèle ne justifie pas d'ajouter des bornes
musculaires à tous les stages internes si ACADOS les imposait seulement aux
nœuds. Les guards mécaniques déjà exportés suivent leur substitution
d'origine.

## Résultats déjà disponibles et vérification

Le [rapport expérimental ACADOS](../acados_ding_local_reduction_experimental.md)
archive le prototype à un muscle : GL4×5 donne une erreur maximale de
reconstruction de \(1,33\,10^{-16}\) et de sensibilité de
\(3,24\,10^{-12}\). Le contrôle Radau IIA cinq stages donne respectivement
\(3,61\,10^{-15}\) et \(9,48\,10^{-12}\). Il s'agit de cartes convergées,
avec 12 Newton et tolérance \(10^{-12}\), sans OCP mécanique couplé.

Le [bilan IPOPT/MA57 Radau-5](../ding_radau5_local_ab.md) décrit un autre
solveur et une autre transcription : deux ordres de rejeu gelé mesurent
−41,0 % et −50,6 % de médiane ; sur 50 RHO réellement propagés, la médiane
passe de 0,798 à 0,551 s (−30,9 %), avec 50/50 succès pour les deux variantes.
Ces chiffres motivent l'expérience ACADOS ; ils n'en mesurent pas le gain.

Reproduire la preuve de carte isolée :

```bash
source .github/scripts/benchmark_env.sh rho32
export ACADOS_SOURCE_DIR="$CONDA_PREFIX"
python scripts/experimental_acados_ding_local_map.py
python -m pytest tests/test_simulation_gui.py tests/test_simulation_core.py -q
```

Pour un A/B OCP, exporter la commande du GUI et conserver les mêmes données,
seed, réglages Newton/SQP, paramètres QP et durées ; seul le flag de réduction
doit varier. Utiliser deux dossiers de génération/sortie distincts. Mesurer
la préparation des profils, `time_sim`, `time_qp`, le temps total, les
résidus, les commandes et la faisabilité dans les états reconstruits. Un
statut de processus nul ne suffit pas à conclure au succès scientifique.

### A/B intégré : 50 Hz, guard actif, contrôles libres

Le 13 septembre 2026, l'intégration complète a été exécutée sur deux fenêtres
RHO avec le même seed certifié 50 Hz, couple résistif `+0,1 Nm`, mécanique
réduite dynamique, SQP/IRK GL4×5, guard de cadence actif et sans slew. Les
deux variantes valident **2/2** fenêtres NLP et physiques; les coûts cumulés
diffèrent de seulement `1,3e-6` relativement. La fenêtre chaude conserve les
mêmes sept itérations SQP, mais passe de `0,488268 s` à `0,207147 s` natif,
soit **−57,6 %**; le temps mur chaud passe de `0,510688 s` à `0,236340 s`
(**−53,7 %**) avec le cache des profils discrets. Une exécution précédente
mesurait `0,185894 s` natif : communiquer une plage de **58–62 %** sur cette
unique fenêtre chaude, pas une promesse générale.

L'audit indépendant sur les quatre muscles, la mécanique et 50 intervalles
donne une erreur de carte `2,36e-16`, de sensibilité `1,33e-15`, un défaut
IRK complet `8,94e-13` et des résidus de substitution coût/contraintes nuls.
Les artefacts sont dans
[`acados-ding-local-integration-20260913`](../../../acados-ding-local-integration-20260913/).

Le smoke `delta_pw_v1` avec slew `100 us` est aussi certifié sur une fenêtre
(`26 → 14` états, cinq SQP, audit de carte `2,50e-16`), mais **n'a pas
d'A/B de performance** et n'est donc pas exposé par le GUI initial.

Reproduction du cas A/B initial :

```bash
bash scripts/run_acados_ding_local_ab.sh 2 baseline \
  acados-ding-local-integration-20260913/baseline2-guard50
bash scripts/run_acados_ding_local_ab.sh 2 local \
  acados-ding-local-integration-20260913/local2-guard50-cached
```

### Benchmark 100 RHO : préfixe certifié commun de 76 cycles

Un benchmark prolongé a été lancé avec exactement le même contrat, seed et
gardes, sans récupération IPOPT. Les deux variantes ont validé les mêmes
**76 cycles physiques/NLP**, puis se sont arrêtées au RHO 77. Ce n'est donc
pas une preuve d'endurance à 100 cycles et ne doit pas être présenté comme
un succès 100/100. C'est en revanche une comparaison directe sur un préfixe
long identique.

| Mesure sur les 75 fenêtres chaudes | ACADOS complet | Réduction locale | Écart |
|---|---:|---:|---:|
| Médiane native | 0,455458 s | 0,167345 s | **−63,3 %** |
| p90 natif | 1,454084 s | 0,389846 s | −73,2 % |
| Médiane mur-à-mur | 0,475349 s | 0,194729 s | −59,0 % |
| p90 mur-à-mur | 1,473534 s | 0,417386 s | −71,7 % |
| Objectif cumulé | 2491,792796 | 2491,792974 | +7,2e−8 relatif |
| Capacité minimale `A` | 0,912530879 | 0,912530900 | +2,2e−8 |

L'audit indépendant de la dernière fenêtre locale conserve une erreur de
carte `2,78e−16`, une erreur de sensibilité `7,77e−16`, un défaut IRK complet
`5,17e−13` et des résidus coût/contraintes nuls. Les deux résultats et le
log sont archivés dans
[`acados-ding-local-integration-20260919`](../../../acados-ding-local-integration-20260919/).

## Références

Les dérivations ci-dessus partent des équations exécutées dans le dépôt ;
les paramètres par défaut ou les adaptations mécaniques ne doivent pas être
attribués implicitement à l'article d'origine.

- Ding et al. (2007), *Mathematical model that predicts the force–intensity
  and force–frequency relationships after spinal cord injuries*, Muscle &
  Nerve 36(2), 214–222, [doi:10.1002/mus.20806](https://doi.org/10.1002/mus.20806).
  Référence physiologique citée par le modèle du dépôt ;
  [texte de l'article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2633444/).
- [Équations Ding et recrutement PW du dépôt](../../../cocofest/models/ding2007/ding2007.py),
  [fatigue](../../../cocofest/models/ding2007/ding2007_with_fatigue.py) et
  [calcium `periodic_node`](../../../cocofest/models/ding2007/ding2007_with_fatigue_periodic_node.py).
- [Documentation officielle `acados_template`](https://docs.acados.org/python_interface/index.html),
  notamment `collocation_type`, `sim_method_num_stages`,
  `sim_method_num_steps`, `set`, `constraints_set` et les statistiques.
  Elle décrit les tableaux GL/Radau et les interfaces de capsule ; la pile
  locale peut différer de la documentation courante.
- [Publication ACADOS](https://arxiv.org/abs/1910.13753),
  *acados: a modular open-source framework for fast embedded optimal control*.
  Référence pour l'architecture du solveur, sans résultat spécifique à Ding.
- [Dérivation Radau-5 précédente](../ding_analytic_reduction_radau5.md),
  [comparaison des solveurs](../README.md) et
  [documentation générale du GUI](../../../cocofest/simulation/README.md).
