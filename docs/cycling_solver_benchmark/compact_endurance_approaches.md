# Anticiper l'endurance avec un RHO compact

## Choix proposé au 10 septembre 2026

Conserver le RHO physique d'un cycle et construire **un coût terminal local à
partir de simulations musculaires rapides avec adaptation des PW**. Les
simulations tournent entre les résolutions et fournissent quelques coefficients
numériques à IPOPT. Aucune trajectoire FHO n'est nécessaire.

Le prototype précédent ajoute, pour trois cycles futurs, 360 PW libres,
90 égalités de moment, 3 240 marges et 780 paramètres fixes Bioptim. Les paramètres
fixes restent présents dans le vecteur NLP générique avant élimination par le
solveur. La propagation des états éliminés crée aussi des dépendances temporelles
longues dans les dérivées. Le faible nombre d'états multibody futurs ne suffit
donc pas à garantir un faible coût de construction ou de résolution.

Le lancement réel précédent n'a produit ni rapport final ni itérations IPOPT
visibles; un essai sans compilation C était également silencieux. **On ne peut
pas attribuer ce délai à MA57, à la compilation C seule, ou à une difficulté de
convergence**, ni annoncer que le test réel a passé. Le petit NLP synthétique
compile et converge; cela ne valide pas les performances du modèle réel.

## Comparaison des approches

Les nombres ci-dessous désignent les **variables futures libres supplémentaires**,
pas les états déjà présents dans le RHO. Un profil transmis via les paramètres
fixes de Bioptim peut néanmoins augmenter le vecteur NLP.

| Approche | Variables futures | Comment elle anticipe | Limite principale |
|---|---:|---|---|
| Prix marginal de la fatigue | 0 | Pénalise la perte des muscles dont la fatigue réduit le plus la marge future de moment | Approximation locale, à réévaluer lorsque l'allocation change |
| Rollout musculaire rapide sous politique adaptative | 0 s'il est calculé hors NLP | Prédit les PW qui maintiennent le moment total, avec les états de fatigue actualisés | Politique de secours choisie, et cinématique future supposée |
| Coût terminal local linéaire/quadratique | 0 | Résume plusieurs rollouts autour de l'état terminal attendu | Domaine local de validité et contrôle de l'erreur nécessaires |
| Allocation par quelques coefficients/blocs | Environ 9–24, selon la base | Optimise quelques redistributions ou corrections PW sur des groupes de cycles | Réduction du nombre de commandes sans réduction automatique du coût de propagation |
| Carte de cycle / horizon à plusieurs résolutions | 0 sous politique fixée, sinon quelques coefficients | Détaille le futur proche, puis avance par cycles entiers pour les états lents | Transitoires et pertes de faisabilité entre les points espacés |
| Fonction de valeur apprise sur simulations du patient | 0 | Approxime le coût futur sur un domaine plus large que le modèle local | Données et validation hors distribution; pas de garantie clinique automatique |

L'approximation de coût terminal par la valeur d'une politique de rollout est
une approche établie en MPC/programmation dynamique. Notre adaptation consiste
à utiliser une politique musculaire issue du modèle patient et des cycles RHO,
puis à en approximer localement la valeur. Voir
[Bertsekas, cadre MPC et programmation dynamique, 2024](https://web.mit.edu/dimitrib/www/Bertsekas_NMPC_IFAC.pdf)
et [Moreno-Mora, Beckenbach et Streif, coûts terminaux appris, 2022](https://arxiv.org/abs/2212.00361).
Ces travaux ne démontrent pas à eux seuls l'amélioration d'endurance de notre modèle.

Les commandes par blocs réduisent les degrés de liberté; la réduction des
contraintes et la faisabilité demandent un traitement distinct. Voir
[Schitz et al., 2024](https://arxiv.org/abs/2408.08020). Les garanties de leur
formulation ne se transfèrent pas directement au pédalage FES non linéaire.

## Une réduction exacte propre aux équations actuellement utilisées

Dans notre code, pour chaque muscle :

\[
\dot z_j=-\frac{z_j-z_{j,0}}{\tau_f}+\alpha_j F,
\qquad z=(A,\tau_1,K_m).
\]

Avec \(\alpha_A<0\), définir

\[
d=1-A/A_0,\qquad
e_j=z_j-z_{j,0}-\frac{\alpha_j}{\alpha_A}(A-A_0),
\quad j\in\{\tau_1,K_m\}.
\]

Alors, exactement :

\[
\dot d=-d/\tau_f-\alpha_A F/A_0,
\qquad e_j(t+h)=e^{-h/\tau_f}e_j(t).
\]

Une seule mémoire par muscle reçoit donc l'effet des stimulations via la force.
Les deux autres coordonnées sont des écarts initiaux dont la décroissance est
connue. La reconstruction des trois états est exacte pour les équations du
dépôt; ce n'est pas une nouvelle loi physiologique validée chez le patient.

**Il ne faut pas imposer arbitrairement les écarts à zéro.** Dans l'archive
réelle analysée, l'écart initial de `Tau1` du deltoïde antérieur représente
environ 18 % de sa valeur de repos au début de la trajectoire. Les autres
écarts sont beaucoup plus faibles. Le prototype les conserve.

Pour un coût terminal général, les quatre mémoires seules peuvent être
insuffisantes : les quatre forces résiduelles à la fin du RHO influencent le
futur proche. Je retiendrais d'abord **8 coordonnées variables** : quatre
dommages normalisés et quatre forces normalisées. Les écarts initiaux, la phase,
le régime calcique et le contexte de tâche doivent rester connus ou être
ajoutés si le contrôleur autorise leur variation. En particulier, une fréquence
de stimulation variable invaliderait l'hypothèse de calcium imposé.

## Prototype implémenté : réponse affine en recrutement

La loi PW actuelle s'écrit avec

\[
r=1-\exp[-(PW-PD_0)/PDT].
\]

Si `A`, `Tau1` et `Km` sont gelés **dans l'équation de force pendant un seul
intervalle de stimulation**, cette équation est affine en \(r\). La carte rapide
calcule alors

\[
F_{k+1}=a_kF_k+b_kr_k,
\qquad
PW_k=PD_0-PDT\log(1-r_k).
\]

Le calcium est calculé analytiquement selon `periodic_node`. Les gains mécaniques
et l'activation calcique sont échantillonnés au milieu de petits sous-intervalles;
sur chacun, la réponse de force et sa convolution exponentielle de fatigue
sont intégrées analytiquement. Les trois états lents sont ensuite actualisés
avec cette convolution. Les forces résiduelles et la récupération sont conservées.
Raffiner les sous-pas réduit l'erreur de quadrature rapide, mais ne supprime pas
l'approximation de gel des états lents dans chaque phase.

À chaque phase future, le petit allocateur borné existant choisit les moments
individuels dont la somme satisfait le moment total demandé. L'inversion PW
devient analytique, au lieu de relancer Ding complet à chaque essai d'une
recherche scalaire. Le QP est utilisé ici **dans une simulation numérique hors
NLP**; ses changements d'ensemble actif ne sont pas introduits dans un objectif
qu'IPOPT doit différencier. Les états et PW futurs peuvent varier à chaque cycle.

Implémentation :

- `cocofest/optimization/compact_muscle_prediction.py` : coordonnées exactes,
  carte affine, allocation et rollout;
- `scripts/explore_compact_endurance_prediction.py` : expérience reproductible,
  rejeu indépendant et figures;
- `tests/test_compact_muscle_prediction.py` : invariants, convolution,
  convergence de l'intégration rapide, antagonistes et impossibilité de tâche.

## Résultats numériques obtenus

Source : `validated-rho-trajectory.npz`, campagne IPOPT/MA57, résistance signée
`+0.1 N·m`, mécanique réduite, 30 phases et période 1 s. Chaque prédiction part
de la **frontière finale** du cycle source. La cinématique et le moment total
de ce cycle servent de tâche future supposée. Aucune donnée des cycles suivants
ne sert à construire la politique.

Les seuils ont été fixés avant les calculs : erreur maximale de moment au
rejeu \(10^{-3}\,N\cdot m\), erreur maximale des états lents normalisée par
leur repos \(10^{-3}\). Le rejeu utilise RK4 à 16 sous-pas avec les équations
complètes et les mêmes PW que la carte compacte.

| Ancre, index à partir de 0 | Horizon | Sous-pas compacts | Temps compact | Erreur max de moment au rejeu | Erreur lente max / repos | Gate |
|---|---:|---:|---:|---:|---:|---|
| 0 | 10 cycles | 4 | 0,078 s | 0,005658 N·m | 0,000453 | échec |
| 0 | 10 cycles | 8 | 0,097 s | 0,001802 N·m | 0,000149 | échec |
| 0 | 10 cycles | 16 | 0,130 s | 0,000480 N·m | 0,0000381 | passé |
| 112 | 10 cycles | 16 | 0,128 s | 0,000483 N·m | 0,0000443 | passé |
| 112 | 10 cycles | 32 | 0,199 s | 0,000114 N·m | 0,0000106 | passé |

Tous ces rollouts et les deux oracles ont complété leurs dix cycles. L'oracle
actuel « Ding complet + inversions PW + allocation » prend environ 40,4 et
39,1 s. La comparaison donne environ 305–310 fois moins de temps pour le
prototype à 16 sous-pas, **par rapport à cette implémentation Python particulière**.
La préparation des gains coûte 0,053–0,054 s, puis est réutilisable. Le rejeu
complet de vérification coûte environ 2,9 s pour dix cycles.

Une vérification supplémentaire du même rejeu à 64 sous-pas, depuis l'ancre
112, modifie le moment total d'au plus `9,01e-6 N·m` et le ratio de capacité
d'au plus `3,91e-7`. La référence à 16 sous-pas n'est donc pas exacte, mais son
erreur estimée par raffinement reste bien inférieure au seuil de `1e-3 N·m`.

Ces mesures uniques ne sont ni un benchmark de latence clinique, ni une
comparaison à un oracle compilé, ni un gain de temps RHO déjà établi. Le
pré-échantillonnage des gains contribue également à l'accélération. Deux ancres ne
suffisent pas à valider un prédicteur d'endurance. Les égalités sont auditées
aux extrémités des phases; le maintien continu de toute la tâche mécanique
demande une validation supplémentaire.

Les rapports et figures locaux sont dans
`.cache/compact-endurance-exploration/cycle0/` et `cycle112/`. Le dossier
`cycle112-final/` contient aussi les tableaux NumPy exportés et une figure
des variations de PW à phase identique, produits par la version finale du script.

La suite ciblée passe **41 tests**, dont six nouveaux : reconstruction exacte
avec écarts initiaux non nuls, convergence rapide d'ordre deux, convolution
vérifiée par une intégration indépendante, antagonistes, cible impossible et
rejet des états hors domaine. Ces tests ne remplacent pas l'essai de contrôle
en boucle fermée.

Commande reproductible :

```bash
MPLBACKEND=Agg conda run -n cocofest-rho32 python \
  scripts/explore_compact_endurance_prediction.py \
  ipopt-linear-solver-150-20260904/resistance-0p10Nm/ipopt-sx-radau5-ma57-150-max2000-reduced/validated-rho-trajectory.npz \
  benchmark-seed/reduced-cycling-fourier12.npz \
  .cache/compact-endurance-exploration/cycle112 \
  --cycles 10 --cycle-index 112 --substeps 16 32 --reference-substeps 16
```

## Comment transformer cela en fonction coût compacte

L'effet des stimulations du cycle courant est déjà transmis à l'état terminal
par la dynamique du RHO. Le coût à ajouter est donc une fonction de cet état :

\[
J=J_{\mathrm{cycle}}+\lambda\widehat V_H(x_T;\theta_k).
\]

Les coefficients \(\theta_k\) sont construits avant la résolution et restent
fixes pendant celle-ci. Une approximation locale simple est

\[
\widehat V_H(\xi)=V_0+g^T(\xi-\bar\xi)
 +\tfrac12(\xi-\bar\xi)^TH(\xi-\bar\xi).
\]

Pour huit coordonnées, le modèle linéaire a huit sensibilités; un modèle
quadratique complet a 45 coefficients avec sa constante. Une Hessienne
diagonale en exige 17, mais perd les interactions de substitution musculaire.
Le nombre de coefficients ne dépend pas des 10, 20 ou 50 cycles simulés.
Le coût devient lisse pour IPOPT même si l'oracle numérique utilisé pour
calculer ses coefficients emploie une allocation par morceaux.

Les sensibilités \(g_i\) jouent le rôle de **poids musculaires calculés depuis
la tâche et l'état de fatigue**, au lieu de poids choisis a priori. Elles ne
sont pas nécessairement constantes ni toutes du même signe : la récupération,
la force résiduelle et les possibilités de redistribution comptent. Elles ne
doivent pas être modifiées en cours d'une même résolution IPOPT.

Le maximum des PW seul peut rester bloqué à `PW_max` tandis que d'autres
muscles permettent de continuer. La valeur future devrait donc inclure la
**marge de production du moment total**, par exemple les distances signées du
moment demandé aux deux bornes de son enveloppe atteignable, normalisées par
une échelle de moment de la tâche. On peut lisser la pire marge sur les phases
et les cycles. Le premier cycle impossible reste un indicateur de rapport;
c'est une mauvaise cible directe de gradient car il est discret.

La faisabilité de cette politique donne une estimation conditionnelle à cette
politique. Son échec ne prouve pas que toute autre stratégie musculaire échoue.
Une amélioration de la marge lissée doit finalement être vérifiée par une
augmentation des cycles RHO physiquement réalisables.

## Suite recommandée et critères de décision

1. Évaluer la carte rapide sur plusieurs ancres, charges et horizons, notamment
   proches de la perte de faisabilité. Tester le classement de stratégies,
   l'accord sur les phases critiques et le rejeu complet; augmenter les
   sous-pas ne corrige pas toutes les approximations.
2. Ajouter l'enveloppe de moment et construire une valeur locale sur les huit
   coordonnées. Dix-sept rollouts suffisent pour des différences centrales du
   gradient; des essais distincts doivent vérifier les erreurs et la direction
   du gradient avant raccordement à IPOPT. Réduire le voisinage ou refuser le
   modèle si l'allocation change trop brutalement.
3. Raccorder un seul petit Mayer au NLP compilé. Transmettre seulement les
   coefficients, le centre et le domaine de validité; réutiliser le graphe.
   Imposer un voisinage de confiance ou détecter toute extrapolation et
   recalculer les coefficients. Ne pas réutiliser une approximation hors domaine.
4. Comparer des RHO appariés avec IPOPT/MA57 : mêmes conditions initiales,
   charge et tolérances, critères de fatigue et de faisabilité, temps total
   incluant la mise à jour de valeur. La fréquence de mise à jour peut être
   ralentie tant que l'erreur de prédiction reste acceptable.
5. Si la valeur locale est trop restrictive, essayer 9–24 coefficients de
   redistribution sur des blocs de cycles, avec propagation détaillée du
   premier cycle et carte de cycle ensuite. La base doit conserver le moment
   total et les bornes PW; le simple blocage ne garantit pas ces propriétés.

## Avancement : coût local et raccordement expérimental

La valeur numérique et son raccordement Bioptim sont maintenant implémentés,
mais **pas activés automatiquement dans les scripts publics de campagne RHO**.
Leur validation sur un petit NLP ne vaut pas validation prospective du pédalage.

- `cocofest/optimization/local_endurance_value.py` fournit les huit coordonnées,
  l'oracle de marge, le polynôme linéaire ou quadratique diagonal et son audit;
- `cocofest/optimization/local_endurance_value_ocp.py` fournit un raccordement
  explicite au moment de la construction du NLP et une mise à jour numérique
  entre résolutions;
- `tests/test_local_endurance_value.py` et `tests/test_local_endurance_value_ocp.py`
  vérifient les signes, les cas invalides, les dérivées et la compilation unique.

La valeur est une pénalité softplus de la pire marge lissée. Pour une phase,
avec une enveloppe de moment total `[L, U]` et une demande `q`, les deux marges
sont `(q-L)/s` et `(U-q)/s`, où `s` est une échelle positive fixe de la tâche.
Cela conserve le signe des muscles antagonistes et reste défini lorsque `q=0`.
La soft-min utilisée est conservatrice; son biais dépend du nombre de phases
et du choix de température. **Comparer des valeurs d'horizons différents exige
donc de traiter ce biais**, même si les variables du NLP restent identiques.

Un échec de la politique future, un état invalide ou une extrémité de
l'enveloppe hors domaine ne produit aucune valeur de remplacement arbitraire.
L'ajustement est alors refusé. Le premier prototype exige des rayons symétriques
strictement positifs : une force résiduelle exactement nulle n'est pas traitée,
et une force presque nulle impose un très petit rayon dans cette direction.
Cette limitation peut rendre le voisinage trop restrictif en pratique.

Les différences centrales utilisent 17 simulations pour huit coordonnées.
Les vérifications supplémentaires utilisent des points distincts : axes,
coins conjoints et directions liées au gradient, notamment les coins où un
objectif linéaire est susceptible d'envoyer le solveur. Les erreurs et le
classement de toutes les paires informatives doivent satisfaire les seuils.
Une valeur plate sans paire informative est refusée. Il s'agit d'un audit
échantillonné, **pas d'une preuve de validité sur toute la boîte**.

Le raccordement transmet, pour quatre muscles, **58 paramètres fixes**,
16 marges de voisinage et 12 résidus de contexte bornés des deux côtés,
indépendamment du nombre de cycles simulés. Il n'ajoute ni état musculaire
futur ni PW future libre. Une empreinte relie chaque ajustement à ses paramètres
musculaires, son centre, ses forces normalisées, son calcium et ses écarts
initiaux : mélanger un ajustement et un autre contexte est refusé avant toute
modification des buffers.

Le contexte calcique doit provenir de la discrétisation réelle. Dans l'archive
Radau5 examinée, le calcium terminal vaut environ `0,16295396`, contre
`0,16298216` pour le point fixe analytique; l'écart `2,82e-5` dépasse une garde
de `1e-7`. Les écarts lents calculés à partir de la trajectoire restent conservés.
Les autres caractéristiques de la tâche (phase, vitesse, période, charge et
cinématique) doivent être vérifiées par le contrôleur appelant : le raccordement
musculaire seul ne les certifie pas.

Le test d'intégration compilée utilise **IPOPT/MA57** avec la bibliothèque HSL
de l'environnement. Trois résolutions du même petit problème synthétique
vérifient successivement les coefficients puis le contexte numérique et
l'échelle des forces. Leurs forces terminales `15`, `25` puis `22,5 N` suivent
la solution analytique attendue. L'identité du solveur, les buffers de paramètres
et la date du fichier C restent inchangés. Ce test utilise `F'=u`, pas le modèle
physiologique : il isole volontairement le raccordement et les mises à jour.

Les résultats sur les ancres RHO réelles et les prochaines limites à lever sont
consignés dans `local_endurance_value_validation.md`. Aucun gain d'endurance
clinique n'est revendiqué à ce stade.

## Accélération par lots et tolérance de suivi explicite

Le calcul des candidats de différences finies et de validation est maintenant
vectorisé dans `batched_compact_muscle_prediction.py` et
`batched_endurance_value.py`. Après une vérification du centre, les 16 points
de différences finies puis les points de validation sont évalués par lots.
Les équations, les sous-pas et la politique d'allocation du mode exact restent
les mêmes; un calcul par points de rupture remplace la recherche scalaire du
multiplicateur du petit QP. Les enveloppes sont conservées pendant le rollout,
ce qui évite une seconde propagation pour calculer le coût.

Cette vectorisation concerne des simulations numériques indépendantes, pas de
nouvelles variables du RHO. Le raccordement conserve ses 58 paramètres fixes
pour quatre muscles; les tests de compilation unique IPOPT/MA57 restent actifs.
Les temps sur les trajectoires réelles, la parité des résultats et les cas
refusés sont détaillés dans `batched_endurance_value_validation.md`.

Une option séparée `tracking_band_nm`, nulle par défaut, permet d'étudier les
petits écarts de moment dans la **politique future uniquement**. Si la demande
initiale est atteignable, elle reste inchangée. Sinon, la politique utilise la
borne atteignable la plus proche seulement si l'écart est inférieur à la bande,
à la tolérance numérique explicitement déclarée près. Au-delà, elle s'arrête.
Les PW ne sont jamais placées hors de leurs bornes pour continuer artificiellement.

La valeur conserve les marges signées par rapport à la demande originale et
ajoute, lorsque la bande est positive, une pénalité

\[
J_{\rm suivi}=w_e\frac{\sum_k\Delta t_k(e_k/s)^2}{\sum_k\Delta t_k},
\qquad e_k=M_{\rm obtenu,k}-M_{\rm demandé,k}.
\]

L'échelle positive `s` et le poids `w_e` ne dépendent pas de la bande. Avec
`w_e=1`, cette pénalité peut être très petite : sa présence ne suffit pas à
garantir l'absence de dérive. Les rapports distinguent coût de marge et coût de
suivi, nombre de phases relâchées, écart maximal et sommes `Σ Δt e` / `Σ Δt |e|`.
Ces sommes portent sur des valeurs aux extrémités des phases; ce ne sont ni des
intégrales continues certifiées ni un travail mécanique.

Un coût ajusté avec une bande positive exige `allow_tracking_band=True` dans le
raccordement. Une mise à jour numérique ne peut pas modifier implicitement la
bande ou son poids. L'empreinte de la tâche et tous les paramètres de coût
accompagnent l'ajustement, en plus de son empreinte de contexte musculaire.
Les échecs restent isolés par candidat et aucun coût n'est fourni pour un
horizon inachevé. Une bande positive change la tâche prédite; elle ne doit pas
être présentée comme un gain d'endurance à suivi exact, ni comme une tolérance
clinique validée.

Le déficit compact H30 (`1,45e-4 N·m`) étant plus petit que le seuil de rejeu
jusqu'ici utilisé (`1e-3 N·m`), un diagnostic indépendant a rejoué le préfixe
réussi avec Ding complet et reconstruit l'enveloppe signée de la phase critique.
Le déficit reste de `6,51e-5 N·m` après raffinement à 128 sous-pas, avec des
variations de bornes inférieures à `8,38e-8 N·m`. Les résultats et limites sont
détaillés dans `compact_rollout_failure_diagnostic.md` : l'erreur de suivi du
préfixe reste non nulle et ce rejeu partiel ne valide pas un horizon complet.

Le prochain essai sera une allocation future anticipant deux ou trois phases,
pour tenir compte de la force résiduelle avant les changements de demande.
Ce petit calcul restera hors du NLP RHO. Il faudra comparer à cible inchangée
la politique actuelle et cette politique anticipative, puis rejouer leurs PW
avec Ding complet avant d'accepter une nouvelle valeur H30. Aucune tolérance
physique supplémentaire ni amélioration d'endurance ne sera présumée.

## Expérience : allocation anticipative sur quelques phases

La politique à une phase peut satisfaire la demande actuelle en laissant une
force résiduelle trop élevée pour la demande suivante. L'expérience suivante
ne répète donc pas les PW : elle choisit une redistribution sur `K=2` ou `K=3`
phases, avec `K=1` comme témoin. Quatre muscles donnent au plus 12 recrutements
dans ce calcul numérique auxiliaire, toujours **hors du NLP RHO**.

Pour une trajectoire nominale d'états lents, la force prédite s'écrit

\[
F_{i,j+1}=a_{i,j}F_{i,j}+b_{i,j}r_{i,j},\qquad
0\le r_{i,j}\le r_{i,\max}.
\]

Le recrutement `r` se convertit analytiquement en PW. Les coefficients sont
gelés pendant le petit QP, mais la force est propagée d'une phase à l'autre :
les décisions présentes changent donc les forces résiduelles futures. Le QP
minimise des écarts quadratiques aux moments musculaires de référence sous
les égalités de **moment total original** à chaque phase. Il conserve les
coefficients mécaniques signés et n'introduit pas de bande de suivi.

Le solveur auxiliaire est SLSQP, avec objectif quadratique et dérivées
analytiques; ce n'est pas le solveur du RHO. Les réglages initiaux sont
60 itérations au maximum, une régularisation du recrutement de `1e-8` après
normalisation commune du coût, une tolérance de moment de `1e-8 N·m` et une
tolérance KKT de `1e-6`. L'audit vérifie séparément faisabilité, stationnarité
et complémentarité. La limite de `0,5 s` est vérifiée lors des appels à
l'objectif : elle n'est pas un plafond dur pour la construction, l'audit KKT
ou le calcul complet d'une valeur locale.

La trajectoire nominale est une aide au calcul, pas une solution certifiée.
Si sa construction nécessite une projection interne sur une enveloppe,
celle-ci est comptabilisée et ne change pas les cibles du QP. Après résolution,
seule la première décision est appliquée avec la carte compacte originale;
les bornes, le domaine physiologique et le résidu de moment sont contrôlés,
puis le nominal et le QP sont reconstruits. Le preview est raccourci en fin
d'horizon pour ne pas imposer de contraintes au-delà de l'horizon annoncé.

Un QP rejeté ne doit pas produire un coût d'horizon partiel. Un repli éventuel
sur la politique à une phase doit être explicitement activé et comptabilisé;
il est désactivé dans la comparaison principale. Un arrêt anticipé peut
signifier que le petit QP ne trouve pas de séquence pour ses prochaines
phases, sans que la phase actuelle soit impossible. Son index d'arrêt n'est
donc pas, à lui seul, une mesure d'endurance physique.

L'adaptateur expérimental `preview_endurance_value.py` réutilise le score de
marges signées et l'audit des enveloppes aux états effectivement appliqués.
Il conserve l'identité de la politique et ses réglages dans les métadonnées.
Il est séquentiel : les gains du backend batch à une phase ne lui sont pas
attribués. Le code des campagnes RHO et le graphe IPOPT/MA57 restent inchangés.

Le protocole compare les mêmes ancres RHO 0 et 112, horizons 10 et 30 cycles,
conditions initiales, profils et tolérances. Les PW complètes sont rejouées
indépendamment avec Ding avant toute conclusion sur la fidélité dynamique.
La latence totale de prédiction et les éventuels échecs sont des critères de
décision, au même titre que les marges et les différences de fatigue. Un
meilleur rollout compact seul ne suffit pas à accepter une nouvelle valeur
terminale ni à revendiquer un gain d'endurance.

### Résultat et décision après le premier écran

Les trois politiques terminent 10 cycles aux deux ancres et 30 cycles à
l'ancre 0; les neuf rejeux Ding correspondants passent les critères numériques.
À l'ancre 112, les politiques à une, deux et trois phases terminent
respectivement 607, 636 et 605 phases sur les 900 demandées. Un contrôle
indépendant par programmation linéaire confirme l'incompatibilité des égalités
et des bornes des deux QP figés à leurs frontières d'arrêt. Cela ne prouve pas
l'infaisabilité du problème musculaire complet. Les résultats, réglages et
figures sont détaillés dans `preview_muscle_allocation_validation.md`.

Le preview à deux phases progresse davantage dans cet exemple, mais son
rollout H10 coûte environ `0,7 s`, avant calcul des marges et ajustement local;
le preview à trois phases coûte environ `1 s`. Aucun fit de 41 points sur les
ancres réelles n'a été chronométré pour ce backend. L'accélération batch du
greedy ne doit pas lui être attribuée. Cette étape ne justifie donc pas une
activation automatique dans le RHO.

La prochaine expérience sera une politique hybride : calcul rapide à une
phase, avec preview à deux phases seulement lorsqu'un critère de risque
annoncé à l'avance détecte une difficulté future. Elle changera explicitement
la politique prédite et devra être comparée au preview systématique, à cibles
et tolérances inchangées. Le critère de décision sera le temps **total** de
calcul et d'audit de la valeur locale, avec le suivi Ding et le classement
des valeurs, avant une comparaison RHO prospective avec IPOPT/MA57.
