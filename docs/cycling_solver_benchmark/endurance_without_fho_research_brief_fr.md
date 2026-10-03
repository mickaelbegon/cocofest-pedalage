# Maximiser l’endurance du pédalage FES sans FHO clinique

Note de recherche et dossier de questions pour d’autres LLMs — état des preuves locales au 2 octobre 2026.

Ce document rassemble le problème scientifique, les résultats numériques disponibles, les échecs utiles et les expériences qui pourraient départager les prochaines directions. Il concerne le modèle et les campagnes de ce dépôt. Il ne présente aucune méthode comme validée sur des patients.

## 1. Le problème à résoudre et le fait qui change la comparaison

Nous voulons produire un travail de pédalage imposé pendant le plus grand nombre de cycles possible, en réglant les largeurs d’impulsion de stimulation électrique fonctionnelle, avec des muscles qui se fatiguent et récupèrent selon un modèle de Ding. Le contrôleur utilisable en ligne doit rester court et rapide. Il ne peut pas résoudre, à chaque cycle clinique, une optimisation portant sur toutes les minutes restantes de l’exercice.

La question centrale est donc : **quelle information sur les conséquences futures des stimulations faut-il ajouter à un RHO court pour prolonger la réalisation de la tâche, sans avoir besoin d’une trajectoire FHO propre au patient ?**

Deux résultats motivent la recherche :

- Un BO de poids musculaires fixes atteint **234 cycles communs certifiés** dans la campagne bilatérale asymétrique à 1,92 Nm équivalents au total.
- Un PACE-RT avec manque de réserve normalisé, rollout de trois cycles et superviseur persistant atteint **169 cycles communs certifiés**.

Mais les fichiers révèlent une différence de protocole majeure : **les poids musculaires du BO sont fixes, tandis que sa répartition droite/gauche du travail est adaptative**. Le PACE-RT normalisé conserve **0,96 Nm par bras**. Le BO transfère une partie de la charge vers le bras droit, plus favorable dans ce modèle. Les deux nombres sont exacts pour leurs archives, mais leur différence de 65 cycles n’isole pas l’effet des seuls poids musculaires ou de la seule anticipation.

Ainsi, deux questions doivent être séparées :

1. Pourquoi la meilleure configuration opérationnelle BO observée atteint-elle 234 cycles alors que le PACE-RT considéré en atteint 169 ?
2. À répartition du travail strictement identique, quelle part du gain vient des poids fixes, de leur adaptation, d’un coût terminal prédictif ou de l’allocation entre bras ?

La première question possède déjà plusieurs éléments de réponse. La seconde nécessite des ablations appariées. Une proposition de nouvel algorithme qui ignore ce point commencerait par expliquer un écart causal qui n’a pas encore été identifié.

Sources principales : [résumé BO](../../asymmetric-sides-r192-rho-bo-20260928/bo-results/summary.json), [configuration de son meilleur candidat](../../asymmetric-sides-r192-rho-bo-20260928/bo-results/candidates/0ed1b7bec477b44f/cycles-0240/input.json), [configuration PACE-RT normalisée](../../asymmetric-sides-r192-rho-bo-20260928/pace-rt-refresh-40-20261002/normalized-endurance-200.json), [résultat PACE-RT normalisé](../../asymmetric-sides-r192-rho-bo-20260928/pace-rt-refresh-40-20261002/persistent-normalized-endurance-200-interactive-results/summary.json).

### 1.1 Statut des affirmations

Les mentions suivantes distinguent les niveaux de preuve :

- **[Mesuré]** : valeur ou comportement présents dans une archive, un reçu ou un rapport identifié. Une mesure reste conditionnelle au modèle, à la tâche, au solveur et au protocole de cette archive.
- **[Code]** : propriété de la formulation ou de l’implémentation inspectée. Elle ne prouve pas un gain de contrôle.
- **[Inférence]** : interprétation soutenue par plusieurs observations, mais qui n’a pas été isolée expérimentalement.
- **[Hypothèse]** : explication plausible à tester.
- **[Proposition]** : expérience ou méthode envisagée, sans résultat revendiqué.

Les archives peuvent décrire des versions différentes d’un composant. Le nom « PACE » a été employé pour plusieurs familles de contrôleurs. Il faut identifier le script, la configuration et le coût réellement connecté, pas seulement le nom du dossier. Le workspace comporte aussi du code expérimental en cours d’évolution : un comportement actuel du code ne doit pas être attribué rétroactivement à une campagne ancienne.

### 1.2 Ce que « sans FHO clinique » impose réellement

Le besoin est de se passer d’une optimisation complète de la séance propre au patient dans le fonctionnement requis du contrôleur. Cela n’interdit pas d’utiliser le modèle identifié, des simulations hors ligne, des continuations RHO, une calibration courte, une estimation d’état ou un modèle de valeur appris sans données FHO.

Il faut préciser séparément si l’on accepte :

- un BO hors ligne de poids constants sur le modèle identifié ;
- une bibliothèque de simulations sur une population de paramètres ;
- des calculs lents sur des cœurs séparés pendant l’exercice ;
- une mise à jour des paramètres physiologiques à partir de mesures ;
- un petit horizon futur exact, par exemple deux ou trois cycles ;
- un FHO uniquement comme référence de recherche hors ligne, jamais comme donnée nécessaire au contrôleur.

Le présent dossier traite la dernière possibilité comme un benchmark scientifique optionnel. Il ne suppose pas que les poids, les gradients ou les valeurs terminales puissent être appris sur une FHO patient disponible par avance.

## 2. La tâche physique : ne pas confondre résistance et travail isocinétique

### 2.1 Pédalage et membres représentés

Le problème actuel concerne le pédalage de manivelle avec deux bras, modélisés ici comme deux OCP unilatéraux indépendants, chacun comportant quatre muscles : deltoïde antérieur, deltoïde postérieur, biceps et triceps. Les exemples historiques comprennent aussi une formulation mécanique dynamique et des modèles bilatéraux couplés. Ces architectures ne sont pas interchangeables.

Les deux bras indépendants partagent cadence et horloge de manivelle, mais possèdent chacun leurs états musculaires, leurs commandes, leur warm start et leur solveur. Le compteur d’endurance bilatérale pertinent est le **préfixe commun certifié** : il s’arrête dès que l’un des bras n’a pas validé le cycle suivant.

Dans la campagne asymétrique considérée, les différences droite/gauche sont construites numériquement. Par rapport au cas de base, le côté droit augmente Fmax et `a_scale` de 5 %, diminue la valeur absolue de `alpha_a` de 5 % et diminue `tau_fat` de 5 %. Le côté gauche fait l’inverse. Il ne s’agit pas d’une identification clinique mesurée. Les paramètres individuels sont dans [model-right.json](../../asymmetric-sides-r192-rho-bo-20260928/model-right.json) et [model-left.json](../../asymmetric-sides-r192-rho-bo-20260928/model-left.json).

### 2.2 Formulation dynamique à résistance constante

Dans les campagnes dynamiques historiques, la vitesse de manivelle résulte des forces et de la charge. Pour la convention de rotation négative utilisée, un couple externe signé positif est résistant. Le contrôleur doit assurer le mouvement et sa fermeture, pas uniquement une quantité de travail.

Une petite erreur persistante de couple peut modifier la vitesse, puis le calendrier angle/stimulation, puis la géométrie musculaire et les forces. Le rejeu d’un profil PW dépend donc de son indexation temporelle, de l’état de départ et de la mécanique. Un cycle de PW répété ne crée pas automatiquement une orbite physique périodique.

Les campagnes à 0,15 ou 0,22 Nm de résistance dynamique ne peuvent pas être rangées dans le même tableau d’endurance que la campagne isocinétique à 1,92 Nm au total. Les nombres n’ont ni la même tâche ni la même définition de charge.

### 2.3 Formulation isocinétique à travail imposé

**[Code]** Dans la formulation isocinétique, la vitesse est prescrite :

\[
\omega^\star=-2\pi\ \mathrm{rad/s},\qquad T_{cycle}=1\ \mathrm{s}.
\]

Pour un bras dont le couple moyen équivalent demandé vaut \(\bar\tau_i\), le travail net par tour vaut :

\[
W_i^\star=2\pi\bar\tau_i.
\]

À 0,96 Nm par bras, cela correspond à `6,031857894892402 J` par bras et par cycle. Le total bilatéral de 1,92 Nm correspond à environ `12,0637 J/cycle`.

Ce n’est **pas** une prescription de couple instantané constant de 0,96 Nm. Le couple de charge instantané est inféré par l’équilibre inverse, sous ses bornes. Le RHO peut déplacer la production de travail entre phases du cycle, dans la mesure où les autres contraintes l’autorisent.

Pour la mécanique réduite documentée, avec coefficients de projection \(b_m(\theta)\), \(b_{ext}(\theta)\), et termes mécaniques \(g,c\) :

\[
\tau_{load}^{req}=\frac{g(\theta)+c(\theta)(\omega^\star)^2-\sum_m b_m(\theta)F_m}{b_{ext}(\theta)},
\]

\[
\dot E_{prod}=\left(\sum_m b_mF_m-g-c(\omega^\star)^2\right)\omega^\star.
\]

À rotation négative, le signe propulsif d’une contribution musculaire est celui de \(\omega^\star b_m F_m>0\). Sélectionner seulement des moments positifs ferait une erreur de signe. Une phase assistive doit diminuer le travail net contre la charge.

L’accumulateur `E_prod` est remis à zéro à chaque nouvelle fenêtre. Cette remise à zéro comptable ne remet pas les muscles au repos. Les états Ding, la force résiduelle, le calcium et l’historique requis doivent être transmis.

Les paramètres généraux et les seuils historiques sont décrits dans [l’OCP isocinétique](../isokinetic_rho_ocp.md). Ils ne doivent pas être supposés identiques dans toutes les campagnes : les bornes PW, les contraintes de demi-cycle futur, les options de slew et les tolérances effectives se relisent dans leurs configurations et reçus.

### 2.4 Pourquoi cette distinction transforme un proxy de réserve

**[Inférence]** Un proxy qui exige à chaque phase la puissance moyenne du cycle peut déclarer artificiellement une mauvaise réserve à une phase où le vrai problème accepte une production plus faible, compensée ailleurs. Un proxy qui maximise le travail positif de chaque muscle peut, à l’inverse, ignorer le travail de freinage, les bornes de couple simultanées et la persistance de force.

Le bon objet n’est donc ni automatiquement « force maximale restante », ni « moment maximal à chaque angle », ni « travail positif additionné ». Il dépend de **l’ensemble exact des tâches admissibles** : travail total, couple instantané, cadence, variation des PW, mémoire musculaire et conditions terminales.

## 3. Ce que le modèle de Ding mémorise

### 3.1 Cinq états par muscle et une commande de largeur d’impulsion

Les états musculaires sont :

\[
z_m=(C_{n,m},F_m,A_m,T_m,K_m),\qquad T_m=\mathrm{Tau1}_m,\quad K_m=K_{m,m}.
\]

`Cn` représente l’état calcique du modèle, `F` la force, `A` la capacité ou le facteur de production de force, et `Tau1`/`Km` deux états qui modifient la dynamique de force. La PW est la décision principale ; la fréquence de stimulation est fixée dans les comparaisons citées, typiquement à 30 Hz.

Une écriture correspondant aux équations décrites dans le dépôt est :

\[
\rho(u)=1-\exp[-(u-u_0)/u_t],\qquad \sigma(C,K)=\frac{C}{K+C},
\]

\[
\begin{aligned}
\dot C&=(H(t)-C)/\tau_c,\\
\dot F&=G(q,\dot q)\left[A\rho(u)\sigma(C,K)-\frac{F}{T+\tau_2\sigma(C,K)}\right],\\
\dot A&=-(A-A_r)/\tau_f+\alpha_A F,\\
\dot T&=-(T-T_r)/\tau_f+\alpha_T F,\\
\dot K&=-(K-K_r)/\tau_f+\alpha_K F.
\end{aligned}
\]

**[Code]** Dans cette implémentation, le gain mécanique \(G\) multiplie également le terme de relaxation de force. Une variante qui ne multiplierait que la production de force changerait le modèle. Le forçage calcique `periodic_node` utilise l’historique de stimulation selon une convention déterminée ; il ne faut pas reconstruire un autre train impulsionnel implicitement.

Source mathématique détaillée : [réduction analytique Ding/Radau-5](ding_analytic_reduction_radau5.md), avec liens vers les classes musculaires effectivement utilisées.

### 3.2 Réduction exacte possible, sous des hypothèses précises

Les trois états lents possèdent le même temps de récupération dans le modèle considéré et sont forcés par la même force. Sous force connue, leur évolution s’écrit :

\[
z(t+h)=z_r+e^{-h/\tau_f}(z(t)-z_r)+\alpha_z\int_0^h e^{-(h-s)/\tau_f}F(t+s)\,ds.
\]

Si \(\alpha_A\ne0\), les offsets

\[
d_T=(T-T_r)-\frac{\alpha_T}{\alpha_A}(A-A_r),\qquad
d_K=(K-K_r)-\frac{\alpha_K}{\alpha_A}(A-A_r)
\]

décroissent exponentiellement. Cela autorise une réduction conservant `A` et ces offsets. Cela n’autorise pas à les mettre à zéro pour un checkpoint quelconque.

Dans un intervalle sans nouvelle stimulation, le calcium est reconstructible exactement à partir de son état initial et du forçage local :

\[
C(t_k+s)=e^{-s/\tau_c}\left(C_k+H_ks/\tau_c\right).
\]

Le calcium reste continu à la frontière ; le forçage change. Remplacer `Cn` par son point fixe périodique à chaque reprise supprimerait un transitoire physique.

**[Inférence]** Cette structure fournit une voie crédible pour réduire le coût de simulation ou la dimension d’un modèle de valeur. Elle ne démontre pas que `A/A_rest` seul suffit pour choisir les PW : deux états de même capacité peuvent avoir des forces résiduelles, des offsets et des perspectives différentes.

### 3.3 Une projection rapide n’est pas une projection autonome de capacité

Une carte lente est exacte **conditionnellement au profil de force imposé**. Or ce profil dépend de la PW, de `Cn`, de la fatigue et de la mécanique. Répéter la force archivée pendant 300 cycles ne prouve pas que les PW bornées pourront encore la produire au cycle 300.

De même, fixer les forces futures peut supprimer la dépendance de la décision présente sur la répartition future. Un horizon plus long devient alors parfois surtout un facteur d’échelle, au lieu de fournir une nouvelle direction de décision. C’est une limite structurelle décrite dans [le recalibrage de réserve mécanique](mechanical_reserve_work_recalibration.md).

## 4. Ce que les contrôleurs optimisent réellement

### 4.1 Endurance, fatigue, charge de stimulation et réserve sont des objectifs différents

Une formalisation utile de l’endurance est le nombre maximal de cycles réalisables à tâche inchangée :

\[
N^*(x)=\sup_\pi\{N:\text{les cycles }1,\dots,N\text{ restent réalisables depuis }x\}.
\]

Cette définition concerne un état complet, une tâche et des contraintes. Dans la pratique, nous observons seulement le préfixe que notre contrôleur et notre procédure de résolution savent certifier, noté ici \(N_{obs}^{\pi}\). Sans preuve d’optimalité ni certificat global d’infaisabilité, \(N_{obs}^{\pi}\) n’est pas \(N^*\).

La fatigue cumulée peut être faible tandis qu’un muscle indispensable à certaines phases ne dispose plus de force exploitable. À l’inverse, une stratégie peut accepter une forte perte de capacité sur un muscle remplaçable et préserver plus longtemps la tâche. La quantité de stimulation, la fatigue quadratique, la capacité minimale et le travail maximal instantané ne classent donc pas nécessairement les politiques dans le même ordre que les cycles restants.

### 4.2 RHO unitaire

**[Code]** Le coût quadratique paramétrique utilisé dans ces conditions est de la forme :

\[
J_{fatigue}=10^4\int_{cycle}\sum_mw_m\left(1-\frac{A_m(t)}{A_{rest,m}}\right)^2dt.
\]

Le cas unitaire prend \(w_m=1\). Dans [custom_objectives.py](../../cocofest/custom_objectives.py), le vecteur retourné contient \(\sqrt{w_m}(1-A_m/A_{rest,m})\), puis l’objectif quadratique Bioptim effectue le carré. Mettre directement `w_m` dans ce vecteur ferait apparaître \(w_m^2\).

Ce coût porte sur la **perte de capacité déjà accumulée**, pas directement sur sa dérivée et pas sur les cycles futurs. Une petite perte supplémentaire \(\Delta d_m\) pour \(d_m=1-A_m/A_{rest,m}\) coûte approximativement \(2w_md_m\Delta d_m\), hors variation temporelle et termes d’ordre supérieur. La pénalisation marginale augmente donc déjà avec la fatigue courante.

Un RHO d’un cycle minimise cette quantité sur son horizon court. Il ne reçoit pas automatiquement le prix de la capacité qui deviendra indispensable dans cent cycles.

### 4.3 RHO-Physio

Les poids physiologiques cherchent à combiner fatigabilité et utilité mécanique. La reproduction des formules de l’annexe du travail de référence a été étudiée séparément des performances d’endurance. Le score initial est notamment lié à une pente de fatigue sous force prescrite et à une contribution dans les régions mécaniques difficiles.

**[Mesuré]** La reproduction de la figure S4 donne les poids `1 ; 0,0943161 ; 0,3892687 ; 0` dans l’ordre deltoïde antérieur, deltoïde postérieur, biceps, triceps. Ce zéro n’est pas une loi générale sur l’utilité du triceps. Une autre variante du panel lui donne un poids positif sans plancher arbitraire.

**[Mesuré]** Les 69 cas de calibration fondés sur les paramètres locaux du panel inspecté sortent tous du domaine de la calibration historique à force élevée imposée ; le nominal local atteint une capacité antérieure non positive dès le 50e cycle de cette calibration. Cela ne prédit pas l’arrêt du RHO au cycle 50 : la calibration impose un profil de force que le RHO n’impose pas.

Autre différence importante : l’équation de fatigue de l’article discutée dans les notes intègre une racine de somme pondérée des pertes **absolues** de capacité au carré. Le coût quadratique relatif du code courant est une autre fonction. Retirer une racine sous une intégrale ou normaliser chaque muscle sans transformer les poids ne conserve généralement pas le même optimum.

Source : [validation des poids physiologiques](physiological_weights_validation.md) et [plan multi-échelle](multilevel_endurance_plan.md). Les 1230/1692/1729 cycles rapportés pour le modèle de l’article appartiennent à un autre protocole ; ils ne sont pas des résultats de la campagne actuelle à 1,92 Nm.

### 4.4 PACE causal fondé sur la capacité

Une famille PACE ajuste lentement les poids à partir de la capacité observée. Une écriture décrite dans le dépôt est :

\[
\log w_m^{target}=\log w_m^{initial}-k\log(A_m/A_{rest,m}),
\]

suivie d’une normalisation, d’une projection dans une boîte et d’un lissage borné. Un muscle plus déplété est davantage pénalisé.

Cette règle est causale et peu coûteuse. Elle n’a cependant pas d’information explicite sur les futures substitutions entre muscles. Puisque le coût quadratique protège déjà davantage un muscle dont la perte accumulée est élevée, augmenter encore son poids peut accentuer cette protection. **[Hypothèse]** Il peut s’agir d’une double pénalisation utile dans certains cas et excessive dans d’autres.

Les variantes Physio-U réévaluent les poids depuis l’état courant et une contribution au travail, notamment via un score de Shapley sur les groupes musculaires. Elles posent un problème distinct : la fidélité de l’enveloppe de travail utilisée pour attribuer cette contribution. Voir [la proposition Physio-U](physio_update_isokinetic_proposal.md).

### 4.5 BO de poids fixes

Le BO effectue des simulations RHO complètes pour rechercher des poids qui améliorent directement un score d’endurance opérationnel. Il n’utilise pas nécessairement une FHO. Dans la représentation actuelle, il recherche des coordonnées de log-poids relatifs centrés ; pour quatre muscles, trois coordonnées suffisent par bras, la moyenne géométrique restant égale à un.

L’espace des poids effectifs doit être distingué des anciens multiplicateurs de poids physiologiques. Une normalisation suivie d’une projection peut transformer plusieurs propositions différentes en un même vecteur appliqué. Les nouveaux reçus permettent de vérifier le vecteur réellement utilisé.

**[Inférence]** Le BO reçoit une récompense bien plus proche du but que le minimum de fatigue sur un cycle. Il peut ainsi découvrir qu’il faut davantage « dépenser » certains muscles. Son avantage ne démontre pas la nécessité d’une commande temporellement complexe : une mauvaise pondération locale peut être une grande part du problème. Son coût de recherche, sa variabilité et sa généralisation restent à mesurer.

### 4.6 PACE prédictif, PACE-VR et PACE-RT

Ces appellations recouvrent des mécanismes différents :

| Famille | Information future | Action transmise au RHO | Limite principale |
|---|---|---|---|
| PACE causal | Capacité déjà observée | Poids de fatigue | Heuristique, sans classement de futurs |
| PACE prédictif historique | Rollouts compacts de plusieurs candidats | Poids retenus par un garde | La politique compacte peut différer du RHO |
| PACE-VR | Réallocation réduite des PW, score de travail/réserve et fit local | Selon la version, poids issus des sensibilités ou sélection de candidats | Une bonne sensibilité ne devient pas automatiquement un bon poids |
| PACE-RT | Rollout compact et modèle local autour du terminal du prochain cycle | Coût terminal à cible bornée et proximité | Objectif conditionnel, domaine local et poids relatif à la fatigue |
| PACE-V proposé | Vraies branches RHO et cycles restants | Choix d’une courte intervention puis retour à une référence | Collecte hors ligne et validation de la valeur |

PACE-RT maintient le graphe NLP fixe et met à jour des paramètres numériques. Sa version inspectée propage les cinq états Ding dans le rollout, mais identifie seulement des directions dans `A/A_rest`, les offsets de mémoire `Tau1`/`Km` restant conditionnés. Trois directions supplémentaires sont utilisées avec `fit_max_samples=3` ; ce n’est pas une identification libre du gradient dans tous les états physiques.

Le modèle affine prédit un **coût** local \(\widehat V(q)\), à diminuer. La cible est construite comme une fraction de la baisse maximale prédite dans une boîte :

\[
\Delta V_{box}=\sum_m|g_m|r_m,\qquad V_{target}=V_0-\eta\Delta V_{box}.
\]

Le coût terminal ajoute une pénalité lissée sur le dépassement de cette cible et une proximité normalisée au centre. Le signe « coût à diminuer » doit rester explicite : parler de réserve à maximiser sans changer le signe du gradient inverserait la décision.

Cette cible est calculée sur une boîte de coordonnées, **pas sur l’ensemble des terminaux réellement atteignables par le RHO**. Même un modèle affine parfaitement ajusté ne démontre pas que sa meilleure direction soit atteignable.

### 4.7 FHO : intérêt réel et limites du raisonnement

Une FHO choisit les stimulations sur un horizon long simultanément. Elle peut coordonner l’usage et la récupération des muscles à travers de nombreux cycles, au prix d’un problème beaucoup plus grand. Elle peut trouver des répartitions qu’un RHO court, mal doté en valeur terminale, ne découvre pas.

Il faut toutefois distinguer :

- un FHO de durée fixée minimisant la fatigue ;
- un problème qui maximise explicitement la durée réalisable ;
- une recherche progressive du plus grand horizon pour lequel on trouve une solution ;
- l’optimum global théorique, rarement certifié par ces NLP non convexes.

**[Inférence]** Une solution FHO faisable plus longue montre qu’une politique peut aller plus loin dans cette formulation. Un échec local du FHO ne donne pas une borne supérieure globale d’endurance. Et un FHO minimisant un coût de fatigue différent n’est pas automatiquement une référence optimale en nombre de cycles.

**Limite actuelle : aucune archive FHO strictement appariée et vérifiée aux deux modèles asymétriques, à la charge totale de 1,92 Nm et aux autres contraintes du BO/PACE-RT ci-dessus n’a été établie dans cette revue.** La possibilité pour une FHO d’aller plus loin est une motivation conceptuelle et historique ; aucun nombre FHO n’est ajouté à leur classement de 234 et 169 cycles.

La [comparaison historique RHO113/FHO113](../../comparison-rho113-fho113/comparison.md) montre des trajectoires différentes sur un horizon égal, dont une RMSE PW de 51,59 µs. Elle ne mesure pas un écart d’endurance de cette campagne bilatérale. « Full/reduced » peut en outre qualifier la mécanique, tandis que « FHO/RHO » qualifie l’horizon : les deux axes doivent rester séparés.

## 5. Les résultats centraux, avec leurs limites exactes

### 5.1 BO : 234 cycles communs et des poids effectivement fixes

**[Mesuré]** Le fichier BO contient 58 entrées d’essai. Le meilleur est le numéro 55, candidat `0ed1b7bec477b44f`, avec l’objectif `common_certified_cycles` et la valeur 234. La fidélité courte valide 80 cycles ; la continuation demandée jusqu’à 240 s’arrête après 234 cycles communs.

| Muscle | Poids droit appliqué | Poids gauche appliqué |
|---|---:|---:|
| Deltoïde antérieur | 1,283620 | 0,771299 |
| Deltoïde postérieur | 1,322479 | 1,967750 |
| Biceps | 0,349946 | 0,310955 |
| Triceps | 1,683345 | 2,118900 |

La relecture de `weights_used` sur les cycles certifiés ne trouve qu’un vecteur unique par bras. Les poids musculaires sont donc bien fixes dans cet essai. Le bras droit certifie 234 cycles puis échoue à la tentative 235 ; le gauche certifie encore sa tentative 235. Le préfixe commun est 234, et non 235.

Le champ `observation.reason` vaut `fatigue_counterfactual_certified_with_pw_saturation`. Au dernier échec droit, le test à objectif nul ne trouve pas de témoin faisable ; les contre-factuels avec travail diminué de 0,5 %, PW maximale augmentée de 5 %, ou états lents remis au repos trouvent des témoins. Cela soutient une limitation conditionnelle liée à la capacité et aux bornes de stimulation, dans le protocole numérique. Le mot `certified` du libellé ne transforme pas l’échec d’un solveur local en preuve globale d’infaisabilité.

### 5.2 Le facteur de confusion : une répartition du travail adaptative dans le BO

**[Mesuré]** `resistance_pace.capacity_feedback=true` dans cet essai BO, avec actualisation tous les 20 cycles. Sur les cycles certifiés :

| Grandeur | Droite | Gauche |
|---|---:|---:|
| Couple équivalent initial | 0,960000 Nm | 0,960000 Nm |
| Minimum observé | 0,960000 Nm | 0,861327 Nm |
| Maximum observé | 1,058673 Nm | 0,960000 Nm |
| Couple au cycle commun 234 | 1,048188 Nm | 0,871812 Nm |

La somme reste 1,92 Nm. Ce résultat est donc un **BO de poids fixes sous superviseur de partage du travail adaptatif**, et non un contrôleur entièrement statique à split 50/50.

**[Inférence]** Le changement du bras limitant — gauche dans le PACE-RT fixe, droite dans le BO — est cohérent avec une utilisation accrue de la réserve droite. Il ne quantifie pas à lui seul l’apport du partage de travail, car les poids musculaires diffèrent aussi.

### 5.3 PACE-RT normalisé : 169 cycles communs

Le résultat retenu ici est explicitement le dossier `persistent-normalized-endurance-200-interactive-results`. Il ne doit pas être confondu avec l’ancienne campagne à quatre conditions ni avec le lancement persistant au nom proche.

Sa configuration déclare :

- tâche isocinétique, 30 stimulations par cycle, un cycle par fenêtre, IPOPT ;
- 0,96 Nm par bras, partage 50/50 fixe, `capacity_feedback=false` ;
- poids de fatigue unitaires fixes ;
- superviseur asynchrone persistant, horizon compact H=3 ;
- mise à jour demandée tous les **2 cycles**, âge maximal **3 cycles**, deadline **2 s** ;
- cible à 25 % de la baisse locale estimée, proximité `0.03`, manque de réserve `1.0` ;
- `normalize_terminal_shortage=true`, rayon de fit `0.01`, trois directions de fit au maximum.

Le document général [pace_rt_validation.md](pace_rt_validation.md) décrit aussi un protocole initial à actualisation tous les dix cycles. Cette cadence ne doit pas être attribuée au résultat normalisé considéré ici.

**[Mesuré]** Le droit certifie 170 cycles, le gauche 169 puis échoue à la tentative 170. `completed_rho_cycles=170` compte ici la frontière tentée ; le score commun certifié vaut donc 169.

**[Mesuré]** Le terme terminal est actif sur 89 des 170 cycles droits certifiés et 129 des 169 cycles gauches certifiés. Il commence au cycle 4. Son dernier cycle actif gauche est 141 ; il n’est plus actif au cycle gauche 169. Cette campagne ne représente donc pas 169 cycles d’intervention terminale continue.

Le journal du superviseur contient 85 soumissions, 73 événements `proposed` et 23 `held`. Les 85 retours gauches comprennent 70 `complete`, un `deadline_expired` et 14 `projected_cycle_not_validated`. Les retours droits comprennent 51 `complete`, 32 `deadline_expired` et deux retours sans statut de rollout. Les événements ne sont pas tous des cycles physiques : ne pas additionner ces compteurs comme des cycles d’endurance.

Au snapshot gauche 168, le rollout ne valide même pas son premier cycle : déficit de travail du premier échec d’environ `0,03628 J`, malgré un message de succès du petit QP. Le fit est refusé parce que le terminal nominal suivant est indisponible. Le résultat n’affirme pas une impossibilité physiologique globale ; il rapporte une défaillance du prédicteur/allocation local.

**[Mesuré]** À l’arrêt réel gauche, la sonde gelée à objectif nul ne trouve pas de témoin. Une diminution de travail de 0,5 % ne suffit pas dans les relances présentes ; une diminution de 2 %, une augmentation de la borne PW de 5 %, ou le contre-factuel `A/Tau1/Km` au repos permettent une résolution faisable. Ces contre-factuels conservent leurs modifications explicites et n’ajoutent aucun cycle au résultat nominal.

### 5.4 La normalisation a activé une influence, pas démontré un gain d’endurance

**[Mesuré]** Sur le pilote de 40 cycles, le PACE-RT non normalisé était indiscernable de « proximité seule » à moins de `1e-6` en capacité. Le score de rollout avait une échelle trop petite par rapport au terme proximal adimensionnel.

La normalisation divise le manque de réserve et son lissage par une échelle fondée sur \(\sum|g_m|r_m\), avec un plancher fini. Elle change effectivement les solutions : dans le rapport, la capacité terminale du triceps droit passe de 0,84187 à 0,84309, tandis que le minimum gauche passe de 0,80389 à 0,80366.

Cela prouve une influence numérique du terme sur le RHO, pas une amélioration globale de la stratégie. La droite gagne sur une coordonnée, la gauche en perd légèrement sur une autre ; ce sont des conséquences à interpréter au regard de la tâche, pas des scores d’endurance.

### 5.5 Autres résultats, utiles mais à ne pas amalgamer

La [comparaison quatre conditions r2](../../asymmetric-sides-r192-rho-bo-20260928/pace-rt-validation-20261002-r2/comparison.json) indique les préfixes communs suivants : unitaire 169, physio 148, PACE causal 170, PACE-RT de cette campagne 169. Tous y portent encore `stopped_requires_frozen_review` dans le rapport. Le protocole y annonce H=3 et une cadence de dix cycles.

Cette table est un contexte utile, mais elle n’est pas l’expérience factorielle appariant le meilleur BO au PACE-RT normalisé K=2. Il faut conserver les variantes et leurs mécanismes d’arrêt distincts.

### 5.6 Pourquoi une capacité minimale plus basse n’est pas automatiquement pire

Au dernier cycle commun du BO, les capacités relatives sont :

| Bras et cycle | Deltoïde antérieur | Deltoïde postérieur | Biceps | Triceps |
|---|---:|---:|---:|---:|
| BO droit, 234 | 0,5370 | 0,5786 | 0,2888 | 0,5299 |
| BO gauche, 234 | 0,4446 | 0,6246 | 0,2603 | 0,4745 |
| PACE-RT droit, 169 | 0,9661 | 0,8122 | 0,7573 | 0,5870 |
| PACE-RT gauche, 169 | 0,5835 | 0,5804 | 0,4266 | 0,4745 |

Ces états sont mesurés à des durées et des allocations différentes ; leur comparaison n’est pas un effet causal à état identique. Elle suffit néanmoins à interdire un raisonnement naïf du type « toute capacité minimale plus basse est une politique moins endurante ». Le BO a réalisé davantage de cycles tout en acceptant un biceps beaucoup plus déplété. La disponibilité mécanique restante dépend de la répartition complète, de la phase et des autres états.

## 6. Chronologie raisonnée des méthodes essayées

Cette chronologie retrace les familles utiles au problème d’endurance, pas tous les commits. Les dates sont celles des rapports et campagnes. Une correction plus récente peut rendre une description de code ancienne obsolète sans changer ce que son expérience avait mesuré.

| Période / famille | Idée et résultat documenté | Enseignement pour la suite |
|---|---|---|
| Mise en place des références RHO/FHO et full/reduced | Comparaisons de solveurs, contraintes mécaniques, warm starts et certification | Une fin de solveur n’est pas une fin physiologique ; un changement de mécanique peut changer le problème |
| Début septembre : primitive de capacité terminale | Soft-min de `A/A_rest`, dommages marginaux, projection à forces figées | Faible dimension et faible coût ne garantissent pas une métrique liée à la tâche |
| 10 septembre : rollout de moment strict | Depuis l’ancre 112, arrêt après 607 phases à H30 ; déficit confirmé au rejeu Ding | Une force résiduelle peut rendre la baisse de moment impossible pour cette allocation |
| Fits locaux H10 | Fit acceptable au repos ; près de l’arrêt, boîte divisée par 16 avant acceptation | Précision locale et étendue exploitable du domaine sont deux exigences différentes |
| Preview à deux phases K2 | K2 systématique modifie les PW et allonge un préfixe compact ; le preview déclenché retrouve souvent H1 | Anticiper seulement au voisinage immédiat de la défaillance peut être trop tard ou trop peu influent |
| 11–12 septembre : PACE prédictif et garde | Un meilleur score compact pouvait détériorer la fatigue réelle ; garde sur fatigue, capacité minimale et déficit | Une métrique intermédiaire doit être validée contre la décision du vrai RHO |
| 12–13 septembre : coût numérique | Radau, fréquence, slew, SX/MX, compilation et réduction analytique Ding | Accélérer sans conserver la précision physique peut déplacer artificiellement l’arrêt |
| 19–20 septembre : deux échelles, playback et retour de phase | Certains replays d’un cycle passent, tous les candidats cinq cycles du screen de phase échouent | Répéter une PW ou corriger proportionnellement la phase ne remplace pas un prédicteur musculaire fermé |
| 22–25 septembre : campagnes de poids et fatigue | Saturation des allocations QP ; variations de poids sans changement de commande | Un superviseur peut être actif informatiquement et inopérant sur la politique |
| 25 septembre : Physio-U et PACE-V proposés | Contribution au travail depuis l’état courant ; valeur de vraies continuations RHO | Deux réponses différentes au décalage entre proxy et endurance |
| 26–28 septembre : deux bras et asymétries | Recherche BO avec paramètres et poids propres à chaque côté ; résultat BO 234 | Distinguer poids par muscle et partage de travail entre bras |
| 28 septembre : réserves mécaniques / travail signé | Correction du signe et de fausses exigences de puissance par phase ; tangentes plus rapides | Corriger le sens physique d’un proxy précède l’optimisation de ses dérivées |
| 29–30 septembre : PACE-VR H100 | Rollouts réallouant les PW, condensation, QP persistant, écrans de candidats | H100 demandé n’équivaut pas à H100 validé ; l’ordre des candidats peut être erroné |
| 30 septembre : checkpoints et réserve de tâche | Reprises exactes instrumentées ; grille de travail au-dessus du nominal | Une grille plafonnée fournit des témoins de réserve, pas la frontière maximale |
| 1er octobre : oracle de charge et sensibilités KKT | Au gauche c140, témoin 1,180210 fois le travail nominal ; gradients locaux validés | Une marge d’un cycle plus exacte est possible, mais reste distincte des cycles restants |
| 1er octobre : proxy PW-max de travail positif | Utilise les cinq états ; intégration directe dans le NLP coûteuse, linéarisation terminale retenue | Le gradient rapide peut favoriser la force résiduelle sans préserver la tâche future |
| 2 octobre : PACE-RT, persistance, normalisation | Influence terminale réelle après normalisation, mais 169 cycles communs au run retenu | Il faut mesurer activation, confiance, fidélité décisionnelle et conflits d’objectif |
| 2 octobre : réserve prioritaire | Mécanisme de cible compilée et de second objectif disponible ; essais margin-only/poids faibles exploratoires | Une réduction de poids de fatigue n’est pas encore une hiérarchie lexicographique validée |

Les sources détaillées sont regroupées en fin de document. Les plans `rho_endurance_horizon_plan.md`, `multilevel_endurance_plan.md` et `pace_value_policy_proposal_20260925.md` contiennent également des propositions qui n’ont pas toutes été exécutées.

## 7. Ce que les échecs de rollout et de fit ont réellement appris

### 7.1 Échec du rollout strict par surproduction résiduelle

**[Mesuré]** Dans [le diagnostic H30 à l’ancre 112](compact_rollout_failure_diagnostic.md), la demande de moment vaut `0,0846647707961 Nm`. La borne inférieure compacte vaut `0,084809796369 Nm`, soit une surproduction minimale de `1,45026e-4 Nm`.

Le rejeu Ding complet depuis la frontière issue du préfixe indépendant conserve un déficit de `6,50881e-5 Nm` avec 128 sous-pas. Le raffinement final change les bornes de moins de `8,38e-8 Nm`. Le signe de l’échec ne disparaît donc pas par simple raffinement d’intégration.

Mais le préfixe rejoué garde environ `4,79e-4 Nm` d’écart maximal au moment cible historique. Il ne s’agit pas d’une autre politique Ding qui aurait réoptimisé les PW pour suivre exactement les cibles à chaque phase. La conclusion est un échec conditionnel de cette allocation et de ces PW, pas l’absence de toute politique réalisable.

Le fait que certaines contributions soient antagonistes est crucial : le PW maximal d’un muscle à coefficient négatif peut contribuer à la borne **inférieure** du moment total. « Tout au minimum » et « tout au maximum » ne sont généralement pas les deux bornes de moment signées.

### 7.2 Un fit exact à l’échelle du score peut mal classer les décisions

**[Mesuré]** Le [fit local historique H10](local_endurance_value_validation.md) de l’ancre 112 commence avec 14 inversions de classement, malgré une erreur absolue maximale de `1,52e-3` jugée acceptable par le seul seuil de valeur. Quatre réductions successives de rayon sont nécessaires avant d’éliminer les inversions testées ; le fit accepté demande 45,62 s cumulées et une boîte seize fois plus petite.

Il faut donc mesurer à la fois : erreur de valeur, erreur de dérivée, classement des candidats, regret de la décision et taille du domaine. Un fit à très faible erreur dans une boîte que le RHO quitte systématiquement a peu d’utilité de contrôle.

L’oracle de ce test agrège 600 marges avec un soft-min. Son signe peut devenir négatif alors que la marge physique minimale reste positive, par le biais de `log-sum-exp`. Changer l’horizon change aussi ce biais. Il est incorrect d’interpréter mécaniquement le signe du score lissé comme un certificat de faisabilité.

### 7.3 Un preview plus cher peut ne transmettre aucune information nouvelle

**[Mesuré]** Le [preview K2 déclenché](triggered_preview_allocation_validation.md) réduit les appels de QP comparé au K2 systématique, mais produit exactement les mêmes PW que le glouton sur les préfixes communs testés. Ses fits de 41 points prennent 17–18 s et reproduisent les mêmes valeurs locales.

**[Inférence]** Le problème n’est pas seulement de calculer plus vite. Si le nouvel allocateur retrouve la même commande et le même classement, son horizon supplémentaire n’améliore pas l’information fournie au RHO. Il faut d’abord montrer une décision différente et utile sur un cas discriminant.

### 7.4 Saturation et manque de sensibilité aux poids

**[Mesuré]** Dans [l’audit de régularisation PACE H50](pace_regularization_audit_20260925.md), cinq projections tardives donnent des métriques identiques pour toutes les alternatives. Sur 61 alternatives, 56 sont rejetées pour absence de gain matériel. Le petit terme quadratique du QP est dominé par son terme linéaire de fatigue ; environ 78 % des recrutements sont aux bornes aux faibles régularisations testées.

Modifier les poids de 10 % peut alors laisser exactement le même ensemble actif. Augmenter la régularisation restaure une sensibilité dans le modèle compact, mais change la politique simulée. Le meilleur réglage est celui qui prédit la réponse du véritable RHO, pas celui qui produit le score compact le plus flatteur.

### 7.5 La sensibilité d’une valeur ne donne pas directement un poids de fatigue

Un gradient terminal \(g=\partial V/\partial x\) et un vecteur de poids \(w\) du coût de fatigue ne sont pas le même objet. Pour convertir une direction souhaitée en poids, il faut tenir compte de la réponse du RHO : \(\partial x_{k+1}/\partial w\), des contraintes actives et de l’échelle des coûts.

**[Mesuré]** Sur le snapshot droit c100 de [l’audit PACE-VR de candidats](pace_vr_candidate_fidelity.md), les quatre candidats terminent H20, mais les marges les classent dans l’ordre : pas opposé `0,046972`, courant `0,046692`, demi-pas `0,046509`, proposition `0,046293`. La proposition effectivement appliquée est dernière selon son propre surrogate, avec un écart de `0,000680` au pas opposé.

Ce résultat réfute une conversion automatiquement bénéfique « sensibilité → poids » pour ce cas. Il ne prouve pas encore lequel des candidats est le meilleur en continuation RHO réelle.

### 7.6 La précision de propagation ne suffit pas à la fidélité de décision

La condensation du propagateur peut reproduire sa version séquentielle à l’arrondi près, sans corriger la politique réduite ni la géométrie figée. Le rejeu DOP853 à PW identiques valide une dynamique de propagation, pas la réponse de la politique lorsque les PW sont à nouveau optimisées depuis des états différents.

Il faut garder trois tests distincts :

1. mêmes PW, comparaison de trajectoire entre intégrateurs ;
2. mêmes états, comparaison des commandes de deux politiques ;
3. mêmes checkpoints, comparaison du classement des politiques par le surrogate et par leur endurance réellement observée.

### 7.7 Le fit de PACE-RT n’est pas identifié dans toutes les directions utiles

**[Code]** Le fit actuel utilise des contrastes de capacités normalisées ; il n’identifie pas librement les directions rapides `Cn/F` ni les offsets indépendants `Tau1/Km`. Trois directions contrastées ne donnent pas, à elles seules, toutes les sensibilités possibles d’un état musculaire complet.

**[Hypothèse]** Une composante manquante peut être le mouvement commun des quatre capacités, ou une combinaison de capacité et force résiduelle qui change l’atteignabilité du cycle suivant. Il faut tester les directions effectivement produites par le RHO, car un gradient très précis dans un sous-espace non visité n’aide pas la commande.

### 7.8 Les modèles locaux sur une seule trajectoire peuvent être non identifiables

**[Mesuré]** Le [fit global de réserve de tâche](task_reserve_local_experiment.md) limité aux quatre `A/a_scale` sur la trajectoire gauche est rejeté, avec erreurs maximales `0,119` en apprentissage et `0,203` en holdout corrigé, et conditionnement proche de 88 322. Le rang algébrique complet n’a pas suffi.

Les quatre capacités covarient au cours du temps. Une très faible courbure peut créer un rang numérique complet sans renseigner l’effet d’une redistribution indépendante. L’identification doit employer des branches atteignables qui excitent les contrastes pertinents, et non compter les cycles voisins comme des expériences indépendantes.

## 8. Asynchronisme : une contrainte scientifique autant qu’informatique

### 8.1 Quatre temps différents

Pour chaque proposition, il faut distinguer :

- le cycle source dont l’état a été copié ;
- la durée de calcul du rollout et du fit ;
- le délai de transport, d’attente et de récupération du résultat ;
- le cycle auquel les paramètres entrent réellement dans le RHO.

Un calcul proprement terminé dans un worker peut être trop vieux ou hors domaine au moment de son application. Inversement, un échec de deadline ne prouve pas que le noyau numérique est trop lent : initialisation de processus, contention CPU, attente et transport peuvent dominer.

### 8.2 Ce qui a été observé

**[Mesuré]** L’[audit d’accélération PACE-VR](pace_vr_rollout_acceleration.md) compare deux threads superviseurs à deux processus numériques `spawn`, sur les mêmes snapshots : 10,31 s contre 6,00 s de temps mural démarrage/fermeture inclus, avec les mêmes préfixes 86/67 et les mêmes refus scientifiques. Une autre répétition à threads atteint 18,64 s. Ce résultat décrit ce benchmark, sans attribuer tout l’écart au seul GIL.

**[Mesuré]** Le run PACE-RT normalisé retenu possède déjà un superviseur persistant et deux CPU distincts de ceux du RHO. Il reste pourtant des expirations de deadline et des périodes d’inactivation, surtout à droite pour les délais et à gauche près de la limite du rollout. Le terme est désactivé lorsqu’aucun modèle admissible ne couvre le terminal actuel.

Sa durée murale totale est environ 399,56 s pour 170 frontières tentées. La médiane de paire enregistrée est 1,591 s, le p95 2,866 s ; l’ultime frontière coûte 92,02 s avec les diagnostics/reprises. La moyenne de 2,321 s inclut cette frontière. Ces chiffres ne démontrent donc pas une chaîne complète sous une seconde par cycle, même si les noyaux compacts sont beaucoup plus rapides.

### 8.3 Question de conception non résolue

Une limite d’âge de trois cycles protège contre un retard grossier. Elle ne garantit pas la pertinence de la direction depuis l’état actuel. Une proposition devrait idéalement être réévaluée sur l’état vivant, ou être accompagnée d’une borne empirique de dégradation en fonction du déplacement d’état.

**[Proposition]** Une expérience hors ligne doit séparer le mérite scientifique du modèle et la livraison asynchrone : mêmes checkpoints et même modèle, avec application immédiatement disponible, retard imposé de 1/2/3 cycles, puis véritable ordonnanceur. Si le modèle n’améliore rien avec livraison parfaite, accélérer le transport ne résout pas le problème central.

## 9. Intégrateurs, solveurs et coûts numériques

### 9.1 Radau-5 ne signifie pas « ordre cinq » dans ces rapports

La référence appelée Radau-5 utilise cinq stages de Radau IIA, d’ordre classique neuf pour une solution suffisamment régulière. Radau-3 utilise trois stages. Aux pas de stimulation utilisés, les transitoires rapides et la non-linéarité peuvent empêcher d’atteindre immédiatement l’ordre asymptotique.

**[Mesuré]** À PW fixées et 30 Hz, le test triceps donne une erreur relative de force terminale d’environ `7,081e-3` pour Radau-3 à un élément par stimulation et `6,251e-4` pour Radau-5. Sur le cas dynamique historique à 50 Hz, l’erreur de phase citée est `0,003352 rad` contre `0,000104859 rad`. Le budget de transcription proposé était `0,0002 rad`.

Passer à Radau-3 n’est donc pas une accélération neutre déjà validée. Source : [ordre Radau observé](ding_radau_observed_order.md).

### 9.2 DOP853 et replays indépendants

DOP853 est utilisé comme référence indépendante de propagation pour plusieurs validations. Il faut couper correctement aux instants de stimulation et conserver le même historique. Les états issus du solveur ne doivent pas réinitialiser le rejeu à chaque intervalle si l’on cherche l’erreur cumulée d’un préfixe.

Un résidu discret NLP très petit ne donne pas la même précision en temps continu. Le document isocinétique historique distingue notamment une égalité de travail de transcription à `1e-6 J` d’une tolérance de replay de smoke test à `0,02 J`. Les erreurs observées de replay, les bornes entre nœuds et la convergence temporelle doivent rester visibles.

### 9.3 IPOPT/MA57, MadNLP, FATROP et acados

IPOPT/MA57 avec Radau-5 sert de référence fréquente, grâce à sa robustesse et aux outils d’audit. MadNLP et FATROP ont été évalués dans les campagnes de solveurs ; acados fournit une transcription et des méthodes SQP/IRK adaptées à des résolutions rapides. Mais obtenir un solve plus rapide ne prouve ni un optimum identique ni un arrêt physiologique identique.

Les points qui doivent rester appariés sont le modèle continu, les contraintes, le coût, la discrétisation ou son erreur contrôlée, le seed, les limites d’itération, le transfert des états, le warm start et les critères de certification. Un hybride acados avec recovery IPOPT doit être décrit et chronométré comme hybride. Un temps acados qui exclut la préparation ou la reprise ne mesure pas le délai clinique complet.

Les rapports de référence sont [l’historique détaillé](development_history.md), [le benchmark principal](README.md) et [la comparaison croisée IPOPT/acados](ipopt_acados_cross_evaluation.md).

### 9.4 Les gains exacts sont préférables aux approximations non auditées

Les réductions du calcium et des états lents peuvent être appliquées au niveau continu ou par condensation des équations Radau déjà discrétisées. Ces deux opérations ne sont pas identiques. La condensation algébrique conserve le problème discret lorsque tous les termes, bornes et multiplicateurs sont reconstruits correctement.

**[Mesuré]** Sur neuf appels chauds gelés du [test local Ding/Radau-5](ding_radau5_local_ab.md), la médiane native IPOPT passe de 1,132 s à 0,668 s, avec 10/10 appels réussis et violations originales sous le seuil. Le test impose les mêmes entrées et utilise la trajectoire baseline pour le transfert externe : c’est un benchmark de résolution à entrées égales, pas une démonstration d’endurance améliorée.

Un A/B en boucle fermée peut diverger simplement parce que la reformulation change le chemin de barrière d’un NLP non convexe et sélectionne un autre bassin local. C’est précisément pourquoi les benchmarks de temps à problème gelé et les expériences de contrôle doivent être distingués.

### 9.5 Rollout compact et petit QP

L’accélération PACE-VR exploite des récurrences affines par intervalle, la vectorisation des sous-pas, le retrait de contraintes de force redondantes sous des hypothèses vérifiées, le report des LP d’enveloppe et un QP qpOASES persistant.

**[Mesuré]** Au snapshot droit c1, la carte condensée et la séquentielle donnent le même préfixe de 86 cycles dans le test qpOASES, avec 4,82 s contre 1,66 s. Ce n’est pas un fit H100 accepté : le rollout s’arrête avant H100 et les perturbations directionnelles ne sont pas toutes lancées.

Le rejeu DOP853 des PW réduites documente encore des erreurs propres au prédicteur : environ 0,4–0,5 % sur la force maximale normalisée et près de 0,4 % sur le travail dans les préfixes examinés. L’erreur ajoutée par la condensation est quasi nulle ; l’erreur du modèle compact préexistant ne l’est pas.

### 9.6 Pourquoi la FHO reste lourde

**[Mesuré]** Dans [l’audit FHO MX du 27 septembre](fho_hessian_and_hybrid_strategy_20260927.md), les solves exacts FHO65/FHO88/FHO91 prennent respectivement environ 756/1632/1578 s, avec 75–77 % du temps dans la Hessienne. Ces cas sont historiques et non appariés au BO bilatéral présenté ici.

Le protocole hybride L-BFGS redémarré en blocs puis solve exact est 2,16 fois plus lent sur FHO65 et 1,45 fois plus lent sur FHO88. Le redémarrage perd la mémoire quasi-Newton et reconstruit le problème. Le parallélisme des évaluations natives et les noyaux locaux compilés sont des directions d’accélération, mais ils ne rendent pas une FHO de séance utilisable en ligne par simple changement de solveur linéaire.

## 10. Pourquoi le coût de fatigue peut entrer en conflit avec l’endurance

### 10.1 Le problème de la valeur manquante

Pour maximiser les cycles restants, une décision de cycle devrait être appréciée par la viabilité de l’état qu’elle laisse. Un principe de programmation dynamique idéal comparerait quelque chose comme :

\[
1+N^*(f_{cycle}(x,u)),\qquad u\text{ satisfaisant la tâche actuelle}.
\]

Le coût \(\int\sum d_m^2dt\) n’est pas cette valeur. Il exprime une préférence sur la fatigue accumulée. Il peut être un bon proxy dans certaines régions, mais il ne reçoit aucune garantie générale d’alignement avec le nombre de cycles.

### 10.2 Moyenne, minimum et composantes ne donnent pas la même ressource

Une somme de pertes au carré privilégie un compromis entre muscles. Un soft-min de capacité protège le plus déplété, même s’il est remplaçable. Une réserve de travail d’un cycle favorise une performance de pointe. Une réserve sur plusieurs cycles mesure une politique de projection. Un nombre de cycles restants sous BO mesure une autre politique encore.

Le choix scientifique est donc d’identifier **quelle valeur conditionnelle** sert à la décision. Une « réserve » sans définition de tâche, de politique de poursuite et d’horizon n’est pas suffisamment spécifiée.

### 10.3 Pondération globale et priorités

Ajouter un terme terminal à la fatigue produit une scalarisation :

\[
J=J_{fatigue}+\lambda J_{reserve}+\mu J_{proximal}+J_{regularisation}.
\]

Une bonne mise à l’échelle rend cette somme interprétable. Elle ne garantit pas que la réserve soit prioritaire. Réduire le poids fatigue de 100 à 1, de 1 à 0,01 ou à `1e-8` peut révéler un conflit, mais reste une ablation de scalarisation.

**[Hypothèse]** Le terme de fatigue peut ramener le RHO vers une préférence locale contraire à une meilleure poursuite, surtout lorsque le gradient de réserve est faible ou très local. L’hypothèse doit être testée sur les directions **réalisées** : différences de coût, gradients ramenés à des variations de PW admissibles, évolution effective de réserve et cycles restants.

Comparer les normes brutes des gradients `Cn/F/A/Tau1/Km` est insuffisant, car ces états ont des unités différentes. Des coordonnées normalisées et, idéalement, la sensibilité composée jusqu’aux décisions PW sont nécessaires.

### 10.4 Un bénéfice local de réserve peut sacrifier la récupération

**[Hypothèse]** Maximiser le travail du cycle suivant peut créer de la force résiduelle, mobiliser un muscle récupérant vite ou lentement d’une façon défavorable plus loin, ou déplacer la limite vers une autre phase. Une politique peut gagner en charge maximale d’un cycle et perdre en endurance à charge nominale.

Un horizon H3 ne devient pas un horizon de plusieurs minutes parce que ses états contiennent de la fatigue. La mémoire lente transmet un effet futur, mais le critère terminal doit encore lui attribuer le bon prix.

## 11. Explications concurrentes de l’écart observé

Les explications suivantes peuvent coexister. Elles ne doivent pas être réduites à « PACE-RT est trop myope » sans expérience de séparation.

| Explication | Statut actuel | Expérience discriminante |
|---|---|---|
| Le BO profite du transfert de charge vers le bras droit | Différence de protocole mesurée ; taille de l’effet inconnue | BO/unit/PACE-RT avec split fixe et avec exactement le même partage adaptatif |
| Les poids BO utilisent mieux les muscles remplaçables | Poids et fatigue observés cohérents ; causalité non isolée | Continuer depuis le même checkpoint avec BO, unit et contrastes de poids |
| Le score compact classe mal les candidats | Contre-exemple mesuré pour PACE-VR c100 | Comparaison surrogate/continuations RHO avec regret et incertitude |
| La conversion gradient → poids est inadéquate | Contre-exemple mesuré pour une version PACE-VR | Évaluer directement les poids proposés et le pas opposé ; comparer à une vraie sensibilité RHO |
| Le coût de fatigue domine ou contredit la réserve | Échelle non normalisée auparavant inactive ; conflit résiduel plausible | Ablations de poids puis vrai protocole lexicographique à deux solves |
| La cible de réserve se trouve hors de l’ensemble terminal atteignable | Possible par construction de la cible de boîte | Première optimisation réelle de la réserve depuis le même état |
| La confiance est trop locale pour les déplacements RHO | Mesuré dans plusieurs fits et bindings | Boîtes dans les directions atteignables, validation prospective et taux de fallback |
| L’anticipation cesse avant l’arrêt | Mesuré : dernier cycle gauche actif 141 pour arrêt commun 169 | Valeur alternative au rollout incomplet ; analyser les 28 derniers cycles sans coût actif |
| Le résultat asynchrone est trop ancien ou trop tardif | Deadlines manquées mesurées | Livraison parfaite hors ligne, retards contrôlés puis ordonnanceur réel |
| Les coordonnées `A` omettent une direction utile | Limite structurelle mesurée ; effet inconnu | Fits complets ou sur variété Ding atteignable ; comparaison de décisions |
| Le petit allocateur réduit trop l’espace de PW | 16 variables avec quatre nœuds/phase et confiance locale ; préfixes réduits courts | Augmenter la base de contrôle à état identique, vérifier faisabilité et rang |
| L’arrêt dépend d’un bassin du NLP | Possible pour tout solveur local | Reprises gelées, seeds alternatifs, Phase-I et replay indépendant |
| La stratégie n’a pas assez de flexibilité temporelle | Plausible, non démontré par 234 contre 169 | BO d’une règle à un changement, puis interventions courtes appariées |

L’ordre raisonnable est de corriger d’abord la comparaison expérimentale, puis de mesurer la fidélité de décision, avant d’ajouter de la complexité à la valeur terminale.

## 12. Objectifs et architectures alternatives à examiner

### 12.1 Expérience prioritaire : séparer poids et partage entre bras

**[Proposition]** Construire une matrice factorielle minimale :

| Pondération / valeur terminale | Split 50/50 fixe | Partage adaptatif identique au BO |
|---|---|---|
| Unitaire | Référence A | Référence B |
| Meilleurs poids BO fixes | Effet des poids à split fixe | Reproduction du 234 |
| PACE-RT normalisé | Reproduction du 169 | Effet de PACE-RT avec partage commun |

Ces six conditions utilisent les mêmes modèles, les mêmes bornes, le même état initial, les mêmes tolérances et le même protocole de reprise/arrêt. Il ne faut pas réoptimiser le BO entre colonnes pour la première ablation : conserver ses poids permet d’isoler le partage. Une nouvelle optimisation BO à split fixe sera ensuite une expérience distincte.

La même répartition adaptative est une **même règle**, pas nécessairement une même trajectoire de couples : les capacités diffèrent selon le contrôleur. Pour isoler encore davantage l’effet, une seconde expérience peut rejouer un calendrier de partage fixé et identique, à condition qu’il soit admissible pour toutes les conditions et que cette expérience soit étiquetée comme telle.

#### Nouveau BO de référence strictement 50/50 : 54 essais terminés

**[Mesuré]** La recherche locale à partage fixe de **0,96/0,96 Nm** a complété 54 essais. Le meilleur, essai 21/candidat `64653f690b1599ea`, atteint **179 cycles communs certifiés**. Le témoin unitaire du même protocole atteint 169 cycles; les poids fixes optimisés apportent donc **+10 cycles** dans cette recherche, à tâche appariée. Le meilleur arrêt à la fidélité 240 est étiqueté `fatigue_counterfactual_certified_with_pw_saturation`; il ne constitue pas une preuve globale d'épuisement physiologique.

Le partage reste fixe : `resistance_capacity_feedback:false` et `initial_right_fraction:0.5` dans le protocole. Les poids relatifs effectivement utilisés sont :

| Muscle | Poids droit | Poids gauche |
|---|---:|---:|
| Deltoïde antérieur | 0,6432 | 0,5671 |
| Deltoïde postérieur | 1,0478 | 0,6035 |
| Biceps | 0,5361 | 0,8894 |
| Triceps | 2,7681 | 3,2853 |

Ce résultat ne remplace pas rétroactivement le BO historique à 234 cycles : celui-ci combine les poids et un partage D/G adaptatif. Il isole en revanche un effet observable des poids fixes, qui explique au plus une partie de l'écart 234–169. Le budget de 54 essais et la borne relative [0,25; 4] font partie du protocole et doivent être reportés avec le résultat.

Sources : `asymmetric-sides-r192-rho-bo-20260928/bo-fixed-split-50-50-results/summary.json` et le candidat `64653f690b1599ea`.

### 12.2 Vraie hiérarchie lexicographique : réserve d’abord, fatigue ensuite

**[Proposition]** Depuis une frontière physique identique, résoudre deux fois le même horizon court sans transfert d’état entre les solves :

1. **Premier solve** : satisfaire les contraintes physiques et minimiser la valeur terminale \(\widehat V\), sous le domaine de confiance validé. Une régularisation nécessaire à l’unicité doit être explicitée et suffisamment contrôlée pour ne pas changer la priorité revendiquée.
2. Auditer la solution et enregistrer \(V_1=\widehat V(x_{end}^{(1)})\).
3. **Second solve** : conserver \(\widehat V(x_{end})\le V_1+\varepsilon_V\), retirer son coût de l’objectif, puis minimiser fatigue et régularisation.
4. Auditer la seconde solution et seulement alors transférer l’état.

Le seuil \(\varepsilon_V\) doit combiner la précision numérique et l’incertitude locale du modèle ; il ne doit pas être choisi après coup pour préserver une solution préférée. Si le premier solve échoue ou sort du domaine, le protocole doit conclure à une valeur non exploitable et conserver une politique de référence certifiable.

**[Code]** `PaceRtObjectiveBinding` possède maintenant une activation séparée du coût et de la contrainte, ainsi que `activate_fatigue_secondary_stage`. Ce mécanisme rend possible l’expérience. Il ne constitue pas une preuve que le superviseur effectue déjà une hiérarchie lexicographique d’endurance validée.

Une cible dérivée uniquement de \(\sum|g|r\) n’est pas l’optimum du premier solve réel. Une expérience avec fatigue multipliée par `1e-8` n’est pas équivalente à ces deux étapes. Enfin, la hiérarchie peut parfaitement optimiser en priorité **un mauvais proxy** : elle diagnostique le conflit entre objectifs, pas la justesse de la réserve.

### 12.3 Valeur d’une vraie politique RHO de référence

**[Proposition]** Définir \(N^{\pi_{ref}}(x)\), nombre de cycles restants sous une politique RHO explicite, par exemple les poids BO avec un partage fixé. Depuis un checkpoint, appliquer une courte intervention de K cycles puis revenir à cette référence. L’avantage est :

\[
\Delta N(x,a)=N^{a\rightarrow\pi_{ref}}(x)-N^{\pi_{ref}}(x).
\]

Cette cible est alignée avec l’endurance opérationnelle et n’exige pas de FHO. Elle coûte des branches RHO hors ligne, mais peut être apprise sous forme compacte ou servir directement à un premier test de faisabilité scientifique.

La [proposition PACE-V](pace_value_policy_proposal_20260925.md) recommande d’abord des branches réelles, avant un modèle de valeur complexe. Pour le cas bilatéral actuel, les checkpoints, actions, partage et horizons doivent être redéfinis pour la limite autour de 169–234 cycles ; les c180/c300/c420 de son exemple historique à 0,22 Nm ne sont pas transposables tels quels.

Un modèle appris doit distinguer les trajectoires censurées, les arrêts numériques et les endpoints suffisamment étayés. Les cycles voisins d’une même trajectoire restent dans la même partition d’apprentissage/test. Le candidat courant doit toujours pouvoir être conservé lorsque l’avantage prédit n’excède pas l’incertitude.

### 12.4 BO d’une règle adaptative très courte

**[Proposition]** Avant de chercher une valeur différentiable de grande dimension, rechercher une règle limitée : deux jeux de poids, un seuil d’état, éventuellement une durée d’intervention. Cette expérience répond à une question simple : existe-t-il, dans ce modèle et cette classe d’actions, une adaptation temporelle qui dépasse les bons poids fixes ?

Les seuils doivent dépendre d’états observables pertinents, pas uniquement du numéro de cycle. Les données de développement et les paramètres tenus à l’écart sont séparés. Un résultat positif sur une seule asymétrie nominale n’est pas une généralisation.

Cette voie garde un coût en ligne très faible. Elle ne dispense pas d’une comparaison contre un BO fixe réoptimisé sous le même protocole, ni d’un budget de recherche comparable.

### 12.5 Oracle exact de charge réalisable sur un cycle

La voie [task-load margin](task_load_margin_oracle.md) remplace la seule égalité de travail terminal par une borne et maximise le travail terminal en gardant les autres contraintes. Le facteur de charge est éliminé analytiquement : \(\lambda=E_{prod}(T)/W_{nominal}\).

**[Mesuré]** Au checkpoint gauche c140, le témoin optimisé atteint `1,1802101892` fois le travail nominal, avec audit complet des 4171 lignes de contrainte et 4288 variables. Le solve prend 0,572 s ; reconstruction, solve et audit de ce problème prennent 7,43 s, auxquels s’ajoutent 5,05 s pour le nominal dans le protocole initial.

Des sensibilités issues des multiplicateurs de bornes initiales sont comparées à des OCP perturbés. Sur un domaine complet de 20 états, des validations locales passent ; une boîte isotrope trop large est refusée. Le gradient peut être beaucoup moins coûteux à estimer par les multiplicateurs que par 40 nouveaux solves, mais les changements d’ensemble actif restent un risque.

**Limite** : il s’agit d’une marge de travail sur un cycle avec les autres contraintes, notamment celles du demi-cycle suivant, laissées à leur demande originale. Ce n’est ni un maximum global prouvé, ni une charge uniformément augmentée partout, ni une valeur de durée restante.

**[Mesuré]** Un essai de rafraîchissement depuis c140 va jusqu’au dernier cycle certifié 169, avec 29 cycles supplémentaires certifiés, 7 cycles candidat acceptés et 23 fallbacks rapportés, ces compteurs incluant les événements de tentative selon le runner. Son résumé laisse `physiological_failure_certified=false`. Les expériences « marge seule » avec confiance dure sont des pilotes locaux, pas un gain d’endurance établi.

Source récente : [refresh-until-stop-w295](../../asymmetric-sides-r192-rho-bo-20260928/task-load-margin-c140-left-refresh-until-stop-w295-20261002/summary.json). La présence d’un gradient mieux validé ne suffit donc pas à conclure qu’il prolonge l’exercice.

### 12.6 Travail positif sous PW maximale : une primitive, pas une frontière

Le [proxy de travail positif PW-max](max_pw_work_capacity.md) simule les cinq états Ding depuis le terminal candidat et additionne :

\[
W_+(x)=\sum_m\int\max(\omega b_m,0)F_m(t;x,u^{policy})\,dt.
\]

Une politique de PW prescrite est utilisée ; il n’y a pas d’optimisation interne. Deux variantes activent soit tous les intervalles à PW maximale, soit les phases propulsives sélectionnées. La force résiduelle rend la seconde différente d’une enveloppe optimale.

**[Mesuré]** L’insertion directe du rollout RK4×16 dans le NLP rend un smoke bilatéral environ 2,2 fois plus lent. Le binding retient donc un gradient numérique rafraîchi au bord du cycle, avec un terme terminal affine. Une ablation conserve seulement les dérivées des états lents.

**[Hypothèse]** La composante `F` peut récompenser une force immédiatement disponible sans considérer son coût de freinage futur. Filtrer les états rapides peut réduire ce comportement, mais supprimer une dérivée physique utile est aussi possible. Le choix doit se faire sur la marge réelle et les cycles restants, pas sur la seule augmentation de \(W_+\).

### 12.7 Approximation de noyau de viabilité ou barrière de réserve

**[Proposition]** Chercher un ensemble d’états depuis lesquels une politique de secours connue peut réaliser encore K cycles. Un coût ou une contrainte pourrait préférer les terminaux qui restent à l’intérieur de cet ensemble.

L’intérêt est de rapprocher la métrique de l’existence d’une poursuite. La difficulté est de construire cet ensemble dans un état musculaire/mécanique de dimension élevée, avec bornes PW et incertitudes. Une frontière obtenue par échecs locaux IPOPT ne peut pas être présentée comme une frontière garantie. Des témoins positifs, des marges robustes et un domaine clairement limité sont nécessaires.

### 12.8 Allongement exact modéré du RHO et terminal appris

**[Proposition]** Un RHO de deux ou trois cycles avec un coût terminal simple peut fournir un meilleur pont entre forces résiduelles et fatigue qu’un cycle plus un rollout réduit mal classant. C’est une comparaison à mesurer, non un choix automatiquement trop coûteux ou automatiquement supérieur.

Le coût clinique doit inclure le solve complet et sa variabilité. Les gains d’une condensation Ding exacte pourraient être réinvestis dans l’horizon plutôt que seulement dans la vitesse. Il faut comparer les deux usages du budget avec les mêmes échéances.

### 12.9 Coût de fatigue incrémentale, flux de dommage ou coût économique

**[Proposition]** Tester séparément : fatigue accumulée, augmentation de fatigue par rapport à la récupération sans stimulation, PW/charge électrique, et coût économique combinant travail utile et réserve.

Une fatigue nette peut diminuer pendant qu’un muscle reçoit une stimulation : la récupération et le dommage induit coexistent. Comparer au contre-factuel sans force aide à isoler l’effet du recrutement. Mais un coût local du dommage, même physiquement défini, n’est toujours pas une valeur de viabilité future. Ces objectifs sont des ablations scientifiques, pas des synonymes d’endurance.

## 13. Protocole de validation qui permettrait de conclure

### 13.1 Niveau 0 : figer le problème comparé

Archiver explicitement : modèles et hashes, paramètres Ding, états initiaux complets, historique de stimulation, géométrie, tâche, partage entre bras, cadence, fréquence, PW min/max et slew, contraintes de fin/mi-cycle, intégrateur, solveur, tolérances et stratégie de reprise.

Conserver la définition du score : cycles tentés, certifiés par bras, préfixe commun, plafond de campagne, raison d’arrêt et résultats de la revue gelée. Un plafond atteint est une observation censurée à droite, pas une mesure exacte de durée maximale.

### 13.2 Niveau 1 : reprise exacte depuis un checkpoint

Un résumé de `A/A_rest` ne suffit pas. La reprise doit contenir les états musculaires et mécaniques, le primal translaté, les PW et leur historique, les bornes, les paramètres numériques, l’horloge de manivelle et le contexte de tâche.

Le [contrat de reprise PACE-VR](pace_vr_checkpoint_validation.md) demande un reçu attesté par de nouveaux workers, avec empreinte du problème préparé identique avant l’application du candidat. Un `prepared-export.json` provisoire n’est pas le `receipt.json` final. Les hashes empêchent des substitutions accidentelles ; ils ne remplacent pas une vérification de restauration et de propagation.

### 13.3 Niveau 2 : exactitude numérique et domaine physique

À PW fixées, vérifier la propagation Ding, les signes, les quadratures, le travail net, les bornes de couple et les états entre nœuds, aux états frais, intermédiaires et proches de l’arrêt. Raffiner pas et tolérances jusqu’à ce que l’erreur soit petite face aux différences à départager.

Une erreur de travail de 0,4 % peut être acceptable pour un certain usage, mais ne justifie pas un classement dont l’écart n’est que `5,7e-6` en marge sans conversion d’échelle et étude d’incertitude. Des tolérances de test arbitraires ne sont pas des intervalles de confiance.

### 13.4 Niveau 3 : identification et action réelle du modèle local

Depuis le même checkpoint, générer des états terminaux atteignables par plusieurs interventions de poids ou PW. Auditer le rang, le conditionnement et le sous-espace excité. Réserver des branches entières pour le holdout.

Mesurer :

- erreur de valeur et de dérivée sur les directions conservées ;
- classement et regret sur des candidats qui diffèrent matériellement ;
- taux de fit refusé et coût complet du fit, y compris les échecs ;
- taille du domaine de confiance dans les déplacements réellement produits ;
- sensibilité du RHO aux paramètres installés et preuve d’activation du coût ;
- prévisions comparées aux terminaux effectivement réalisés.

Une Hessienne diagonale locale, un fit affine de rang trois et un gradient KKT ne répondent pas aux mêmes hypothèses de régularité. Leur audit doit être adapté, et leurs coefficients ne doivent pas être comparés comme s’ils étaient la même valeur.

### 13.5 Niveau 4 : fidélité décisionnelle contre de vraies continuations

À au moins trois niveaux de fatigue, comparer depuis la même frontière : candidat courant, BO fixe, unitaire, proposition, demi-pas et pas opposé. Au premier écran, des branches de 1/5/20 cycles sont utiles pour identifier les effets immédiats. La cible finale reste une poursuite jusqu’à l’arrêt ou au plafond, sous une politique de queue commune explicitement définie.

Rapporter la concordance des paires, un regret en cycles et les cas non départageables. Un candidat qui gagne seulement en proxy et perd en cycles doit être présenté comme tel. Une bonne corrélation sur une trajectoire globale peut masquer un classement local inutilisable.

### 13.6 Niveau 5 : boucle fermée et contraintes temporelles

Comparer les contrôleurs depuis les mêmes conditions initiales, avec plusieurs répétitions si la trajectoire dépend des effets numériques ou de l’ordonnancement. Alterner les lancements ou contrôler l’affinité CPU pour éviter un biais de charge machine.

Rapporter à la fois : endurance commune, cause d’arrêt, coût de fatigue commun non repondéré, PW/saturation, travail/couple, états musculaires, temps solveur, temps de préparation, certification, transfert, superviseur, délais manqués, ancienneté des propositions et cycles avec coût réellement actif.

Une moyenne inférieure à une seconde ne suffit pas si les échéances ne tolèrent aucun dépassement. Le besoin exact — moyenne, p95, pire cas ou calcul anticipé avec tampon — doit être défini par l’application. Les diagnostics lourds de fin d’exercice ne doivent pas être cachés dans une moyenne de solve chaud ni confondus avec une latence habituelle.

### 13.7 Niveau 6 : attribution de l’arrêt

Au dernier état certifié, ne jamais avancer la physique avec l’itéré échoué. Effectuer une sonde gelée au même problème, éventuellement à objectif nul ou avec restauration Phase-I, puis des seeds alternatifs et contre-factuels documentés.

Une solution admissible constitue un témoin positif pour le problème numérique audité. Plusieurs échecs locaux renforcent un diagnostic opérationnel, mais ne prouvent pas globalement l’infaisabilité. Une diminution de charge ou un relâchement de PW diagnostique une sensibilité ; la solution ainsi obtenue ne devient pas un cycle nominal.

Conserver au moins les catégories : plafond atteint, interruption technique, échec numérique non résolu, témoin de poursuite nominale trouvé, limitation liée à la fatigue/bornes étayée par contre-factuels, et éventuelle preuve plus forte si elle existe réellement.

### 13.8 Niveau 7 : généralisation

Garder des modèles et des charges hors développement : différences droite/gauche, fatigabilités, récupérations, force disponible, paramètres mal identifiés, cadence et fréquence. Tester ensuite le bruit et les erreurs d’estimation d’état selon un protocole séparé.

Le fait que le contrôleur ait accès à l’état exact dans la simulation est une hypothèse importante. Une valeur qui dépend fortement d’un état difficile à estimer peut être excellente dans le modèle et peu exploitable en pratique. Il faut distinguer limite de contrôle, limite d’identification et limite de calcul.

## 14. Questions ouvertes à soumettre à d’autres LLMs

Les réponses recherchées doivent proposer des mécanismes falsifiables, des expériences petites mais discriminantes et des critères d’abandon. Une liste générique de MPC, RL, GP ou réseaux de neurones sans lien avec les échecs ci-dessus ne suffit pas.

### 14.1 Prompt général prêt à transmettre

> Nous cherchons à maximiser le nombre de cycles réalisables de pédalage FES avec fatigue Ding, sans utiliser une trajectoire FHO patient en clinique. Le RHO rapide optimise un cycle d’une seconde, 30 Hz, quatre PW par bras, avec dynamique musculaire Cn/F/A/Tau1/Km, travail net isocinétique imposé et bornes mécaniques/stimulation. Le coût de base est une intégrale de pertes relatives de capacité au carré, pondérées par muscle. Deux bras asymétriques sont simulés par des OCP indépendants synchronisés.
>
> Un BO de poids musculaires fixes a produit 234 cycles communs ; cependant il possède un partage adaptatif du travail droite/gauche, total 1,92 Nm. Un PACE-RT à poids unitaires, split fixe 0,96/0,96 Nm, rollout compact H3 et coût terminal local normalisé a produit 169 cycles communs. Il ne faut donc pas attribuer causalement leurs 65 cycles d’écart aux seuls poids. Le PACE-RT n’était actif que sur 129/169 cycles gauches, avec dernier cycle actif gauche 141 ; des rollouts tardifs échouent avant leur horizon.
>
> Les difficultés déjà observées sont : politiques compactes différentes de la réponse RHO ; saturation du petit QP rendant les poids inopérants ; proposition issue de sensibilité parfois dernière selon le propre score du surrogate ; fits précis mais valides dans des boîtes trop petites ; gradients seulement en A alors que les forces et mémoires comptent ; erreurs de signe ou d’exigence de puissance dans certains anciens proxies ; opposition possible entre fatigue accumulée, réserve d’un cycle et endurance ; délais et inactivation asynchrones. Une FHO peut coordonner de longues répartitions, mais aucune FHO strictement appariée à la campagne 234/169 n’est actuellement vérifiée.
>
> Propose une stratégie de recherche sans données FHO nécessaires au contrôleur. Commence par identifier les comparaisons non causales et ce qu’il faut maintenir identique. Distingue les faits ci-dessus de tes hypothèses. Donne une définition mathématique de la valeur que tu proposes, sa politique de poursuite, ses coordonnées, ses contraintes, sa sensibilité à l’horizon et son coût de calcul. Explique comment elle influence réellement les PW et pourquoi elle pourrait classer les politiques selon leurs cycles restants. Propose d’abord deux ou trois expériences locales capables de réfuter ton idée, puis une validation en boucle fermée. Indique ce qui ferait abandonner la méthode. Ne présente pas un échec IPOPT, une marge compacte négative ou une faible capacité minimale comme une preuve globale d’épuisement.

### 14.2 Prompt ciblé : quel objectif faut-il prioriser ?

> Analyse si la fatigue quadratique accumulée doit être l’objectif principal, un objectif secondaire, une contrainte, ou seulement une métrique de suivi. Dérive la différence entre fatigue accumulée, incrément de dommage par rapport à la récupération, travail disponible et cycles restants. Construis un petit contre-exemple où minimiser la fatigue totale ou protéger le muscle le plus fatigué réduit l’endurance de tâche. Propose un protocole lexicographique à deux solves, avec domaine de confiance, tolérance et politique de secours. Distingue le fait de rendre une réserve prioritaire du fait de disposer d’une réserve scientifiquement valide.

### 14.3 Prompt ciblé : valeur sans FHO et sans long rollout fragile

> Définis une valeur de poursuite fondée seulement sur des checkpoints et continuations RHO. Compare : nombre de cycles restants sous BO, avantage d’une intervention K cycles puis retour à BO, marge de travail exacte sur un cycle, ensemble de viabilité K cycles et transition apprise de cycle RHO. Pour chaque option, donne données minimales, biais de politique, traitement de la censure et des échecs numériques, partage train/test, incertitude et coût clinique. Choisis une première expérience de moins de quelques dizaines de branches, et justifie pourquoi elle répond mieux à la question d’endurance qu’un fit de marge sur les cycles voisins d’une seule trajectoire.

### 14.4 Prompt ciblé : pourquoi le rollout perd-il le classement ?

> Le rollout compact Ding utilise 16 décisions réduites de recrutement, quatre nœuds par muscle, une géométrie échantillonnée et une SQP/QP locale avec travail exact dans le modèle. Un RHO réel dispose des PW par intervalle et d’un autre coût. À état identique, une proposition PACE-VR a été classée dernière par le surrogate face au pas opposé, malgré son acceptation. Décompose les erreurs possibles : propagation, base de contrôle, coût d’allocation, domaine local, linéarisation des contraintes, conversion gradient → poids, état source et asynchronisme. Propose un ordre d’ablations qui identifie chacune sans changer simultanément toutes les briques.

### 14.5 Prompt ciblé : réduction exacte de Ding et coordonnées utiles

> Les équations lentes A/Tau1/Km partagent la même constante de récupération et la même force, avec offsets indépendants à conserver. Le calcium est reconstructible analytiquement entre stimulations ; la force demeure non linéaire et porte une mémoire rapide. Propose des coordonnées de valeur ou de viabilité qui respectent cette structure et les contraintes d’atteignabilité. Discute les cas alpha_A proche de zéro, les changements d’historique de stimulation et les paramètres incertains. Explique comment vérifier qu’un gradient identifié dans ces coordonnées prédit la variation produite par des décisions PW réalisables.

### 14.6 Prompt ciblé : partage du travail entre bras

> Les deux bras ont des modèles asymétriques ; le total de travail est imposé, mais sa répartition peut être un degré de liberté. Dans le BO à 234 cycles, la droite passe de 0,96 Nm à environ 1,05 Nm et la gauche descend vers 0,87 Nm ; le PACE-RT à 169 reste à 0,96/0,96. Propose une expérience factorielle et une stratégie d’allocation entre bras qui n’utilise pas simplement la capacité minimale. Faut-il équilibrer une marge mécanique, une probabilité de survie ou une valeur de cycles restants ? Comment éviter d’optimiser séparément les bras en oubliant que l’objectif bilatéral est le minimum de leurs durées ?

### 14.7 Prompt ciblé : système temps réel à deux vitesses

> Un superviseur peut mettre à jour des paramètres numériques d’un NLP déjà construit, mais ses propositions peuvent être retardées ou hors domaine. Propose un contrat de publication/application avec état complet, hashes, cycle source, modèle de confiance et politique de repli. Sépare vitesse du noyau, fit complet, transport, reconstruction et délai de la paire de bras. Décris une validation avec livraison parfaite hors ligne, retards imposés puis exécution asynchrone réelle, et explique comment attribuer une absence de gain au modèle ou à l’ordonnancement.

### 14.8 Format de réponse demandé aux LLMs consultés

Pour chaque proposition retenue, demander :

1. Le mécanisme supposé et les observations du dossier qu’il explique.
2. La quantité exactement optimisée, ses unités et la différence avec l’endurance.
3. Les hypothèses mathématiques ou numériques nécessaires.
4. Les données déjà disponibles et les données manquantes.
5. Une expérience de réfutation locale, avec témoin et seuils préfixés.
6. Un protocole d’endurance apparié et son critère d’arrêt.
7. Le budget estimé hors ligne/en ligne, explicitement marqué comme estimation.
8. Le risque de surapprentissage, de mauvaise attribution ou de faux certificat.
9. La condition d’abandon et l’alternative la plus simple à tester ensuite.

## 15. Registre des sources locales à relire avant de lancer une campagne

Les chemins ci-dessous sont relatifs à la racine du dépôt lorsqu’ils sont écrits en code. Les liens pointent vers les fichiers. Les JSON détaillés ont parfois plusieurs mégaoctets : lire des champs sélectionnés plutôt que d’imprimer toute leur arborescence.

### Provenance Git à compléter après commit chirurgical

Cette version référence des **chemins locaux et des artefacts**, pas un commit censé représenter tout le workspace. Des changements de code, de documentation et des résultats non suivis coexistent ; il serait incorrect d’attribuer tout ce contenu à la seule révision Git courante.

Après une sélection et un commit ciblés des fichiers effectivement concernés, compléter : identifiant du commit documentaire, liste des versions de code correspondant aux campagnes citées, configurations et empreintes des artefacts principaux, fichiers volontairement laissés hors commit, et date de l’extraction des compteurs. Un commit du présent texte ne garantit pas à lui seul que tous les résultats volumineux ou les fichiers expérimentaux auxquels il renvoie soient archivés avec lui.

| Objet | Source prioritaire |
|---|---|
| Définition isocinétique, travail et signes | [docs/isokinetic_rho_ocp.md](../isokinetic_rho_ocp.md) |
| Architecture des deux bras | [independent_isokinetic_arms.md](independent_isokinetic_arms.md) |
| Équations et réduction Ding | [ding_analytic_reduction_radau5.md](ding_analytic_reduction_radau5.md) |
| Coût de fatigue effectivement implémenté | [cocofest/custom_objectives.py](../../cocofest/custom_objectives.py) |
| PACE causal, normalisation et géométrie BO | [rho_pace.md](rho_pace.md) |
| Paramètres asymétriques | [model-right.json](../../asymmetric-sides-r192-rho-bo-20260928/model-right.json), [model-left.json](../../asymmetric-sides-r192-rho-bo-20260928/model-left.json) |
| BO 234 et poids gagnants | [bo-results/summary.json](../../asymmetric-sides-r192-rho-bo-20260928/bo-results/summary.json) |
| Configuration BO gagnante et partage adaptatif | [candidat 0ed1b7bec477b44f/input.json](../../asymmetric-sides-r192-rho-bo-20260928/bo-results/candidates/0ed1b7bec477b44f/cycles-0240/input.json) |
| Cycles, poids appliqués et contre-factuels BO | [candidat 0ed1b7bec477b44f/summary.json](../../asymmetric-sides-r192-rho-bo-20260928/bo-results/candidates/0ed1b7bec477b44f/cycles-0240/summary.json) |
| Configuration PACE-RT normalisée H3/K2 | [normalized-endurance-200.json](../../asymmetric-sides-r192-rho-bo-20260928/pace-rt-refresh-40-20261002/normalized-endurance-200.json) |
| PACE-RT normalisé, cycles physiques et confiance | [persistent-normalized-endurance-200-interactive-results/summary.json](../../asymmetric-sides-r192-rho-bo-20260928/pace-rt-refresh-40-20261002/persistent-normalized-endurance-200-interactive-results/summary.json) |
| PACE-RT normalisé, soumissions et refus | [pace_vr_supervisor.json](../../asymmetric-sides-r192-rho-bo-20260928/pace-rt-refresh-40-20261002/persistent-normalized-endurance-200-interactive-results/pace_vr_supervisor.json) |
| Quatre conditions initiales, contexte séparé | [pace-rt-validation-20261002-r2/comparison.json](../../asymmetric-sides-r192-rho-bo-20260928/pace-rt-validation-20261002-r2/comparison.json) |
| Normalisation, ablations et prochaine hiérarchie | [pace_rt_validation.md](pace_rt_validation.md), [pace_rt_ocp.py](../../cocofest/optimization/pace_rt_ocp.py) |
| Poids de l’annexe et domaine de calibration | [physiological_weights_validation.md](physiological_weights_validation.md) |
| Actualisation physiologique | [physio_update_isokinetic_proposal.md](physio_update_isokinetic_proposal.md) |
| Plan d’ensemble sans données FHO | [rho_endurance_horizon_plan.md](rho_endurance_horizon_plan.md), [multilevel_endurance_plan.md](multilevel_endurance_plan.md) |
| Valeur locale et échec H30 | [local_endurance_value_validation.md](local_endurance_value_validation.md), [compact_rollout_failure_diagnostic.md](compact_rollout_failure_diagnostic.md) |
| Preview déclenché | [triggered_preview_allocation_validation.md](triggered_preview_allocation_validation.md) |
| Garde et saturation des allocations | [rho_pace_guarded_weights.md](rho_pace_guarded_weights.md), [pace_regularization_audit_20260925.md](pace_regularization_audit_20260925.md) |
| Accélération et fidélité PACE-VR | [pace_vr_rollout_acceleration.md](pace_vr_rollout_acceleration.md), [pace_vr_candidate_fidelity.md](pace_vr_candidate_fidelity.md) |
| Reprises exactes et comparaison causale | [pace_vr_checkpoint_validation.md](pace_vr_checkpoint_validation.md) |
| Réserve de tâche, calibration et branches | [task_reserve_local_experiment.md](task_reserve_local_experiment.md), [task_reserve_branch_execution.md](task_reserve_branch_execution.md) |
| Oracle de charge et sensibilités | [task_load_margin_oracle.md](task_load_margin_oracle.md), [task_load_margin_contract.md](task_load_margin_contract.md) |
| Correction du proxy de travail | [mechanical_reserve_work_recalibration.md](mechanical_reserve_work_recalibration.md) |
| Travail positif PW-max | [max_pw_work_capacity.md](max_pw_work_capacity.md) |
| Valeur de vraies continuations RHO | [pace_value_policy_proposal_20260925.md](pace_value_policy_proposal_20260925.md) |
| Contrôle de phase et playback rejetés | [phase_feedback_screen.md](phase_feedback_screen.md) |
| Précision de Radau et réduction exacte | [ding_radau_observed_order.md](ding_radau_observed_order.md), [ding_radau5_local_ab.md](ding_radau5_local_ab.md) |
| Solveurs et campagnes historiques | [development_history.md](development_history.md), [README.md](README.md) |
| FHO historique, coût et limitations numériques | [fho_hessian_and_hybrid_strategy_20260927.md](fho_hessian_and_hybrid_strategy_20260927.md) |
| Diagnostic gelé de poursuite | [frozen_state_viability_probe.md](frozen_state_viability_probe.md) |

### 15.1 Relecture minimale des deux nombres centraux

Ces commandes sont uniquement des lectures. Elles évitent de compter le dernier cycle tenté comme certifié :

```bash
jq '{objective, trials: (.trials|length), best: .best}' \
  asymmetric-sides-r192-rho-bo-20260928/bo-results/summary.json

jq '{arms: (.arms | with_entries(.value = {
  certified_cycles: ([.value.cycles[] | select(.certified)] | length),
  torque_min: ([.value.cycles[] | select(.certified) | .equivalent_mean_torque_nm] | min),
  torque_max: ([.value.cycles[] | select(.certified) | .equivalent_mean_torque_nm] | max),
  unique_weights: ([.value.cycles[] | select(.certified) | .weights_used] | unique | length)
}))}' \
  asymmetric-sides-r192-rho-bo-20260928/bo-results/candidates/0ed1b7bec477b44f/cycles-0240/summary.json

jq '{arms: (.arms | with_entries(.value = {
  certified_cycles: ([.value.cycles[] | select(.certified)] | length),
  active_terminal_cycles: ([.value.cycles[] | select(.certified and .pace_rt_terminal.active)] | length),
  last_active_cycle: ([.value.cycles[] | select(.certified and .pace_rt_terminal.active) | .cycle] | max)
}))}' \
  asymmetric-sides-r192-rho-bo-20260928/pace-rt-refresh-40-20261002/persistent-normalized-endurance-200-interactive-results/summary.json
```

### 15.2 Décisions que les nouvelles expériences doivent rendre possibles

Le dossier doit permettre de décider, sur des preuves séparées, si le prochain effort doit porter sur le partage entre bras, l’objectif de fatigue, l’identification d’une valeur de poursuite, l’allocateur compact, le domaine de confiance, l’ordonnancement ou le coût du NLP.

Le résultat attendu n’est pas seulement « un score de réserve plus élevé ». C’est une politique dont l’avantage en cycles réalisables est démontré à tâche égale, dont les interventions sont réellement appliquées, dont les arrêts sont correctement classés et dont le coût complet tient dans le budget choisi.
