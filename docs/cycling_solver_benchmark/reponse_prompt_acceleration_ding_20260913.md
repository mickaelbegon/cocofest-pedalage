# Réponse au prompt d'accélération Ding — application au cycling reduced

Date de l'audit : 13 septembre 2026.

Document examiné : [PROMPT_acceleration_Ding_Cocofest.md](/home/mickaelbegon/Downloads/PROMPT_acceleration_Ding_Cocofest.md). Ses consignes et objectifs chiffrés sont traités comme le contenu à évaluer, et non comme une autorisation de modifier le modèle ou les solveurs. Cette réponse s'appuie sur le code local et les campagnes archivées ; elle ne reproduit pas les benchmarks SciPy annoncés dans le prompt et ne certifie pas indépendamment chacune de ses références bibliographiques.

## Résumé exécutif

Le prompt contient deux idées justes : exploiter le caractère exogène du calcium lorsque le calendrier de stimulation est fixé, et réduire les trois états de fatigue à une seule variable dynamique. Toutefois, son principal gain annoncé — supprimer une convolution symbolique de vingt impulsions — est **déjà largement réalisé dans notre formulation `periodic_node`**. Il ne constitue donc pas un gain supplémentaire de ×10 à ×40 pour la référence actuelle.

La référence pertinente reste **IPOPT/MA57, SX, collocation Radau à cinq stages, modèle mécanique reduced**, avec les réglages physiques et de warm-start de la campagne considérée. Ici, « Radau-5 » signifie cinq stages, ordre classique neuf sur un intervalle lisse ; il ne désigne pas le Radau à trois stages d'ordre cinq de certains logiciels IVP.

Une expérience supplémentaire mérite d'être envisagée : **garder `F,A` à tous les nœuds et stages, précalculer `Cn` et reconstruire `Tau1,Km` localement avec les mêmes coefficients Radau-5**. Cette variante diffère de la condensation maximale déjà testée et défavorable. Elle pourrait préserver la structure creuse et convertir plusieurs bornes physiques en bornes sur `A`, au lieu d'ajouter des contraintes générales. Son gain reste une hypothèse : aucun résultat OCP ne permet actuellement de lui attribuer un facteur d'accélération.

Les propositions de carte semi-analytique, moyennisation ou changement de transcription sont des pistes de recherche avec erreur numérique ou changement de problème. Elles ne justifient ni d'abandonner Radau-5 ni de promettre une carte complète exacte pour le cyclage couplé.

## Ce qui est déjà établi dans notre dépôt

Les références de lignes ci-dessous décrivent l'état du dépôt pendant cet audit et peuvent évoluer.

| Fait vérifié | Preuve locale | Conséquence |
|---|---|---|
| Le forçage calcium est une amplitude numérique multipliée par une exponentielle | [periodic_node](../../cocofest/models/ding2007/ding2007_with_fatigue_periodic_node.py), lignes 140–146 et 249–252 | La boucle symbolique sur vingt impulsions ne domine plus cette formulation |
| `R0` utilise `Km_rest`; le PW agit sur le recrutement de force | Même fichier, lignes 75–81 et 176–194 ; [Ding 2003](../../cocofest/models/ding2003/ding2003.py), lignes 320–324 | Le calcium est indépendant des PW pour notre calendrier et nos paramètres fixés |
| Un intervalle correspond à une stimulation | `periodic_node`, lignes 239–246 | L'alignement impulsionnel demandé par le prompt est déjà imposé |
| La fatigue partage un temps de récupération et un forçage `F` | [Ding 2007 avec fatigue](../../cocofest/models/ding2007/ding2007_with_fatigue.py), lignes 204–247 | La réduction affine commune est valide muscle par muscle |
| La condensation maximale n'accélère pas IPOPT/MA57 | [A/B symbolique](ding_radau5_symbolic_ab.md), lignes 93–106 | Une réduction du nombre de variables ne suffit pas |
| La référence A/B utilisait déjà MA57, SX, Radau-5 et scaling `full` | Même rapport, lignes 81–88 | Les gains génériques de ces réglages ne peuvent pas être recomptés |

Le pilote de condensation a réduit les variables de **4 103 à 1 703**, mais conservé **3 960 lignes de contraintes** : 2 400 défauts supprimés ont été remplacés par 2 400 contraintes de bornes reconstruites. La médiane chaude est passée de **0,947 à 1,022 s**, et la Hessienne de **5,931 à 7,611 ms par évaluation**. Les deux variantes ont réussi 10/10 fenêtres. Le smoke initial retrouve la même solution à précision numérique ; les trajectoires RHO fermées divergent ensuite, ce qui limite l'interprétation des temps chauds comme comparaison à problème identique.

Les essais [FATROP et DMS](README.md#64-fatrop-face-à-ipoptma57--rho-150-radau-5) ont aussi fourni des signaux défavorables : FATROP/collocation est environ cinq fois plus lent que la référence historique sur le RHO-150, avec une réserve sur le warm-up ; le smoke RK8/DMS réussi coûte 8,275 s ; l'IRK/MX a rencontré des NaN au démarrage. Ces observations ne prouvent pas que toute carte dédiée serait lente, mais empêchent d'inférer son efficacité à partir d'un benchmark de simulation musculaire isolée.

## Affirmations du prompt et verdict pour notre cas

| Affirmation ou proposition du prompt | Verdict pour notre application | Preuves, limites ou correction |
|---|---|---|
| Supprimer la convolution historique ferait gagner ×10–40 | **Levier largement déjà appliqué** | Le forçage `periodic_node` contient une exponentielle, avec amplitude numérique précalculée. Les chiffres du prompt concernent une autre expression de la dynamique. |
| `Cn` est précalculable pour Ding 2007 avec fatigue | **Oui sous conditions** | PW seul optimisé, calendrier, paramètres et calcium entrant fixés. Si durée, instants, intensité, paramètres ou état initial deviennent des décisions, conserver leurs dépendances et dérivées. |
| La récurrence calcium représente exactement le modèle courant sans troncature | **Pas automatiquement** | La réalisation à mémoire est exacte pour la convolution choisie. Notre amplitude reproduit une histoire tronquée dont le plus ancien coefficient vaut un ; remplacer cette convention par une histoire infinie change légèrement le modèle. Voir `periodic_node`, lignes 83–108. |
| Le remplacement de `A,Tau1,Km` par une convolution commune est exact | **Oui au niveau continu** | Les trois équations ont le même `tau_fat` et le même forçage. Les paramètres doivent rester constants et les offsets initiaux doivent être conservés. Voir [développement existant](ding_analytic_reduction_radau5.md), sections 3 et 5. |
| La réduction exacte implique des gains NLP importants | **Non démontré** | La condensation maximale mesurée ralentit le solveur et sa Hessienne. La variante locale `F,A` reste différente et non mesurée en OCP. |
| Le modèle n'est pas raide et le seuil RK4 est 55,7 ms | **Conclusion non transposable telle quelle** | Notre Ding 2007 utilise `tauc=0,011 s`, non `0,020 s` : [paramètres](../../cocofest/models/ding2007/ding2007.py), lignes 50–66. Le mode calcium vaut −90,91 s⁻¹ et le seuil linéaire RK4 environ 30,64 ms, inférieur à 33,33 ms à 30 Hz. Le bloc mécanique et les couplages demandent aussi une analyse. |
| Radau est plus lent que RK45 dans SciPy, donc il faut changer la collocation OCP | **Comparaison insuffisante** | Simulation adaptative, collocation NLP, coût des sensibilités et factorisation KKT sont des objets différents. Les mesures SciPy du document n'ont pas été reproduites ici. |
| F est linéaire et s'intègre exactement par facteur intégrant | **Identité conditionnelle, pas carte complète explicite** | F est affine si les autres trajectoires sont prescrites. Fatigue et mécanique dépendent de F. Le [RHS réel](../../cocofest/models/ding2003/ding2003.py), lignes 361–363, comporte aussi les facteurs mécaniques. |
| La fatigue exponentielle est exacte avec la force moyenne du pas | **Faux pour une moyenne arithmétique générale** | L'intégrale requiert une moyenne exponentiellement pondérée ; la formule simple est exacte si F est constante. Geler les coefficients et utiliser Gauss introduit également une approximation. |
| Une carte semi-analytique complète ne laisse rien à intégrer numériquement | **Faux pour le schéma décrit** | Les quadratures, l'approximation de coefficients et le gel des états lents restent des opérations numériques approchées. Leur erreur doit être mesurée. |
| `Sigma,F,Phi` suffisent pour Hmed à intensité optimisée | **Comptage généralement incomplet** | Une réalisation locale du filtre à pôle double doit porter `Sigma` et `Cn`, soit quatre états avec F et Phi. Éliminer Cn exige une reconstruction avec mémoire ou une représentation équivalente de son information. |
| Une convolution commune prouve la non-identifiabilité des coefficients de fatigue | **Non établi** | Cette propriété ne démontre pas une symétrie d'identification. La conclusion dépend des sorties observées et des paramètres inconnus. |
| Une carte discrète entraîne une Hessienne plus creuse | **Non garanti** | Une élimination ou une propagation globale peut créer des dépendances entre stages ou sur tout l'horizon. Notre condensation en donne un contre-exemple de performance. |
| MA57, SX et scaling sont des nouveaux gains faciles | **Déjà présents dans la référence ciblée** | Protocole A/B local. Les options de compilation, warm-start et expansion doivent être rapportées par campagne, sans appliquer les multiplicateurs de gains génériques du prompt. |
| Réduire la troncature et supprimer le bourrage accélérerait notre solveur | **Faible intérêt pour `periodic_node`** | La somme est évaluée numériquement pendant la préparation. Cette optimisation peut être utile ailleurs dans Cocofest, pas comme levier principal de notre hot solve. |
| Enlever `dt` des défauts est une correction certaine | **À auditer avant toute modification** | Le code actuel [Ding 2003](../../cocofest/models/ding2003/ding2003.py), lignes 232–246, utilise aussi `dt`, contrairement à la prémisse du prompt. Vérifier les conventions de temps et dérivées de Bioptim ainsi que le scaling effectif. |
| Le modèle moyenné et un NLP de quelques impulsions conviennent aux horizons longs | **Possibles pour un autre objectif** | Ils peuvent aider la planification grossière ou l'initialisation ; ils ne conservent pas automatiquement notre mécanique cyclique, les PW par muscle et les contraintes aux stages. |

## Direction prioritaire : réduction locale en conservant Radau-5

La variante à tester conserve les états mécaniques et, pour chaque muscle, `F,A`. Elle retire les autres états uniquement lorsque leurs valeurs sont déterminées par les données d'entrée et les variables conservées. Cette restriction est essentielle si une condition initiale est libre ou relâchée.

### Calcium : conserver la discrétisation de référence

Avec le tableau Radau \(\mathcal A\), les stages de forçage \(\mathbf H\), un pas fixe \(h\) et un calcium entrant fixé \(C_k\), les stages sont

\[
\mathbf C=\left(I+\frac h{\tau_c}\mathcal A\right)^{-1}
\left(\mathbf1 C_k+\frac h{\tau_c}\mathcal A\mathbf H\right).
\]

Ce calcul est numérique et indépendant des PW. On transmet ensuite la sortie avec la règle Radau originale. Utiliser immédiatement l'exponentielle exacte du modèle continu donnerait une autre discrétisation de la force ; c'est une expérience possible, mais elle ne doit pas être confondue avec un A/B strictement équivalent.

### Fatigue : garder la dépendance au même nœud ou stage

Pour \(\alpha_A\ne0\), définissons

\[
\beta_T=\frac{\alpha_T}{\alpha_A},\quad
\beta_K=\frac{\alpha_K}{\alpha_A},\quad
d_T=(T-T_r)-\beta_T(A-A_r),\quad
d_K=(K-K_r)-\beta_K(A-A_r).
\]

Les termes en force s'annulent :

\[
\dot d_T=-d_T/\tau_f,\qquad \dot d_K=-d_K/\tau_f.
\]

Pour un offset entrant fixé \(d_k\), ses stages Radau se calculent donc une fois :

\[
\mathbf d=\left(I+\frac h{\tau_f}\mathcal A\right)^{-1}\mathbf1 d_k.
\]

La reconstruction au point \(i\) reste locale :

\[
T_i=T_r+\beta_T(A_i-A_r)+d_{T,i},\qquad
K_i=K_r+\beta_K(A_i-A_r)+d_{K,i}.
\]

La différence avec le test précédent est précise : ce dernier supprimait également `A` aux stages, ce qui couplait chaque état reconstruit à plusieurs forces internes. Ici, `A_i` reste une variable du NLP et la reconstruction de `T_i,K_i` dépend de ce même `A_i`.

### Bornes et parcimonie

Lorsque \(z_i=a_i+b A_i\), une borne \(\ell_z\le z_i\le u_z\) peut devenir une borne sur \(A_i\). On inverse les extrémités si \(b<0\), puis on intersecte les bornes obtenues pour A, Tau1 et Km. Si \(b=0\), la condition porte uniquement sur une donnée connue. Les bornes sur le calcium deviennent elles aussi des vérifications numériques.

Cette transformation peut éviter d'ajouter une contrainte générale pour chaque état éliminé. Elle doit être vérifiée avec les bornes effectives de chaque point, sans retirer aucun objectif ou aucune contrainte physiologique. Les multiplicateurs des bornes actives et les transferts RHO nécessitent une correspondance dédiée ; la conversion des bornes peut changer le chemin de la méthode de barrière même si l'ensemble faisable reste identique.

Le potentiel vient donc de **moins d'états et de défauts avec une reconstruction locale**, et non simplement d'un compte de variables plus petit. Le temps MA57 et les non-zéros du KKT décideront si cette hypothèse est utile.

## Recommandations hiérarchisées

1. **Priorité 1 : ablation calcium seul, Radau-5 strict.** Précalculer les stages et nœuds calciques d'une fenêtre figée, maintenir les autres états. Mesurer le bénéfice isolé et vérifier les conditions de précalcul.
2. **Priorité 2 : variante locale `F,A`.** Ajouter les offsets Radau pour Tau1/Km, convertir les bornes lorsque c'est équivalent et vérifier les dérivées. Cette expérience est la contribution la plus pertinente du prompt à la suite du travail local.
3. **Priorité 3 : audit de précision impulsionnelle.** Rejouer des commandes imposées, examiner le stage terminal de chaque intervalle et comparer à une référence segmentée DOP853 ou Radau raffinée. Ce test évalue la précision ; son résultat ne présuppose pas un gain de temps.
4. **Priorité conditionnelle : carte semi-analytique.** À considérer si les dérivées ou l'intégrateur restent dominants après mesure. Conserver explicitement l'erreur de quadrature, le couplage mécanique et la nécessité de certifier les contraintes internes.
5. **Recherche distincte : modèle moyenné.** Réserver son étude à l'initialisation ou à la planification fatigue sur des temps longs, avec une formulation explicite de la perte d'information intra-cycle.

## Ce qu'il n'est pas recommandé de faire

- Promettre ou dimensionner le projet sur les facteurs ×10, ×100 ou ×1000 du prompt : ils ne sont pas démontrés pour notre OCP et plusieurs gains y sont déjà présents.
- Remplacer Radau-5 ou les paramètres physiologiques sur la seule base d'un benchmark SciPy ou de valeurs provenant d'un autre modèle Ding.
- Relancer directement une campagne FATROP/DMS ou 150 RHO avant d'avoir un candidat qui améliore un test apparié court.
- Réutiliser la condensation maximale comme si elle était la variante `F,A` : l'A/B négatif concernait une autre structure.
- Déclarer la carte semi-analytique « exacte », négliger les bornes des états supprimés, ou imposer des offsets de fatigue nuls à un état entrant arbitraire.
- Changer les défauts `dt`, le lissage ou les bornes physiques sans vérifier les conventions du code et l'équivalence du problème.
- Réduire la troncature du calcium en supposant un gain hot solve significatif pour la formulation périodique déjà précalculée.

## Validation minimale avant décision

1. **Figer le cas.** Enregistrer révisions, matériel, solveur, bibliothèques, threads, options de compilation, état initial complet, durée, paramètres musculaires, contraintes et warm-start. Utiliser la baseline courante, pas un temps historique d'une autre campagne.
2. **Prouver le domaine de la réduction.** Vérifier que calendrier, Cn entrant et offsets sont fixés. Inclure des états entrants fatigués avec offsets non nuls. Vérifier la convention d'histoire tronquée et les paramètres muscle par muscle.
3. **Auditer l'équivalence discrète.** Sur des décisions imposées, reconstruire chaque stage et nœud, puis évaluer les défauts, bornes, contraintes et objectifs originaux. Vérifier Jacobienne et Hessienne de la représentation réduite. Rejeter une transformation qui supprime une dépendance décisionnelle.
4. **Mesurer une fenêtre puis plusieurs fenêtres figées.** Comparer baseline, calcium seul, puis `F,A`, avec mêmes entrées et initialisations physiques. Rapporter nombre de variables, égalités, inégalités, non-zéros Jacobienne/Hessienne, construction, évaluations, itérations et temps solveur. Séparer le transfert et l'audit Python du temps natif.
5. **Contrôler la précision physique.** Pour la variante strictement Radau, vérifier l'équivalence à la baseline ; pour toute variante exponentielle ou approchée, ajouter un rejeu à commandes imposées contre DOP853 segmenté et une étude de raffinement. Inclure PW variables, limites et contraintes mécaniques internes.
6. **Passer au RHO fermé si le test court est favorable.** Faire dix fenêtres, analyser aussi les divergences de trajectoire et la qualité de solution, puis envisager 150 RHO seulement si le gain dépasse clairement la variabilité de mesure et si la certification reste valide.

## Décision à ce jour

**IPOPT/MA57 reduced avec Radau-5 reste la meilleure référence scientifique mesurée dans cette exploration.** Le prompt ne renverse pas les résultats précédents. Il précise une expérience encore défendable — calcium précalculé et fatigue réduite localement en gardant `F,A` — dont le bénéfice doit être démontré au niveau NLP. Les cartes approchées et modèles moyennés restent des projets distincts, avec des critères de précision à établir.

Documents locaux complémentaires : [développements mathématiques](ding_analytic_reduction_radau5.md), [A/B de condensation](ding_radau5_symbolic_ab.md), [bilan des solveurs](README.md).
