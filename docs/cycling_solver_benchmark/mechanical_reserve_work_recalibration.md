# Recalibrage de la réserve mécanique : travail signé par cycle

## Diagnostic des ablations

Les trois essais `reserve-h1*-l003-240-results` arrêtent le bras gauche à la
tentative 169, après 168 cycles certifiés. Les marges minimales de cette
tentative valent −6,9755 (H1), −6,9740 (H1+5) et −6,9659 (H1+5+20).
Cette similarité finale ne signifie pas que les horizons ont toujours donné
le même coût : au cycle 2, les valeurs étaient respectivement −5,3478,
−5,7691 et −7,1309. Ces nombres proviennent des `left/result.json` locaux.

Le calibrateur historique fait un replay de **tous les muscles à PWmax** et
extrait une marge pour chaque phase, puis une marge de travail. Deux problèmes
distincts en résultent :

1. Un muscle peut produire du travail négatif à certaines phases. PWmax
   simultanément pour tous les muscles n'est pas une enveloppe maximale du
   travail mécanique, même si elle donne de grandes forces.
2. Comparer la puissance de chaque phase à la puissance moyenne demandée
   ajoute au proxy une exigence absente d'une cible de travail par cycle.
   La soft-min peut alors se concentrer sur une phase antagoniste tout en
   ignorant une réserve de travail globalement positive.

Les unités du Jacobien historique sont cohérentes : dérivée des marges
normalisées par rapport aux états physiques A/Tau1/Km. Il n'y a pas de facteur
de normalisation manquant identifié dans cette chaîne. Le problème principal
est la signification de la quantité différentiée.

Une troisième limite concerne l'horizon : dans le binding actuel, le profil
de forces répété est une donnée du dernier cycle certifié. Le coût dépend des
PW courantes **via les états lents terminaux**, mais pas directement via le
profil courant de forces. À profil figé, la dérivée de l'état projeté par
rapport à l'état terminal est `exp(-H*T/tau_fat)`. Si les muscles ont le même
tau_fat, changer H modifie surtout l'échelle et les offsets, pas la direction
des préférences. Recalibrer la marge ne supprime pas cette limite structurelle.

## Variante expérimentale `cycle_work_gated_v1`

`calibrate_isokinetic_ding_margin_model(...,
calibration_policy="cycle_work_gated_v1")` retourne une unique marge :

`(travail du replay − travail requis) / (échelle de puissance × durée du cycle)`.

L'échelle reste un argument explicite. Choisir la puissance moyenne requise
pour une charge positive exprime la marge en fraction de la demande. Les
contributions mécaniques signées et les termes non musculaires sont conservés.

Pour chaque muscle, cinq calendriers bornés sont évalués depuis le même état
complet Cn/F/A/Tau1/Km : PWmax, PW au seuil de recrutement `pd0`, puis PWmax
durant les phases productives anticipées de 0, Tau1 ou 2×Tau1. Chaque calendrier
est propagé par les équations complètes de Ding. Les forces résiduelles et la
mémoire du calcium sont donc conservées; on ne sélectionne pas indépendamment
une force maximum à chaque phase. Les candidats identiques sont dédupliqués.

Le calendrier donnant le plus de travail signé est sélectionné séparément pour
chaque muscle. La mécanique prescrite et les dynamiques musculaires séparables
permettent cette sélection sans QP/OCP additionnel. Elle contient toujours
PWmax parmi ses candidats; le travail de référence ne peut donc pas être
inférieur à celui du replay historique, à quadrature et états identiques.

La sélection est **figée pendant les différences finies** sur les états lents.
Le Jacobien décrit ainsi la même stratégie de PW autour du point de référence;
il ne mélange pas les changements discrets de calendrier. Les replays inchangés
des autres muscles sont mis en cache pendant la calibration. Le coût reste
affine en états projetés dans le NLP et ne contient aucune sélection discrète.

La variante historique reste disponible par défaut via
`calibration_policy="simultaneous_pwmax_v1"`. La nouvelle variante a une seule
marge au lieu de `nombre_de_phases+1`, ce qui se choisit à la construction du
NLP; la structure ne change ensuite pas lors des actualisations numériques.

## Portée scientifique

Ce score reste une opportunité de travail par replay, **pas une faisabilité
certifiée**. Les calendriers ne satisfont pas nécessairement les limites de
couple, de variation des PW, de demi-cycle suivant ou la fermeture des forces.
La sélection est limitée à cinq candidats par muscle; elle n'est pas un
maximum global de travail. Les géométries et forces aux fins de phases sont
utilisées pour une quadrature approchée. Le replay conserve les états rapides
au point de référence, puis seule la sensibilité initiale des états lents est
linéarisée. Une projection longue peut quitter son domaine local de précision.

Une marge positive ne prouve pas qu'un cycle RHO fermé est réalisable; une
marge négative ne prouve pas un échec physiologique. Aucun ajustement ne dépend
de données FHO ou des poids optimisés par BO.

## Tests numériques et protocole suivant

`tests/test_mechanical_reserve_work_calibration.py` vérifie :

- un exemple où une puissance de phase est négative mais le travail de cycle
  est positif;
- une augmentation du travail signé par rapport à PWmax, et l'identité entre
  les forces retenues et un replay complet indépendant des PW sélectionnées;
- l'invariance du score et du Jacobien à une mise à l'échelle mécanique
  cohérente;
- les six sensibilités A/Tau1/Km de deux muscles par différences finies;
- le gradient CasADi et une préférence correcte entre allocations produisant
  le même travail dans un exemple synthétique à fatigabilités différentes.

Ce dernier test vérifie le potentiel décisionnel de la primitive lorsque
les forces sont variables. Il ne démontre pas que le binding RHO actuel,
dont les forces de rollout sont figées, transmet la même sensibilité.

Avant une campagne longue :

1. Sur un cycle certifié réel, enregistrer marges phase/travail historiques,
   travail recalibré, calendrier choisi et Jacobien normalisé. Vérifier les
   domaines ainsi que la sensibilité à une variation de 1–5 % de chaque état.
2. Faire un test local du **coût réellement connecté au NLP** : dérivées par
   rapport aux PW et classement de plusieurs répartitions admissibles.
   Comparer les sensibilités à H1 et H20 et le poids relatif au coût de base.
3. Un smoke test de 3 cycles vérifie activation au cycle 2, taille constante
   des paramètres et réutilisation du solveur. Un essai de 21 cycles franchit
   ensuite la deuxième mise à jour de calibration.
4. Seulement après ces portes, faire l'ablation longue à même charge, mêmes
   modèles et mêmes récupérations : unitaire, ancien proxy travail-seul,
   puis nouveau proxy travail-gaté. Ce contrôle intermédiaire permet de
   distinguer l'effet d'enlever les fausses contraintes de phase de l'effet
   de sélectionner des calendriers moins antagonistes. Tester la force
   courante dans la projection serait une ablation séparée.

Rapporter endurance certifiée, raisons d'arrêt, temps de solve médian/p95,
temps de calibration séparé et moyenné sur sa période d'actualisation. Les
gains synthétiques de travail ou la correction du signe du proxy ne constituent
pas encore un bénéfice d'endurance sur le modèle bilatéral.

## Accélération et parallélisme (2026-09-28)

Les deux bras sont les unités de parallélisme : deux processus persistants,
un cœur par solveur IPOPT/MA57, synchronisés uniquement à la frontière de
cycle. Une mise à jour lente est donc calculée simultanément à gauche et à
droite; son délai est le maximum des deux temps, pas leur somme. Ajouter des
threads ou des processus par muscle à l'intérieur de ces workers créerait une
sur-allocation des cœurs et coûte plus que les petits replays Ding.

Une tentative de vectorisation des petites différences finies n'a pas réduit
le temps mesuré et a été écartée. La version retenue propage à la place la
matrice tangentielle de Ding avec le même RK4 que le replay. Elle remplace les
six replays +/- par muscle par un replay et ses trois dérivées, tandis que le
calendrier PW sélectionné reste figé.

Sur le smoke test bilatéral 21 cycles à 30 Hz (MA57, deux cœurs, actualisations
aux cycles 1 et 20), le délai critique de préparation est passé de 2.92 s puis
3.29 s à 1.89 s puis 2.11 s (maximum des deux bras), soit environ 36 % de moins
sur ces deux actualisations. Les 21 cycles ont été certifiés; la médiane du
temps de paire était 1.12 s et le p95 1.48 s. Les tangentes sont comparées aux
différences finies dans les tests unitaires et à l'ancienne calibration
complète (tolérance relative 4e-7). Les deux formulations restent disponibles
pour une validation de référence via `sensitivity_mode="finite_difference"`;
le mode par défaut est `"tangent"`.
