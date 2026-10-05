# Protocole d'identification Ding pour le pédalage FES

## Objectif et périmètre

Ce protocole identifie séparément les paramètres de force fraîche puis de
fatigue du modèle Ding pour les quatre muscles de chaque côté. Chaque côté est
estimé indépendamment : une symétrie éventuelle est une hypothèse testée a
posteriori, jamais une contrainte initiale. Le résultat visé est un modèle qui
prédit le couple, le travail par cycle et l'évolution de fatigue sous la
politique de pédalage, et non seulement une force moyenne sur des contractions
isolées.

Le protocole est réservé à une tâche isocinétique instrumentée. Toute session
doit avoir une procédure clinique et de sécurité approuvée indépendamment de
ce document.

## Mesures, horloge et observation mécanique

Synchroniser sur une même horloge : angle et vitesse du pédalier, stimulation
(temps, courant et PW), force tangentielle de chaque pédale et, si disponible,
force radiale. L'observable primaire est le couple utile :

\[
  \tau_{pedal}(t) = r F_t(t).
\]

Le modèle d'observation est

\[
  \tau_{pedal}(t) = \sum_m g_m(\theta(t),\dot\theta(t))F_m(t)
    + \tau_{passif}(\theta(t),\dot\theta(t)) + \epsilon(t).
\]

Les gains mécaniques `g_m` proviennent du modèle réduit, avec un facteur
muscle-spécifique éventuellement identifié durant la première session. Avec
un seul DoF, la force tangentielle suffit à identifier le couple utile. Avec
deux DoF, la force radiale est une observation secondaire : elle sert à
détecter posture, glissement, compensation et force parasite; elle ne doit
pas être confondue avec une contribution directe au travail du pédalier.

Avant toute identification, estimer les offsets, bruit et dérive des capteurs
sur des cycles passifs. Les cycles qui échouent des critères de contact ou de
stabilité de `F_r` sont exclus ou repondérés, avec leur motif conservé.

## Phase 1 — force fraîche

Identifier la force sans fatigue appréciable, un muscle à la fois. Pour chaque
côté et muscle :

1. mesurer un court bloc passif;
2. appliquer des trains courts, séparés par un repos pré-déclaré;
3. varier PW, fréquence et phase de pédalage dans les secteurs où le gain
   mécanique est informatif;
4. vérifier le retour du couple de référence avant le train suivant.

Ne pas limiter l'identification à 30 Hz : un petit ensemble de fréquences et
d'intervalles inter-impulsion est nécessaire pour séparer recrutement,
calcium et dynamique force-fréquence. Les paramètres de recrutement PW,
amplitude de force et dynamique rapide sont ajustés ici. La force des huit
muscles ne doit pas être estimée simultanément à partir du seul couple total;
les essais mono-muscle rendent ce problème identifiable. Les combinaisons de
muscles sont réservées à la validation.

## Phase 2 — fatigue et récupération

Fixer, ou régulariser fortement, les paramètres de force fraîche estimés à la
phase 1. Pour chaque muscle et côté, appliquer des blocs de stimulation à la
cadence et dans les fenêtres angulaires de la tâche, à plusieurs niveaux de
demande. Insérer régulièrement une sonde de référence inchangée et une période
de récupération contrôlée.

Les sondes permettent de distinguer une baisse de couple produite par une
commande différente d'une diminution de capacité. Les états Ding `Cn`, `F`,
`A`, `Tau1` et `Km` ne sont pas tous libérés par défaut : l'excitation doit
justifier chaque paramètre, sinon il est fixé par un prior physiologique ou
partagé hiérarchiquement. Une fréquence fixe à 30 Hz ne suffit généralement
pas à identifier les paramètres calcium/force-fréquence.

## Conception avancée des essais

Construire un catalogue d'essais candidats, chacun défini par côté, muscle,
phase de départ, durée, séquence PW, fréquence et repos suivant. Pour chaque
candidat, calculer les sensibilités du couple prédit par rapport aux
paramètres libres et la matrice d'information de Fisher pondérée par le bruit :

\[
  I(\xi) = S(\xi)^T W S(\xi).
\]

Sélectionner séquentiellement les essais par un critère D-optimal robuste,
maximisant l'augmentation de `log det(I + prior)` tout en pénalisant :

- charge de stimulation et fatigue prédite;
- proximité des bornes PW/courant;
- répétition excessive du même secteur angulaire;
- incertitude de posture détectée par la force radiale;
- durée totale et temps de récupération.

Utiliser les valeurs singulières et le conditionnement de `I` comme critères
d'arrêt : un essai très énergique qui n'améliore pas la plus petite direction
observable n'est pas retenu. Après chaque bloc, mettre à jour le posterior
des paramètres et re-sélectionner dans le catalogue restant. Les candidats de
force fraîche et de fatigue doivent avoir des ensembles de paramètres cibles
distincts.

## Estimation et validation

Estimer les paramètres par MAP/multiples shooting, en intégrant les états Ding
avec le même schéma que celui retenu pour la tâche. Les états initiaux de
fatigue sont propres à chaque bloc; les paramètres physiologiques sont globaux
pour un muscle/côté. Rapporter le rang, les valeurs singulières, les
corrélations de paramètres, les bornes actives et l'incertitude locale.

Conserver des essais et des séquences de pédalage complets hors ajustement.
Valider séparément l'erreur de couple tangentielle, la phase, le travail par
cycle, la réponse des sondes de référence et la prédiction de capacité sur
plusieurs cycles. Des résidus corrélés à l'angle, la cadence ou `F_r` indiquent
une insuffisance mécanique ou instrumentale, et non automatiquement un mauvais
paramètre Ding.

## Prototype logiciel et limites actuelles

Le prototype versionné sépare quatre composants :

- `cocofest.identification.synthetic_cycling` génère des essais mono-muscle
  synthétiques pour les huit muscles, avec couple tangentiel, force radiale
  optionnelle, bruit et biais configurables;
- `cocofest.identification.optimal_design` calcule les sensibilités blanchies,
  l'information Fisher, le rang et une sélection D-optimale sous budgets;
- `cocofest.identification.cycling_design` construit des catalogues force
  (PW × fréquence) et fatigue (train, récupération, sonde);
- `cocofest.identification.sequential` réalise l'ajustement MAP gaussien
  borné, puis bloque les paramètres de force avant l'ajustement fatigue.
- `cocofest.identification.cycling_geometry` convertit un profil réduit du
  bioMod en bras de levier signé et gain force--longueur/vitesse à chaque
  phase; la force radiale reste une observation du capteur, pas une grandeur
  déductible du couple.
- `cocofest.identification.uncertainty` rééchantillonne des essais/blocs
  complets et propage les réajustements force vers fatigue avant de former un
  ensemble de paramètres pour le design suivant.

Le prototype utilise actuellement une résolution moindres carrés bornée et
des différences finies pour les sensibilités. Son information Fisher est
locale, fondée sur les données et sur un bruit temporel indépendant. Le
bootstrap par blocs fournit une dispersion empirique et un ensemble à injecter
dans le design fatigue, mais ne corrige ni un biais de modèle ni une dépendance
temporelle mal modélisée. Les données corrélées doivent être préblanchies ou
sous-échantillonnées avant l'ajustement. De même, le générateur conserve une
géométrie constante par défaut uniquement pour les démos : les essais réels
doivent utiliser l'adaptateur du profil réduit, avec une cadence cohérente avec
l'horloge et l'orientation du pédalage.

## Livrables par participant

- paramètres force fraîche et fatigue, avec unités, bornes et covariance;
- définition exacte des capteurs, conventions de signe et calibration;
- catalogue et ordre effectif des essais, y compris essais rejetés;
- données brutes synchronisées et données nettoyées traçables;
- rapport d'identifiabilité et validation hors échantillon;
- fichier de configuration directement utilisable par le RHO, distinct des
  données ayant servi à l'identification.
