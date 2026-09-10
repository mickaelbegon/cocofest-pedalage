# Validation numérique de la preview K2 déclenchée

## Résultat principal

Le déclencheur évite presque tous les QP K2, mais il ne change aucune commande
par rapport à l'allocation gloutonne sur les préfixes communs de cette campagne.
Les PW et les états compacts des deux hybrides sont exactement identiques à H1,
y compris lorsque le seuil de 1 % déclenche 10 QP à l'ancre 112/H10 et 21 avant
l'arrêt H30. Les QP déclenchés acceptés ont donc retrouvé la première commande
gloutonne.

À l'ancre 0, toutes les politiques complètent 10 et 30 cycles. À l'ancre 112,
elles complètent H10, puis H1 s'arrête après 607 phases, K2 systématique après
636, et les deux hybrides après 606. L'hybride refuse ici le résultat de son
QP à deux phases après audit, une phase avant le refus courant de H1. Ce n'est
ni un cycle d'endurance perdu ni une preuve d'épuisement du modèle Ding complet.
Cette campagne n'a pas établi un certificat indépendant d'infaisabilité du QP.

Les six trajectoires hybrides complètes ont été rejouées avec Ding complet et
passent les deux gates numériques. Aucun replay H30 complet n'est disponible à
l'ancre 112, car les séquences PW hybrides y sont incomplètes.

## Politique et protocole

L'hybride calcule d'abord la commande gloutonne de la phase courante. Il examine
ensuite la marge disponible à la phase suivante. Quand cette marge est supérieure
au seuil déclaré, la commande gloutonne est appliquée directement. Près d'une
borne, il résout le même QP K2 que la preview systématique. Une solution K2
refusée n'est jamais remplacée silencieusement par H1.

Les choix ont été fixés avant le calcul :

- source RHO réelle et profil réduit identiques aux rapports précédents;
- frontières finales des cycles source d'index zéro 0 et 112;
- horizons de 10 et 30 cycles, 30 phases par cycle;
- témoins H1 glouton et K2 systématique réellement rejoués une fois;
- seuils hybrides 0 % et 1 % de la demande totale maximale absolue de la tâche;
- hybride 1 % désigné favori avant les résultats;
- carte compacte et replay Ding avec 16 sous-pas;
- suivi du moment total original avec tolérance numérique `1e-8 N·m`;
- 60 itérations SLSQP, tolérance d'optimalité `1e-6`, régularisation `1e-8`;
- aucun repli glouton après déclenchement;
- mesures séquentielles, une passe, sans médiane ni calcul concurrent.

Les rapports d'entrée sont refusés si la source, le profil, le contexte de valeur,
les lissages, la bande de suivi nulle, la tolérance, les rayons ou le nombre de
points diffèrent. Le contexte courant reconstruit reproduit les SHA de tâche des
deux cas historiques. Les rayons proviennent du baseline H10 « physical-box »;
ils bornent des amplitudes physiologiques locales, sans prouver que tout point de
la boîte est une frontière terminale atteignable par l'OCP.

## Progression, déclenchements et coût hors NLP

| Ancre | Horizon | H1 | K2 systématique | Hybride 0 % | Hybride 1 % |
|---:|---:|---|---|---|---|
| 0 | 10 | complet, `0,226 s` | complet, `0,746 s` | complet, 0 déclenchement, `0,311 s` | complet, 0 déclenchement, `0,314 s` |
| 0 | 30 | complet, `0,653 s` | complet, `2,189 s` | complet, 0 déclenchement, `0,941 s` | complet, 0 déclenchement, `0,950 s` |
| 112 | 10 | complet, `0,224 s` | complet, `0,738 s` | complet, 0 déclenchement, `0,324 s` | complet, 10 déclenchements, `0,343 s` |
| 112 | 30 | arrêt après 607, `0,452 s` | arrêt après 636, `1,601 s` | arrêt après 606, 1 déclenchement, `0,661 s` | arrêt après 606, 21 déclenchements, `0,706 s` |

Sur les trajectoires complètes, l'hybride demande environ 42 à 46 % du temps de
K2 systématique, mais 1,4 à 1,5 fois celui de H1. Cette différence vient du
screening de la phase suivante : entre 299 et 899 écrans sont calculés selon
l'horizon. Aucun des cas n'a utilisé de repli, et aucun écran n'a échoué.

Le seuil de 1 % vaut `5,969e-3 N·m` à l'ancre 0. La plus petite marge H30 y reste
à `6,435e-3 N·m`, donc aucun QP n'est appelé. À l'ancre 112, le seuil vaut
`5,365e-3 N·m`; dix marges H10 le franchissent. Le seuil 0 % ne s'active que
lorsque la marge devient non positive, à l'approche de l'arrêt H30.

À l'ancre 112/H30, les deux hybrides refusent le QP déclenché à la phase 6 du
cycle futur d'index 20. Ce QP inclut la phase suivante, celle que H1 refuse après
607 phases. K2 systématique suit une autre trajectoire compacte et va jusqu'à
636 phases; les compteurs ne comparent donc pas une endurance physique commune.

## Commandes et états

Les différences maximales des hybrides par rapport à H1 sont exactement nulles
sur chaque préfixe commun : PW, calcium, force et états lents. Par rapport à K2
systématique, les écarts compacts sont :

| Cas | PW max | Force max | État lent max / repos |
|---|---:|---:|---:|
| ancre 0, H10 | `15,61 µs` | `1,600 N` | `1,660e-3` |
| ancre 0, H30 | `33,49 µs` | `1,992 N` | `6,140e-3` |
| ancre 112, H10 | `9,55 µs` | `0,560 N` | `4,967e-4` |
| ancre 112, préfixe H30 | `106,20 µs` | `1,354 N` | `2,083e-3` |

La dernière ligne compare seulement les 606 phases communes. Ces écarts montrent
que K2 systématique modifie réellement l'allocation; ils ne constituent pas une
mesure de bénéfice d'endurance.

## Replay Ding des hybrides complets

Le replay applique les PW compactes aux équations Ding complètes avec RK4 et 16
sous-pas. Il audite le suivi aux extrémités de phase et les états lents, sans
refermer la boucle hybride sur les états Ding rejoués.

| Ancre | Horizon | Erreur moment max, hybride 0 % / 1 % | Erreur lente max / repos | Gate |
|---:|---:|---:|---:|---|
| 0 | 10 | `0,4804 / 0,4804 mN·m` | `3,814e-5 / 3,814e-5` | passé |
| 0 | 30 | `0,5116 / 0,5116 mN·m` | `1,128e-4 / 1,128e-4` | passé |
| 112 | 10 | `0,4827 / 0,4827 mN·m` | `4,427e-5 / 4,427e-5` | passé |

Les seuils sont `1e-3 N·m` pour le moment et `1e-3` pour l'état lent normalisé.
L'identité des deux colonnes découle de leur identité avec H1 dans ces cas.
Seules les trajectoires centrales ci-dessus ont été validées par Ding. Les 41
points de chaque fit ci-dessous sont contrôlés contre l'oracle compact, pas par
41 replays Ding complets.

## Coût et audit du fit local H10

Le favori pré-déclaré, hybride 1 %, a été ajusté une fois aux rayons H10 acceptés
du baseline physique. Chaque fit comprend 17 points de construction et 24 points
déterministes tenus à l'écart.

| Ancre | Temps hybride 41 points | Décision | Erreur tenue à l'écart max | Erreur mise à l'échelle | Classements |
|---:|---:|---|---:|---:|---:|
| 0 | `17,017 s` | accepté | `3,228e-6` | `2,237e-4` | `247`, aucun échec |
| 112 | `17,686 s` | accepté | `2,156e-5` | `1,477e-3` | `208`, aucun échec |

Les 41 coordonnées, valeurs compactes, prédictions, coefficients, pas de
différences finies et métadonnées de politique sont conservés dans le JSON. Un
fit accepté valide localement les résidus et l'ordre des directions testées; il
ne certifie ni toute la boîte ni un gain clinique.

Un audit indépendant contre les tentatives acceptées du baseline physique donne
une différence exactement nulle sur les 41 coordonnées et les 41 valeurs, aux
deux ancres. Les classements recalculés retrouvent `247` et `208` paires sans
faute. L'hybride n'a donc apporté aucune amélioration de valeur locale observable
sur cet échantillon.

Le rapport batch compatible mesurait historiquement `0,638 s` par fit glouton
de 41 points aux mêmes rayons. Les fits hybrides sont ici environ 26,7 et 27,7
fois plus lents. Cette comparaison historique n'est pas un benchmark simultané,
mais elle suffit à montrer que le backend preview séquentiel n'hérite pas de
l'accélération batch gloutonne. Une mise à jour terminale à chaque cycle n'est
donc pas démontrée assez bon marché par ce prototype, et son surcoût n'est pas
justifié pour cette application testée. Un prochain essai devrait modifier
l'allocation avant la proximité immédiate du blocage, puis vectoriser les 41
évaluations, en conservant les mêmes gates indépendants.

Aucun fit H30 n'a été lancé. En particulier, l'oracle hybride central de l'ancre
112 échoue avant 30 cycles et doit refuser une valeur plutôt que fabriquer une
pénalité finie.

## Portée

Cette campagne valide un mécanisme numérique hors NLP. Elle n'ajoute aucune
variable future au RHO, n'utilise aucune donnée FHO et n'active aucune option
publique du solveur. Le seuil de 1 % réduit le coût par rapport à K2 systématique,
mais aucune modification de trajectoire, aucune amélioration d'endurance et
aucune aptitude clinique ou temps réel ne sont établies.

L'arrêt H30 vient du QP affine anticipé selon le modèle compact. Il ne prouve pas
une limite physiologique globale. L'absence de replay complet n'interdit pas un
futur diagnostic Ding du préfixe, mais ce diagnostic serait distinct d'une
validation de trajectoire H30 complète.

## Artefacts et reproduction

Les artefacts sont dans `.cache/triggered-preview-allocation-validation/` :

- `report.json` : protocole, compatibilité, résultats, 41 points par fit et onze
  empreintes SHA-256 du code réellement exécuté;
- `validation_arrays.npz` : PW, états, moments et replays Ding disponibles;
- `validation.png` : progression, coût, erreur de moment et erreur d'états lents;
  `NR` signifie non rejoué et `FAIL` un replay tenté mais échoué.

Commande reproductible :

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mplconfig \
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -u \
scripts/validate_triggered_preview_allocation.py
```

Tests ciblés :

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mplconfig \
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest -q \
tests/test_triggered_preview_validation.py \
tests/test_triggered_preview_allocation.py \
tests/test_preview_endurance_value.py
```

La campagne complète a pris `75,24 s`, replays Ding et deux fits inclus. Les
temps sont ceux d'une seule passe séquentielle, pas des médianes.
