# Validation numérique de la valeur locale d'endurance

## Résultat

La fonction quadratique diagonale locale est acceptée par les essais tenus à
l'écart pour les deux frontières finales de cycle à horizon 10. À l'ancre 0,
la boîte physiologique initiale suffit. À l'ancre 112, quatre boîtes sont
rejetées avant qu'une cinquième, seize fois plus petite, soit acceptée. Ce
dernier domaine est très étroit : la valeur locale doit donc être recalculée ou
refusée dès que le terminal RHO le quitte.

À l'horizon demandé de 30 cycles depuis l'ancre 112, la politique compacte
échoue après 607 phases complétées, à la phase 7 du cycle futur d'index 20. Le
fit refuse alors de produire un modèle après une seule évaluation centrale. Ce
résultat décrit l'échec de **cette politique d'allocation**, pas une impossibilité
globale, une mesure clinique d'épuisement ou une preuve sur toute stratégie PW.

Toutes les expériences partent de la colonne terminale réelle du cycle source,
y compris son `Cn` archivé. Elles ne lisent aucune trajectoire FHO, n'ajoutent
aucune variable future au RHO et ne lancent aucun contrôleur prospectif.

## Protocole déterministe

Les huit coordonnées sont les quatre dommages
`d = 1 - A/A_rest`, puis les quatre forces divisées par une échelle explicite.
Les deux écarts indépendants de `Tau1` et `Km`, ainsi que `Cn`, restent ceux de
la frontière source. L'échelle de force est `max(F_terminal, 1 N)` par muscle.

La boîte initiale utilise au plus `0,0025` en dommage et `0,02` en force
normalisée. Chaque rayon est aussi limité à 25 % de la coordonnée centrale :
les échantillons gardent donc `d >= 0` et `F >= 0`. Cette boîte respecte ces
bounds physiologiques simples; elle ne prouve pas que chaque point est un état
terminal dynamiquement atteignable par l'OCP d'origine. Une coordonnée centrale
nulle ferait échouer explicitement ce protocole symétrique.

Chaque essai emploie 17 valeurs d'entraînement distinctes : le centre et les
deux différences finies centrales sur chacun des huit axes, avec un pas égal au
quart du rayon. Les 24 points tenus à l'écart comprennent les extrémités des
axes, quatre coins déterministes, deux directions de gradient et les deux coins
du signe du gradient. Aucun point tenu à l'écart n'est utilisé pour ajuster les
coefficients. Un fit est refusé si un rollout échoue, si l'erreur dépasse
`0,002 + 0,05 |V|`, si une paire informative inverse son classement, ou si les
points ne sont pas informatifs. Après un refus, tous les rayons sont divisés par
deux, jusqu'à cinq essais. Une évaluation du polynôme hors boîte est également
testée et doit lever une erreur.

Ces vérifications servent aussi à choisir le rayon après un refus : elles
constituent un jeu de validation du voisinage, pas un troisième jeu de test
prospectif indépendant. Leur réussite reste un contrôle échantillonné et ne
garantit ni toute la boîte ni le comportement du contrôleur.

L'oracle mesure les deux distances signées à l'enveloppe de moment total, puis
leur minimum lissé et une pénalité de marge. Le `soft minimum` peut être négatif
malgré une marge physique minimale positive, à cause du biais `log-sum-exp` sur
les 600 marges de dix cycles. Il ne faut pas interpréter son signe comme une
infaisabilité physique, ni comparer directement des valeurs construites avec
des horizons différents.

## Fits et essais tenus à l'écart

| Ancre | Horizon | Résultat | Essais / temps de fit | Rayon dommage accepté | Rayon force normalisé accepté | Erreur tenue à l'écart max | Classements informatifs |
|---:|---:|---|---:|---|---|---:|---:|
| 0 | 10 | accepté | 1 / 9,13 s | `[2,50e-3, 2,50e-3, 1,13e-3, 1,20e-4]` | `[1,17e-3, 2,15e-3, 2,00e-2, 2,00e-2]` | `3,23e-6` | 247, zéro inversion |
| 112 | 10 | accepté | 5 / 45,62 s cumulées | `[1,56e-4, 1,22e-4, 1,56e-4, 1,56e-4]` | `[9,71e-5, 2,06e-4, 1,25e-3, 1,25e-3]` | `2,16e-5` | 208, zéro inversion |
| 112 | 30 | refusé | 1 / 0,263 s | aucun | aucun | aucune valeur | aucun |

Le premier fit de l'ancre 112 avait une erreur maximale `1,52e-3` et 14
inversions de classement. Les trois réductions suivantes avaient encore 3, 2
et 1 inversions. Le protocole ne les compte pas comme des succès, même si leur
erreur absolue satisfaisait à elle seule la tolérance. Au total, les fits H10
ont observé 246 évaluations complètes (`41 + 5×41`) et aucun rollout compact
échoué. Le cas H30 a observé une évaluation centrale `policy_failed`, sans lui
substituer de coût artificiel.

Les deux modèles acceptés rejettent bien le point volontairement placé hors de
leur boîte. Les rayons très faibles à l'ancre 112 et le coût cumulé des quatre
refits rejetés montrent qu'une mise à jour n'est pas encore une opération
« gratuite », même si elle reste hors du NLP.

## Rejeu indépendant de Ding complet

Pour chaque fit H10, les PW compactes du centre et du coin de descente prédit
ont été rejouées avec les équations Ding complètes, RK4 à 16 sous-pas. Ce rejeu
contrôle la dynamique et le suivi du moment obtenus avec ces PW; il ne remplace
pas l'enveloppe compacte par une fonction de valeur Ding complète et ne valide
donc pas indépendamment le gradient de la valeur.

| Ancre / point | Temps compact | Temps rejeu Ding | Erreur moment max | Erreur lente max / repos | Gate `1e-3` / `1e-3` |
|---|---:|---:|---:|---:|---|
| 0 / centre | 0,132 s | 2,906 s | `4,804e-4 N·m` | `3,814e-5` | passé |
| 0 / coin de descente | 0,132 s | 2,885 s | `4,800e-4 N·m` | `3,814e-5` | passé |
| 112 / centre | 0,132 s | 2,924 s | `4,827e-4 N·m` | `4,427e-5` | passé |
| 112 / coin de descente | 0,130 s | 2,918 s | `4,825e-4 N·m` | `4,421e-5` | passé |

Pour l'échec H30, un diagnostic numérique séparé au point de défaillance donne
un moment demandé de `0,08466477 N·m` et une borne compacte inférieure de
`0,08480980 N·m`, soit une surproduction minimale de `1,450e-4 N·m`. Un calcul
ponctuel Ding raffiné donne le même sens avec des écarts de l'ordre de
`0,6–1,5e-4 N·m`, inférieurs au seuil de rejeu `1e-3 N·m`. Il s'agit donc d'une
frontière très proche imposée par le suivi exact de la politique. La suite
utile est d'étudier une tolérance de suivi bornée ou une allocation plus
anticipative, pas de tronquer silencieusement le rollout.

## Portée et artefacts

Ces nombres valident localement un modèle numérique conditionnel à une tâche,
une phase, un régime calcique et une politique donnés. Ils ne démontrent pas
une amélioration d'endurance, une robustesse clinique, la validité sur toute la
boîte, ni la performance d'un RHO muni de ce coût terminal.

Le premier dossier `.cache/local-endurance-value-validation/` provenait d'un
diagnostic dont la boîte autorisait des dommages négatifs (`A > A_rest`). Il est
supplanté et ne doit pas être cité comme validation. Les résultats finaux sont :

- `.cache/local-endurance-value-validation/physical-box/report.json` : toutes
  les tentatives, statuts, timings, coefficients, erreurs et empreintes SHA-256;
- `.cache/local-endurance-value-validation/physical-box/validation_arrays.npz` :
  points tenus à l'écart, valeurs, coefficients et replays;
- `.cache/local-endurance-value-validation/physical-box/validation.png` :
  résidus, effets locaux et erreurs du rejeu Ding.

Commande reproductible :

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mplconfig \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/validate_local_endurance_value.py
```

Les quatre tests ciblés du runner passent avec :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest -q \
  tests/test_local_endurance_value_validation.py
```
