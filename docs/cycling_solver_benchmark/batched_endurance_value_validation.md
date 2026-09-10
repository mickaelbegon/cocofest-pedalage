# Validation numérique de la valeur d'endurance vectorisée

## Conclusion

La version vectorisée reproduit la valeur stricte et la décision d'acceptation
des fits scalaires sur les mêmes points et les mêmes rayons. Pour 41 valeurs
H10, elle prend `0,270–0,271 s` contre `8,75–8,94 s` en appels scalaires
séquentiels, soit `32,35–33,08×` plus rapide dans ces deux mesures uniques.
Un fit complet prend environ `0,64 s` par tentative au lieu de `9,1 s` dans le
rapport scalaire antérieur, soit `14,2–14,3×` plus rapide. Ces temps Python ne
sont ni un benchmark temps réel, ni une mesure du temps total d'un contrôleur.

L'écran H30 pré-déclaré à l'ancre 112 ne produit aucune politique complète :
les bandes `0` et `1e-4 N·m` échouent après 607 phases, et `5e-4 N·m` avance
jusqu'à 697 phases avec trois phases relâchées avant d'échouer. Aucun coût H30,
fit H30 ou rejeu Ding H30 n'est donc inventé.

L'expérience part de la frontière finale réelle des cycles RHO source 0 et 112,
emploie le même profil réduit et ne lit aucune trajectoire FHO. La vectorisation
et le coût sont calculés hors NLP et n'ajoutent aucune variable future au RHO.

## Parité stricte

Les 41 points de chaque cas sont exactement les 17 points d'entraînement et les
24 points tenus à l'écart du rapport `local_endurance_value_validation.md`.
Avant tout chronométrage, le runner refuse ce baseline si son schéma, ses
empreintes SHA-256 source/profil, ses seuils, son type de fit ou ses sous-pas
diffèrent. Pour chaque ancre, il vérifie aussi l'horizon, l'échelle de moment,
le centre et l'empreinte complète du contexte de coordonnées. Des rayons ou
timings issus d'une autre tâche ne peuvent donc pas être comparés silencieusement.
Le timing est un appel batch suivi d'un passage scalaire séquentiel, sans
exécution concurrente ni médiane de répétitions. La construction du prédicteur
compact coûte environ `0,054 s`; l'enveloppe batch ajoute moins de `7 µs` dans
ces mesures.

| Ancre, H10 | Temps batch, 41 valeurs | Temps scalaire | Rapport | Écart valeur max |
|---:|---:|---:|---:|---:|
| 0 | `0,2706 s` | `8,7542 s` | `32,35×` | `5,55e-17` |
| 112 | `0,2703 s` | `8,9424 s` | `33,08×` | `1,67e-16` |

Sur cinq points représentatifs par ancre, tous les statuts et nombres de phases
complétées sont identiques. Les écarts maximaux batch/scalaires observés sont :

- états : `4,55e-13`;
- PW : `4,28e-18 s`;
- moments musculaires atteints : `1,21e-15 N·m`;
- moments alloués : `1,21e-15 N·m`.

Ces différences sont du niveau de l'ordre des opérations en virgule flottante.
Elles ne changent aucun statut dans cet échantillon; elles ne constituent pas
une preuve d'identité pour tous les états possibles.

## Parité des fits locaux

Les rayons, pas de différences finies, tolérances de résidu et gate de
classement sont ceux du rapport scalaire physique. La séquence complète est
rejouée avec le backend batch :

| Ancre | Séquence d'acceptation | Temps batch cumulé | Scalaire antérieur | Classements du fit accepté | Erreur tenue à l'écart max |
|---:|---|---:|---:|---:|---:|
| 0 | accepté | `0,638 s` | `9,130 s` | 247, zéro inversion | `3,23e-6` |
| 112 | refusé, refusé, refusé, refusé, accepté | `3,183 s` | `45,617 s` | 208, zéro inversion | `2,16e-5` |

Chaque tentative de l'ancre 112 reproduit le nombre d'inversions scalaire
(`14, 3, 2, 1, 0`). L'écart absolu maximal entre les vecteurs de coefficients
acceptés vaut `6,15e-8` à l'ancre 0 et `3,77e-7` à l'ancre 112. Les très petits
pas peuvent amplifier l'arrondi dans les courbures obtenues par seconde
différence; la parité des 41 valeurs, des résidus et des classements est le test
opérationnel pertinent ici.

Les métadonnées figent l'empreinte de tâche, l'horizon, les échelles, les
températures, la bande et son poids. Avec une bande nulle, le coût de suivi est
désactivé et la valeur est celle de l'oracle scalaire strict.

## Écran exploratoire de bande de suivi H30

Une bande non nulle autorise seulement la projection du moment total demandé
sur l'enveloppe atteignable lorsqu'il s'en écarte d'au plus la bande. Elle
**change la tâche prédite** : un résultat avec bande ne doit jamais être appelé
« suivi exact ». La tolérance numérique du QP reste distincte. Les erreurs
ci-dessous sont toujours signées par rapport à la cible originale.

| Bande | Statut de la politique | Phases / 900 | Phases relâchées | Erreur max sur phases réussies | Intégrale rectangulaire `Σ|e_k|Δt` | Première défaillance |
|---:|---|---:|---:|---:|---:|---|
| `0` | échec | 607 | 0 | `1,11e-16 N·m` | `1,75e-16 N·m·s` | cycle 20, phase 7 |
| `1e-4 N·m` | échec | 607 | 0 | `1,11e-16 N·m` | `1,75e-16 N·m·s` | cycle 20, phase 7 |
| `5e-4 N·m` | échec | 697 | 3 | `4,779e-4 N·m` | `3,115e-5 N·m·s` | cycle 23, phase 7 |

Pour `5e-4 N·m`, l'intégrale signée vaut aussi `+3,115e-5 N·m·s` : selon le
modèle compact, les trois écarts sont donc des surproductions dans cet écran.
Ce n'est pas encore une preuve physique. Cette somme aux extrémités de phase
n'est ni une intégrale continue de suivi, ni un travail mécanique.

À la défaillance stricte, la borne compacte inférieure dépasse la cible de
`1,450e-4 N·m`; la bande `1e-4` est donc insuffisante. Après les trois
relâchements permis par `5e-4`, la défaillance suivante présente un dépassement
de `6,438e-4 N·m`, supérieur à la bande. La politique avec bande n'est pas
complète, donc l'oracle refuse une valeur scalaire et ne calcule ni coût de
marge final ni coût de suivi. Le poids exploratoire fixé à 1 n'a ainsi produit
aucun candidat H30 utilisable et ne doit pas être interprété comme calibré.

Aucun des trois cas H30 pré-déclarés n'étant complet, aucun rejeu Ding sur
30 cycles complets n'a été lancé. On ne revendique donc pas un contrôle
`<= 1e-3 N·m` sur cet horizon. Cela n'interdit pas un diagnostic Ding
explicitement limité au préfixe complété. Les trois bandes ont été fixées
avant le calcul.

## Portée, artefacts et reproduction

La validation montre une parité numérique locale et une accélération de cette
implémentation Python particulière. Elle ne valide pas un gain d'endurance, la
robustesse clinique, la performance dans le RHO, ni les interactions croisées
qu'une Hessienne diagonale omet encore.

Les artefacts sont dans `.cache/batched-endurance-value-validation/` :

- `report.json` contient les statuts, parités, timings, fits, erreurs de suivi
  et empreintes SHA-256;
- `validation_arrays.npz` contient les points, valeurs et séries de cible,
  enveloppe et erreur;
- `validation.png` compare parité, latence, tentatives de fit et progression H30.

Commande reproductible :

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mplconfig \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/validate_batched_endurance_value.py
```

Tests ciblés du runner et du prédicteur batch :

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mplconfig \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest -q \
  tests/test_batched_endurance_validation.py \
  tests/test_batched_compact_muscle_prediction.py
```
