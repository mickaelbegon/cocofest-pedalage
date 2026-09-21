# Validation Radau à 50 Hz contre DOP853

Le but est de mesurer l'erreur d'intégration qui peut créer un arrêt RHO artificiel.
Une convergence IPOPT à `1e-8` mesure le respect du problème discrétisé et ne
garantit pas la précision de sa dynamique continue. À 30 Hz, le cas fatigué
`frozen-cycle76-fatigue-degree5-dop853-x0p5` présentait déjà une erreur maximale
de phase de 0,02964 rad avec Radau-5. Radau-5 est donc un point de départ de
validation, pas une certification universelle.

## Pilote reproductible

`scripts/validate_radau_frequency.py` prépare une solution à 50 Hz avec Radau-5,
puis relance un cycle Radau-3 et un cycle Radau-5 depuis le même seed complet.
Le modèle est dynamique, la résistance constante, le solveur IPOPT/MA57, le cycle
dure une seconde et les deux degrés ont **50 décisions PW par muscle et par
cycle**. Augmenter le nombre de stimulations ne constitue pas un raffinement du
maillage: cela changerait aussi la fréquence et l'espace de commande.

Chaque trajectoire optimisée est rejouée avec ses propres PW, en DOP853,
`rtol=1e-11`, `atol=1e-13`. Le rejeu repart de son état initial complet et ne
réinitialise jamais les états aux nœuds optimisés. Les intervalles de stimulation
sont intégrés séparément pour respecter les changements de PW et le forçage
calcique. Les états initiaux, la fréquence et le nombre de PW sont comparés
explicitement dans le résumé. La préparation sert à apparier les conditions;
elle n'est pas un cycle d'endurance certifié.

```bash
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
"$PY" scripts/validate_radau_frequency.py \
  --model-config examples/fes_multibody/cycling/two_model_ding_campaign/ding_triceps_alpha_a_x0p5.json \
  --reduced-profile resistance-fho-pilots-20260910/seed-0p10/reduced-cycling-fourier12.npz \
  --hsl-library "$HSL" --frequency-hz 50 --resistance-nm 0.22 \
  --output-directory radau-frequency-validation-50hz-new --run
```

Le pilote utilise un cœur par résolution et lance les trois OCP séquentiellement.
Les sorties doivent être nouvelles. `manifest.json` conserve les commandes,
`launcher.log` les sorties des solveurs, `summary.json` les métriques. L'option
`--summarize-only` relit les résultats sans lancer d'OCP.

## Mesures et décision

- Erreur de trajectoire par état aux extrémités de chaque intervalle, maximum
  absolu et maximum normalisé; présenter séparément phase, vitesse, force,
  capacité et autres états de Ding, car leurs unités diffèrent.
- Erreur DOP853 de fermeture du cycle relativement au déplacement de `-2π`;
  la tolérance de tâche actuelle est environ 0,002 rad.
- Budget de précision proposé: erreur de phase due à la transcription au plus
  **0,0002 rad**, soit 10 % de cette tolérance de tâche. Ce seuil est un choix
  d'ingénierie explicite à vérifier, pas un seuil physiologique.
- Coût: temps solveur, temps total, nombre d'évaluations du rejeu et dimensions
  du problème. Ces durées incluent un démarrage et ne mesurent pas la latence
  RHO chaude avec réutilisation du NLP compilé.

La réussite de ce pilote signifie seulement que le budget de phase est respecté
pour ce cas. La fermeture du geste et le budget d'erreur sont deux mesures
distinctes: une erreur faible peut accompagner une trajectoire réellement
incapable de fermer le cycle, et une fermeture fortuite peut masquer une erreur
d'intégration élevée.

## Extension nécessaire avant les campagnes d'endurance

1. Répéter à 50 Hz à état peu fatigué, intermédiaire et proche de l'arrêt, pour
   les deux jeux de paramètres Ding et plusieurs résistances constantes. Une
   trajectoire à 30 Hz ne peut pas être présentée comme un historique à 50 Hz:
   les états calciques et l'historique de stimulation doivent être cohérents.
2. Effectuer aussi un test **à PW identiques imposées**, pour isoler la seule
   intégration. Le pilote OCP actuel autorise des PW optimales différentes entre
   degrés et mesure donc l'erreur des solutions obtenues par chaque méthode.
3. Vérifier la convergence de la référence en abaissant les tolérances DOP853
   d'un facteur 10; sa variation doit rester bien en dessous du budget de
   transcription. Le pilote actuel emploie une seule paire de tolérances.
4. Tester Radau-7 ou plusieurs sous-intervalles **à PW constante entre deux
   stimulations**, sans ajouter de décisions ni augmenter la fréquence. Ce
   raffinement h nécessite une représentation distincte du maillage et des
   instants de stimulation dans le code actuel.
5. Ajouter des contrôles denses des bornes de vitesse, force et états musculaires
   ainsi que de la phase terminale, puis rejouer plusieurs cycles sans
   réinitialiser les états afin de quantifier l'accumulation des erreurs.

Un échec numérique, un désaccord entre degrés, ou un rejeu continu hors budget
laisse le cas indéterminé pour l'endurance. La sonde de déficit de réalisation
à état figé ne devient interprétable qu'après ces vérifications numériques.

## Premiers résultats du 12 septembre 2026

Six OCP ont été exécutés séquentiellement dans `cocofest-rho32`, IPOPT/MA57,
avec le modèle `ding_triceps_alpha_a_x0p5`: une préparation et les deux degrés
pour chacune des résistances 0,10 et 0,22 Nm. Dossiers de résultats:
`radau-frequency-validation-20260912/50hz-x0p5-r010` et `50hz-x0p5-r022`.

| Résistance | Degré | Statut NLP | Max erreur θ / DOP853 | Max erreur ω / DOP853 | Erreur de fermeture DOP853 |
| --- | --- | --- | --- | --- | --- |
| 0,10 Nm | 3 | Infaisabilité locale, 79 itérations | non mesurée sur solution acceptée | non mesurée | non mesurée |
| 0,10 Nm | 5 | Convergence | 0,00010486 rad | 0,00060678 rad/s | 0,00191802 rad |
| 0,22 Nm | 3 | Infaisabilité locale, 73 itérations | non mesurée sur solution acceptée | non mesurée | non mesurée |
| 0,22 Nm | 5 | Convergence | 0,00086926 rad | 0,00593735 rad/s | 0,00275389 rad |

La violation maximale de continuité Radau-3 vaut respectivement 0,01297284 et
0,01297178. Aucun rejeu d'un Radau-3 accepté n'est disponible; le comparateur
ne remplace pas cette donnée absente par zéro. L'état initial est imposé par
le seed commun dans les commandes, mais la comparaison des états effectivement
exportés reste non vérifiable pour Radau-3 puisque son préfixe accepté est vide.

Radau-5 respecte le budget de phase proposé et la fermeture à 0,10 Nm, mais pas
à 0,22 Nm. Cela interdit de généraliser sa précision à partir d'un cas facile.
Ces résultats concernent des états peu fatigués: ils ne sont pas directement
comparables au cas à 30 Hz après 75 cycles. La prochaine mesure prioritaire est
le test d'intégration à PW imposées, qui permettra de chiffrer Radau-3 même
lorsqu'un OCP complet n'aboutit pas.

## Contrôles causaux après correction du seed

Le transfert des états de warm-up Radau-5 vers Radau-3 utilisait auparavant un
maillage `linspace`. C'était incorrect pour les nœuds internes Radau non
uniformes et les frontières de shooting dupliquées. Il est maintenant construit
sur les instants physiques de collocation; aux frontières dupliquées, la valeur
de shooting (limite droite) est retenue. Les tests unitaires couvrent en
particulier le transfert Radau-5 -> Radau-3.

Avec exactement le même cas à 0,10 Nm, la relance après cette correction
(`radau-frequency-validation-20260912-radau-grid-fix`) fait converger
Radau-3 : 88 itérations, violation maximale `9,45e-9`. Son rejeu DOP853 reste
toutefois hors du budget de phase : `0,00335197 rad`, contre `0,000104859 rad`
pour Radau-5. Le seed défectueux expliquait donc l'ancienne infaisabilité locale,
mais non l'erreur de transcription de Radau-3.

Deux expériences sans OCP complètent ce contrôle :

- `scripts/validate_ding_fixed_pulse_width.py` impose l'état Ding initial et
  les 50 PW à chaque muscle, avec mécanique neutre. À 50 Hz, l'erreur maximale
  de `Cn` est `2,34025e-3` pour Radau-3 et `7,96185e-7` pour Radau-5; l'erreur de
  force maximale du triceps est respectivement `0,0693 N` et `0,00168 N`.
- `scripts/validate_radau_coupled_mechanics.py` impose en plus la cinématique
  Radau-5 et réévalue les relations force-longueur, force-vitesse et passive.
  Les résultats restent du même ordre (`Cn`: `2,340e-3` vs `5,010e-6`; force
  triceps: `0,0608 N` vs `0,00151 N`). La différence est donc déjà présente
  dans l'intégration de Ding à PW fixées, et ne peut pas être attribuée au seul
  warm-start ou aux PW optimisées.

Ces contrôles restent à un cycle et à mécanique prescrite. Ils justifient
l'usage de Radau-5 pour les campagnes, et laissent à faire la convergence de
Radau-5/7 près de la fatigue ainsi que l'accumulation multi-cycle.

## Ding à PW imposées : comparaison 30 contre 50 Hz

La sonde Ding isolée a été exécutée avec les archives Radau-5 à `0,10 Nm` du
même variant musculaire. Dans chaque ligne, Radau-3 et Radau-5 partent du
**même état Ding** et appliquent les **mêmes PW** de cette fréquence; DOP853 est
la référence. Les séquences PW et les états initiaux diffèrent nécessairement
entre 30 et 50 Hz : cette table compare donc les erreurs représentatives des
deux problèmes de stimulation, et non une étude à état/PW strictement égaux
entre fréquences.

| Fréquence | Degré | Max erreur `Cn` | Max erreur `F` | Muscle pour `F` |
| --- | --- | ---: | ---: | --- |
| 30 Hz | Radau-3 | 0,0104086 | 0,649199 N | Triceps |
| 30 Hz | Radau-5 | 2,81968e-5 | 0,0459346 N | Triceps |
| 50 Hz | Radau-3 | 0,00234025 | 0,0692948 N | Triceps |
| 50 Hz | Radau-5 | 7,96185e-7 | 0,00168108 N | Triceps |

Les résidus des équations de collocation sont tous inférieurs à `1e-9`; les
écarts sont donc de la discrétisation de Ding, non des résolutions directes des
petits problèmes de collocation. Pour l'archive historique 30 Hz, les
paramètres effectifs ne sont pas présents dans le seed : le rapport cite
explicitement le fichier de configuration x0p5 utilisé, au lieu de supposer
des valeurs par défaut.

## Arbitrage Radau-3 à 50 Hz

Sur le même OCP froid à 50 Hz, IPOPT/MA57 a passé `2,59 s` dans Radau-3 contre
`4,08 s` dans Radau-5, soit environ 36 % de temps solveur en moins. Ce gain ne
suffit pas pour le benchmark : le rejeu DOP853 donne une erreur de phase de
`0,003352 rad` avec Radau-3, plus de 16 fois le budget de `0,0002 rad`; Radau-5
est à `0,000105 rad`.

Dans la sonde Ding à PW fixes, l'erreur de force triceps Radau-3 est `0,0693 N`
(`0,079 %` de la force maximale vue sur ce cycle) contre `0,00168 N` pour
Radau-5. Elle persiste à la fin du cycle (`0,0632 N`), donc elle ne se comporte
pas comme un simple bruit local qui s'annulerait avant le cycle suivant.

Raffiner Radau-3 sans modifier les 50 PW aide, mais détruit son intérêt de
vitesse : deux sous-pas par période ramènent l'erreur force à `0,00748 N`, et
trois à `0,00187 N`. Ils imposeraient respectivement 6 et 9 stades de
collocation par PW, contre 5 pour Radau-5. Cette expérience ne mesure que le
map Ding direct, pas le temps IPOPT d'un OCP raffiné; elle est donc une borne
optimiste pour Radau-3 raffiné.

Conclusion opérationnelle : Radau-3 simple peut servir à un écran exploratoire
non certifiant, mais ne constitue pas une alternative acceptable pour le RHO
qui pilote le geste ou les comparaisons d'endurance. Radau-5 reste le choix
minimal; une variante Radau-3 sous-pas n'est à envisager qu'après un benchmark
OCP chaud complet démontrant un vrai gain de latence et le respect des budgets
de phase et de force.
