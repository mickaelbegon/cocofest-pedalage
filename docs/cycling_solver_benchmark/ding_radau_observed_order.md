# Ordre observé Radau à PW fixées — 13 septembre 2026

Le test 1 du `PLAN_TEST_mediane_chaude_Ding.md` a été exécuté sur les paramètres
et les PW d'une archive locale reduced à 30 Hz. La convention impulsion/étage
terminal est cohérente. La diminution Radau-5 → Radau-3 accroît bien l'erreur
de force d'environ un ordre de grandeur. Elle ne devient pas une accélération
acceptable par défaut : les anciens essais mécaniques à 50 Hz échouaient déjà
au budget de phase avec Radau-3.

## Convention effectivement implémentée

Dans `ding2007_with_fatigue_periodic_node.py`, `calcium_history` reçoit
l'amplitude après stimulation et le début de l'intervalle dans les données
numériques. Pendant tout l'élément, elle évalue

\[
H(t)=H_k\exp(-(t-t_k)/\tau_c).
\]

Dans la dépendance Bioptim locale, `OdeSolver.COLLOCATION.d_ode` fournit
`nlp.numerical_timeseries.cx_start`. La boucle des stages de
`integrator.COLLOCATION.dxdt` transmet cette même donnée, y compris au stage
terminal de Radau. Celui-ci voit donc `H_k exp(-h/τc)` ; le nouvel intervalle
voit `H_{k+1}`. `Cn` reste continu, son forçage change à la frontière.
Il n'y a pas d'activation anticipée de la stimulation suivante dans ce chemin.

Le modèle impose actuellement un intervalle de shooting par stimulation.
Le raffinement présenté ici est une sonde d'intégration autonome, pas une
modification de ce contrat ni une nouvelle discrétisation OCP.

Attention à l'annexe du document externe : sa branche `consistent=False`
construit l'ensemble des impulsions actives avec la fin de l'élément, puis
l'utilise à **tous** ses stages. Elle ne représente donc pas exclusivement
l'erreur décrite dans le texte (« seul le stage terminal voit la prochaine
impulsion »). Les chiffres de cette branche ne doivent pas être attribués
à notre implémentation.

## Protocole et contrôles

`scripts/validate_ding_radau_order.py` reprend les fonctions du validateur
`validate_ding_fixed_pulse_width.py`. Il conserve l'état initial, les 30 PW
de chaque muscle et les paramètres déclarés du cas
`repository_current--ofat--triceps--alpha_a--x0p5`. La mécanique est neutralisée.
Chaque intervalle de stimulation est subdivisé en 1, 2, 4, 8 ou 16 éléments,
sans ajouter d'impulsion ou de décision PW.

La référence DOP853 intègre séparément chaque intervalle avec `rtol=1e-13`,
`atol=1e-15`. Sa comparaison à `1e-12/1e-14` donne au maximum `1,12e-10 N`
d'écart de force sur les quatre muscles. Les équations Radau sont résolues
dans la forme Butcher, équivalente à la collocation directe, avec un résidu
normalisé par état. Cela évite d'amplifier les erreurs d'arrondi par la
division par le très petit pas. Les tests vérifient l'équivalence avec le
validateur existant et la solution calcique exacte après subdivisions.

```bash
source .github/scripts/benchmark_env.sh rho32
python scripts/validate_ding_radau_order.py \
  --common-seed resistance-fho-pilots-20260910/seed-0p10/common-reduced.npz \
  --model-config examples/fes_multibody/cycling/two_model_ding_campaign/ding_triceps_alpha_a_x0p5.json \
  --output-json ding-hot-median-plan-20260913/radau-order-30hz.json
python -m pytest tests/test_ding_radau_order.py -q
```

Les sorties existent déjà ; utiliser un nouveau nom pour une répétition.

## Mesures triceps

L'ordre est `log2(e_h/e_{h/2})`, pour l'erreur relative de force en fin de
cycle. L'erreur maximale est mesurée aux frontières de stimulation pendant
le cycle, sans réinitialiser les états entre intervalles.

| Degré | Éléments / stimulation | Erreur relative F(T) | Ordre observé | Max erreur F (N) |
| --- | ---: | ---: | ---: | ---: |
| 3 | 1 | 7,081e-3 | — | 0,6492 |
| 3 | 2 | 1,091e-3 | 2,70 | 0,08397 |
| 3 | 4 | 2,284e-4 | 2,26 | 0,01711 |
| 3 | 8 | 2,479e-5 | 3,20 | 0,001892 |
| 3 | 16 | 1,536e-6 | 4,01 | 0,0001197 |
| 5 | 1 | 6,251e-4 | — | 0,04593 |
| 5 | 2 | 6,565e-5 | 3,25 | 0,004984 |
| 5 | 4 | 3,275e-6 | 4,32 | 0,0002596 |
| 5 | 8 | 6,584e-8 | 5,64 | 0,000005472 |
| 5 | 16 | 5,407e-10 | 6,93 | 0,00000004688 |

Les autres muscles présentent la même montée d'ordre. À un élément par
impulsion, l'erreur maximale calcique est `0,0104086` pour Radau-3 et
`2,81968e-5` pour Radau-5. On ne voit pas le plateau d'ordre un suspecté
dans le document. La force converge plus lentement vers l'ordre asymptotique
que le calcium : le pas de départ `h/τc ≈ 3,03` reste grossier vis-à-vis
des transitoires non linéaires.

## Décision et critères restants

Radau-3 donne ici environ 0,7 % d'erreur **terminale de force isolée**.
Ce nombre ne certifie pas une tolérance physiologique de 1 % : les erreurs
de couple et de phase dépendent de tous les muscles et de la mécanique.
L'étude précédente à 50 Hz mesurait `0,003352 rad` d'erreur de phase sous
Radau-3 contre `0,000104859 rad` sous Radau-5. Le budget de transcription
proposé dans le dépôt est `0,0002 rad`, soit 10 % de la tolérance de tâche
de `0,002 rad`. Cette preuve existante interdit de retenir Radau-3 comme
accélération générale avant de nouvelles validations couplées.

À 30 Hz, Radau-5 à un élément domine aussi Radau-3 à deux éléments dans cette
sonde : cinq stages au lieu de six et une erreur force inférieure. Cela ne
prouve pas que tout raffinement est inutile : près de la fatigue, Radau-5
peut lui-même exiger une précision supérieure. Le cas historique après
75 cycles avait une erreur de phase d'environ `0,02964 rad` à 30 Hz.

Avant de réduire le degré, il faudrait démontrer, à plusieurs niveaux de
fatigue et plusieurs PW/résistances, le respect du budget de phase, l'écart
de couple, les erreurs absolue et normalisée de force/capacité, les bornes
physiologiques entre nœuds et l'accumulation sur plusieurs cycles. Les seuils
physiologiques de force/capacité restent à définir avec la tâche scientifique.
Le test présent confirme le besoin de ces critères ; il ne les invente pas.

Artefact complet : `ding-hot-median-plan-20260913/radau-order-30hz.json`.
Deux tests unitaires passent. Aucun solveur ni modèle de production modifié.
