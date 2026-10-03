# Propagation lente de Ding par convolution exponentielle — 2 octobre 2026

## Résultat mesuré

La propagation des trois états lents `A`, `Tau1` et `Km` peut être faite avec un
coût négligeable à partir du profil de force d'un cycle. Sur 29 cycles consécutifs
certifiés, les états finaux sont reproduits à environ `10^-12` relatif aux valeurs
de repos. Le coût médian de construction et application de la carte est de
**0,132 ms par muscle et par cycle**, soit environ **0,53 ms pour quatre muscles**
sur cette machine et ce passage. Ce temps exclut lecture JSON/NPZ et construction
du profil de force futur.

Cette mesure valide un **rejeu conditionnel aux forces observées**. Elle ne prouve
pas que ces forces pourront être produites dans un futur cycle, et ne prédit pas
encore l'endurance. Elle ne constitue pas une validation indépendante de la
dynamique complète, du calcium ou des forces entre nœuds.

## Méthode

Pour chaque muscle, les trois équations lentes du modèle actuellement implémenté
ont la même constante de récupération `tau_fat` et le même forçage `F` :

`z_dot = -(z - z_rest) / tau_fat + alpha * F`, avec `z = (A, Tau1, Km)`.

La carte d'un cycle de durée `T` est donc

`z_next = z_rest + exp(-T/tau_fat) * (z - z_rest) + alpha * I_F`,

où `I_F` est l'intégrale de `F(t)` pondérée par `exp(-(T-t)/tau_fat)`.
La force est reconstruite intervalle par intervalle avec le polynôme porté par
le nœud initial et les cinq nœuds Radau archivés. L'intégrale exponentielle de ce
polynôme est calculée à partir de moments analytiques stables pour `dt/tau << 1`.
Les cartes des 30 intervalles sont composées. L'intégration est exacte pour ce
polynôme, à l'arrondi près; cette qualification ne concerne pas une force
continue non observée entre les nœuds.

Le module conserve les trois états initiaux. En particulier, les deux écarts

`Tau1 - Tau1_rest - (alpha_tau1/alpha_a)*(A-A_rest)` et
`Km - Km_rest - (alpha_km/alpha_a)*(A-A_rest)`

suivent une récupération exponentielle homogène. Ils ne doivent être forcés à
zéro que si l'état initial le justifie. Ceci est important pour les états
perturbés dans un calcul de gradient : réduire `Tau1` et `Km` à des fonctions de
`A` sans transporter ces écarts changerait la dynamique initialisée.

## Protocole et portée

- Source : `task-load-margin-c140-left-intervention-until-stop-w100-20261001`,
  bras gauche, cycles **141 à 169**, quatre muscles, 30 Hz, 1 s/cycle, Radau-5.
- Seuls les fichiers `witness-cycle-N.npz` dont la ligne du résumé est
  explicitement `certified: true` sont utilisés. Les exports de primal décalé
  `prepared` ne sont pas interprétés comme des trajectoires résolues.
- Les paramètres complets proviennent du résultat archivé PACE-RT du même modèle
  gauche; le chemin, l'empreinte du modèle et ses substitutions de paramètres
  sont vérifiés contre le protocole des témoins. Les empreintes des témoins sont
  conservées dans le JSON de diagnostic.
- Le rejeu est évalué en repartant de l'état observé à chaque cycle et en
  enchaînant les 29 cycles sans remise à l'état observé. La continuité des états
  observés entre fichiers est exactement satisfaite dans cette archive.
- Aucun OCP n'a été relancé. Aucun résultat source n'a été modifié.

Les erreurs ci-dessous sont normalisées par la valeur de repos propre à chaque
état, et non par une variation de fatigue parfois presque nulle.

| Erreur absolue maximale relative au repos | A | Tau1 | Km |
|---|---:|---:|---:|
| Convolution, un cycle depuis l'état observé | 1,28 × 10^-14 | 6,60 × 10^-14 | 3,44 × 10^-14 |
| Convolution, 29 cycles enchaînés | 2,73 × 10^-13 | 1,31 × 10^-12 | 7,01 × 10^-13 |
| Force moyenne, 29 cycles enchaînés | 1,78 × 10^-4 | 8,00 × 10^-4 | 3,20 × 10^-4 |

L'ablation « force moyenne » utilise exactement la même intégrale ordinaire du
polynôme de force, puis la remplace par une force constante sur le cycle entier.
Les différences cumulées restent modestes dans ce cas : **0,0178 %**, **0,0800 %**
et **0,0320 %** du repos. La convolution corrige donc une approximation réelle,
mais ces mesures ne permettent pas de lui attribuer les grands écarts
d'endurance déjà observés entre fonctions coût.

## Validation indépendante du calcul

Les tests couvrent les moments exponentiels contre une quadrature indépendante,
une force constante et une force polynomiale contre DOP853 à tolérance serrée,
des durées d'intervalles différentes, la sensibilité à la position temporelle
de la force à moyenne identique, les écarts initiaux couplés non nuls, la formule
directe de répétition sur 500 cycles, ainsi que des entrées invalides.

Commande :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python -m pytest -q tests/test_ding_slow_cycle.py
```

Résultat : **15 tests passés**. La vérification jointe avec le module historique
`tests/test_ding_fatigue_rollout.py` donne **31 tests passés**, avec deux
avertissements de dépréciation SWIG sans rapport avec ce calcul.

## Réutilisation et prochaine décision scientifique

Le module `cocofest/optimization/ding_slow_cycle.py` fournit une carte affine
indépendante du solveur. Répéter un profil constant pendant 500 cycles se calcule
en temps constant par somme géométrique. Cette répétition est une hypothèse de
force prescrite, pas une affirmation de faisabilité future.

La suite pertinente est de brancher cette carte à un modèle de capacité de
travail qui tienne compte des PW admissibles et des autres états de Ding, puis
de tester les gradients à checkpoints gelés. La propagation lente est validée
conditionnellement dans ce jeu de données; la génération des forces futures et
la valeur mécanique terminale restent les hypothèses à réfuter.

Les journaux BO et PACE-RT usuels contiennent surtout des états terminaux et des
PW, sans traces de forces intra-cycle complètes. On ne peut donc pas annoncer
une validation sur leurs 169–234 cycles à partir de ces seuls journaux. Une
campagne instrumentée devra enregistrer les forces ou leurs intégrales
exponentielles pour élargir la couverture au début et au milieu de l'exercice.

Diagnostic reproductible :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python scripts/validate_ding_slow_cycle_archives.py \
  --witness-directory asymmetric-sides-r192-rho-bo-20260928/task-load-margin-c140-left-intervention-until-stop-w100-20261001 \
  --model-result asymmetric-sides-r192-rho-bo-20260928/pace-rt-refresh-40-20261002/persistent-normalized-endurance-200-interactive-results/left/result.json \
  --output-directory endurance-costate-diagnostics-20261002/slow-ding-witness-c141-169-r2
```

Le fichier `report.json` contient les résultats par muscle et cycle, les états
observés/prédits, les erreurs, les intégrales et la provenance. Le script exige
un nouveau dossier de sortie pour protéger les diagnostics précédents.
