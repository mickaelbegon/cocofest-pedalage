# Microbenchmark préenregistré de la projection RHO-Réserve

Ce benchmark autonome évalue une **brique de calcul**, sans résoudre de RHO ni
prétendre mesurer une capacité mécanique certifiée. Le contrat fixe un profil
de force candidat de huit phases, répété pendant 1, 20 ou 100 cycles pour
quatre muscles Ding synthétiques connus. Les phases ont une force constante
dans chaque intervalle; deux cas sont testés : force identique à toutes les
phases et force variable par phase. Les états initiaux, repos, coefficients et
gains sont déterministes et codés dans le script. Les trois états lents
`A, Tau1, Km` suivent

\[
\dot z=-(z-z_{rest})/\tau_{fat}+\alpha F.
\]

La carte analytique exacte pour chaque intervalle est comparée à DOP853
(`rtol=1e-11`, `atol=1e-13`), redémarré à chaque discontinuité de force. Pour
une phase et un muscle, la marge **locale affine** est

\[
m=m_{ref}+\sum_j g_j (z_j-z_{rest,j})/z_{rest,j}.
\]

Les gains sont signés : une baisse de `A` pénalise certains muscles, alors
qu'elle peut améliorer la marge locale attribuée à un muscle antagoniste.
Les signes de `Tau1` et `Km` varient également. Ces gains sont des données
du substitut local; ils ne transforment pas la force Ding en enveloppe de
couple physiquement atteignable. Avant un raccordement réel, `m_ref` et `g`
devront provenir d'une linéarisation mécanique validée sur des solutions RHO.

## Score et contrôle négatif

Pour toutes les marges futures `m_i`, le score utilisé pour le risque est
`-T log(sum_i exp(-m_i/T))`, avec `T=0,001`. Il est inférieur ou égal au
minimum brut : une marge négative ne peut donc pas être déclarée positive.
Son conservatisme est borné par `T log(n)`, exporté avec le minimum brut.

Le JSON conserve séparément l'ancien minimum lissé **normalisé**
`-T log(mean_i exp(-m_i/T))` à `T=0,01`. Il est optimiste. Dans les deux cas
à 100 cycles, le minimum brut vaut environ `-0,014`, tandis que ce diagnostic
normalisé vaut environ `+0,0156`. C'est un contrôle négatif explicite : ce
score ne doit pas servir de signe de sécurité. Le score conservateur vaut
environ `-0,0159` dans les mêmes cas.

## Portes numériques fixées avant intégration

Le script encode les seuils dans `GATES` et rend un statut non nul si l'un
d'eux échoue :

| Contrôle | Seuil |
| --- | ---: |
| Écart maximal d'état, divisé par l'état de repos correspondant | `2e-9` |
| Écart maximal de marge locale | `2e-9` |
| Écart absolu du score conservateur | `2e-9` |
| Gradient du score par rapport aux 12 états lents initiaux, écart absolu | `2e-7` |
| Même gradient, écart relatif avec plancher `1e-6` | `2e-5` |
| Coût P90 analytique H=100, projection et score | `≤20 ms` |
| Accélération H=100 par rapport à DOP853 | `≥5×` |
| Score de risque positif avec minimum brut négatif | **interdit** |

Le gradient analytique est contrôlé par différences finies centrales. Les
mesures de temps sont des mesures par appel, médiane de 31 appels analytiques;
la référence DOP853 est exécutée une fois par cas par défaut. Ce gate de temps
sert au filtrage sur une machine partagée, pas à prédire le temps total IPOPT.
Le JSON inclut aussi le coût de projection avec gradient et les P90
analytiques. Les nombres peuvent bouger avec la charge du calculateur.

Une intégration dans RHO demandera **en plus** ces contrôles préalables, sur
des profils RHO gelés et des états terminaux réels : (1) linéarisation
`m_ref,g` vérifiée sur des perturbations tenues à l'écart, avec erreur absolue
de marge au plus `0,005` et aucun faux signe de sécurité lorsque la vraie
marge est inférieure à `-0,005`; (2) gradient du chemin CasADi/NLP contre
différences finies avec les mêmes seuils ci-dessus; (3) temps P90 complet de
mise à jour du terme au plus `20 ms` à H=100 sur le matériel cible;
(4) pilote RHO apparié, mêmes états et contraintes, sans perte de certificat
ni dégradation des résidus mécaniques. Ce microbenchmark ne valide ni cette
linéarisation, ni les PW futures, ni l'endurance. Une revendication de gain
d'endurance exigera ensuite une campagne prospective RHO séparée.

## Reproduction

Depuis la racine du dépôt, avec un Python contenant NumPy, SciPy et pytest :

```bash
python scripts/benchmark_mechanical_reserve_projection.py \
  --output /tmp/mechanical_reserve_projection_benchmark.json
python -m pytest -q tests/test_benchmark_mechanical_reserve_projection.py
```

Le JSON est également imprimé sur stdout. `--horizons`, `--analytic-repeats`
et `--reference-repeats` permettent une vérification rapide, mais les seuils
et la campagne de référence portent sur les valeurs par défaut.

Sur `cocofest-rho32` (Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1), les six
cas ont passé les portes numériques; le plus grand écart d'état normalisé
est `6,49e-14`. À H=100, projection et score analytiques ont pris environ
`4,6–7,7 ms` par appel médian dans cette exécution, contre `250–293 ms` pour DOP853.
Les 11 tests ciblés passent. Le diagnostic normalisé à T=0,01 échoue au
contrôle de signe attendu sur les deux cas H=100; il reste exclu du score
utilisé pour le risque.
