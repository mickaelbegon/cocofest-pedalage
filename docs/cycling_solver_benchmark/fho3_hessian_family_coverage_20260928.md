# FHO3 : couverture structurelle de la Hessienne par familles — 28 septembre 2026

## Question

Avant d'investir dans une callback Hessienne en paquets, il faut savoir si les
termes répétés représentent une fraction significative de la Hessienne exacte
réellement donnée à IPOPT. Cette mesure répond uniquement à cette question de
**couverture structurelle**. Elle ne résout aucun OCP, ne remplace aucune
callback et ne constitue donc pas encore un benchmark de temps de solve.

## Protocole reproductible

Le script
[`audit_fho_hessian_family_coverage.py`](../../scripts/audit_fho_hessian_family_coverage.py)
rejoue la commande historique FHO3 jusqu'à la construction du NLP MX, intercepte
`nlp_hess_l`, et arrête le chemin avant le premier appel IPOPT. Il utilise la
branche isolée Bioptim `codex/postshake-penalty-registry` uniquement pour
inspecter les contributions. Les paires `(i,j)` sont canonisées, de sorte que
le choix CasADi de triangle supérieur ou inférieur ne modifie pas les comptes.

```bash
MPLCONFIGDIR=/tmp/mpl-hessian-audit OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/audit_fho_hessian_family_coverage.py \
  --command local-results/fho3-postshake-registry-benchmark-20260927-v10/report.json \
  --output local-results/fho3-hessian-family-coverage-20260928-v5 \
  --hsl-library /path/to/libhsl.so
```

La sortie brute est
[`report.json`](../../local-results/fho3-hessian-family-coverage-20260928-v5/report.json).
Le FHO de référence est le cas MX non compilé à 0,30 N·m, avec 12 263 variables
et 11 911 contraintes.

## Résultat

La Hessienne native triangulaire de `nlp_hess_l` contient **49 077** paires
structurelles. Les familles suivantes sont disjointes dans ce cas :

| Famille de provenance | Termes | Paires locales brutes | Paires dans `nlp_hess_l` | Couverture |
|---|---:|---:|---:|---:|
| `STATE_CONTINUITY` (`ThreadMap`) | 91 | 97 020 | 48 510 | **98,845 %** |
| objectif `minimize_overall_muscle_fatigue` | 91 | 364 | 364 | 0,742 % |
| contraintes restantes | 31 | 203 | 203 | 0,414 % |
| union du registre | 213 | — | 49 077 | **100,000 %** |

Les 97 020 paires brutes de continuité deviennent 48 510 après coalescence des
contributions entre intervalles : les paquets se recouvrent aux frontières. Ce
n'est pas une perte de couverture, mais une propriété que la future callback
doit reproduire par *scatter-add* exact.

La famille dominante est appelée `STATE_CONTINUITY` par Bioptim. Dans cette
formulation Radau, elle porte la propagation/collocation entre noeuds ; elle ne
doit donc pas être confondue avec une simple petite contrainte de raccord. Le
registre post-*shake* ne fournit pas encore les fragments par intervalle à coût
acceptable, mais ce résultat identifie sans ambiguïté la famille à préserver
**avant** son agrégation `ThreadMap`.

## Coût de l'instrumentation actuelle

La matérialisation exhaustive post-*shake* a pris **82,50 s** et a atteint
environ **9,8 Gio** de mémoire. C'est délibérément un diagnostic, pas une voie
de production. Elle confirme qu'extraire les localisations depuis le grand
graphe MX après *shake* est un non-sens opérationnel, même si les fonctions
ainsi obtenues sont exactes.

## Décision

**GO, mais seulement pour un registre pré-*shake* de `STATE_CONTINUITY`.** Le
plafond de couverture est 98,845 %, suffisamment élevé pour justifier un
prototype de préparation et d'évaluation. Le résultat négatif précédent sur
la contrainte de vitesse (0,709 %) ne s'applique pas à cette famille dominante.

**NO-GO** pour toute callback IPOPT à ce stade. Il manque encore :

1. la conservation, pendant `generic_get_all_penalties`, de chaque fragment
   de continuité avant `sum2`/agrégation `ThreadMap`, avec ses indices globaux
   de décision et ses lignes `g` ;
2. le regroupement par signature locale et un noyau exact (valeur, Jacobienne,
   Hessienne du Lagrangien) par signature ;
3. un scatter-add qui reproduit les 48 510 paires, y compris les recouvrements
   de frontière, à erreur <= 1e-10 sur `g`, `J` et `H` ;
4. un benchmark isolé qui compare la préparation, l'évaluation `nlp_hess_l`,
   l'assemblage et le nombre d'itérations IPOPT à la référence native.

À titre de borne purement théorique, si ces 98,845 % étaient accélérés d'un
facteur `s` sans coût d'assemblage, la borne d'Amdahl serait
`1 / (0,01155 + 0,98845 / s)`: ×1,98 pour `s=2`, ×2,96 pour `s=3,7`. Ce ne sont
pas des prédictions : le `ThreadMap` natif est déjà parallélisé et le
scatter-add peut annuler une grande part du gain. Le prochain test doit donc
mesurer le groupe dominant, jamais extrapoler le microbenchmark de la petite
contrainte de vitesse.
