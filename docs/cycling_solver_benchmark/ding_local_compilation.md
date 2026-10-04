# Compilation après réduction locale de Ding

Le blocage précédent était lié à l'ordre d'installation des adaptateurs : Bioptim
compile le NLP complet avant que l'adaptateur expérimental ne construise le NLP
réduit. Les callbacks ainsi compilés ont les mauvaises dimensions pour le
solveur réduit. Il n'existe pas d'incompatibilité mathématique entre la
substitution affine de `Cn/Tau1/Km` et la génération C des évaluateurs exacts.

## Chemin expérimental ajouté

`scripts/benchmark_ding_local_compiled.py` reprend intégralement l'adaptateur
local Radau-5 et intercepte sa fabrique **après** `build_local_map` :

1. Construire la substitution discrète exacte et le NLP réduit.
2. Préparer ses évaluateurs IPOPT natifs CasADi.
3. Compiler la Hessienne seule (`--callbacks hessian`) ou les cinq évaluateurs
   (`--callbacks all`) avec le cache de callbacks Bioptim et `-O1`.
4. Réutiliser le même solveur en actualisant les offsets de reconstruction `p`
   à chaque fenêtre RHO.
5. Reconstruire solution et multiplicateurs, puis auditer les bornes et
   contraintes dans le NLP complet, comme pour l'adaptateur interprété.

L'objectif, la transcription Radau-5, les tolérances, les contraintes et les
audits ne changent pas. La compilation n'emploie pas `fast-math`. Les offsets ne
sont jamais figés dans le code C : ils restent des entrées des callbacks. La
compilation est amortie au fil des cycles, tant que la structure du NLP reste
inchangée. Le cache est indexé par le code généré et la chaîne de compilation.

Exemple, avec les arguments du benchmark unilatéral prescrit :

```bash
"$PY" scripts/benchmark_ding_local_compiled.py \
  --callbacks hessian --cache-dir "$RUN/cache" local -- \
  [arguments du comparateur, sans --ipopt-c-compile ni --ipopt-c-compile-callback]
```

Les métriques existantes restent dans `symbolic-audit/nlp0_metrics.json`.
`symbolic-audit/nlp0_compilation.json` ajoute temps de préparation interprétée,
temps de compilation/rechargement, dimensions réduites, nombre de paramètres
runtime, noms des callbacks natifs et statut du cache.

## Validation disponible

Les tests `test_ding_local_compiled.py` comparent les valeurs des cinq callbacks,
les gradients, Jacobiennes et Hessiennes, puis des résolutions répétées avec
trois jeux d'offsets différents. La Hessienne est aussi vérifiée contre la règle
de chaîne de la projection affine du problème complet. Les deux modes de
compilation sont couverts; un second montage vérifie le réemploi du cache.
Tolérance de comparaison des callbacks : `1e-13` absolue et relative.

Les tests existants `test_ding_radau5_local_ab.py` conservent la vérification de
la reconstruction discrète Radau-5 avec états entrants arbitraires, les bornes
affines et les gardes d'infaisabilité. Ensemble : **7 tests passent** dans
`cocofest-rho32` le 4 octobre 2026.

Ces tests ne constituent pas un benchmark d'endurance ni une preuve de gain de
temps sur le modèle complet. Le A/B unilatéral prescrit doit établir le gain
net et les audits physiques; compilation à froid, solveur chaud, mapping/audit,
sortie et latence de la paire parallèle doivent être mesurés séparément.

## Limites conservées

- Le registre de capacités de production continue à refuser la combinaison.
  Ce nouveau lanceur est réservé au benchmark jusqu'à validation du vrai RHO.
- Le graphe **source** doit être SX et respecter toutes les conditions actuelles
  de `build_local_map`, dont Radau-5 et états Ding entrants fixés. Compiler le
  graphe après réduction ne supprime pas ces conditions.
- Le chemin dépend du cache de callbacks de la révision locale Bioptim
  (`bioptim.interfaces.c_compile_cache.cached_nlpsol_callbacks`).
- La reconstruction, la factorisation des offsets, le transport des duaux et
  l'audit complet restent interprétés. `all` désigne les cinq callbacks IPOPT,
  pas l'intégralité du contrôleur Python.
- La cinématique prescrite est compatible en principe : elle doit avoir été
  éliminée avant la construction du NLP passé à l'adaptateur. La validation
  effective de cette combinaison appartient au nouveau benchmark prescrit.
- Le minimum local et le nombre d'itérations peuvent varier au niveau des
  arrondis. Rapporter les certificats et la faisabilité originale reste
  indispensable; aucun gain ne se déduit du seul nombre de variables.
