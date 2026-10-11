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

## Benchmark réel prescrit du 4 octobre 2026

Cas unilatéral à quatre muscles, fatigue unitaire, vitesse prescrite
`−2π rad/s`, couple équivalent `0.2 Nm`, assistance/résistance bornée à
`[−3,3] Nm`, 30 Hz, Radau-5, SX, IPOPT/MA57, tolérance NLP `1e-6`, seuil
de faisabilité `1e-5`, sortie compacte et un thread numérique. Les appels
complets proviennent du baseline de `paired100` et sont gelés à l'identique.
La Hessienne seule est compilée; le problème réduit comprend 1750 variables,
1651 contraintes et 2172 offsets runtime.

Le smoke compilé valide 3/3 fenêtres, puis le benchmark valide 100/100 NLP.
Sur les 100 appels, les sorties candidates `x`, `g`, multiplicateurs et
objectifs sont **strictement identiques** aux sorties locales interprétées
archivées; tous les nombres d'itérations sont identiques. Le certificat maximal
vaut `8.00e-15` et la violation maximale du NLP original `7.39e-7`.

| Mesure chaude (première fenêtre exclue), CPU 12 | Local interprété | Local + Hessienne C |
|---|---:|---:|
| Médiane IPOPT | 0.08288 s | 0.07460 s |
| p90 IPOPT | 0.09336 s | 0.08508 s |
| Médiane itération complète du banc d'essai | 0.17636 s | 0.17076 s |
| p90 itération complète | 0.18643 s | 0.18074 s |
| Médiane cumul évaluations Hessienne/appel | 0.008388 s | 0.000450 s |
| Médiane itérations IPOPT | 36 | 36 |

Le gain marginal de compilation est donc modeste : `−10.0 %` au solveur,
`−3.2 %` sur l'itération complète de ce banc d'essai, après le gain beaucoup
plus important apporté par la réduction elle-même. La compilation à froid et
le rechargement demandent 4.50 s; un réemploi du cache demande 0.116 s, en plus
de la préparation interprétée (0.200 s). Sur seulement 100 cycles, compiler à
froid n'est pas amorti par les ~0.8 s gagnées au solveur. Un cache déjà préparé
change ce bilan.

Les deux runs du tableau sont épinglés au CPU 12. Le premier run interprété
`paired100/local`, dont l'affinité n'était pas enregistrée dans ce rapport,
donnait 0.07850 s au solveur et 0.17463 s à l'itération : sa comparaison au run
compilé donne plutôt `−5.0 %` et `−2.2 %`. La variabilité inter-run est donc de
plusieurs millisecondes. Il faut distinguer la réduction, dont le bénéfice est
net, de l'ampleur exacte du petit gain supplémentaire de compilation. Toutes
les solutions candidates des deux références interprétées restent strictement
identiques à celles du compilé.

Artefacts :

- `local-results/prescribed-ding-local-ab-20261004/local-hessian-smoke-v2` ;
- `local-results/prescribed-ding-local-ab-20261004/local-hessian-100` ;
- `local-results/prescribed-ding-local-ab-20261004/local-hessian-comparison.json` ;
- `local-results/prescribed-ding-local-ab-20261004/local-hessian-comparison-cpu12.json` ;
- `local-results/prescribed-ding-local-ab-20261004/local-interpreted-cpu12-100`.

Le temps d'itération rapporté inclut les opérations du banc d'essai, dont les
audits et le rejeu. Le contrôleur extérieur reçoit la trajectoire baseline
enregistrée : ce résultat valide la résolution numérique gelée, pas encore
l'endurance d'un RHO compilé propageant ses propres solutions. Le registre de
capacités de production reste inchangé.
