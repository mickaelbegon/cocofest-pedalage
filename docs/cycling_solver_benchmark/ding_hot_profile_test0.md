# Test 0 — profil chaud IPOPT/MA57, reduced, Radau-5

Analyse du 13 septembre 2026 des mesures natives archivées le 12 septembre.
Le plan externe `PLAN_TEST_mediane_chaude_Ding.md` propose une hypothèse utile
à tester, mais **son hypothèse de fenêtres chaudes à dix itérations ou moins
est infirmée** : la référence demande 43 à 68 itérations, moyenne 52,78.
Les évaluations NLP occupent 55,72 % du temps solveur chaud ; les deux leviers,
coût des évaluations et nombre d'itérations, restent donc ouverts.

## Contrat et méthode

On réanalyse les neuf fenêtres 1 à 9 de l'[A/B symbolique](ding_radau5_symbolic_ab.md),
après exclusion de la fenêtre 0. Le modèle et les trajectoires ne sont pas
relancés ni modifiés : quatre muscles, 22 états, 30 intervalles, Radau cinq
stages, SX interprété, IPOPT/MA57, couple +0,1 Nm, un thread, scaling `full`,
tolérance NLP `1e-6`, gate primal `1e-5`. Les bornes de vitesse aux stages
internes sont désactivées dans cet ancien protocole. Ce point limite sa
représentativité pour des configurations actuelles plus contraintes.

Les compteurs viennent de `symbolic-audit/nlp0_metrics.json` dans
`ding-radau5-symbolic-ab-20260912/{baseline10,condensed10}`. Pour chaque
fenêtre, on somme tous les `t_wall_nlp_*`, puis on calcule
`t_wall_total - somme`. Le même calcul est effectué avec les temps CPU
`t_proc_*`. Aucun timer NLP n'est ajouté à un autre timer NLP englobant : le
compteur `nlp_grad`, en particulier, vaut zéro dans ces données. Les temps
excluent la construction du solveur et les audits Python extérieurs.

La différence entre temps total et évaluations est appelée **hors évaluations**.
Elle englobe MA57, assemblage, recherche linéaire, gestion de barrière et autres
coûts IPOPT/CasADi non comptés dans les callbacks. Elle ne constitue pas une
mesure directe du temps de factorisation MA57. Les lignes CPU détaillées
IPOPT ne sont pas présentes dans ce log silencieux ; les compteurs natifs
archivés donnent ici une ventilation instrumentée de substitution.

## Mesures

| Mesure chaude | Référence | Condensé maximal |
|---|---:|---:|
| Variables / contraintes | 4 103 / 3 960 | 1 703 / 3 960 |
| Itérations, fenêtres 1–9 | 68, 54, 56, 54, 52, 54, 49, 43, 45 | 72, 59, 53, 51, 58, 57, 52, 55, 44 |
| Itérations moyennes | 52,78 | 55,67 |
| Temps solveur médian | 0,947445 s | 1,022261 s |
| Temps cumulé | 8,385713 s | 9,163801 s |
| Évaluations NLP cumulées | 4,672323 s | 5,915006 s |
| Hors évaluations cumulé | 3,713390 s | 3,248795 s |
| Fraction évaluations NLP | 55,72 % | 64,55 % |
| Coût/itération, temps cumulé / itérations cumulées | 17,654 ms | 18,291 ms |

Sur la référence, la Hessienne compte pour **33,77 %** du temps total, la
Jacobienne **14,09 %**, les contraintes **7,65 %**, l'objectif et son gradient
**0,21 %**, le reste hors évaluations **44,28 %**. La fraction CPU évaluations
est 55,7204 %, très proche des 55,7177 % muraux : la conclusion n'est pas due
à un écart important entre compteurs CPU et muraux dans cette archive.

La fenêtre médiane est la fenêtre 6 : 54 itérations, 0,947445 s, dont
0,535357 s d'évaluations (56,51 %) et 0,412087 s hors évaluations.

Le nombre de non-zéros Hessienne/Jacobienne et le remplissage des facteurs MA57
ne sont pas archivés dans cet A/B. Ils ne sont pas déduits artificiellement
des dimensions du NLP ou du nombre de stages. Une capture symbolique fournit
les deux premiers ; le remplissage MA57 exige une instrumentation distincte.

## Ce que cela change dans l'interprétation du plan

La condensation ajoute **5,47 % d'itérations chaudes** et **3,61 % de coût
moyen par itération**. Leur produit explique exactement les **9,28 %** de
temps chaud cumulé supplémentaire. La médiane augmente de 7,90 %, mais les
ratios de médianes ne se factorisent pas comme les sommes. Le ralentissement
chaud ne s'explique donc pas uniquement par une Hessienne plus chère diluée
dans un coût d'itération inchangé. Le travail hors évaluations diminue même
dans cette variante, tandis que les évaluations augmentent.

Les différences de trajectoire sur les neuf fenêtres empêchent de considérer
cette ventilation comme une ablation à problèmes chauds identiques. Elle
suffit en revanche à réfuter l'hypothèse « peu d'itérations, donc ne plus
travailler le warm-start ». Elle ne prouve ni que les 53 itérations peuvent
être fortement réduites, ni qu'un meilleur warm-start donnera un gain.

À itérations et autres coûts fixes, rendre **toutes** les évaluations quatre
fois plus rapides donnerait théoriquement ×1,72 sur le temps total cumulé ;
les rendre gratuites plafonne à ×2,26. Ce sont des bornes arithmétiques du
profil actuel, pas une prédiction de gain mesuré ni de médiane future.

## Priorités tenant compte des expériences antérieures

1. Garder Radau-5 pour les essais à problème scientifique inchangé. R3 change
   la précision ; le [bilan existant](README.md) rapporte déjà une erreur du
   calcium périodique isolé de 6,3864 % en R3 contre 0,0173 % en R5. Le facteur
   ×2–2,5 annoncé dans le plan n'est pas démontré pour notre NLP. Le contrôle
   de convergence et une cible de précision explicite doivent précéder son
   éventuelle adoption.
2. Cibler les Hessiennes/Jacobiennes ou leur structure de calcul. Le coût
   d'évaluation constitue un levier mesuré, mais le C monolithique a déjà
   échoué à terminer dans les budgets de compilation : [112,6 MB, arrêt à
   300 s, environ 17 GiB RSS](../../casadi38-transform-rho-pilot-20260912/C_REPORT.md).
   La compilation séparée de `f/g/grad_f` et son cache sont déjà validés sur
   de vrais RHO, sans accélération globale significative dans ce petit pilote
   ([rapport](../../casadi38-transform-rho-pilot-20260912/SPLIT_REPORT.md)).
   Un nouvel essai doit donc viser des blocs de dérivées bornés, avec contrôle
   de compilation, ou la reconstruction locale qui préserve la creusité.
3. Le transform CasADi 3.8 réduit déjà le coût des dérivées, mais son pilote
   50 stimulations ne montre qu'environ −4,5 % sur la médiane solveur et
   aucun gain démontré de latence applicative
   ([rapport](../../casadi38-transform-rho-pilot-20260912/REPORT.md)). Ces
   chiffres proviennent d'un autre contrat et ne doivent pas être fusionnés
   avec l'A/B 30 intervalles ci-dessus.
4. Conserver un axe réduction d'itérations : les expériences pertinentes sont
   des rejeux de fenêtres figées avec les mêmes primales, duales, bornes et
   tolérances, puis un A/B séquentiel. Aucun gain de warm-start IPOPT n'est
   attribuable au test FATROP précédent.

La reconstruction locale `Tau1,Km` depuis `A` reste une expérience distincte
de la condensation maximale. Son intérêt serait de réduire les évaluations
et la taille des blocs sans dépendances entre forces de stages. L'exactitude
continue d'un offset exponentiel ne garantit cependant pas l'identité du NLP
Radau discret ; cet aspect doit être vérifié par l'expérience dédiée.

## Reproduction

```bash
python scripts/analyze_ding_hot_profile.py \
  --output ding-hot-profile-test0-20260913/profile.json
```

Le [script](../../scripts/analyze_ding_hot_profile.py) utilise uniquement la
bibliothèque standard, vérifie la cohérence des compteurs et des statuts,
exporte chaque fenêtre et les empreintes SHA256 des métriques sources. Les
[résultats JSON](../../ding-hot-profile-test0-20260913/profile.json) permettent
de recalculer tous les ratios. Matériel constaté lors de la réanalyse : AMD
Ryzen 9 5950X, 16 cœurs / 32 threads ; la réanalyse n'établit pas rétroactivement
la charge machine ou l'affinité CPU lors des solves archivés.
