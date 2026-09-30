# FHO3 — noyau local de Hessienne de continuité compilé, 28 septembre 2026

## Question

Peut-on accélérer exactement le terme dominant de la Hessienne FHO tout en
laissant le graphe global IPOPT/FHO en MX et non compilé ?

Le test utilise FHO3 (12 263 variables, 11 911 contraintes) et les 90
fragments `STATE_CONTINUITY` capturés avant `Function.map`. Chaque fragment a
158 variables locales, 132 lignes de contrainte et 1 004 non-zéros de
Hessienne locale. La valeur compilée est comparée bit-à-bit (au sens double)
à la fonction CasADi VM au vecteur initial physique.

## Résultat

Le noyau exact local `lagrangian_hessian(local_x, local_lambda)` a été généré
en C puis chargé par `casadi.external`; seul le noyau local est natif, pas le
FHO global ni un callback IPOPT.

| Évaluateur des 90 fragments | VM (s) | C `-O0` (s) | gain | erreur infinie |
|---|---:|---:|---:|---:|
| `map(..., "serial", 1)` | 1,0557 | 0,1879 | 5,62x | 0 |
| `map(..., "thread", 12)` | 0,2439 | 0,04227 | 5,77x | 0 |

Le précédent `scatter-add` Python mesurait 0,038–0,047 s. Avec la voie C
parallèle, le paquet continuité complet est donc estimé à environ 0,089 s,
contre 0,237 s pour l'évaluation native de la contribution de continuité avec
les mêmes multiplicateurs : **environ 2,7x** sur cette contribution.

Le C généré fait 13 Mo. Sa compilation `-O0` prend 11,6 s; `-O3` n'a pas
terminé dans la fenêtre interactive disponible. Ce coût doit être mis en cache
par structure de NLP et n'est acceptable que hors résolution.

## Contraintes observées

- L'environnement de la campagne est CasADi **3.7.2**, pas 3.8 :
  `Function.transform` n'est donc pas disponible. Les passes `cse`,
  `ref_count` et `const_folding` ne sont pas une option mesurable ici.
- CasADi a été construit sans `WITH_OPENMP`; `map(..., "openmp", 12)` avertit
  et retombe en série. La voie valable est `map(..., "thread", 12)`.
- Le benchmark appelle directement le noyau Hessien local compilé. Il ne peut
  pas être substitué au milieu de `nlp_hess_l` sans callback : pour conserver
  IPOPT natif, il faut insérer *dès la construction du graphe* un `External`
  pour le primal local, avec ses dérivées exactes C (Jacobienne et
  forward/reverse ou Hessienne) également exportées. CasADi construira alors
  le callback global MX ordinaire autour de ces atomes compilés.

## Recommandation

1. Garder le registre pré-`map` en cache, une fois par structure FHO.
2. Ajouter une option expérimentale Bioptim, limitée aux `ThreadMap`
   homogènes : générer/compiler le primal de continuité **et toutes ses
   dérivées exactes**, puis remplacer le noyau scalaire avant `Function.map`.
   Ne pas installer de callback IPOPT ni faire de `scatter-add` Python.
3. Mesurer alors le véritable `nlp_hess_l`, `nlp_jac_g`, et une résolution FHO
   entière. Le gain de 2,7x ci-dessus est une borne optimiste sur la seule
   continuité, pas encore une promesse sur une itération IPOPT complète.
4. Exiger `max(abs(H_VM-H_external)) <= 1e-10` et une solution/certificat
   IPOPT identiques avant de proposer cette voie en production.

Cette approche est la seule qui évite un callback personnalisé tout en ayant
une chance crédible de dépasser le FHO MX natif : compiler seulement le
Hessien local est insuffisant pour l'intégrer; compiler seulement le primal
est insuffisant pour préserver les dérivées exactes.

Un mini-test CasADi 3.7.2 confirme le contrat `External` : pour qu'une
Hessienne symbolique d'un primal externe reste disponible, le générateur doit
exporter le primal, sa Jacobienne, `forward(1)`, `reverse(1)`, **et les
dérivées de `reverse(1)`** (au minimum `reverse(1).forward(1)` et
`reverse(1).reverse(1)`). Exporter seulement le primal/Jacobienne produit
`Derivatives cannot be calculated` lors de la Hessienne. C'est donc une
exigence explicite du futur POC, pas un détail d'implémentation.
