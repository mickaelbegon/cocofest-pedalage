# FHO3 MX post-shake : audit des termes Hessien locaux — 27 septembre 2026

## Résultat

La récupération d'un **terme de contrainte local exact** est possible après le
`shake` Bioptim, sans résoudre l'OCP et sans compiler le NLP MX global. Sur le
FHO3 historique à 0,30 Nm, huit lignes de contraintes réparties dans le
vecteur final ont été réécrites avec seulement les variables appartenant à la
ligne de leur Jacobienne, puis comparées à `nlp_hess_l` avec un unique
multiplicateur non nul. La différence maximale est `4.55e-13`; les huit
Hessiennes natives n'ont aucun non-zéro hors de leur support local.

| Problème post-shake | Valeur |
|---|---:|
| Variables | 12 263 |
| Contraintes | 11 911 |
| Jacobienne, nnz | 84 236 |
| Hessienne triangulaire, nnz | 49 077 |
| Lignes locales éligibles (≤128 variables) | 11 911 / 11 911 |
| Échantillon audité | 8 lignes, 2–13 variables |
| Erreur max locale vs native | `4.55e-13` |

La commande ne réalise aucun solve : elle intercepte le dernier NLP MX avant
`nlpsol`, construit la Hessienne native uniquement pour la référence, puis
s'arrête. Les résultats bruts sont dans
`local-results/fho3-postshake-packet-audit-20260927-v6/report.json`.

## Compilation locale réelle

Les Hessiennes locales exactes des huit lignes ont ensuite été générées en C
et chargées par `casadi.external`. L'erreur VM/C est strictement nulle. En
revanche, **appeler une bibliothèque externe par ligne est plus lent** ici :
les deux lignes non linéaires mesurées donnent environ ×1,15 et ×0,60
(compilé/interprété), les autres ont une Hessienne nulle. Le coût de franchir
l'ABI `.so` domine une expression de 6–13 variables.

Ce résultat invalide une intégration naïve « une `.so` par contrainte » : elle
ne réduirait pas le temps FHO. Il confirme cependant la correction de la
partition et du contrat de dérivée.

## Ce qui doit changer pour obtenir un gain

Le bénéfice du microbenchmark précédent venait d'un appel `map` d'un **même
noyau** sur de nombreux stages. Après shake, Bioptim ne publie que le graphe
agrégé `f,g`; il ne publie pas l'association :

```
type de pénalité/stage -> indices x globaux -> indices g globaux
```

Il faut exposer ce registre lors de la construction `PenaltyOption` / de
`ThreadMap`, avant l'agrégation du NLP. On pourra alors :

1. construire une Hessienne locale par type (dynamique/collocation,
   contrainte mécanique, objectif) ;
2. compiler une fois par type ;
3. mapper le type sur tous ses stages et scatter-add ses valeurs dans
   `hess_lag` ;
4. laisser les objectifs ou termes vraiment globaux au chemin MX natif ;
5. auditer valeurs, triplets, produit Hessienne-vecteur, objectif et
   certificat avant l'A/B FHO65.

La présence déjà observée de `ThreadMap` dans `nlp_hess_l` montre que cette
répétition existe dans le graphe natif; elle n'est simplement pas accessible
avec les indices post-shake nécessaires au callback externe.

## Protocole A/B prêt

1. Ajouter ce registre dans Bioptim et le conserver après `shake`.
2. Rejouer cet audit FHO3 avec la partition complète et exiger erreur ≤`1e-10`
   sur plusieurs `(x, sigma, lambda)`.
3. Faire un **évaluation-only** FHO65 : `nlp_hess_l` native contre callback
   paquet, avec 1/8/12 workers, même affinité et OMP/BLAS=1.
4. Ne lancer un solve 30 itérations FHO65 que si le callback donne au moins
   20 % de baisse sur Hessienne + assemblage et aucune hausse RSS excessive.
5. Ne retenir qu'un A/B certifié à tolérance, objectif et audit de contraintes
   identiques.

Le script reproductible est
[`audit_fho_postshake_packet_terms.py`](../../scripts/audit_fho_postshake_packet_terms.py).
