# Applicabilité de la réduction Ding locale au FHO

## Conclusion et domaine

Oui, la réduction algébrique locale validée pour le RHO s'applique aussi à un
FHO **à une phase couvrant plusieurs cycles**, avec un état initial fixé et
le même modèle Ding, calendrier de stimulation et schéma Radau-5. Un smoke
single-shot de deux cycles l'a vérifié. Le prototype est expérimental; le
chemin de production FHO n'a pas été modifié.

Pour un FHO dont la taille augmente à chaque résolution, commencer par la
réduction exacte **interprétée**. La compilation de chaque nouvelle taille
coûte beaucoup plus que son gain sur une résolution dans ce smoke.

## Pourquoi l'élimination reste exacte

Les équations de récupération de `A`, `Tau1` et `Km` du modèle Ding utilisé
ont le même temps caractéristique et la même force d'entrée, avec des
coefficients constants propres à chaque muscle. En soustrayant un multiple
de l'équation de `A`, on obtient une équation affine homogène/forcée connue
pour `Tau1 - alpha_tau1/alpha_a * A`, et de même pour `Km`.

Le code calcule ces décalages avec **l'opérateur discret Radau d'origine**,
incluant les continuités. Il ne remplace pas Radau par une solution continue
qui changerait le NLP. `F` et `A` restent variables à tous les nœuds/stades.
Les contraintes et le coût originaux sont évalués après reconstruction;
les bornes des états éliminés sont intersectées avec celles de `A`. Une
relation symbolique entre défauts éliminés et défauts retenus est certifiée.

Étendre une phase à deux ou plusieurs tours allonge cet opérateur et conserve
les mêmes relations. Il n'y a pas de remise à zéro de la fatigue entre tours.
Le pivot actuel demande `alpha_a != 0`; une variante sans fatigue doit choisir
une élimination différente ou reprendre le chemin complet.

## Calcium périodique et réduction : deux décisions distinctes

L'élimination numérique de `Cn` est exacte pour le forcing déjà présent dans
le NLP, le calendrier connu et sa condition initiale. Elle ne justifie pas à
elle seule un régime périodique du calcium.

Le smoke utilise le modèle `periodic_node` déjà employé dans le benchmark
RHO. Un FHO démarrant au repos doit représenter le transitoire initial et
l'historique effectif des impulsions, s'ils diffèrent de cette convention.
Il ne faut pas appliquer la table périodique au début de chaque cycle.
Les changements de fréquence, de cadence ou de calendrier imposent de
recalculer le forcing connu et les opérateurs correspondants. Si le timing
devient une décision du NLP, le facteur constant de ce prototype n'est plus
valable sans extension.

## Smoke contrôlé : deux cycles, une phase

Protocole : commande single-shot, n_windows = cycles_per_window = 2; mécanique
isocinétique prescrite, 30 Hz, Radau-5, fatigue unitaire, travail équivalent à
0,2 Nm, bornes de couple −3/+3 Nm, MA57, un thread, CPU 14. Les trois runs
emploient les mêmes vecteurs initiaux et bornes archivés dans l'espace original.

| | Complet interprété | Local interprété | Local Hessienne compilée |
|---|---:|---:|---:|
| Variables | 7822 | 3490 | 3490 |
| Contraintes | 7621 | 3301 | 3301 |
| Cycles validés | 2/2 | 2/2 | 2/2 |
| Temps IPOPT | 1,580 s | 0,156 s | 0,149 s |
| Itérations IPOPT | 69 | 34 | 34 |
| Construction solveur | 0,823 s | 0,439 s | 12,923 s |
| Construction réduction | — | 0,992 s | 1,005 s |
| Objectif | 0,002540649 | 0,002274894 | 0,002274894 |
| Violation originale max | 2,55e−9 | 4,16e−9 | 4,16e−9 |
| Certificat local max | — | 3,56e−15 | 3,56e−15 |

Les vecteurs primaux, contraintes et duals locaux interprétés/compilés sont
identiques bit à bit. Le complet et le réduit atteignent des solutions
locales différentes; le changement de trajectoire d'IPOPT interdit de
présenter le rapport de temps ~10 comme un gain général garanti. La
substitution conserve le problème discret, pas nécessairement l'optimum
local sélectionné par le solveur.

La compilation froide ajoute ici 12,48 s pour un gain de solveur d'environ
7 ms. Le point de rentabilité brut de cette mesure isolée dépasserait mille
résolutions de même topologie. Il n'est pas une prévision fiable pour de
grands FHO : matrices et temps de génération évoluent avec l'horizon. Un
cache déjà construit réduit le coût de démarrage, mais sa signature change
quand la taille ou la structure du NLP change.

Résultats : `local-results/ding-local-fho-smoke-20261004/`.
Reproduction :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/run_ding_local_fho_smoke.py \
  --output local-results/ding-local-fho-smoke-new --cycles 2 --cpu 14
```

Le runner borne les horizons à 2 ou 3 cycles, vérifie CPU/MA57, limite chaque
processus à 180 s, écrit les commandes et refuse d'écraser un résultat terminé.

## Restrictions et étapes avant utilisation FHO générale

1. **Une phase longue** : le cas testé est supporté. Vérifier les certificats
   à plusieurs tailles, sur un seed RHO représentatif et sur des états plus
   fatigués; vérifier aussi chaque contrainte mécanique du FHO de référence.
2. **Plusieurs phases** : `build_local_map` ajoute une condition initiale
   numérique par phase et le wrapper exige qu'elle soit fixée. Les états
   initiaux de phases suivantes d'un FHO sont généralement liés à la phase
   précédente, pas fixés. Il faut propager les décalages à travers ces liens
   ou les garder symboliquement dépendants des variables retenues. Appliquer
   directement le wrapper actuel serait incorrect; son contrôle de bornes
   doit continuer à refuser ce cas.
3. **Transitoire calcium** : construire un cas au repos et un cas avec
   historique non périodique, puis comparer complet/réduit sur le même
   forcing. Rejouer indépendamment avec DOP853 pour quantifier l'erreur du
   schéma discret sans confondre cette erreur avec celle de l'élimination.
4. **Warm start entre tailles** : interpoler/transférer le seed dans l'espace
   physique complet, puis réduire pour la nouvelle taille. Ne pas transporter
   aveuglément les offsets ou la numérotation des duals d'un horizon à un
   autre. Vérifier les bornes de PW et les conditions au raccordement.
5. **Politique compilation** : mesurer froid/cache/solve séparément pour des
   tailles représentatives. Préférer interprété pour une succession de
   tailles uniques; activer la compilation après réduction pour des tailles
   réellement réutilisées. Aucun résultat présent ne valide encore un FHO
   de centaines de cycles.
