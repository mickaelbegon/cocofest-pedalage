# Ding local compilé : RHO unilatéral propagé et paire synchronisée

## Protocole

La référence K5-I1 et le candidat ont chacun propagé **leurs propres** solutions
pendant 100 cycles. Le cas est isocinétique à −2π rad/s, 4 muscles Ding,
30 stimulations/cycle, Radau-5, mécanique réduite, fatigue unitaire,
IPOPT/MA57 et un thread numérique. Le travail visé vaut 0,2 Nm × 2π rad
par cycle. Le candidat reconstruit `Cn`, `Tau1`, `Km` par la carte discrète exacte
et compile la Hessienne du NLP réduit. Les sorties du candidat sont réinjectées
dans son propre RHO ; aucune entrée ni solution n'est gelée par la référence.

Le préflight a vérifié MA57, la compilation des scripts et la référence K5-I1
100/100. Un smoke candidat de 3 cycles et un smoke de paire de 3 cycles ont
précédé les campagnes de 100 cycles.

## Validation unilatérale autonome

| Mesure | Ding complet K5-I1 | Ding local + Hessienne compilée |
|---|---:|---:|
| Cycles certifiés | 100/100 | 100/100 |
| Médiane IPOPT chaude | 0,3797 s | **0,0756 s** |
| Médiane itération RHO chaude | 0,4662 s | **0,1715 s** |
| P90 itération RHO | 0,5262 s | 0,1844 s |
| Temps de la boucle de résolution RHO | 49,58 s | 23,63 s |
| Temps bout en bout, export et replay compris | 130,94 s | 107,87 s |
| Violation maximale du NLP complet | 9,09×10⁻⁷ | 7,39×10⁻⁷ |
| Certificat discret Radau-5 maximal | sans objet | 7,71×10⁻¹⁵ |
| Écart maximal au travail prescrit, quadrature | 1,49×10⁻¹³ J | 1,49×10⁻¹³ J |
| Écart maximal au travail prescrit, DOP853 | 0,001315 J | 0,001219 J |
| Écart nodal `Cn` contre DOP853, maximum | 2,82×10⁻⁵ | 2,82×10⁻⁵ |
| Écart nodal force triceps contre DOP853, maximum | 0,1485 N | 0,1488 N |
| Écart nodal normalisé contre DOP853, maximum | 0,00468 | 0,00292 |
| Dépassement de borne du couple de charge | 0 | 0 |
| Fatigue exécutée cumulée | 1502,703 | 1501,819 |
| Ratio de capacité `A` minimal | 0,940457 | 0,940464 |

Tous les appels IPOPT ont réussi et tous les audits de travail isocinétique
ont passé. Les trajectoires ne sont pas identiques : le plus grand écart de PW
est 13,45 µs, essentiellement au premier cycle ; la médiane de l'écart sur
tous les contrôles et cycles est 0,000028 µs. Le plus grand écart de travail
DOP853 par cycle est 0,0000954 J. L'écart final de fatigue cumulée est de
0,884 unité, soit 0,059 % de la référence complète. Des minima locaux
différents sont possibles dans ce NLP non convexe.

Le temps « bout en bout » comprend le replay dense DOP853 et l'export du
rapport, qui ne sont pas nécessaires avant le cycle de contrôle suivant. Il
ne doit pas être interprété comme la latence du contrôleur.

## Paire sur deux cœurs

Deux processus persistants, épinglés respectivement aux CPU 12 et 13,
ont exécuté chacun 100 cycles autonomes. Une barrière à chaque appel NLP
aligne les cycles. Les deux processus ont réutilisé la Hessienne compilée
depuis le cache (`hit=true`, rechargement ≈0,118 s chacun).

| Mesure de paire | Valeur |
|---|---:|
| Cycles certifiés | 100/100 par processus |
| Médiane latence paire, entrée NLP → audit et archive | **0,0920 s** |
| P90 latence paire | 0,1047 s |
| Maximum latence paire | 0,1306 s |
| Décalage médian de sortie de barrière | 0,563 ms |
| Médiane itération RHO, gauche / droite | 0,1746 / 0,1748 s |
| Temps total des processus, exports compris | 106,91 s |

Chaque processus conserve 100/100 succès physiques et scientifiques, tous
les audits de travail, les bornes de charge et les certificats du cas
unilatéral. Les deux processus emploient ici le **même modèle musculaire** :
cela mesure la contention, le cache et la synchronisation des calculs ; un
test avec paramètres gauche/droite distincts reste requis pour valider un
cas bilatéral physiologique. La barrière se trouve à l'entrée du NLP : la
latence de paire mesurée inclut mapping, résolution, reconstruction et audit
original, mais pas la préparation entre fenêtres ni l'export final.

## Reproduction et artefacts

```bash
bash scripts/run_prescribed_ding_local_propagated.sh compiled \
  local-results/prescribed-ding-local-autonomous-20261004/compiled100 100
bash scripts/run_prescribed_ding_local_pair.sh \
  local-results/prescribed-ding-local-autonomous-20261004/pair100 \
  100 12 13 \
  local-results/prescribed-ding-local-autonomous-20261004/compiled100/cache
```

Résumé détaillé :
`local-results/prescribed-ding-local-autonomous-20261004/full-vs-autonomous-summary.json`
et `local-results/prescribed-ding-local-autonomous-20261004/pair100/summary.json`.
Les archives `symbolic-audit` contiennent les entrées et sorties originales de
chaque NLP ainsi que les certificats de réduction. Aucun échec physiologique
n'a été observé sur ces 100 cycles ; une panne éventuelle d'IPOPT serait
classée comme panne de résolution jusqu'à preuve explicite d'un déficit de
faisabilité.
