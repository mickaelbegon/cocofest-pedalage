# Point de reprise du benchmark RHO

État consolidé au 12 août 2026 sur la branche
`codex/full-horizon-homotopy`. Le dernier SHA Cocofest cité ci-dessous comme
résultat CI reste celui de la campagne correspondante; les changements locaux
explicitement signalés ne sont pas encore des résultats Linux.

Ce document répond à deux questions :

1. quelles conclusions peuvent déjà être considérées comme acquises;
2. quelles tâches doivent être reprises, dans quel ordre et avec quel critère
   de succès.

Pour les justifications complètes et les liens vers chaque run, consulter
l'[historique des développements](development_history.md). La méthode active
est décrite dans le [README](README.md).

Pour reprendre sur un autre calculateur, copier le
[prompt de continuation](continuation_prompt.md) et suivre la
[procédure Linux 32 cœurs](linux_32core_setup.md).

## 1. Configuration scientifique courante

- Objectif : minimiser uniquement la fatigue musculaire.
- Couple externe : `0 N.m`; aucune assistance ne masque l'arrivée de la
  fatigue.
- Un RHO résout un OCP d'un cycle, avec 30 décisions de PW par muscle.
- Quatre muscles, soit 20 états de Ding.
- PW admissibles dans $[pd0,600\,\mu\mathrm{s}]$.
- Angle terminal absolu, sans accumulation de la solution du cycle précédent.
- Graphes SX pour les campagnes actives.
- MUMPS pour IPOPT et MadNLP.
- Mécanique reduced : deux états $[\theta,\omega]$, sans imposer une cadence
  constante.
- Mécanique full : six états mécaniques, utilisée comme contrôle scientifique.
- Force passive active en full et en reduced.

Une chaîne est valide seulement jusqu'au premier RHO qui échoue au statut
solveur ou à l'audit physique. Les fenêtres qui convergent après cet échec
restent diagnostiques et ne sont pas agrégées comme endurance.

## 2. Conclusions considérées comme acquises

### 2.1 La référence historique n'est plus l'oracle

Deux défauts empêchent de prendre l'ancienne transcription comme cible
scientifique :

1. un ancien chemin `updating_model()` pouvait perdre l'activation de la
   relation de force passive;
2. la collocation Radau degré 3 sous-estime de `6.39 %` le calcium périodique
   isolé étudié.

La nouvelle règle est de corriger le modèle ou la transcription, puis de
recertifier les solveurs sur ce problème commun. Un ancien coût n'est conservé
que comme donnée historique.

### 2.2 La force passive est maintenant conservée

Le chemin full transmet explicitement
`activate_passive_force_relationship`. La mécanique reduced l'active par
défaut et évalue le même coefficient périodique. Un test ciblé vérifie que la
mise à jour du modèle conserve simultanément les relations force-longueur,
force-vitesse et passive.

Une campagne sans force passive doit désormais porter le statut d'ablation.
Elle ne doit pas être mélangée à une comparaison de solveurs.

### 2.3 La résolution du calcium doit être raffinée

Pour le régime périodique isolé :

| Méthode | $C_N$ |
|---|---:|
| Analytique | `0.162982158353` |
| ACADOS IRK, 4 étages et 5 sous-pas | `0.162982158637` |
| Radau degré 3 | `0.152573519058` |
| ERK testé | `0.232903256` |

Radau 3 est numériquement cohérent avec sa transcription, mais insuffisamment
fidèle à la dynamique calcique. ERK est exclu avec les réglages testés. La
bonne approche est de garder les 30 décisions de PW tout en augmentant la
résolution interne des états, par Radau 5 ou sous-pas IRK.

MadNLP/MUMPS et IPOPT/MUMPS Radau 5 convergent maintenant tous deux `5/5`.
La désactivation des duals transférés a ramené le RHO 2 IPOPT de 2 000 à 140
itérations. La convergence solveur n'est cependant pas une certification de
la transcription : R5--R6 diffère encore de `0.343–0.398 %` sur la fatigue et
de `0.580–0.645 %` sur l'AUC. Le raffinement n'est donc pas certifié à 30 ou
100 RHO.

### 2.4 La réduction mécanique est validée pour le régime testé

La réduction conserve exactement les 20 états de Ding et remplace seulement
la mécanique par :

```math
q=\Phi(\theta),
\qquad
\dot q=T(\theta)\omega.
```

La campagne IPOPT appariée à 100 RHO a donné :

| IPOPT/MUMPS | Full | Reduced |
|---|---:|---:|
| Préfixe physique | `100` | `100` |
| Fatigue exécutée | `4347.708565` | `4343.497299` |
| Écart relatif |  | environ `-0.097 %` |
| Médiane chaude | `4.595 s` | `1.031 s` |
| Gain reduced |  | `4.46x` |

La capacité finale du Biceps ne diffère que d'environ `8.15e-6`. Le grand
écart de fatigue observé dans les premières campagnes provenait donc de
différences de transcription et de bornes, pas d'une réserve artificielle
créée par la réduction.

Cette conclusion reste à confirmer sur la transcription calcique raffinée.

### 2.5 SX et compilation persistante sont les leviers logiciels sûrs

- SX réduit de `57.5 %` à `60.5 %` la médiane chaude par rapport à MX sur les
  comparaisons appariées.
- La bibliothèque C doit être construite une fois et réutilisée pendant tous
  les RHO.
- L'état initial, l'angle terminal et les bornes mobiles doivent être des
  paramètres runtime.
- Le temps de construction et de compilation doit être séparé du temps chaud
  du RHO.

Augmenter le nombre de cœurs ne multipliera pas la vitesse par le nombre de
cœurs. La construction de graphes, de nombreuses évaluations CasADi et une
partie de l'algorithme restent sérielles. Le parallélisme est surtout utile
entre solveurs ou formulations; le gain intra-solveur dépend de la part de la
factorisation MUMPS réellement parallélisable.

## 3. État des solveurs

### 3.1 IPOPT/MUMPS

- Seul solveur ayant certifié full et reduced sur les 100 RHO appariés.
- Reduced/SX compilé : environ `1.0 s` chaud par RHO sur Radau 3.
- Choix robuste actuel pour produire une chaîne complète.
- Radau 5 reste sensible à la stationnarité malgré une bonne faisabilité
  primale.

### 3.2 MadNLP/MUMPS

- MUMPS est transmis à libMad sous le nom typé `MumpsSolver`.
- PARDISO/MKL n'a pas apporté de gain et est abandonné.
- Sur 100 RHO Radau 3, le full a un préfixe strict de 80 et le reduced de 98.
- Les RHO suivants peuvent converger, mais ils partent alors d'un état non
  certifié et ne forment pas une endurance valide.
- Radau 5 est prometteur : `5/5` et environ `1.99 s` chaud.
- La compilation persistante fonctionne techniquement, mais son gain doit être
  remesuré sur une campagne assez longue pour amortir le coût initial.

Le prochain levier MadNLP n'est pas un budget d'itérations plus élevé. Le fast
path est maintenant plafonné au P90 exact observé sur les fenêtres Linux R5
terminées du run `31589698184`, soit `73` itérations, avec une garde murale
native de `20 s`. Après le premier échec, IPOPT/Radau cible reconstruit une
primale du **même RHO** et MadNLP tente de la certifier. Après le second échec,
une solution IPOPT ne peut avancer ce RHO que si elle est convergée et passe
l'audit commun. Ce chemin hybride est implémenté localement mais pas encore
mesuré en CI; il faut donc séparer dans le bilan les succès MadNLP natifs, les
recoveries seed-only, les fallbacks IPOPT et le temps pipeline complet.

Le run `30856972707` précise ce diagnostic : MadNLP/Radau-5 reduced certifie
`140` RHO, puis reproduit deux fois exactement le même échec au RHO 141
(`353` itérations, `inf_pr = 3.503992`). Ce stop n'est pas attribuable à la
fatigue avec les preuves actuelles (`4.33 %` seulement de saturation PW haute).
Une restauration IPOPT du même RHO est maintenant disponible via
`--nlp-ipopt-recovery`; elle doit encore être certifiée sur Linux avant de
revendiquer un préfixe plus long.

Le contrôle `30873302850` sur le code corrigé certifie finalement `145/145`
avec MadNLP sans appeler cette restauration. Le RHO 141 converge en `57`
itérations (`1.299 s`) et la médiane chaude vaut `1.458 s`. L'arrêt précédent
n'est donc pas reproductible. La priorité devient une ablation appariée de la
préparation inter-RHO (`none`, rollout Radau, Phase I adaptative), avec le temps
de préparation inclus dans le temps effectif.

### 3.3 FATROP

- Reduced/SX/collocation : `100/100` et solution physiologique cohérente avec
  IPOPT reduced.
- Plus lent qu'IPOPT reduced dans la campagne active.
- Full : l'échec structurel était causé par l'aplatissement global des
  contraintes multi-thread dans Bioptim. Le commit `4179bf07` les redistribue
  par stage; le vrai OCP full passe localement `1/1`. Le gate Linux reste à
  exécuter avant toute revendication d'endurance.
- Compilation full : non retenue. Le C CasADi fait environ `201 Mo`; Clang n'a
  pas terminé après plus de 40 minutes et deux runners Linux ont été repris
  pendant GCC. Le gate full est donc interprété; reduced reste compilé et
  mesure la réutilisation persistante.
- RK4 est abandonné.

FATROP reste un excellent contrôle indépendant de l'optimum reduced, mais pas
le meilleur chemin de production actuel.

### 3.4 ACADOS 0.5.5

La formulation full avec garde de cadence asymétrique à `2.60 rad/s` est le
seul candidat ayant régulièrement résolu une fenêtre très sous la seconde.
Elle certifie `30/30`, puis `80/100` RHO : médiane solveur `0.102 s`, médiane
murale effective `0.121 s` et `p90 = 0.304 s` sur le run `30760775027`.
L'ancien préfixe limité à 13 RHO provenait notamment d'une contrainte de
cadence full appliquée à la mauvaise variable mécanique et ne décrit plus
l'état actuel.

Les analyses suivantes ont été menées :

- pénalité douce de cadence;
- homotopie terminale;
- retry primal seul puis primal-dual complet;
- `SQP_WITH_FEASIBLE_QP` avec Byrd–Omojokun;
- préservation sélective des multiplicateurs;
- sélection entre shift projeté et rollout IRK projeté.

Aucune n'a déplacé le premier échec. Byrd–Omojokun réduit parfois les résidus
d'environ un ordre de grandeur, mais reste au-dessus du seuil physique. La
préservation des duals est instable. L'homotopie multiplie le temps effectif
sans prolonger le préfixe. Le rollout IRK concurrent ajoute environ `0.245 s`
par transfert et est rejeté dans la plupart des cas.

La chaîne s'arrête proprement au RHO 81. Son meilleur itéré est presque
faisable (`6.74e-6` de défaut dynamique), mais sa stationnarité `1.56e-3`
reste au-dessus de la tolérance. Deux reprises primal-dual depuis cet itéré
reproduisent exactement le même échec; répéter ce retry est donc inutile. La
configuration active en conserve un seul pour diagnostiquer et empêcher la
propagation d'une solution MAXITER.

Le run apparié `30763188906` a depuis testé ce levier. Byrd--Omojokun reproduit
le même préfixe `80/100`, tandis qu'une Phase-I qui ne projette que `q/qdot`
atteint `100/100`. Elle préserve exactement les 20 états Ding du shift dans le
guess et ne change pas l'OCP final. Son temps chaud complet, projection
incluse, vaut `0.739 s` en médiane, `0.909 s` au P90 et `0.963 s` au maximum.
La projection coûte `60.436 s` sur 99 transferts et n'est acceptée que 34 fois;
une présélection moins coûteuse reste donc possible.

Le commit `a8f1955` ajoute une récupération lazy opt-in. Elle restaure le
checkpoint du même RHO seulement après un échec, projette `q/qdot`, garantit
l'invariance exacte des états Ding et des PW, réinitialise la mémoire native
SQP/QP puis exige une nouvelle certification ACADOS. Le mode CI
`acados_lazy_recovery` exécute baseline et lazy successivement sur la même
machine. La campagne Linux 100 RHO
[31390381640](https://github.com/mickaelbegon/cocofest/actions/runs/31390381640)
donne `80/100` dans les deux cas. Au RHO 81, la Phase I lazy modifie `q` de
`0.154 rad` et `qdot` de `1.413 rad/s`, tout en gardant Ding/PW exactement
inchangés. Le retry reste néanmoins à `2.37e-2` de stationnarité et `1.09e-3`
de défaut dynamique. Le déclenchement après échec est donc trop tardif.

La Phase-I sur tous les états atteint aussi `100/100` et semble plus rapide
(`0.450 s` médian, projection incluse), mais elle modifie les états Ding du
guess jusqu'à `33.46`, double presque le temps solveur et dégrade fortement le
rollout DOP853 : `+3.519 %` sur le coût et drift normalisé maximal `109.3`.
Elle est rejetée. Pour la Phase-I mécanique, les écarts DOP853 sont beaucoup
plus faibles (`+0.358 %` coût, `+0.113 %` AUC), mais le drift mécanique full
atteint encore `2.856` après 100 cycles sans reprojection de contact. Le
`100/100` certifie donc le solveur et les nœuds, pas encore la trajectoire
continue full.

Le run `30754413003` ajoute un défaut plus précis : les nœuds respectent
presque la borne de cadence, mais la progression angulaire entre deux nœuds
implique un dépassement rapide de `0.376–0.403 rad/s`. Deux écrans sont prêts :
garde interne ACADOS asymétrique à `2.60`, puis `2.55 rad/s`, tout en gardant
la marge physique et l'audit à `3.0 rad/s`. Les tests unitaires séparent ces
deux notions. La garde 2.60 passe 80 RHO; 2.55 échoue au premier. Pour rendre
la garantie inter-nœuds plus rigoureuse, tester 60 nœuds avec move-blocking
des 30 PW; augmenter seulement les sous-pas IRK ne contraint pas les stages
internes.

### 3.5 Alpaqa et autres voies archivées

- Alpaqa : intégration non fonctionnelle sur cette formulation.
- PARDISO pour MadNLP : aucun gain démontré.
- MA57 : résultat IPOPT historique, non retenu dans la matrice portable.
- Surrogate neuronal : non prioritaire avant profilage détaillé.
- Full horizon MadNLP/MX : question distincte du RHO temps réel.

## 4. Full horizon MadNLP/MX

Le monolithique full/MX certifie un cycle. Deux cycles échouent très loin des
limites mémoire : environ `2.35 GiB` de RSS sur une limite CI de `12.5 GiB`.
Le témoin reduced/MX échoue lui aussi à deux cycles. La cause principale n'est
donc pas la mémoire ni uniquement la mécanique full, mais la construction et
la couture du seed multi-cycle avec la transcription non convexe.

Ce chantier ne doit pas bloquer l'optimisation du RHO. Il ne redevient
prioritaire que si l'objectif est explicitement un horizon monolithique.

## 5. TODO priorisés

### P0 — Définir et certifier le problème scientifique corrigé

- [x] Ajouter le profil verrouillé `scientific-radau5`, distinct de la
  transcription diagnostique Radau 3.
- [x] Enregistrer explicitement dans chaque JSON le statut de la force passive,
  le degré de collocation, le nombre d'étages et les sous-pas.
- [x] Ajouter un test isolé du calcium périodique contre la valeur analytique.
- [x] Conserver 30 décisions de PW indépendamment du degré de collocation des
  états.
- [x] Certifier le calcium isolé aux degrés Radau 3/4/5/6. Radau 4 a une erreur
  de `0.4518 %`, Radau 5 de `0.01730 %` et Radau 6 de `0.000415 %`.
- [x] Exécuter le premier gate Linux 5 RHO Radau 4/5/6. Il réfute la
  certification immédiate : MadNLP Radau 5--6 diffère de `0.3977 %` sur la
  fatigue et de `0.6452 %` sur l'AUC; IPOPT Radau 5 ne valide que `1/5`.
- [x] Refaire le gate avec raffinement cible, duals coupés, profils hashés et
  contrôle full. IPOPT et MadNLP passent `5/5`, mais reproduisent l'écart
  R5--R6; le problème n'était donc pas seulement la convergence d'IPOPT.
- [ ] Transférer R4 vers R5 et R6 vers R5, puis réintégrer les mêmes PW avec un
  intégrateur haute précision pour distinguer discrétisation et bassin local.
- [x] Exporter automatiquement la trajectoire complète certifiée de chaque cas
  scientifique sous `validated-rho-trajectory.npz`, afin que le transfert
  croisé conserve les PW exactes plutôt que les valeurs JSON arrondies.
- [x] Ajouter un rollout DOP853 continu des PW exportées, sans remise à zéro
  aux nœuds, avec quadrature commune du coût, de l'AUC et des quatre muscles.
- [x] Exécuter le nouveau gate et comparer les métriques DOP853 R5/R6. Sur le
  run `30754413003`, le drift tombe d'environ `2 %` en R5 à `0.094 %` avec
  IPOPT/R6 et `0.336 %` avec MadNLP/R6; MadNLP/R6 devient la cible provisoire.
- [ ] Transférer les mêmes PW entre R5/R6 et full/reduced si le gate 30 laisse
  encore apparaître un écart de bassin optimal.
- [ ] Lancer `5`, puis `30`, puis `100` RHO seulement lorsque le gate précédent
  est physiquement certifié.

Critère de sortie proposé : erreur relative du calcium inférieure à `0.1 %`,
variation de fatigue et d'AUC inférieure à `0.1 %` au raffinement suivant, et
aucune violation interne des bornes.

Gate Linux disponible : les deux solveurs convergent strictement sur les trois
degrés. R4 et R5 sont proches (`0.080–0.095 %` fatigue), mais R5 et R6 ne le
sont pas (`0.343–0.398 %` fatigue, `0.580–0.645 %` AUC). Les PW diffèrent
surtout au premier cycle sur le Biceps et le Triceps. En Radau 5, IPOPT et
MadNLP reduced s'accordent à `0.0205 %`, et MadNLP full/reduced à `0.00194 %`.
IPOPT full trouve toutefois une branche de fatigue `0.400 %` plus basse que
son reduced : la prochaine étape est un transfert croisé des mêmes contrôles,
pas un palier d'endurance plus long.

### P1 — Établir la nouvelle baseline de performance

- [ ] Comparer IPOPT/MUMPS et MadNLP/MUMPS reduced sur exactement le même NLP
  raffiné.
- [ ] Conserver IPOPT full comme contrôle apparié aux cycles 5, 30 et 100.
- [ ] Réutiliser une seule bibliothèque compilée par cas et vérifier son hash.
- [ ] Séparer construction, compilation, préparation initiale, solveur chaud et
  temps mural effectif.
- [ ] Comparer fatigue et AUC des quatre muscles ainsi que les PW aux cycles
  10, 30 et 100.

Critère de sortie : un tableau unique où tous les solveurs ont le même modèle,
le même maillage des états, les mêmes PW admissibles et le même préfixe
physique.

### P1 — Rendre MadNLP robuste au premier RHO difficile

- [x] Exclure explicitement le premier RHO du budget temps réel : budget
  MadNLP initial de 2000 itérations sans limite murale, puis capsule rapide à
  73 itérations/20 s reconstruite une seule fois avant la séquence chaude.
- [ ] Valider cette séparation en CI et vérifier dans les logs la présence du
  reset de capsule, puis rapporter sur les RHO 2..N la médiane, les P90/P99, le
  maximum et la proportion sous la durée physique du cycle.
- [ ] Sauvegarder le dernier checkpoint physique avant l'échec full RHO 81 et
  reduced RHO 99.
- [x] Ne jamais avancer la fenêtre depuis une solution non certifiée.
- [x] Donner une seconde chance au même RHO depuis une primale IPOPT distincte
  et documentée; la campagne Linux reste à valider.
- [x] Réutiliser la solution récupérée uniquement si sa faisabilité mesurée
  passe, puis exiger une certification finale par MadNLP.
- [x] Implémenter localement une récupération **conditionnelle** commune à
  IPOPT, MadNLP et Fatrop : après le premier échec seulement, restaurer la
  primale préparée du même RHO, projeter uniquement la mécanique par Phase I,
  préserver exactement les états de Ding et les PW, remettre les duals à zéro,
  puis exiger une nouvelle résolution du problème physique strict.
- [x] Ajouter les tests unitaires du restore exact, de l'invariance Ding/PW,
  du reset des duals et de la seconde tentative obligatoire. Validation locale
  ciblée : `5 passed` le 10 août 2026.
- [x] Exécuter un vrai RHO local IPOPT reduced/SX/Radau 3 avec le recovery
  armé : `1/1` certifié, statut `0`, `85` itérations, `2.309 s` solveur et
  aucune Phase I déclenchée. Cela valide le chemin nominal dormant, mais ne
  remplace pas le gate Linux ni un échec forcé/réel.
- [x] Forcer localement un échec avec `max_iter=1`. Ce smoke a découvert puis
  corrigé un budget interne Bioptim trop court : après correction, deux appels
  solveur ciblent bien le même RHO, sans avancement; un seul recovery est
  enregistré, le checkpoint est restauré exactement et Ding/PW restent
  inchangés. Les deux solves échouent volontairement au plafond d'une
  itération; ce résultat ne mesure ni convergence ni fatigue.
- [x] Comparer localement IPOPT reduced/SX/Radau 3 sur `5/5` RHO avec et sans
  recovery armé. Aucun recovery ne se déclenche; l'écart relatif est seulement
  `8.05e-12` sur l'objectif, `7.99e-12` sur la fatigue exécutée et `6.53e-12`
  sur l'AUC. Les itérations et temps fluctuent entre les deux processus et ne
  constituent pas une mesure de gain sur un seul passage.
- [x] Exposer ce mode comme input CI opt-in et vérifier dans le runner que le
  contrat est bien sérialisé dans le JSON; les campagnes nominales restent
  inchangées.
- [ ] Lancer le smoke CI de 5 RHO et le comparer à la baseline sans recovery.
  Tant qu'aucun échec naturel ne survient, le nombre de Phase I doit être nul
  et les résultats physiques identiques.
- [ ] Rejouer ensuite le premier checkpoint d'échec naturel IPOPT et MadNLP;
  mesurer séparément temps de restauration, Phase I et retry solveur.
- [ ] Comparer les ensembles actifs PW, multiplicateurs, stationnarité et
  conditionnement avant/après récupération.

Critère de sortie : préfixe strict `100/100` sans propagation d'un terminal
invalide.

### P1 — Refaire le transfert ACADOS sur la dynamique discrète

- [x] Ne jamais avancer après un statut MAXITER; rejouer et arrêter au même
  RHO si la reprise échoue.
- [x] Tester deux reprises successives du meilleur itéré. Elles sont
  déterministes et n'étendent pas le préfixe de 80 RHO; revenir à une seule
  reprise diagnostique.
- [x] Construire une projection proximale des états mécaniques `q/qdot` de la
  nouvelle fenêtre, tout en préservant exactement le shift du calcium et des
  autres états Ding.
- [x] Exécuter le mode apparié `acados_recovery` : baseline, Phase-I mécanique,
  Phase-I complète et Byrd--Omojokun, tous avec export du préfixe exact.
- [ ] Auditer explicitement l'angle absolu et la variété de contact du
  **candidat Phase-I** avant le solve. L'OCP final les impose et les 100
  solutions les respectent, mais le critère d'acceptation de la projection ne
  les certifie pas encore séparément.
- [ ] Ajouter un écran bon marché avant la Phase-I mécanique : elle coûte
  actuellement environ `0.61 s` à chacun des 99 transferts, mais 65 candidats
  sur 99 sont finalement rejetés.
- [x] Implémenter une alternative plus stricte à l'écran : déclencher la
  Phase-I mécanique seulement après le premier échec certifié du même RHO,
  restaurer le checkpoint exact et réinitialiser complètement ACADOS.
- [x] Ajouter les tests du contrat CLI, de l'invariance Ding/PW et du reset
  natif. Le module ciblé passe `333/333` localement le 10 août 2026.
- [x] Lire la campagne `31390381640` et comparer baseline/lazy sur 100 RHO :
  préfixe strict, premier échec, nombre et coût des recoveries, temps médian,
  P90, maximum, objectif et fatigue par muscle.
- [ ] Produire et comparer les checkpoints exacts baseline/proactif aux RHO 17,
  35 et 80. Les projections proactives acceptées aux RHO 18--35 sont le premier
  endroit où les deux histoires de warm start peuvent diverger durablement.
- [ ] Rejouer le RHO 81 depuis le checkpoint 80 de la chaîne proactive sans
  nouvelle Phase I. Si ce RHO converge, l'histoire du bassin est confirmée;
  sinon, rechercher une différence de mémoire SQP/QP ou de paramètres runtime.
- [ ] Si nécessaire, précompiler deux capsules synchronisées : restauration de
  faisabilité puis objectif de fatigue.
- [ ] N'évaluer RTI qu'après plusieurs chaînes SQP complètes certifiées.
- [x] Réintégrer densément chaque intervalle accepté pour vérifier cadence et
  calcium entre les nœuds.
- [ ] Ajouter en complément un rollout DOP853 remis à l'état certifié au début
  de chaque RHO, puis rejouer les mêmes PW en mécanique reduced. Le rollout
  full global actuel accumule le drift de la contrainte de pédalier sur 100
  cycles et ne permet pas encore d'attribuer l'écart à ACADOS seul.
- [ ] Relancer IPOPT R5, MadNLP R5 et ACADOS hybride depuis un fichier d'état
  initial strictement commun, avec comparaison bit à bit des 20 états Ding,
  de `theta/omega`, de la cible terminale et des bornes mobiles avant le RHO 1.
  Rejouer ensuite les trois séries de PW avec le même intégrateur reduced
  indépendant. La comparaison historique à 145 RHO démontre le gain de temps
  ACADOS, mais son gain apparent de fatigue est confondu avec un seed initial
  moins fatigué.
  Le workflow ACADOS recentre maintenant les bornes du premier nœud sur le
  seed commun; il reste à certifier ce nouveau chemin en CI et à comparer les
  trois premiers états exportés.
  Le premier essai (`31442152939`) a exposé des bornes ACADOS mises en cache
  avant le recentrage : RHO 1 converge mais atteint `omega=-10.212 rad/s`,
  puis RHO 2 échoue. La capsule native resynchronise désormais les bornes
  courantes avant le rollout IRK et le premier SQP; ce correctif doit être
  recertifié.

Le critère intermédiaire est atteint : le RHO 81 est franchi sans relâcher les
contraintes et le solveur passe `100/100` sous `1 s` projection incluse. Le
critère final exige maintenant l'audit continu full/reduced des mêmes PW.

### P2 — Recertifier full contre reduced sur le modèle corrigé

- [ ] Rejouer exactement les mêmes PW et les mêmes 20 états Ding dans les deux
  mécaniques.
- [ ] Comparer RHS musculaires, couple, accélération, force passive et calcium
  point par point.
- [ ] Refaire la comparaison IPOPT full/reduced à 5, 30 puis 100 RHO avec le
  calcium raffiné.
- [ ] Expliquer tout écart supérieur à `0.1 %` par muscle, pas seulement sur la
  somme.

Critère de sortie : coûts, AUC et capacités finales appariés, sans différence
de modèle cachée.

### P2 — Profiler avant de paralléliser davantage

- [ ] Mesurer séparément temps des fonctions, Jacobien, Hessienne,
  factorisation et backsolve.
- [ ] Tester MUMPS avec `1`, `2`, `4`, `8`, `16`, `30` et éventuellement `48`
  threads sur le même NLP chaud.
- [ ] Répéter chaque cas au moins trois fois et fixer les autres sources de
  parallélisme.
- [ ] Comparer le parallélisme intra-solveur au lancement parallèle des quatre
  familles de solveurs.

Critère de sortie : courbe de speedup et d'efficacité; ne pas extrapoler un
gain `30x` ou `48x` sans mesure.

### P3 — Nettoyer le langage et les artefacts du benchmark

- [ ] Remplacer dans les nouveaux cas le terme ambigu `reference` par
  `legacy-radau3`, `scientific-radau5` ou un nom descriptif équivalent.
- [ ] Garder les anciens noms uniquement dans l'archive et les liens de runs.
- [ ] Ajouter dans le rapport un badge `historique`, `certifié numérique` ou
  `certifié scientifique`.
- [ ] Faire échouer la CI si la force passive ou la transcription diffèrent
  entre deux cas annoncés comme appariés.
- [x] Mettre à jour le tableau de synthèse pour Kevin avec la campagne
  reduced hybride 300 RHO (`31428024125`), ses temps chauds, ses fatigues et
  ses réserves sur le seed, l'overhead et la régularité des PW.

### P3 — Atteindre un échec réellement causé par la fatigue

- [ ] Après certification à 100 RHO, prolonger reduced vers 300 RHO.
- [ ] Si la chaîne reste faisable, prolonger vers 1000 RHO.
- [ ] Conserver deux chances de non-convergence au même RHO sans avancer depuis
  un échec.
- [ ] Distinguer clairement échec de fatigue, changement d'ensemble actif,
  limite d'itérations et erreur d'infrastructure.

## 6. Pistes à ne pas relancer sans nouvel élément

- PARDISO/MKL pour MadNLP;
- Alpaqa sur l'interface actuelle;
- FATROP/RK4;
- simple augmentation du budget SQP ACADOS;
- homotopie terminale ACADOS actuelle;
- retry ACADOS primal ou primal-dual inchangé;
- conservation globale des duals ACADOS;
- rollout IRK concurrent actuel;
- réseau neuronal avant d'avoir identifié une fonction dominante;
- estimation de la capacité full horizon à partir de la RAM avant d'avoir une
  seed multi-cycle certifiée.

Une piste archivée peut être réouverte uniquement si un changement précis de
modèle, d'interface ou d'algorithme invalide le résultat négatif précédent.

## 7. Prochaine session de reprise conseillée

1. Lire les artefacts complets des runs `31389585968` et `31390381640`.
2. Vérifier que le recovery ACADOS cible deux fois le même angle absolu et le
   même état de fatigue, sans fenêtre intermédiaire exportée.
3. [fait] Exporter les checkpoints proactifs aux RHO 17, 35 et 80 et rejouer
   le RHO 81. Le primal exact échoue dans une capsule neuve malgré des résidus
   identiques : l'état interne ACADOS/HPIPM contribue au succès en chaîne.
4. [fait, run `31394895014`] Phase I au seul RHO 19 échoue encore au RHO 81;
   la séquence 19--36 atteint `100/100` avec 18 projections (`8.479 s`).
5. [fait, run `31396677025`] `19--27` valide 85 RHO puis échoue au 86;
   `19--31` valide 86 RHO puis échoue au 87. La borne minimale est dans
   `32--36`.
6. [résultat scientifique disponible, run `31398686286`] `19--33` échoue au
   RHO 88; `19--35` et `19--36` atteignent `100/100`. Le run est rouge à cause
   d'un post-gate qui cherchait encore l'ancien fichier lazy, pas du solveur.
7. [fait, run `31399758587`] `19--34` échoue encore au RHO 88; `19--35`
   atteint `100/100`. La fenêtre minimale certifiée contient 17 projections.
8. [fait, run `31400668993`] `19--35` valide 140 RHO puis échoue au 141;
   capacité minimale `0.946`, arrêt fatigue non confirmé. C'est une nouvelle
   perte de bassin numérique.
9. [fait, run `31401580984`] La Phase I proactive échoue également au RHO 141,
   avec une trajectoire quasi identique à `19--35`. Une seconde fenêtre
   mécanique n'est pas une solution.
10. [fait, run `31414366905`] Étendre le recovery IPOPT/Radau-5 du reduced
    vers le full : même RHO gelé, mêmes bornes/targets, audit structurel,
    injection du primal certifié, reset natif puis obligation d'un retry
    ACADOS réussi avant d'avancer. Le premier smoke a confirmé l'identité
    structurelle mais a exposé un seed `common-full` non physique; le gate
    corrigé part d'une trajectoire ACADOS full native certifiée par le NLP et
    journalise explicitement cette provenance. Le second smoke a produit ce
    seed, puis un post-gate l'a rejeté pour l'excursion inter-nœuds connue du
    bridge non gardé; l'audit physique est maintenant exigé au bon niveau,
    sur l'OCP cible muni de la garde rapide `2.60`. Le troisième smoke a
    ensuite exposé un index absolu de cycle mal établi lors du rechargement du
    seed natif; le consommateur utilise désormais le pipeline standard qui a
    produit ce seed, sans raffinement IPOPT initial redondant. Le smoke final
    valide `5/5`, l'audit physique, l'injection, le reset et la recertification
    ACADOS du même RHO.
11. [fait numériquement, run `31419405169`] La campagne reduced naturelle
    atteint `150/150` sans recovery IPOPT. ACADOS chaud vaut `0.127 s` en
    médiane solveur et `0.140 s` en médiane murale; le préfixe certifié est
    exporté. Le rouge final est un faux négatif du post-gate sur le chemin du
    fichier, corrigé par un contrôle absolu et diagnostiqué.
12. [fait numériquement, run `31420496210`] La campagne atteint `300/300`,
    avec cinq recoveries IPOPT aux RHO 5 (deux appels), 120, 183 et 286. Le
    solve ACADOS reste sous la seconde, mais les recoveries portent le coût
    après préparation à `1.90 s/RHO`. Le gate final est encore rouge à cause
    de l'`exit 0` anticipé du shell Conda, après `jq` et `ls` réussis.
13. [fait, run `31422321005`] Remplacer l'`exit 0` par une branche `if/else`
    et laisser le shell terminer normalement. Le smoke est vert et reproduit
    deux recoveries au RHO 5 (`105.1 s`, puis `20.7 s`).
14. Comparer le préfixe 1--150 des runs `31419405169` et `31420496210` : le
    premier n'a aucun recovery, tandis que les runs `31420496210` et
    `31422321005` échouent tous deux au RHO 5. Comparer runner/CPU,
    configuration résolue, résidus et itérés initiaux avant de relier cet écart
    à la fatigue.
    [diagnostic] Le seed Intel du premier run diffère du seed AMD; les deux
    runs AMD ont des seeds bit-à-bit identiques. La PW biceps diffère jusqu'à
    `180.6 µs`. L'input temporaire `acados_seed_source_run_id` permet maintenant
    de rejouer le seed Intel sur un nouveau CPU, avec SHA journalisés.
    [confirmé, run `31423661232`] Le seed Intel reproduit exactement les cinq
    premiers RHO du run 150 sur un nouveau runner, sans recovery. Le seed, et
    non le CPU ACADOS, cause le changement de branche précoce.
15. [fait, run `31424509992`] Le seed Intel supprime le recovery du RHO 5 et
    valide 233 RHO. IPOPT converge lors des recoveries aux RHO 218 et 234,
    mais ACADOS ne recertifie pas le RHO 234 après deux injections. L'audit
    mécanique passe et la capacité biceps vaut encore `0.8969`; l'arrêt reste
    `unconfirmed_endurance_stop`.
16. [fait, runs `31426862847` et `31428024125`] Tester le fallback hybride explicitement
    étiqueté : si
    IPOPT/Radau-5 converge et passe tous les audits du RHO gelé après un échec
    ACADOS, avancer exceptionnellement depuis sa solution adaptée puis rendre
    le RHO suivant à ACADOS. Conserver en parallèle le mode strict qui exige
    une recertification ACADOS. Le mode `acados_reduced_fallback` exige
    `status=0`, rééchantillonne Radau sur les shooting nodes ACADOS et compte
    séparément les RHO certifiés par IPOPT. Le run `31426862847` valide
    235/235 RHO : IPOPT certifie le RHO 234 et ACADOS reprend au RHO 235.
    Les états sont continus, mais les PW changent jusqu'à `69.3 µs` à l'entrée
    du fallback et `468.6 µs` au retour; auditer ce changement de branche sur
    300 RHO avant d'introduire une borne de slew. La campagne 300 valide tous
    les RHO avec un seul fallback au RHO 234, puis ACADOS reprend jusqu'au RHO
    300. Les grands sauts existent aussi avant le fallback; ils ne sont donc
    pas causés uniquement par l'adaptateur IPOPT.
17. Versionner le seed commun retenu avec son SHA et sa provenance; l'input
    inter-run actuel expire avec l'artefact et ne suffit pas à la
    reproductibilité du benchmark.
18. Réduire le coût du recovery : les sorties IPOPT faisables mais arrêtées à
    2 000 itérations coûtent à elles seules environ `280.6 s`. Tester une
    terminaison acceptable ou une Phase I de faisabilité bornée, sans relâcher
    la certification ACADOS finale.
19. En alternative contrôlée, tester une Phase I de faisabilité qui peut
    déplacer les états Ding dans une trust region stricte; auditer calcium,
    force, capacités et PW avant toute acceptation.
20. Construire ensuite un prédicteur déterministe et bon marché des projections
    utiles à partir des défauts `q/qdot`, du changement d'ensemble actif PW et
    de la distance aux bornes; mesurer faux positifs, faux négatifs et coût.
21. Implémenter ensuite le DOP853 remis à l'état certifié par RHO et le replay
    des mêmes PW en mécanique reduced.
22. En parallèle scientifique seulement, poursuivre le transfert croisé R5/R6;
    ne pas confondre cette validation de transcription avec l'ablation ACADOS.
23. [profilé, runs `31429249538` et `31435870919`] Séparer les postes online et offline. La
    boucle RHO complète coûte `0.401 s/RHO` sur le runner instrumenté, dont
    `0.203 s/RHO` dans les appels solveurs et `0.198 s/RHO` d'orchestration
    Bioptim. Le post-traitement final ne coûte que `6.62 s`; supprimer les
    audits n'est donc pas la priorité. Mesurer désormais explicitement les
    `~53.9 s` de setup pré-solve unique et réduire la création/fusion des
    objets solution dans la boucle. Le smoke 235 confirme `42.15 s` de setup
    pré-solve et `0.308 s/RHO` online, dont `0.163 s/RHO` d'orchestration.
24. Tester une régularisation de variation de PW et/ou une trust region mobile
    par rapport au cycle précédent. Comparer au mode non régularisé le coût de
    fatigue, les quatre AUC, le nombre d'itérations, les recoveries et les
    quantiles/maxima de `|PW_k-PW_{k-1}|`; ne pas imposer arbitrairement un
    slew avant cette ablation.
25. [en cours] Comparaison appariée reduced IPOPT/MadNLP/ACADOS+IPOPT sur 145
    RHO. Le producteur NLP `31441917891` est vert. Les deux premiers
    consommateurs ACADOS (`31442152939`, `31442920674`) ont exposé un couplage
    erroné entre le recalage du premier nœud et celui de toutes les bornes
    cinématiques : le path et le terminal de `omega` étaient élargis jusqu'aux
    valeurs du seed. Le correctif sépare ces opérations et possède une
    régression locale. Relancer ACADOS avec le même artefact, puis régénérer les
    figures, les quatre fatigues finales/AUC et les temps sur le préfixe commun.
    Le run `31443831558` confirme que le recalage cinématique est désormais
    désactivé, mais le seed NLP a été produit avec la boîte physique symétrique
    de `3.0 rad/s`, alors que le cas ACADOS imposait une garde nodale rapide de
    `2.55 rad/s`. Le seed est alors tronqué et le raffinement reste à
    `inf_pr=4.281`. Le workflow expose maintenant cette marge : tester d'abord
    `3.0` pour une comparaison à domaine physique identique et conserver
    l'audit dense; la garde conservatrice `2.55` restera une ablation séparée.
    Le run `31444656648` à `3.0` montre toutefois que la vraie violation
    dominante est la borne de position absolue `theta` restée dans l'ancien
    cycle (`5.090 rad` au nœud 29). Le nouveau correctif translate uniquement
    les bornes `q/theta`, fixe le premier état commun et préserve exactement la
    boîte physique `qdot/omega`. Le run `31445243270` à `3.0` résout bien
    `145/145`, mais l'audit inter-nœuds rejette un overshoot de `0.4000 rad/s`;
    le run `31445706759` à `2.55` échoue avant le premier RHO depuis le seed
    commun. Une continuation initiale automatique resserre désormais seulement
    la borne basse reduced de `omega`, sans avancer le RHO. Le run
    `31448682164` accepte `3.000 -> 2.8875 -> 2.775`, puis atteint
    `ACADOS_MINSTEP` à `2.6625`. Le run `31449332788` avec un pas effectif de
    `0.045` échoue encore à `2.730` après 100 SQP. La dernière ablation ramène
    le pas à `0.01 rad/s` et porte uniquement le budget offline de continuation
    à 300 SQP. Relancer le cas
    apparié strict. S'il passe, régénérer les quatre figures et le tableau
    fatigue/temps depuis les trois artefacts. S'il échoue, conserver les deux
    ablations comme résultat négatif et ne pas présenter le run `3.0` comme
    physiquement certifié.
    [clos négativement, run `31449892535`] Les pas de `0.01 rad/s` et 300 SQP
    offline atteignent `2.76`, puis échouent à `2.75` avec un résidu de
    contrainte `3.33e-3`. Ne plus réduire le pas. Construire une Phase I depuis
    les PW d'un seed ACADOS déjà certifié à `2.55`, tout en remplaçant son état
    initial par l'état commun; auditer les 20 états Ding avant comparaison.
26. [premier smoke analysé; second en attente] Construire le seed hybride depuis
    l'état commun du run `31441917891` et le cycle 1 du préfixe ACADOS strict du
    run `31428024125`. Le générateur préserve les 22 blocs d'état, sélectionne
    exactement 30 PW par muscle, valide le profil physique et archive les SHA.
    Sous la garde `2.55`, exécuter la préparation Ding puis la continuation à
    contrôles fixes/`1e-8`/`1e-7 s`; exiger un rayon fini accepté avant le RHO.
    Commencer par 5 RHO. Si ce smoke passe, lancer 145 RHO et régénérer la
    comparaison appariée. S'il échoue, comparer les quatre cycles sources
    `1`, `30`, `100` et `145` avant de modifier les tolérances ou la physique.
    Le run `31484139710` confirme le transfert, mais montre que le raffinement
    IPOPT était exécuté avant la Phase I complète : `inf_pr=0.309`, puis défaut
    scaled réduit à `0.0749`, et enfin `ACADOS_MINSTEP` avec résidu d'égalité
    `0.1058`. Relancer maintenant IPOPT/Radau-5 depuis ce primal amélioré avant
    la continuation ACADOS; ne pas relâcher la garde `2.55` ni les critères de
    certification.
    Le run `31485687969` reproduit exactement `inf_pr=0.309278139` après ce
    second raffinement : le warm-start n'est plus l'explication dominante.
    Exporter maintenant l'index de la contrainte finale violée et les violations
    indépendantes de `g(x)`/`x`; si elles confirment l'incompatibilité, remplacer
    la garde nodale proxy par une contrainte de vitesse aux points internes de
    l'intégrateur, plutôt que relâcher l'angle terminal absolu.
    [cause confirmée, run `31487564843`] Avec `0.35 rad` de slack, IPOPT est
    faisable (`1.96e-6`) et ACADOS résout le RHO en `0.372 s`; l'erreur finale
    vaut précisément `0.3092 rad`. Ne pas conserver ce slack. Implémenter
    ensuite soit des contraintes `omega` aux points internes IRK avec la boîte
    nodale `3.0`, soit deux sous-intervalles mécaniques par PW constante, puis
    rétablir le slack absolu `0.002 rad` et l'audit dense.
    [implémenté, validation CI en attente] La première option utilise désormais
    le prédicteur demi-pas
    `omega + dt/2*f_omega(theta, omega, F, tau_ext)` aux 30 shooting nodes,
    avec la boîte nodale et interne physique de `3.0 rad/s`. Le slack terminal
    reste `0.002 rad` et les 30 PW sont inchangées. Lancer d'abord 5 RHO; exiger
    fermeture stricte, convergence de la Phase I et audit dense, puis seulement
    étendre à 145 RHO et régénérer la comparaison appariée.
    Le run `31490977836` a validé le câblage CI et la configuration exportée,
    puis a échoué avant le solve : Bioptim transmet directement le
    `ReducedFesCyclingModel` au custom constraint, sans wrapper `.bio_model`.
    L'accès accepte maintenant les deux interfaces et possède une régression
    ciblée; relancer le même smoke.
    [smoke validé, run `31491664684`] `5/5` RHO, aucun recovery, médiane chaude
    `0.354 s` solveur et `0.366 s` murale. La trace IRK dense respecte la borne
    rapide à `1.2e-9 rad/s` près, contre `0.4000 rad/s` d'overshoot auparavant;
    erreur angulaire finale `0.00199996 rad`. Lancer 145 RHO depuis le même
    seed commun, conserver le préfixe certifié, puis comparer contrôles,
    fatigues finales/AUC et temps à IPOPT et MadNLP sur le préfixe commun.
    [fait, runs `31492324017`, `31493109875`, `31494271965`] Le run strict
    s'arrête au RHO 25 malgré deux recoveries IPOPT certifiées. Le fallback
    initial a ensuite exposé le compteur d'échecs natifs Bioptim, qui ne se
    réinitialisait pas après un RHO avancé par IPOPT. La correction conserve
    les statuts ACADOS bruts pour l'audit mais confie l'arrêt au compteur
    physique. Le run final valide `145/145`, avec quatre fallbacks aux RHO
    25, 26, 30 et 31, puis ACADOS seul jusqu'au RHO 145. Pipeline : `107.23 s`;
    audit dense valide; figures appariées régénérées.
27. Rejouer les contrôles ACADOS des RHO 1, 25, 31, 100 et 145 dans le modèle
    Radau-5, puis lancer un raffinement IPOPT depuis le primal ACADOS. Comparer
    le coût avant/après, les défauts Ding et les PW afin de séparer bassin
    local et erreur de transcription IRK/Radau.
28. Tester une petite régularisation inter-RHO uniquement sur les transitions
    d'ensemble actif des premiers 35 cycles. Exiger que les quatre AUC et le
    coût exécuté restent dans une tolérance annoncée; ne pas borner le slew
    globalement avant cette ablation.
29. [fait, run `31589712434`] Avec `+0.15 N.m` signé, donc résistif pour
    `qdot < 0`, ACADOS reduced
    certifie `660` RHO et produit un arrêt candidat de fatigue au RHO `661`.
    Les `152` appels de recovery coûtent `316.4 s`; l'orchestration restante
    coûte environ `158.8 s`. Après le RHO `450`, MINSTEP alterne presque un
    cycle sur deux : traiter d'abord le transfert, pas la fatigue.
30. [partiellement validé, run `31622939297`] Instrumenter chaque sous-étape du
    recovery et comparer un budget ACADOS adaptatif `30 + 70` avec R5 seul et
    R3 seed-only. R3 ne possède jamais le droit d'avancer un RHO; R5 reste le
    certifieur final. L'audit IRK diagnostique mesure l'écart sur tous les
    nœuds sans modifier le seed. Les tests ciblés et la suite complète passent
    localement (`351/351` plus `2/2` pour l'extracteur).
    Le replay IPOPT isolé au checkpoint 430 est négatif pour R3 : infeasibility
    `7.53e-2` avec 200 comme avec 2 000 itérations. R5 avec seulement 200
    itérations atteint d'abord `4.95e-2`, mais ne certifie pas non plus; garder
    2 000 pour le stage R5 final. La CI hybride doit accepter le rejet R3,
    journaliser ce résultat, faire recertifier par ACADOS et n'appeler R5 que
    si cette recertification échoue encore.
    Les trois cas à un cycle valident `5/5` avec exactement le même objectif,
    la même AUC, la même capacité minimale et environ `4.20 s` de solve ACADOS
    cumulé. Les seeds IPOPT forcés sont tous rejetés; R3 rejette en `5.93 s`
    contre `14.39 s` pour R5, sans contribuer à la solution finale. C'est un
    gain diagnostique de `2.43x`, pas encore un gain de recovery utile. Garder
    R3 hors du chemin de certification tant qu'un échec naturel n'établit pas
    qu'il améliore ensuite la recertification ACADOS.
31. [implémenté localement, CI à lancer] Extraire sans interpolation un
    checkpoint un ou deux cycles depuis le préfixe certifié, puis comparer au
    cycle 430 `extrapolate`, `repeat` et un horizon de deux cycles sur le même
    runner. Utiliser le mode `cycles=acados_recovery_speed` et le run
    `31589712434` comme `acados_control_seed_source_run_id`.
    Le run `31622939297` valide les trois cas à un cycle mais le cas deux cycles
    s'arrêtait avant le solve : la borne ACADOS de couture cherchait le bloc
    full `q[2]` dans le modèle reduced. Le branchement emploie maintenant
    `position_state_key/wheel_state_index`, donc `theta[0]` en reduced, avec
    une régression dédiée. Relancer uniquement cette ablation avant de conclure
    sur l'intérêt d'un horizon de deux cycles.
32. [à faire après 30--31] Si R3 réduit réellement le mur recovery sans
    dégrader la recertification IRK/ACADOS, conserver deux capsules IPOPT R3/R5
    préconstruites. Sinon supprimer R3 du chemin online et ne le garder que
    comme diagnostic de sensibilité de transcription.
33. [à faire après l'ablation de transfert] Prototyper une capsule ACADOS de
    faisabilité séparée, compilée une fois. Elle peut restaurer le primal mais
    ne doit jamais certifier l'objectif de fatigue; la capsule d'optimalité
    doit résoudre à nouveau le même RHO. Comparer ce chemin au simple
    `SQP_WITH_FEASIBLE_QP` avant d'ajouter une nouvelle fonction objectif.
34. [bloqué par un hang MadNLP; remplacé par le protocole 35] La campagne IPOPT/MadNLP R5 avec le même couple résistif signé
    `+0.15 N.m`, run `31589698184`,
    a atteint environ 779 fenêtres MadNLP puis n'a plus produit de sortie. Le
    simple `max_iter=2000` ne protège donc pas le runner contre un blocage dans
    Julia/MUMPS. Ne pas utiliser ce run incomplet pour conclure sur la fatigue.
35. [implémenté localement, CI à lancer] Tester MadNLP/R5 reduced avec le budget
    rapide `73` itérations et `20 s`, puis IPOPT/R5 seed-only après le premier
    échec et fallback certifiant après le second. Exiger : aucun hang, arrêt
    propre ou 2 000 RHO, journal natif non masqué, et temps MadNLP/recovery/
    pipeline séparés. Recalculer le percentile si le fallback dépasse `10 %`.
36. [implémenté localement, CI à lancer] Borner le DOP853 continu à 30 cycles et
    auditer localement les cycles 430, 660, 779 et le dernier cycle certifié.
    Les résumés conservent aussi les PW et les frontières de tous les états aux
    mêmes jalons pour comparer IPOPT, MadNLP et ACADOS.
37. [validé, run `31622939297`] Installer `t_renderer 0.2.0` avec
    checksum dans le cache ACADOS. Le solve ne doit plus télécharger de binaire
    à runtime. La pile mise en cache et le renderer épinglé passent la
    validation, puis quatre constructions de capsule atteignent le solve sans
    téléchargement runtime.
38. [implémenté localement, CI à lancer] Injecter de vraies limites ACADOS de
    `10`, `20` et `30` SQP aux RHO `100`, `150` et `430`, restaurer le budget
    nominal avant le retry du même RHO et poursuivre 30 cycles. Comparer sur le
    même runner à la chaîne nominale : statut, recoveries/fallbacks, temps,
    angle/cadence, PW, objectif, AUC et capacité des quatre muscles. Aucun
    primal interrompu ne peut avancer le RHO; les audits DOP853 locaux restent
    requis avant toute interprétation de la fatigue.
39. [implémenté localement, CI à lancer] Corriger le filtre des tentatives RHO
    pour `cycles_per_window > 1`. Bioptim exporte le premier cycle de chaque
    tentative puis le reliquat de la dernière fenêtre. Le run `31630846332`
    avait convergé sur ses quatre fenêtres two-cycle, avec des résidus
    dynamiques de l'ordre de $10^{-13}$, mais l'ancien filtre concaténait des
    horizons superposés et créait artificiellement un saut de force Triceps de
    `57.55 N`.
40. [CI analysée; répétition corrective préparée localement] Comparer sur 100 RHO
    reduced/SX/Radau-5 compilés la répétition des PW et leur extrapolation
    phase-par-phase avec $\alpha\in\{0.25,0.5,1\}$, séparément pour IPOPT et
    MadNLP/MUMPS. Le run `31654690299` certifie `100/100` RHO pour les quatre
    cas IPOPT. Les médianes sont proches (`2.475--2.549 s`); `alpha=1` améliore
    le P90 de `4.088` à `3.746 s`, mais ne réduit pas la médiane. Les quatre
    cas MadNLP sont invalides : le plafond rapide de `73` itérations a été
    appliqué au premier RHO. Le run correctif `31710207813` montre ensuite que
    ce premier RHO consomme `99` itérations, mais que le passage `2000 -> 73`
    reconstruit une seconde capsule C de `74 MB` et que tous les cas s'arrêtent
    au RHO 2. La prochaine répétition utilise une capsule unique à `100`
    itérations pour supprimer ce biais; le protocole d'endurance `73 + fallback
    IPOPT` reste inchangé. Si le seed PW seul est bénéfique, tester ensuite un
    rollout des états de Ding cohérent avec ces contrôles.
41. [validé, run `31740586301`] Exécuter `cycles=acados_pw_stability`
    sur 30 RHO reduced avec assistance nulle. Comparer la référence `repeat`,
    `lag2`, puis une borne terminale absolue sur `omega` de `±0.5` et
    `±0.3 rad/s`. Le résultat JSON calcule les erreurs a posteriori de
    `repeat`, `lag2` et de l'extrapolation linéaire. Retenir la borne terminale
    seulement si elle supprime l'orbite paire/impaire et rend le cycle
    précédent proche sans augmenter le coût de fatigue, les recoveries ou la
    fatigue des quatre muscles. Tester ensuite, séparément, une faible
    régularisation proximale des PW. Le run `31710221449` échoue au RHO 2 pour
    `repeat` et `lag2` à cause du rayon PW permanent de `10 us`; les variantes
    terminales échouent avant RHO 1 à cause de la troncature brutale du seed
    de `-8.98` vers la cible `-2 pi rad/s`. La correction libère le rayon PW
    après Phase I, puis resserre hors mesure la seule borne terminale de
    `omega` selon `3,2.5,2,1.5,1,0.5[,0.3]`. Les variantes terminales
    certifient toutes `30/30`; sans elles, `repeat` s'arrête à 9 et `lag2` à
    24. La marge `0.3` conserve une médiane chaude de `0.053 s` et un P90 de
    `0.054 s`. `repeat` redevient plus précis que `lag2` (`0.082` contre
    `0.123 us` en moyenne groupée). Retenir provisoirement la borne terminale
    `+/-0.3`, sans cadence constante à l'intérieur du cycle.
42. [validé négatif, run `31742437770`] Au premier RHO ACADOS non certifié, restaurer le
    checkpoint préparé, remplacer uniquement le prédicteur PW `repeat` par
    `lag2`, réinitialiser la mémoire SQP/QP et réessayer le même RHO. Exiger
    que le résumé prouve l'exécution du chemin et qu'aucun RHO non certifié
    n'avance la fatigue. Le chemin s'exécute au RHO 10, mais `lag2` change une
    PW de `468.6 us` et dégrade le résidu primal de `1.27e-3` à `9`; `MINSTEP`
    survient après une itération et le préfixe reste `9/30`. Ne pas retenir ce
    retry PW seul. Si cette piste est reprise, transférer une primale complète
    phase-alignée puis corriger fatigue et état initial par rollout/sensibilité.
43. [validé négatif, run `31743379091`] Comparer avec la borne terminale `+/-0.3` une très faible
    régularisation proximale des PW et une faible pénalité terminale de cadence.
    Les tester séparément : elles changent l'objectif. Rejeter toute variante
    qui n'améliore pas le P90 ou qui change matériellement le coût de fatigue,
    les quatre AUC ou les ensembles actifs. L'ablation emploie des poids `100`
    et `1000` sur les contrôles scaled, avec une cible recentrée sur la primale
    préparée à chaque RHO. Elle n'ajoute pas un second solve de continuation.
    Les trois cas certifient `30/30` avec 38 itérations cumulées. La référence,
    `100` et `1000` donnent respectivement des médianes/P90 muraux de
    `0.0530/0.0555`, `0.0550/0.0590` et `0.0551/0.0554 s`. La fatigue exécutée
    reste `110.598886` et les capacités finales des quatre muscles sont
    identiques aux chiffres utiles; les PW et les ensembles actifs ne changent
    pas matériellement. La proximité permanente est donc rejetée : elle
    n'apporte ni vitesse ni stabilité mesurable.
44. [à prototyper] Construire une capsule MadNLP/MUMPS reduced/SX/Radau-5 une
    seule fois avant la période online. Les états initiaux et les bornes
    terminales changent comme `lbx/ubx` à runtime; la bibliothèque C de
    l'objectif et des contraintes doit rester identique. Au premier échec
    ACADOS : MadNLP borné restaure le même RHO, ACADOS recertifie, puis IPOPT
    R5 reste le certifieur final si nécessaire. Mesurer séparément compilation,
    MadNLP, recertification et fallback; ne pas conclure à la robustesse sur la
    seule médiane.
45. [directions warm-start non testées, ordre recommandé] (a) prédicteur
    advanced-step par sensibilité KKT primal-dual aux états de fatigue et aux
    bornes; (b) hysteresis des ensembles actifs autour de `pd0`; (c) homotopie
    sur charge/fatigue au seul RHO difficile; (d) banque de seeds par régime,
    au minimum paire/impaire tant que l'orbite n'est pas éliminée; (e)
    restauration PW dans une base spline/Fourier/POD; (f) prédicteur appris
    seulement après constitution d'un jeu de solutions certifiées. Prioriser
    (a)--(c), qui conservent la structure et offrent des critères KKT audités.
    Pour IPOPT/MadNLP, commencer toutefois par le contrôle moins coûteux :
    comparer `off`, `bounds`, `constraints` et `all` pour le transfert de
    `lam_x/lam_g` entre deux RHO de la **même** transcription Radau-5. Remettre
    les multiplicateurs à zéro dès qu'une Phase I, une projection ou un
    rollout modifie la primale. Mesurer le P90, le nombre d'itérations et les
    échecs; un simple gain médian ne suffit pas. Tester ensuite le prédicteur
    KKT complet. Une extrapolation limitée aux PW est déjà insuffisante parce
    qu'elle casse la cohérence avec les états de Ding.
46. [validé, run `31744177514`] Promouvoir le candidat ACADOS reduced sans biais : IRK/SQP,
    borne terminale absolue `omega = -2*pi +/- 0.3 rad/s`, répétition des PW et
    rayon Phase-I relâché. Exécuter seul `100` RHO via
    `cycles=acados_reduced_100`; comparer au témoin 30 RHO les temps, les
    itérations, l'angle absolu, la cadence, la fatigue et les quatre capacités.
    Résultat : `100/100`, aucun recovery, 108 itérations, médiane/P90 solveur
    `0.0401/0.0405 s`, médiane/P90 de l'appel `0.0583/0.0588 s`, aucun drift
    angulaire et aucune violation mécanique. La boucle complète vaut toutefois
    `0.693 s/RHO`, dont `0.632 s/RHO` d'orchestration hors solveur.
47. [validé sur 100 RHO, runs `31746256920`, `31746803352` et
    `31747174680`] Le profil attribue l'essentiel du temps résiduel à deux
    diagnostics online redondants : un second sweep RK4 après le rollout IRK,
    puis l'impression détaillée de la primale à chaque RHO. Leur retrait ne
    change ni l'objectif ni la fatigue à environ `1e-11`, conserve `30/30`,
    38 itérations et le même audit mécanique. La boucle passe de `0.693` à
    `0.457`, puis à `0.138 s/RHO`; `update_functions` vaut maintenant
    `0.042 s/RHO`. Les résidus ACADOS et l'audit final restent calculés et
    sérialisés. Le gate final certifie `100/100`, sans recovery, en 108
    itérations : `0.121 s/RHO` pour la boucle après construction et
    `0.0531/0.0535 s` pour la médiane/P90 de l'appel solveur. Conserver
    `--acados-diagnostics` uniquement pour les runs causaux.
48. [prêt à exécuter sur le Ryzen 9 5950X] Le protocole
    `ryzen5950x_endurance.md` et le lanceur
    `.github/scripts/run_ryzen5950x_endurance_sweep.sh` comparent reduced
    Radau-5 IPOPT compilé, MadNLP/MUMPS compilé avec fallback IPOPT et ACADOS
    SQP/IRK avec fallback IPOPT, pour les résistances signées `+0.10`, `+0.15`
    et `+0.20 N.m`, jusqu'à 2 000 RHO ou deux échecs. Calibrer d'abord 16
    contre 30 threads sur 30 RHO et trois répétitions; la baseline reste 16
    cœurs physiques, sans parallélisme BLAS/OpenMP imbriqué. Après exécution,
    distinguer obligatoirement `fatigue_limited_candidate` de
    `unconfirmed_endurance_stop` et les succès natifs des fallbacks.
49. [audit macOS et Linux négatif] Ne pas interpréter
    `madnlp_dual_warm_start: applied=True` comme un warm-start dual effectif.
    La branche Bioptim copie bien `lam_g/lam_x` vers `lam_g0/lam_x0`, mais le
    runtime CasADi 3.8.0/libMad macOS testé produit exactement le même premier
    pas avec des multiplicateurs nuls ou de norme `1e6`, même avec
    `dual_initialized=true`. Le test Linux CasADi `3.7.2+` du run
    `31765363296` confirme `dual_inputs_consumed=false` avec un écart maximal
    de `8.87e-11`. Ajouter au runtime libMad une vraie
    initialisation primal-dual, puis exiger un test discriminant `max_iter=1`
    avant l'ablation `off/constraints/bounds/all` sur le RHO. Tant que ce test
    échoue, conserver MadNLP en warm-start primal seulement.
50. [prototype implémenté, campagne ciblée prête] Une hystérésis de Schmitt autour
    de `pd0`, avec `delta_off=2 us` et `delta_on=5 us`, ne change que 5 valeurs
    de classification sur 12 000 PW IPOPT et 2 sur 12 000 PW MadNLP dans les
    trajectoires 100 RHO disponibles. Elle ne supprime pas les vraies
    transitions de recrutement, dont les sauts atteignent `468.6 us`.
    L'implémenter d'abord uniquement dans la construction de l'ensemble actif
    **prévu** : un nœud inactif n'est libéré qu'au-dessus de
    `pd0+delta_on`, un nœud actif n'est marqué inactif qu'en dessous de
    `pd0+delta_off`. Ne jamais imposer ce verrou au solve final : relâcher les
    bornes temporaires et certifier le NLP original. La campagne
    `cycles=acados_active_set` compare désormais référence et hystérésis sur
    une machine; commencer à couple nul, car le bridge reduced à
    `signed:+0.15` échoue encore avant ACADOS. Journaliser transitions, faux
    verrous, coût, fatigue et P90.
51. [noyau et second membre bornes testés, branchement OCP à faire] Prototyper le prédicteur paramétrique
    KKT sur le reduced/SX/Radau-5 avant de l'appliquer à ACADOS ou MadNLP :
    (a) rendre explicite le vecteur `p` regroupant état initial et bornes
    terminales; (b) extraire Hessienne de Lagrange, Jacobienne et ensemble
    actif du dernier RHO certifié; (c) résoudre un seul système KKT pour
    prédire la primale **et** les duaux; (d) appliquer trust region,
    fraction-to-boundary et rollout; (e) rejeter le prédicteur si son résidu
    initial dépasse celui de `repeat`; (f) comparer sur 30 puis 100 RHO les
    résidus initiaux, itérations, médiane/P90, recoveries, coût et quatre AUC.
    L'hystérésis du point 50 protège les changements d'ensemble actif; elle ne
    remplace pas le prédicteur. Exploiter l'intervalle d'un cycle pour calculer
    le RHO suivant en advanced-step. Le code sait maintenant convertir les
    déplacements `lbx/ubx/lbg/ubg` des lignes actives en second membre
    `J_A Delta z = Delta b_A`; il reste à construire Hessienne/Jacobienne
    sparse depuis l'interface Bioptim et à comparer son résidu à `repeat`.

La question causale est maintenant resserrée : une seule projection au RHO 19
ne suffit pas, mais la séquence 19--36 conserve le bassin franchissant le RHO
81. Il reste à déterminer la plus petite borne supérieure robuste.
