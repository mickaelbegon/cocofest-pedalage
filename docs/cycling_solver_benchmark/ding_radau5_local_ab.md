# Réduction locale Ding, Radau-5 et IPOPT/MA57 reduced

Le pilote du 13 septembre 2026 donne un **premier signal positif modéré** :
la médiane chaude passe de **1,0207 s à 0,8415 s**, soit **−17,6 %**, avec
10/10 fenêtres valides dans chaque variante. Ce gain n'est pas encore certifié
à trajectoire identique : les deux RHO fermés divergent. La production reste
inchangée.

Cette expérience teste la variante locale ouverte dans
[la revue du prompt](prompt_acceleration_ding_review.md), après le résultat
négatif de [la condensation maximale](ding_radau5_symbolic_ab.md).
L'adaptateur conserve `F,A` à tous les nœuds et stages, et les deux états
mécaniques reduced. Il élimine `Cn,Tau1,Km`, y compris aux nœuds, après fixation
des états entrants. Il reste dix états décisionnels par nœud/stage au lieu de
vingt-deux.

## Équivalence discrète : conserver les offsets Radau

Les états lents d'un muscle vérifient

\[
\dot y=-(y-y_r)/\tau_f+\alpha_yF,
\qquad y\in\{A,\tau_1,K_m\}.
\]

La combinaison \(d_y=y-c_yA\), avec \(c_y=\alpha_y/\alpha_A\), satisfait

\[
\dot d_y=-(d_y-d_{y,r})/\tau_f,
\qquad d_{y,r}=y_r-c_yA_r.
\]

La solution continue contient une exponentielle. Sa substitution directe
dans une collocation Radau modifierait légèrement le NLP discret. Le prototype
utilise donc **l'opérateur Radau original**. Si \(\mathcal A\) est le tableau
à cinq stages, les offsets de stage sont obtenus par

\[
\left(I+\frac h{\tau_f}\mathcal A\right)d
=\mathbf1 d_0+\frac h{\tau_f}\mathcal A\mathbf1 d_{y,r}.
\]

Le dernier stage fournit l'offset au nœud suivant. La reconstruction reste
locale dans les décisions : \(y_j=c_yA_j+d_j\).

L'implémentation extrait le système affine original de `Cn,A,Tau1,Km`, ajoute
les conditions initiales fixes, puis le résout numériquement à force nulle.
Si ses solutions sont \(\bar A,\bar y\), elle calcule
\(d=\bar y-c_y\bar A\). Le forçage commun par la force s'annule exactement.
Le système creux est factorisé à la construction ; seul son second membre
initial change par RHO. `Cn` est calculé avec ce même système, son forçage
`periodic_node` étant numérique avec calendrier et paramètres fixés.

Les facteurs d'échelle des états sont inclus dans \(c_y\). Un certificat
vérifie les coefficients de l'identité

\[
G_{\mathrm{supprimé}}(Pz+d)=B\,G_A(Pz+d),
\]

puis son terme constant pour les offsets de chaque fenêtre. Les défauts
supprimés découlent donc des défauts `A` conservés ; ils ne doivent pas être
nuls à un itéré arbitraire infaisable. Le résidu maximal du certificat est
**7,04e−15** sur dix fenêtres. Les états et contraintes complets sont ensuite
reconstruits et évalués dans **le NLP original** après chaque solve.

## Les bornes restent locales

Pour chaque borne \(\ell_y\le c_yA_j+d_j\le u_y\), l'adaptateur calcule
les extrémités \((\ell_y-d_j)/c_y\) et \((u_y-d_j)/c_y\), prend leur
minimum/maximum selon le signe, puis intersecte avec les bornes de `A`.
Une intersection vide déclenche une erreur. Pour `Cn`, le coefficient est
nul : sa valeur numérique est contrôlée directement.

Il n'y a **aucune contrainte de borne reconstruite supplémentaire**.
Les multiplicateurs des bornes effectives sont réattribués à la borne physique
qui les a engendrées ; les duales d'égalités éliminées sont reconstruites
par la stationnarité. Le transfert RHO Bioptim retrouve les dimensions
originales et conserve son warm-start de bornes standard.

## Protocole et mesures

Baseline relancée avec le code courant, puis variante locale, séquentiellement :
IPOPT/MA57, SX, `periodic_node`, dynamique reduced, Radau cinq stages,
30 stimulations par cycle, une fenêtre d'un cycle, couple +0,1 Nm,
un thread numérique, scaling `full`, tolérance NLP `1e-6`, gate primal `1e-5`.
Même seed commune et adoption de son cycle de warm-up. Comme l'ancien A/B,
ce protocole désactive la borne de vitesse aux stages internes. Les résultats
ne certifient donc pas les configurations où cette contrainte est activée.

| Mesure, dix RHO | Baseline | Réduction locale |
|---|---:|---:|
| Fenêtres réussies / physiquement valides | 10/10 | 10/10 |
| Variables | 4 103 | 1 931 |
| Contraintes | 3 960 | 1 800 |
| Non-zéros Jacobienne | 28 020 | 13 500 |
| Non-zéros Hessienne complète du Lagrangien | 30 244 | 11 044 |
| Médiane chaude, fenêtres 1–9 | 1,020674 s | 0,841479 s |
| p90 chaud | 1,120803 s | 0,916302 s |
| Itérations chaudes moyennes | 52,78 | 42,56 |
| Temps chaud par itération, ratio des sommes | 19,251 ms | 20,189 ms |
| Hessienne chaude par évaluation | 6,360 ms | 9,323 ms |
| Jacobienne chaude par évaluation | 2,596 ms | 3,469 ms |
| Évaluations NLP chaudes cumulées | 5,008 s | 5,675 s |
| Temps chaud hors évaluations cumulé | 4,136 s | 2,058 s |
| Première résolution | 11,731 s | 12,248 s |
| Itérations première résolution | 554 | 528 |
| Violation maximale, bornes/contraintes originales | 5,58e−7 | 6,32e−7 |
| Objectif cumulé des fenêtres | 8,636724 | 8,560113 |

La Hessienne complète est comptée symétriquement dans les deux colonnes.
Ce `nnz` n'est pas celui des facteurs MA57. « Hors évaluations » englobe
assemblage, factorisation, recherche linéaire et autres tâches IPOPT ;
il ne mesure pas MA57 isolément.

Les itérations chaudes sont respectivement `68,54,56,54,52,54,49,43,45` et
`55,43,43,42,40,40,39,41,40`. Le gain associe une baisse de 19,4 % des itérations
et du temps hors évaluations. **L'évaluation des dérivées n'est pas accélérée
dans ce pilote**, malgré la réduction de leur `nnz`.

La construction de la réduction et de son certificat prend 2,69 s ; celle
du solveur 5,42 s contre 5,25 s. Offsets, fusion de bornes et certificat
ajoutent une médiane de 4,39 ms par fenêtre. Les temps natifs du tableau
excluent ces audits et la reconstruction/export Python.

## Limites et suite utile

Au premier RHO, seed et bornes originales sont identiques bit à bit, mais
les solutions ne le sont pas : écart maximal de décision échelonnée
**0,00389**, objectif local inférieur de **1,09 %**. La réduction change la
représentation de la barrière et le chemin numérique d'IPOPT dans ce problème
non convexe, même si le problème faisable est équivalent.

Les états entrants divergent ensuite : écart maximal du vecteur complet
échelonné **0,2358**, objectif cumulé **−0,887 %**. Après le premier transfert,
bornes et seed ne sont plus identiques. Ce pilote mesure deux contrôleurs
fermés, **pas neuf problèmes chauds figés identiques**.

Les temps varient aussi entre lancements : le smoke local précédent prenait
8,38 s à froid et environ 6,39 ms par Hessienne, contre 12,25 s et 9,12 ms
lors du pilote, avec les mêmes 528 itérations et la même solution initiale.
L'ajout du certificat de validation a précédé le second lancement. Les rôles
de l'instrumentation, de la mémoire et de la charge machine restent à isoler ;
on ne doit pas attribuer cette variation à la seule formulation.

La direction mérite maintenant **un rejeu figé de fenêtres chaudes**, avec
états entrants, seed, multiplicateurs et tolérances partagés, puis des
répétitions alternées pour mesurer la variabilité. Si le gain persiste,
tester les bornes internes de vitesse et un horizon plus long. Ce pilote
n'autorise pas une bascule de production ni une extrapolation à FHO,
FATROP, ACADOS ou à un calendrier variable.

## Rejeu gelé des dix fenêtres (13 septembre 2026)

Le premier A/B mélangeait le coût de résolution et l'effet contrôleur : une
solution locale différente modifiait l'état entrant de la fenêtre suivante.
Un second harnais conserve donc les archives originales de la baseline
(`x0`, bornes, multiplicateurs) et les injecte à chaque appel. Pour que la
simulation RHO garde la même continuité mécanique, elle reçoit **seulement
pour son transfert externe** la solution baseline archivée. Les chronomètres,
les itérations, les solutions auditées et les violations restent ceux du
solveur effectivement exécuté. C'est donc un benchmark de coût NLP à entrée
identique, et non une nouvelle comparaison de deux contrôleurs.

Les dix appels ont abouti dans les deux variantes. Les neuf appels chauds
sont les mêmes données initiales, bornes et multiplicateurs, avec la même
propagation RHO baseline et le même environnement à un thread.

| Mesure, neuf appels chauds gelés | Baseline | Réduction locale | Écart |
|---|---:|---:|---:|
| Médiane du temps natif IPOPT | 1,131652 s | 0,667686 s | **−41,0 %** |
| p90 du temps natif IPOPT | 1,235541 s | 0,822103 s | −33,5 % |
| Somme des temps | 10,072906 s | 6,284075 s | −37,6 % |
| Itérations moyennes | 52,78 | 43,67 | −17,3 % |
| Succès | 10/10 | 10/10 | — |
| Violation originale maximale | 5,58e−7 | 9,93e−7 | sous le gate 1e−5 |
| Objectif candidat cumulé | 8,636724 | 8,645864 | +0,106 % |

La trajectoire externe, son objectif exécuté (`8,636724`) et ses dix cycles
sont exactement ceux de la baseline par construction. L'objectif « candidat »
du tableau est en revanche recalculé dans le NLP original pour chaque solution
renvoyée par la variante locale : sa petite différence confirme que l'algorithme
peut suivre un autre minimum local, tout en satisfaisant le problème original.
Le certificat discret reste inférieur à `5e−15` dans la variante locale.

Ce résultat renforce le signal de performance : même en retirant la dérive
des états entrants, le gain dépasse 30 % au p90. Il ne certifie pas encore une
équivalence de solution ni la variabilité interlancement. La prochaine mesure
utile est de répéter ce rejeu en ordre alterné baseline/local, puis de valider
la solution locale réellement propagée sur un horizon long avant intégration.

### Réplication alternée

Une seconde paire a été exécutée immédiatement en ordre inverse (`local-v4`,
puis `baseline-v4`). La machine était globalement plus rapide : il faut donc
comparer **à l'intérieur de chaque paire**, jamais les valeurs absolues entre
les paires. Les itérations sont déterministes dans chaque formulation (52,78
baseline et 43,67 locale dans les deux répétitions).

| Paire gelée | Ordre | Médiane baseline | Médiane locale | Gain médian | Gain sur somme chaude |
|---|---|---:|---:|---:|---:|
| P1 (`v3`) | baseline → locale | 1,131652 s | 0,667686 s | **−41,00 %** | −37,61 % |
| P2 (`v4`) | locale → baseline | 0,986948 s | 0,487390 s | **−50,62 %** | −46,97 % |

Les deux ordres confirment donc un gain de l'ordre de 40–50 % sur ces fenêtres
figées. Cette réplication réduit fortement l'hypothèse d'un effet de cache ou
d'ordre, mais deux paires ne constituent pas encore une caractérisation
statistique complète de la variabilité machine.

## Contrôleur RHO réellement propagé : 50 fenêtres

La solution locale a ensuite été transmise à la fenêtre suivante (aucune
archive baseline et aucun rejeu gelé), avec la même configuration et un run
baseline exécuté immédiatement après. Les deux contrôleurs ont validé les
**50/50** cycles, sans échec de solveur ni de faisabilité physique.

| Mesure, fenêtres 1–49 | Baseline | Réduction locale | Écart |
|---|---:|---:|---:|
| Médiane chaude | 0,798194 s | 0,551213 s | **−30,94 %** |
| p90 chaud | 1,008795 s | 0,621216 s | −38,42 % |
| Somme chaude | 41,015763 s | 27,424142 s | −33,14 % |
| Itérations chaudes moyennes | 45,59 | 40,10 | −12,04 % |
| Violation originale maximale | 9,85e−7 | 8,56e−7 | sous le gate 1e−5 |
| Objectif cumulé candidat | 1035,878491 | 1029,996329 | −0,568 % |

La force, la fatigue et la trajectoire ne sont pas identiques : le problème
reste non convexe et les deux formulations suivent des minima locaux proches.
Mais la variante locale est stable sur cet horizon, conserve la faisabilité,
et le gain reste supérieur à 30 % dans le contrôleur réellement employé. Elle
devient donc la candidate prioritaire pour une intégration expérimentale,
après tests de scénarios (couple, fréquence, fatigue initiale) et bornes de
vitesse internes réactivées.

## Reproduction

```bash
bash scripts/run_ding_radau5_local_ab.sh 10 baseline
bash scripts/run_ding_radau5_local_ab.sh 10 local
source .github/scripts/benchmark_env.sh rho32
python scripts/summarize_ding_radau5_local_ab.py ding-local-radau5-ab-20260913
python -m pytest tests/test_ding_radau5_local_ab.py -q
```

Rejeu gelé, à partir des entrées de la baseline à dix fenêtres :

```bash
bash scripts/run_ding_radau5_local_frozen_ab.sh baseline \
  ding-local-radau5-ab-20260913/baseline10/symbolic-audit \
  ding-local-radau5-frozen-20260913/baseline-v3 10
bash scripts/run_ding_radau5_local_frozen_ab.sh local \
  ding-local-radau5-ab-20260913/baseline10/symbolic-audit \
  ding-local-radau5-frozen-20260913/local-v3 10
```

Trois tests couvrent les intersections signées/constantes de bornes, leur
infaisabilité et la reconstruction Radau avec offsets entrants arbitraires.
Le certificat algébrique est aussi évalué dans chaque solve réel.

- [Adaptateur](../../scripts/benchmark_ding_radau5_local_ab.py)
- [Lanceur](../../scripts/run_ding_radau5_local_ab.sh)
- [Synthèse](../../scripts/summarize_ding_radau5_local_ab.py)
- [Données comparatives](../../ding-local-radau5-ab-20260913/comparison10.json)
- [Baseline](../../ding-local-radau5-ab-20260913/baseline10/result.json)
- [Locale](../../ding-local-radau5-ab-20260913/local10/result.json)

Tous les ajouts sont expérimentaux et isolés. Aucun fichier applicatif ou
fichier Bioptim préexistant n'a été modifié pour cette expérience.
