# Fonctions objectif RHO — première validation à 0,73 Nm

État au 8 octobre 2026. Tous les termes sont formulés pour un RHO à un cycle,
30 stimulations/cycle, isocinétique prescrit, réduction mécanique/Ding-local,
IPOPT-MA57. Les arrêts IPOPT ne sont pas assimilés à une impossibilité
physiologique : seul un témoin de faisabilité du prochain cycle, à état gelé,
peut confirmer que la tâche demeure réalisable.

## Notation et coûts

Pour le muscle `i`, `a_i = A_i/A_scale,i` est sa capacité normalisée et
`phi_i = 1-a_i` sa fatigue normalisée. Pour la stimulation `k`,
`p_i,k = PW_i,k/PWmax_i`. `W_req` est le travail requis par cycle.

### Fatigue intégrée — référence

\[
J_F = \frac{1}{T M}\sum_i\int_0^T \phi_i(t)^2\,dt.
\]

Le facteur `1/(T M)` ne change pas l’optimum seul, mais est utile pour
comparer des poids entre variantes. Le benchmark historique peut conserver son
échelle Bioptim `10000`.

### Dose de stimulation

\[
J_{PW}=\frac{1}{M K}\sum_{i,k}p_{i,k}^2.
\]

Le terme `control` existant emploie la PW normalisée et, avec l’objectif
quadratique, minimise cette dose. Dans la formulation isocinétique à travail
par cycle fixé, une « efficacité travail/dose » naïve partage une grande partie
de l’information avec `J_PW`; il ne faut pas la traiter comme un terme
indépendant avant d’avoir vérifié sa corrélation empirique.

### Réserve musculaire minimale au terminal

\[
R_\tau(\mathbf a_T)=-\tau\log\left(\frac1M\sum_i e^{-a_i(T)/\tau}\right),
\qquad J_R=1-R_\tau.
\]

Le RHO peut minimiser `J_F + lambda_R J_R`. `J_R` protège le muscle ayant la
plus faible réserve normalisée; à température finie, il s’agit d’une
approximation lisse du minimum.

### Équilibrage de fatigue (nouveau terme implémenté)

\[
\bar\phi_T=\frac1M\sum_i\phi_i(T),\qquad
J_B=\frac1M\sum_i(\phi_i(T)-\bar\phi_T)^2,
\qquad J=J_F+\lambda_B J_B.
\]

`J_B` vaut zéro si les muscles ont la même fatigue normalisée. Il n’essaie pas
de réduire la fatigue totale à lui seul : le terme de base `J_F` ou un autre
objectif de tâche doit rester actif. L’option expérimentale
`--fatigue-balance-weight` règle `lambda_B`; zéro omet complètement le terme
et conserve le graphe historique.

### Variation des PW

\[
J_S=\frac{1}{M(K-1)}\sum_{i,k}
\left(\frac{PW_{i,k+1}-PW_{i,k}}{PW_{i,\max}}\right)^2.
\]

Le mécanisme de slew actuellement disponible ajoute aussi une borne dure sur
`Delta PW`. Il change donc l’ensemble admissible, pas uniquement la préférence
de coût; toute comparaison doit rapporter les deux paramètres.

### Marge de travail de la tâche

Soit `W_max(x_c)` le travail positif que l’état musculaire `x_c` pourrait
produire avec les stimulations admissibles, et
`m_c=(W_max(x_c)-W_req)/W_req`. Un coût lisse prospectif possible est

\[
J_M=\sum_c\operatorname{softplus}
\left(\frac{m_{\mathrm{cible}}-m_c}{s}\right)^2.
\]

Le RHO ne devrait pas optimiser ce terme avant que `W_max` soit validé avec les
bornes de PW, l’historique de stimulation et une intégration assez précise.
Les branches de réserve mécanique existantes sont expérimentales et ne sont
pas encore comparées ici jusqu’à une faisabilité gelée.

## Validation logicielle et runs

- `fatigue_imbalance_penalty` NumPy/CasADi : invariance par permutation,
  nullité en cas de fatigue normalisée égale et gradient comparé aux
  différences finies.
- `tests/test_muscle_reserve.py`: **46 tests réussis**.
- Smoke RHO `lambda_B=0.2`: **3/3 cycles convergés**; le résultat rapporte
  `fatigue_balance_weight=0.2` et poids effectif `2000`, moyenne IPOPT
  `0.140 s/cycle`, temps total par fenêtre `0.149 s`.
- Long RHO `lambda_B=0.2`: le checkpoint enregistre
  `producer_completed_windows=844`. Le solve suivant n’a pas produit de
  résultat final après plusieurs minutes; le processus a été interrompu. Ce
  nombre est donc un préfixe certifié, **pas** un seuil d’épuisement.
- Runs longs terminés par échec du solveur au cycle suivant, avec préfixes
  certifiés de **1 022** (`PW seule`), **1 046** (`fatigue + PW`) et **900**
  (`fatigue + réserve`, poids `0.1`). Le témoin de faisabilité mécanique et
  l'audit isocinétique passent sur les fenêtres certifiées, mais aucun de ces
  reçus ne contient le test de déficit de travail à état gelé : ces nombres ne
  sont donc pas des cycles jusqu'à l'échec physiologique. Temps IPOPT moyen par
  fenêtre : respectivement **0,062**, **0,063** et **0,168 s**; temps total par
  fenêtre **0,096**, **0,097** et **0,198 s**. Le ralentissement marqué du
  terme réserve est un résultat à prendre au sérieux avant d'envisager son
  emploi clinique.
- Le RHO long associé à l'équilibrage terminal (`lambda_B=0.2`) a un
  checkpoint à **844 fenêtres terminées**, mais pas de reçu final; le solve
  suivant est resté longtemps sans progrès et a été interrompu. Les dernières
  lignes de journal visibles sont décalées par rapport au checkpoint (jusqu'à
  828), donc le checkpoint est la borne fiable du préfixe. Le smoke test
  correspondant, lui, a convergé sur **3/3** fenêtres. Ni l'un ni l'autre ne
  certifie l'épuisement.
- Référence archivée fatigue seule : 845 cycles rapportés; son arrêt est
  étiqueté `unconfirmed_endurance_stop`.

Les journaux et checkpoints détaillés se trouvent sous
`local-results/rho-objective-study-0p73-20261008/`.

## Limite scientifique et prochaine validation

Un arrêt ou un solveur IPOPT qui ne termine pas ne prouve pas l’infaisabilité.
Il faut, sur le checkpoint du dernier cycle certifié, figer tous les états de
Ding, l’état mécanique et l’historique de stimulation, puis rechercher un PW
admissible en minimisant explicitement le déficit de travail normalisé. Un
témoin de travail nominal prouve que le cycle suivant reste faisable; un
déficit local positif reste un diagnostic, pas une preuve globale d’échec.
Le runner courant sauvegarde bien le checkpoint, mais le probe générique de
déficit n’est pas encore branché à ce protocole unilatéral; l’endurance ne doit
pas être comparée comme une mesure physiologique définitive avant ce câblage.
