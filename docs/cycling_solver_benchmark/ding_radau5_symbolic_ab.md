# Condensation symbolique stricte Ding/Radau-5 : A/B IPOPT/MA57 reduced

L'adaptateur expérimental résout maintenant l'OCP de cyclage reduced réel avec
condensation des stages Ding. Sur le pilote du 12 septembre 2026, les deux
variantes réussissent 10/10 fenêtres RHO, mais **la condensation ne donne pas
d'accélération** : médiane chaude 1,022 s contre 0,947 s pour la référence.

Cette expérience prolonge les
[développements mathématiques de la réduction Ding](ding_analytic_reduction_radau5.md).
Elle conserve la collocation Radau à cinq stages et IPOPT/MA57 ; elle n'utilise
ni multiple shooting, ni intégration exponentielle substituée à Radau.

## Ce qui est effectivement éliminé

L'OCP original a 22 états physiques, quatre muscles et 30 intervalles par
fenêtre. Les cinq stages internes portent, par muscle, `Cn, F, A, Tau1, Km`.
Les états de début et de fin d'intervalle sont conservés intégralement.

L'adaptateur élimine les cinq stages de `Cn`, `A`, `Tau1` et `Km` pour chaque
muscle. Il reste six variables d'état par stage : les quatre forces et les
deux états mécaniques. Cette condensation maximale est plus agressive que la
variante conservant `F` et `A` aux stages, qui n'a pas été mesurée ici.

En séparant le vecteur décisionnel en variables conservées \(y\) et stages
éliminés \(z\), les équations sélectionnées s'écrivent

\[
G_E(y,z)=Mz+b(y)=0,\qquad z=\psi(y)=-M^{-1}b(y).
\]

Le graphe SX est extrait après préparation des pénalités Bioptim. L'adaptateur
vérifie que les lignes sélectionnées sont des égalités nulles, affines dans
le vecteur complet, et que les coefficients de \(M\) sont constants. Il
vérifie ensuite les blocs inversibles de taille cinq. Chaque reconstruction
emploie exactement les coefficients du graphe Radau original : aucune
équation Ding n'est réécrite ou approximée à cette étape.

Pour toutes les fonctions restantes, y compris objectifs et contraintes
mécaniques, la substitution est

\[
\widetilde f(y)=f(y,\psi(y)),\qquad
\widetilde G_R(y)=G_R(y,\psi(y)).
\]

Les bornes physiques des variables éliminées deviennent explicitement

\[
\ell_z\le\psi(y)\le u_z.
\]

Ainsi, le nombre de variables passe de **4 103 à 1 703**. Le nombre de lignes
de contraintes reste **3 960** : 2 400 égalités de stage sont retirées et
2 400 contraintes de bornes reconstruites sont ajoutées. La taille réduite
du vecteur décisionnel ne garantit donc pas une réduction équivalente du
système traité par la méthode de point intérieur.

CasADi différentie les fonctions substituées. Comme la reconstruction est
affine pour le cas testé, si \(P=\partial(y,\psi(y))/\partial y\),

\[
\nabla_y\widetilde f=P^T\nabla_xf,\qquad
\nabla_y^2\widetilde{\mathcal L}=P^T\nabla_x^2\mathcal L P.
\]

Les multiplicateurs des bornes éliminées sont transférés aux nouvelles
contraintes. Les multiplicateurs des égalités retirées sont reconstruits par
la stationnarité dans les directions éliminées :

\[
M^T\lambda_E=-\left(\nabla_zf+
                  J_{R,z}^T\lambda_R+\lambda_z\right).
\]

Bioptim reçoit donc à nouveau les états, contraintes et multiplicateurs du
problème original, avec leurs dimensions originales. Son transfert RHO reste
utilisable. Le pilote utilise le warm-start de bornes configuré par défaut.

## Protocole et résultats

Les exécutions sont séquentielles, avec un thread numérique et les mêmes
options : dynamique reduced, couple résistif +0,1 Nm, 30 stimulations par
cycle, une fenêtre d'un cycle, SX, Radau-5, IPOPT/MA57, scaling `full`,
tolérance NLP \(10^{-6}\), seuil primal \(10^{-5}\). Les deux variantes
reçoivent la même seed physique commune, avec adoption de son cycle de
warm-up. Le contrôle de vitesse aux stages internes est désactivé dans les
deux variantes, comme dans ce protocole de référence ; l'adaptateur conserve
toutes les contraintes effectivement présentes.

La référence est relancée avec le code courant. Les temps historiques IPOPT
d'autres séances ne sont pas utilisés comme dénominateur de l'A/B.

| Mesure, pilote 10 RHO | Référence Radau-5 | Condensé Radau-5 |
|---|---:|---:|
| Fenêtres réussies et physiquement validées | 10/10 | 10/10 |
| Construction solveur CasADi/IPOPT | 4,494 s | 5,478 s |
| Construction condensation supplémentaire | — | 1,384 s |
| Première résolution | 10,797 s | 26,675 s |
| Itérations première résolution | 554 | 1 133 |
| Médiane chaude, fenêtres 1 à 9 | 0,947 s | 1,022 s |
| p90 chaud | 1,040 s | 1,141 s |
| Itérations chaudes moyennes | 52,78 | 55,67 |
| Total solveur, dix fenêtres | 19,183 s | 35,839 s |
| Hessienne, temps moyen par évaluation | 5,931 ms | 7,611 ms |
| Jacobienne, temps moyen par évaluation | 2,388 ms | 2,485 ms |
| Violation maximale, contraintes/bornes originales | 5,58e-7 | 6,86e-7 |

La préparation de la seed et la construction OCP commune précèdent ces
mesures. Les temps solveur sont les `t_wall_total` natifs, hors reconstruction,
export et audits Python. L'audit initial du pilote matérialisait une
Jacobienne dense et ajoutait environ 0,26 s par fenêtre condensée au temps
extérieur ; le script inspecte désormais ses seuls non-zéros. Cette correction
d'instrumentation ne change pas les temps natifs ni le problème optimisé
archivés dans le tableau.

## Équivalence et limites de la comparaison

Le smoke indépendant d'une fenêtre retrouve la même solution : écart maximal
du vecteur décisionnel complet mis à l'échelle **1,35e-10**, écart d'objectif
**1,58e-14**, seed originale identique bit à bit. Les défauts et les
sensibilités des équations éliminées sont audités avant chaque résolution.
Douze tests vérifient la condensation numérique de Ding, les Jacobiennes et
la Hessienne du Lagrangien après substitution, ainsi que le maintien des
contraintes de chemin et des bornes.

Sur les dix fenêtres fermées, les trajectoires ne restent pas identiques :
écart maximal du vecteur décisionnel mis à l'échelle 0,232 ; objectifs
cumulés 8,636724 pour la référence et 8,601863 pour le condensé (environ
−0,40 %). Les états transférés divergent donc progressivement. Un problème
non convexe équivalent peut conduire IPOPT à d'autres solutions lorsque la
représentation, les barrières et les sensibilités numériques changent. Ce
pilote démontre la faisabilité et l'équivalence algébrique ; il ne constitue
pas un benchmark chaud à trajectoire imposée identique.

Le constat de performance est négatif pour cette condensation maximale. Les
évaluations de Hessienne coûtent environ 28 % de plus et la phase froide
demande deux fois plus d'itérations. L'élimination introduit des dépendances
entre stages dans les expressions reconstruites ; conserver une structure
locale plus creuse, par exemple avec `F,A` aux stages, est une autre variante
à mesurer, sans gain actuellement démontré. Un rejeu de fenêtres figées avec
mêmes bornes, états entrants et warm-start serait requis pour une comparaison
chaude plus contrôlée.

## Reproduction et fichiers

```bash
bash scripts/run_ding_radau5_symbolic_ab.sh 10
source .github/scripts/benchmark_env.sh rho32
python scripts/summarize_ding_radau5_symbolic_ab.py \
  ding-radau5-symbolic-ab-20260912 --windows 10 \
  --output ding-radau5-symbolic-ab-20260912/comparison10.json
python -m pytest tests/test_ding_radau5_symbolic_ab.py \
  tests/test_ding_radau5_condensation.py -q
```

Le lanceur accepte ensuite un répertoire de sortie et un chemin de seed
optionnels. Le chemin de seed par défaut est celui de cette machine. Les
scripts et tests ajoutés sont isolés : aucun fichier applicatif existant ni
fichier Bioptim n'a été modifié.

- [Adaptateur symbolique](../../scripts/benchmark_ding_radau5_symbolic_ab.py).
- [Lanceur séquentiel](../../scripts/run_ding_radau5_symbolic_ab.sh).
- [Synthèse 1 RHO](../../ding-radau5-symbolic-ab-20260912/comparison1.json).
- [Synthèse 10 RHO](../../ding-radau5-symbolic-ab-20260912/comparison10.json).
- [Référence 10 RHO](../../ding-radau5-symbolic-ab-20260912/baseline10/result.json).
- [Condensé 10 RHO](../../ding-radau5-symbolic-ab-20260912/condensed10/result.json).

La prise en charge actuelle est volontairement limitée à SX, Radau à cinq
stages, coefficients d'élimination constants et intervalles fixes. Un temps
de phase optimisé, un autre intégrateur, la compilation C ou des callbacks
solveur alternatifs nécessitent une adaptation et une validation explicites.
