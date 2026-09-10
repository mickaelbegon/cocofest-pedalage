# Validation des poids de l'annexe et sensibilité élargie

## Résultat principal

Le calcul reproduit les facteurs intermédiaires et conserve les poids min–max
de la figure S4 pour audit. Pour le contrôleur, chaque cas reçoit désormais
des poids bruts divisés par le plus grand poids brut, sans donnée FHO. Sur 69 cas construits
à partir des paramètres publiés, 52 restent admissibles pour la calibration
à force prescrite. Neuf de ces cas donnent un poids triceps strictement positif,
sans régularisation ni plancher. Les 69 cas construits à partir des paramètres
actuels du dépôt sortent tous du domaine de cette calibration.

**Ces résultats ne sont pas une comparaison d'endurance des contrôleurs.**
Ils valident la reproduction de la formule et identifient ses sensibilités
et ses limites. Aucun paramètre ni coût du RHO public n'a été modifié.

## Sources et reproduction

Référence : Supplementary Material 1.docx fourni, tableaux S3/S4/S6 et
figure S4, et [script publié au commit 31e064f4](https://github.com/pyomeca/cocofest/blob/31e064f4d80741c8b5444e6ce4d91bfe4eb78c3d/examples/fes_multibody/cycling/physiological_weight_calculation.py).
La figure S4 fournit des valeurs arrondies, pas une archive de précision
machine : les contrôles utilisent donc sa précision d'affichage.

| Muscle | Fatigabilité calculée | Criticité calculée | Poids calculé | Poids publié |
| --- | ---: | ---: | ---: | ---: |
| Deltoïde antérieur | 0,000652544 | 0,0814867 | 1 | 1 |
| Deltoïde postérieur | 0,000303836 | 0,0354497 | 0,0943161 | 0,0943 |
| Biceps | 0,000152398 | 0,5815631 | 0,3892687 | 0,389 |
| Triceps | 0,0000361591 | 0 | 0 | 0 |

Le protocole conserve les 1500 cycles d'une seconde, 80 % de Fmax imposé,
les 90° de pré-risque, le seuil de somme de moments positifs de 0,20 N·m,
les contributions moyennées sur les cycles. La reproduction min–max est
archivée; la normalisation utilisée pour les poids contrôleur divise par le
score brut maximal.
Les équations lentes à force prescrite sont composées analytiquement; aucune
PW future ni solution OCP n'est supposée connue.

## Un triceps non nul sans plancher

Le cas `published_code--ofat--biceps--alpha_a--x4` donne :

| Muscle | Poids nominal | Poids avec alpha A du biceps ×4 |
| --- | ---: | ---: |
| Deltoïde antérieur | 1 | 0,558948 |
| Deltoïde postérieur | 0,0943161 | 0,128413 |
| Biceps | 0,3892687 | 1 |
| Triceps | 0 | 0,0308463 |

Une fatigabilité du biceps quatre fois plus marquée modifie les régions à
risque et la contribution attribuée au triceps. Avec la normalisation par le
maximum, ce triceps reste strictement positif et aucun muscle à score brut
positif n'est artificiellement mis à zéro. Cela confirme l'intérêt de
modifier les paramètres des autres muscles, pas seulement ceux du triceps.

Dans ce panel, les variations isolées des paramètres du triceps ne lui donnent
pas de poids positif admissible. Le poids non nul obtenu reste faible; un
effet sur les PW ou sur l'endurance ne peut pas être déduit de sa seule valeur.
La variation indépendante de Fmax est une expérience numérique, pas un profil
clinique de faiblesse musculaire identifié.

## Calibration non admissible avec les paramètres locaux

Le nominal local atteint une capacité antérieure non positive pendant la
phase active du **50e cycle de calibration**. Le minimum ultérieur de A/A au
repos vaut environ −8,16, ce qui interdit l'interprétation physiologique.
Les poids sont alors `None`; le calcul historique après sortie du domaine
est conservé séparément uniquement pour audit.

Ce n'est pas une prédiction d'arrêt du RHO au cycle 50. La calibration impose
une force constante élevée, tandis que le RHO adapte ses stimulations et ses
forces. Les coefficients alpha locaux valent dix fois les valeurs du code
publié pour trois muscles; le triceps diffère également en A, Fmax et tau de
récupération. Les différences entre code, S3, S4 et dépôt sont détaillées dans
[le protocole](physiological_weights_validation_plan.md).

Avant une comparaison des contrôleurs, il faut choisir et harmoniser les
paramètres de référence. Réduire automatiquement la force de calibration ou
sa durée pour éviter ce refus serait une variante méthodologique distincte,
pas la reproduction exacte de l'annexe.

## Contrôles numériques et mécaniques

Les 45 tests ciblés couvrent le noyau (21), le catalogue de cas (9), la
géométrie et la reproduction de S4 (11), et le rapport/sélecteur de cas (4).
Ils vérifient notamment une boucle scalaire indépendante de 1500 cycles,
les facteurs intermédiaires, les permutations, les poids nuls, le domaine
avant et après récupération et l'absence d'héritage des poids entre cas.

Les moments ont été vérifiés contre un second modèle biorbd, à plusieurs
angles et pour deux vecteurs Fmax. Changer le Fmax d'un muscle inactif peut
changer le moment d'une autre colonne par les forces passives : cette
dépendance est conservée. Le résidu maximal de reconstruction des couples
articulaires effectivement utilisés est inférieur à 9e-14 N·m sur le panel.

En revanche, le problème cinématique du code publié conserve un écart maximal
de marqueurs d'environ 0,20147 m en 3D et de 0,009706 m dans le plan XY. Le
rayon effectivement réalisé n'est donc pas assimilé à la cible exacte de
0,1 m. La reproduction des poids ne valide pas ce contact pour le RHO.
L'adaptateur préserve les solutions de moindres carrés et expose ces écarts.

La convention de pré-risque est définie sur des angles triés croissants,
alors que la cinématique de référence parcourt le cycle dans le sens négatif.
Son interprétation temporelle doit être vérifiée avant usage en ligne; elle
n'a pas été corrigée silencieusement dans cette reproduction.

### Raffinement angulaire

| Cas | Poids à 120 intervalles | Poids à 240 intervalles |
| --- | --- | --- |
| Nominal | 1; 0,094316; 0,389269; 0 | 1; 0,095209; 0,379410; 0 |
| Biceps alpha A ×4 | 0,558948; 0,128413; 1; 0,030846 | 0,553067; 0,127383; 1; 0,029529 |
| Deltoïde postérieur Fmax ×2 | 1; 0,388819; 0,389269; 0 | 1; 0,395871; 0,379410; 0 |

Les cas raffinés restent admissibles, et le triceps reste positif dans le cas
prévu. Le changement maximal de poids est environ 0,00986 pour le nominal :
les poids sont sensibles au maillage et ne sont pas certifiés convergés.

## Reproduire et consulter les figures

Depuis la racine du dépôt, dans l'environnement conda `cocofest-rho32` :

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mplconfig OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_physiological_weights.py --output .cache/physiological-weight-sensitivity
```

La commande calcule 69 cas pour chacune des deux références, puis raffine les
trois cas admissibles présélectionnés. Elle produit un JSON avec tous les
paramètres, facteurs, statuts et empreintes du code, des tableaux NPZ et six
figures. Le passage enregistré a pris environ 11 s, figures incluses : ce
n'est ni un benchmark répété ni une mesure de latence du RHO.

Résultats du passage vérifié :

- [Rapport JSON](../../.cache/physiological-weight-sensitivity-20260910-final/report.json).
- [Référence publiée](../../.cache/physiological-weight-sensitivity-20260910-final/published_reference.png).
- [Panel publié](../../.cache/physiological-weight-sensitivity-20260910-final/published_code_panel.png).
- [Cas donnant un poids triceps positif](../../.cache/physiological-weight-sensitivity-20260910-final/published_code_positive_triceps.png).

La prochaine comparaison distingue RHO uniforme, RHO à poids physiologiques
fixes et RHO avec superviseur lent; le FHO reste optionnel sur les trois cas
gelés et n'intervient jamais dans le calcul ou le réglage des poids.
