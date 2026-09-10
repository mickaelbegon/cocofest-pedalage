# Réutiliser et éprouver les poids physiologiques de l'article

## Décision et état des sources

La prochaine référence sera le calcul des poids de l'article, et non un jeu
de poids réglé arbitrairement autour de un. On cherchera ensuite si une
correction lente apporte quelque chose à ces poids initiaux, sans données FHO.

Source lue : manuscrit de Co, Puchaud, Moissenet et Begon, *Maximizing Task
Endurance through Muscle Fatigue Minimization with Consideration of Muscle
Fatigability and Task Contribution: an in-Silico FES Study of Handcycling*,
PDF fourni de 38 pages. Les pages 22–23 décrivent la méthode; la page 19
donne le coût de fatigue pondéré. Le Supplementary Material 1 et son tableau
S6 sont cités mais ne sont pas inclus. Aucune formule manquante de l'annexe
n'est reconstruite ou présentée comme publiée ici.

Le texte principal indique :

- un produit normalisé entre la fatigabilité au carré et la criticité mécanique;
- une fatigabilité obtenue à partir de la diminution de capacité lors d'une
  activation de 80 % dans les zones de moment positif, pendant 1500 cycles;
- une criticité qui additionne une contribution positive exclusive et une
  contribution avant la zone où tous les muscles ont un moment négatif;
- les poids finaux, dans l'ordre deltoïde antérieur, deltoïde postérieur,
  biceps, triceps : **1; 0,0943; 0,389; 0**.

Il manque notamment la définition opérationnelle de la diminution de capacité
(pente, intervalle et normalisation), les formules et conventions angulaires
des deux contributions, leur normalisation relative, ainsi que la
normalisation finale. « 80 % d'activation » ne sera pas assimilé arbitrairement
à 80 % de PW, de recrutement maximal ou de force maximale.

## Étape 1 — Reproduire avant de généraliser

1. Transcrire les formules exactes du supplément, en associant chaque fonction
   à son équation ou ligne de tableau et à ses unités.
2. Reproduire les valeurs intermédiaires du tableau S6 avant les quatre poids.
   Les tolérances dépendront de la précision publiée, pas d'un ajustement a
   posteriori des paramètres pour retrouver le résultat.
3. Conserver les poids nuls. Si tous les scores sont nuls, signaler une
   normalisation indéfinie au lieu d'inventer une préférence. Une éventuelle
   régularisation positive sera une variante annoncée, pas la reproduction.
4. Vérifier les permutations des muscles, les cas symétriques, les unités,
   les limites sans fatigue et la sensibilité au maillage angulaire.
5. Chronométrer séparément puis ensemble la calibration de fatigabilité,
   la géométrie et le calcul final des poids. Le calcul arithmétique des poids
   seul ne représente pas nécessairement le coût de toute la méthode.

L'article annonce des poids calculés en moins d'une seconde, mais un OCP moyen
de 3,62 s dans la condition physiologique. Cela ne démontre pas encore la cible
d'un RHO complet sous une seconde.

## Étape 2 — Auditer les paramètres réellement utilisés

Les valeurs suivantes sont celles exécutées par `set_fes_model` dans
`examples/fes_multibody/cycling/cycling_pulse_width_mhe.py`, et non une
retranscription certifiée du supplément :

| Muscle | A au repos (`a_scale`, N/s) | `alpha_a` | `tau_fat` (s) | `fmax` (N) |
| --- | ---: | ---: | ---: | ---: |
| Biceps | 3314,7 | −0,56 | 179,6 | 149 |
| Triceps | 7036,3 | −0,24 | 76,2 | 617 |
| Deltoïde antérieur | 1148,6 | −1,4 | 445,5 | 48 |
| Deltoïde postérieur | 1234,5 | −1,1 | 342,7 | 51 |

Attention à la notation Python existante : `10e-2` vaut `0.1`, donc
`-5.6 * 10e-2` vaut bien `-0.56`. Ce constat n'établit pas une erreur du code;
il exige une comparaison avec les unités et valeurs de l'annexe avant tout
changement. Les commentaires de ce fichier décrivent des couplages liés à la
PCSA et aux fibres; ils ne remplacent pas la vérification de leur source.

Dans ce code, `fmax` est distinct de `a_scale`. Il intervient notamment dans
les bornes de force du RHO. Le prédicteur compact n'a pas de champ `fmax` :
varier cette borne n'est pas équivalent à varier la génération de force, et
un test compact seul ne garantit pas le respect des mêmes bornes que le RHO.

Chaque cas devra conserver les paramètres, leur origine, leurs unités, les
états initiaux et le profil mécanique. Les variantes modifieront des copies
explicites des paramètres, pas la configuration publique du modèle.

Le dépôt fournit déjà l'efficacité géométrique signée en fonction de l'angle
dans `cocofest/dynamics/reduced_cycling.py` (`muscle_effectiveness`). Elle
permettra de construire les indicateurs exacts de l'annexe; sa valeur absolue
seule ne représente pas la criticité mécanique. L'exemple
`cycling_bayesian_mhe.py` applique des poids à la perte **relative** de capacité
au carré, séparément pour chaque muscle. Ce coût ne reproduit donc pas
automatiquement la racine des pertes **brutes** de l'équation 16.

## Étape 3 — Séparer deux familles de sensibilité

### A. Un paramètre à la fois : comprendre le mécanisme

Premier criblage proposé avant calcul : multiplicateurs 0,9; 1; 1,1 autour
de chaque valeur nominale non nulle, muscle par muscle. Pour trois paramètres
et quatre muscles, cela donne 24 perturbations et le cas nominal. Une grille
plus large de 0,5 à 2 et des interactions ciblées ne viendront qu'ensuite,
pour les facteurs retenus. Ces amplitudes sont des tests numériques, pas des
intervalles physiologiques validés.

- Intensité de fatigue : varier la magnitude de `alpha_a` en gardant son signe.
- Récupération : varier `tau_fat`, en séparant état initial au repos et état
  déjà fatigué. Une constante plus grande signifie une récupération plus lente.
- Capacité : varier `a_scale` en déclarant explicitement si `fmax` et l'état
  initial suivent ou restent fixes.
- Dynamique de fatigue complète : étudier aussi `alpha_tau1`, `alpha_km`,
  `tau1_rest` et `km_rest`, puisque la force produite dépend de ces états.
- Recrutement : étudier les paramètres PW si la définition exacte du protocole
  d'activation fait intervenir la conversion stimulation–activation.

Changer `a_scale` impose un état initial au repos propre au nouveau modèle
ou un protocole de préparation commun : réutiliser une valeur absolue de A
issue d'une ancienne archive changerait aussi la fatigue relative initiale.
Changer `tauc` ou `km_rest` impose de reconstruire les amplitudes et historiques
calciques. Changer `pd0` ou `pdt` impose de recalculer le recrutement accessible
sous la borne physique de PW déclarée. Les anciens intervalles pré-calculés
ne doivent pas être réutilisés aveuglément après ces modifications.

On ne postulera pas que tous les poids normalisés sont monotones avec chacun
de ces paramètres : la force produite, la récupération et les autres muscles
peuvent modifier la réponse. Les tests de monotonie porteront seulement sur
des situations analytiques à force imposée où cette propriété est démontrée.

### B. Paramètres couplés : comparer des profils cohérents

Après validation des relations de mise à l'échelle, faire varier PCSA et
proportion de fibres, puis recalculer ensemble les paramètres concernés.
Ajouter des jeux identifiés expérimentalement s'ils sont disponibles.
Ne pas appeler « profils de patients » une grille synthétique non calibrée.

Séparer les changements de physiologie des changements de tâche : géométrie,
cadence, assistance/résistance et zones de contribution. L'article utilise
une assistance de 0,20 N·m; les archives actuelles sélectionnées pour le
prototype indiquent une résistance de 0,10 N·m. Une identité de paramètres
musculaires ne suffirait donc pas à garantir les mêmes poids mécaniques.

Le protocole de 1500 cycles et 80 % sera reproduit d'abord, puis ses propres
sensibilités seront examinées. Dans l'article, 1500 est choisi à partir des
endurances déjà observées : ce n'est pas une constante indépendante à
supposer universelle pour l'usage clinique.

## Étape 4 — Juger si les poids ont du sens

Pour chaque cas, enregistrer séparément fatigabilité brute, contribution
exclusive, contribution avant la zone critique, score non normalisé et poids
final. Cela distingue un véritable changement de fatigabilité d'un simple
changement de normalisation entre muscles.

Figures prévues :

1. Courbes des scores bruts et poids selon chaque paramètre, avec les quatre
   muscles et le point nominal clairement identifié.
2. Carte PCSA × proportion de fibres, montrant le poids et les régions où le
   modèle ou le protocole de calibration sort de son domaine admissible.
3. Profil angulaire des contributions mécaniques, avec les zones exclusives
   et la zone déficitaire, pour expliquer les poids faibles ou nuls.
4. Comparaison de trois politiques : poids uniformes, poids de l'article
   recalculés, puis ces mêmes poids avec correction lente.

Un poids plausible n'est pas encore une endurance améliorée. La dernière
comparaison exige le même objectif de perte de capacité que l'article, les
mêmes bornes, la même charge, le même état initial et les mêmes critères
d'arrêt. Les PW seront rejouées avec les équations complètes avant une
conclusion sur le contrôleur. Un arrêt numérique sera distingué d'un manque
de capacité à produire le moment demandé.

## Lien avec le superviseur à plusieurs minutes

La direction retenue est : **poids calculés à partir du modèle → corrections
lentes vérifiées → RHO court avec poids fixes pendant chaque résolution**.
Les coefficients nuls et les rapports de poids de l'article demandent une
politique de candidats adaptée : les bornes positives 0,25–4 du premier test
technique ne les représentent pas.

Le module de carte de fatigue à force imposée et le superviseur générique
sont des briques réutilisables. Ils ne remplacent ni les formules S6 ni la
validation du coût réel du RHO. En l'absence du supplément, la reproduction
numérique des poids reste explicitement en attente; aucun résultat de
sensibilité des poids publiés n'est revendiqué.
