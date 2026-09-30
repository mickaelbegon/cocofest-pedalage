# PACE H50 corrigé : sensibilité aux poids et régularisation

Analyse du modèle nominal, résistance 0,22 Nm, QP `predicted_ding_fatigue_v1`,
H=50, mise à jour toutes les 60 cycles. Aucun calcul FHO ni résultat FHO
n'intervient. Le QP utilise bien A_rest depuis la correction précédente.

## Verrou observé dans les projections archivées

Dans la campagne `four-model-r022-1500-unitpace-h050-fatigue-qp-arest-eps12-20260925`,
les décisions nominales après le cycle 120 ne changent plus les poids.
Les cinq projections aux cycles 180, 240, 300, 360 et 420 produisent des
coûts de fatigue, capacités minimales et déficits identiques à la précision
numérique pour toutes les alternatives. Sur 61 alternatives, 56 sont
rejetées pour absence de gain matériel ; seulement deux dégradent la fatigue,
deux la capacité minimale et une le déficit mécanique. Ces catégories peuvent
se recouvrir.

Le problème n'est donc pas principalement une tolérance excessive du garde.
Une fois fatigué, le terme linéaire du QP de fatigue domine sa très faible
courbure. Les allocations se placent souvent aux bornes, et une perturbation
de poids de 10 % ne change plus l'ensemble des muscles recrutés.

## Expérience locale de régularisation

Le script `scripts/audit_pace_regularization.py` conserve ce diagnostic exact
sur les journaux et effectue séparément un sweep depuis le checkpoint préparé
du cycle 300. Ce checkpoint est un primal décalé du prochain RHO ; son profil
n'est **pas** le cycle certifié original utilisé par la projection archivée.
Le sweep est un diagnostic local de sensibilité, pas une reproduction exacte
de la décision ni une preuve de mouvement réalisable.

| Régularisation | Recrutements aux bornes, candidat courant | Fatigue moyenne prédite | Déficit moyen normalisé | Choix avec le même garde |
|---:|---:|---:|---:|---|
| 1e-12 | 78,0 % | 0,115412 | 0,003426 | poids courants |
| 1e-8 | 78,0 % | 0,115412 | 0,003426 | poids courants |
| 1e-6 | 78,0 % | 0,115412 | 0,003426 | poids courants |
| 1e-5 | 75,7 % | 0,114435 | 0,000801 | baisse deltoïde postérieur |
| 1e-4 | 73,4 % | 0,113033 | 0,000244 | baisse deltoïde postérieur |

Chaque batch de dix candidats H50 prend environ 1,5 s. La sensibilité maximale
des PW entre candidats passe de 1,29 µs à 79,48 µs pour 1e-5, puis 10,48 µs
pour 1e-4. À 1e-5 et 1e-4, le candidat retenu est `[1,1 ; 0,853124 ; 1,1 ;
0,968729]`, dans l'ordre deltoïde antérieur, deltoïde postérieur, biceps,
triceps. Ces métriques ne permettent pas de comparer l'endurance entre deux
prédicteurs : chaque régularisation change la politique simulée.

Une régularisation autour du recrutement correspondant au cycle RHO observé
est un moyen de rester proche de la politique réelle et d'éviter cette
dégénérescence. Le passage à 1e-12, motivé par la très petite courbure de
fatigue, ne tenait pas suffisamment compte du terme linéaire une fois la
fatigue accumulée. Il ne faut toutefois pas choisir 1e-5 seulement parce que
sa projection paraît meilleure : la fidélité au RHO doit départager les
régularisations.

## Expériences préparées, sans lancement automatique

Le dossier `pace-regularization-audit-20260925b` contient :

- `audit.json` : journaux analysés, candidats, mesures de sensibilité et sweep ;
- `pilot-epsilon-1em05/command.json` : pilote nominal depuis la même graine,
  1500 cycles maximum, H50/K60, seul epsilon modifié ;
- `continuations/manifest.json` : 15 continuations appariées H1/H5/H20 depuis
  le checkpoint 300, avec les poids courants, unitaires, BO, proposition
  historique et candidat expérimental. Les epsilon qui choisissent des poids
  identiques partagent une seule continuation.

Les continuations réelles utilisent le checkpoint exact et les mêmes
paramètres mécaniques/musculaires. Le candidat du sweep est testé tel quel ;
la politique expérimentale de projection n'est pas restaurée dans ces
continuations statiques.

Critères avant promotion : conserver la validité mécanique et numérique,
confronter le classement prédit à la fatigue/capacité RHO réellement
observées, puis comparer les cycles validés du pilote et son coût temporel.
Un bénéfice d'endurance exige une campagne jusqu'à l'arrêt et une analyse
de viabilité de cet arrêt. Aucun défaut de production n'a été changé.
