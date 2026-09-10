# Diagnostic de l'arrêt strict H30, ancre 112

Le script `scripts/diagnose_compact_rollout_failure.py` isole l'arrêt de la
politique compacte à l'ancre RHO 112, sans utiliser de trajectoire FHO. Il part
de l'état terminal **effectivement archivé**, notamment de son calcium et de
ses écarts de fatigue non nuls. Il rejoue les PW des 607 phases compactes
réussies avec les équations Ding complètes, depuis cet état initial à chaque
raffinement. Le rejeu n'est jamais réinitialisé sur les états compacts
intermédiaires.

À la phase critique, deux expériences sont comparées : Ding complet depuis
l'état compact critique, puis Ding complet depuis l'état obtenu par le rejeu
indépendant. La première isole l'approximation de propagation sur cette phase;
la seconde comprend aussi la différence accumulée pendant le préfixe.

Chaque muscle est testé à neuf PW régulièrement espacées entre `PD0` et
`PW_max`. Les forces doivent croître sur cette grille. La borne inférieure de
moment additionne la force minimale pour les coefficients mécaniques positifs
et la force maximale pour les coefficients négatifs; la borne supérieure
utilise les choix opposés. Une grille non monotone invalide ce diagnostic
d'enveloppe aux extrêmes. La monotonie entre les points n'est pas démontrée.

Les rejeux commencent à 16, 32 et 64 sous-pas RK4 par phase, puis sont raffinés
à 128 ou 256 si nécessaire. Le seuil de variation entre les deux derniers
rejeux vaut le minimum de `1e-6 N·m` et de 1 % du déficit compact. Il couvre le
moment total du préfixe, les deux bornes critiques et les moments musculaires
sur la grille de PW, pour chacune des deux frontières étudiées. Ce contrôle
par raffinement est une estimation numérique, pas une borne d'erreur prouvée.

## Résultat au 10 septembre 2026

Le rollout compact strict termine 607 phases puis échoue au cycle futur 20,
phase 7 (indices à partir de zéro). Le moment demandé est
`0,0846647707961 N·m`. Le deltoïde postérieur a un coefficient de moment
négatif à cette phase : son PW maximal intervient donc dans la **borne
inférieure** totale. Utiliser tous les PW minimaux ne donnerait pas cette
borne signée.

| Propagation et état de départ de la phase critique | Borne inférieure (N·m) | Déficit par rapport à la cible (N·m) |
|---|---:|---:|
| Carte compacte, frontière compacte | 0,084809796369 | `1,45026e-4` |
| Ding complet à 128 sous-pas, frontière compacte | 0,084813375869 | `1,48605e-4` |
| Ding complet à 128 sous-pas, frontière après rejeu indépendant | 0,084729858919 | `6,50881e-5` |

L'erreur de propagation compacte sur la seule phase critique modifie ici la
borne inférieure d'environ `3,58e-6 N·m`. Le changement de frontière après
rejeu indépendant du préfixe la réduit d'environ `8,35e-5 N·m`. Cette différence
accumulée atténue le déficit sans l'annuler.

| Sous-pas Ding | Déficit après rejeu indépendant (N·m) | Erreur maximale du préfixe par rapport aux cibles originales (N·m) |
|---:|---:|---:|
| 16 | `6,34554e-5` | `4,88377e-4` |
| 32 | `6,49687e-5` | `4,79907e-4` |
| 64 | `6,50807e-5` | `4,79279e-4` |
| 128 | `6,50881e-5` | `4,79238e-4` |

Le raffinement 32→64 échoue au seuil choisi : une borne critique varie encore
de `1,27e-6 N·m`. Entre 64 et 128 sous-pas, la variation maximale des bornes
critiques après rejeu est `8,38e-8 N·m`, celle du moment total du préfixe est
`4,14e-8 N·m` et celle de la borne inférieure critique est `7,39e-9 N·m`.
Ces variations sont bien inférieures au déficit résiduel. La grille des neuf
PW est monotone pour les quatre muscles, à tous les raffinements et pour les
deux frontières.

Le déficit persiste donc dans ce diagnostic Ding indépendant; ce n'est pas
seulement un artefact de l'intégration RK4 à 16 sous-pas. En revanche, le
préfixe conserve une erreur de suivi maximale de `4,79e-4 N·m` et une différence
maximale des états lents avec le compact de `8,64e-5` après normalisation par
le repos. Il faut parler de la défaillance conditionnelle de **ces PW rejouées**,
pas de l'impossibilité d'une politique Ding qui réajusterait les PW pour
maintenir les cibles originales.

## Reproduction

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cocofest-mplconfig \
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/diagnose_compact_rollout_failure.py
```

Les chemins de l'archive RHO, du profil mécanique et du dossier de sortie sont
remplaçables en ligne de commande. Les résultats sont enregistrés dans
`.cache/compact-rollout-failure-diagnostic/diagnostic_report.json` et
`diagnostic_arrays.npz`. Le JSON contient les empreintes SHA256 des données et
des sources utilisées; le NPZ conserve les PW, les états et les moments de
chaque rejeu ainsi que la grille critique.

Empreintes des deux entrées : archive RHO
`aee992779366c4f9c4395d218edafedd003f336e730fecda65a4a4ffead6495e`;
profil mécanique
`3f8042f2c1b67f70c5d2469897a86ff07a920fa28de44dcee384449771448700`.

Le diagnostic distingue explicitement erreur d'intégration, approximation
compacte et échec conditionnel de cette séquence PW. Le préfixe rejoué a ses
propres erreurs de suivi du moment original : il ne constitue pas une nouvelle
politique Ding optimisée pour maintenir le suivi exact. Même un déficit
critique confirmé ne prouve ni l'impossibilité d'une autre allocation, ni
l'échec global de la tâche, ni un gain d'endurance. L'horizon H30 complet n'est
pas validé.
