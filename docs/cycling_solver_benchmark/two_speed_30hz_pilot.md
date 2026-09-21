# Pilote à deux vitesses, 30 Hz

Le pilote du 12 septembre 2026 exécute un RHO de référence et le RHO-PACE
existant depuis le même état initial complet. Les trois cycles sont résolus
dans les deux bras, avec une adaptation PACE réellement appliquée à l'objectif
du troisième cycle. L'écart DOP853/Radau est maintenant accepté pour ce
pilote exploratoire à 30 Hz (borne 0,06 rad) : ce run court ne démontre
toutefois pas un gain d'endurance.

Le fichier `scripts/run_two_speed_30hz_pilot.py` ajoute seulement un lanceur et
un audit. Aucun fichier du contrôleur existant n'est modifié. Son manifeste
contient les commandes exactes, les empreintes des entrées et les critères.
Il distingue deux expériences :

- Le bras fermé PACE modifie les poids suivant la capacité `A/A_scale`, toutes
  les deux révolutions. Il utilise la loi causale existante, sans horizon prédictif.
- L'audit indépendant appelle `EnduranceWeightSupervisor` et
  `WeightedCyclePredictor` depuis la fin du premier cycle RHO. Il compare neuf
  candidats sur trois cycles futurs de moment et cinématique prescrits.
  À la date de ce pilote, **cette proposition n'était pas appliquée au RHO**.
  La boucle fermée ultérieure est documentée dans
  `predictive_moment_supervisor.md`; elle conserve ce pilote comme audit
  séparé et reproductible.

Le module `muscle_horizon_ocp` a également été inspecté : il introduit des PW
futures dans le NLP rapide. Il n'est pas activé ici, car cela ne représenterait
pas un superviseur lent indépendant du NLP.

## Conditions et reproduction

Variante Ding `ding_triceps_alpha_a_x0p5.json`, couple résistant positif
0,10 N·m, rotation négative, période 1 s, 30 stimulations et 30 PW par muscle et
par révolution, mécanique réduite. Préparation de 1 cycle puis 3 cycles par
bras; objectif fatigue quadratique, poids initiaux uniformes. PACE : lissage
0,2, gain capacité 1, poids relatifs dans [0,25; 4], pas logarithmique maximal
`log(1.1)`, mise à jour toutes les 2 révolutions.

Tous les OCP, y compris la préparation, utilisent IPOPT/MA57, tolérance NLP
1e-8, budget 4000 itérations et un thread. La transcription est une
**collocation Radau de degré 5**, distincte de la méthode Radau IIA d'ordre 5
de SciPy. Le replay continu indépendant déjà fourni par le benchmark est
DOP853 (`rtol=1e-11`, `atol=1e-13`). Il ne réinitialise pas les états entre
intervalles ou révolutions dans la trace globale.

Le profil géométrique réutilisé est stocké dans un dossier historique nommé
`resistance-fho-pilots-20260910`; aucune trajectoire ni politique FHO n'entre
dans le pilote. Les paramètres musculaires de la projection sont appliqués
par `configured_model_factories`, après vérification de l'empreinte du seed.
Le rapport exige que chaque `actual_parameters` corresponde à la configuration
résolue, afin de ne pas remplacer silencieusement la variante par les muscles
nominaux.

Depuis la racine du dépôt, avec un nouveau dossier de sortie :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python scripts/run_two_speed_30hz_pilot.py prepare \
  --python /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  --model-config examples/fes_multibody/cycling/two_model_ding_campaign/ding_triceps_alpha_a_x0p5.json \
  --reduced-profile resistance-fho-pilots-20260910/seed-0p10/reduced-cycling-fourier12.npz \
  --hsl-library /home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so \
  --output-directory two-speed-30hz-new-run \
  --cycles 3 --update-every-cycles 2 --projection-horizon 3 --run
```

Omettre `--run` prépare seulement le manifeste. La sous-commande
`run chemin/manifest.json` l'exécute ensuite. `summarize chemin/manifest.json`
relit les résultats sans nouvelle résolution; `--output nouveau.json` conserve
une synthèse actualisée dans un fichier neuf. Le lanceur refuse d'écraser un
cas existant. Son code de retour est 2 si une exécution ou la gate numérique
échoue; l'exécution technique et la validation scientifique restent séparées.

## Résultats du pilote court

Les artefacts sont dans `two-speed-30hz-pilot-20260912/`. La synthèse finale
est `summary-final.json`. `summary.json` conserve l'état du premier lancement,
avant correction d'une vérification qui supposait présent un champ optionnel
de fréquence dans l'export. Seul l'audit de projection a été relancé ensuite.

| Mesure | RHO | PACE lent |
|---|---:|---:|
| Cycles certifiés par le benchmark | 3/3 | 3/3 |
| État initial complet | commun | commun |
| Temps solveur médian des cycles suivants | 1,0555 s | 1,0724 s |
| Minimum `A/A_scale` | 0,994484805 | 0,994484994 |
| Aire de fatigue du benchmark | 0,014410790 | 0,014411132 |
| Écart maximal theta du replay global | 0,056909449 rad | 0,056910347 rad |
| Borne exploratoire theta 0,06 rad | accepté | accepté |

La mise à jour PACE après deux cycles est
`[0.9999036374, 0.9997418851, 1.0005163026, 0.9998383592]`
dans l'ordre Delt_ant, Delt_post, Biceps, Triceps. Le journal confirme
`bioptim.update_objectives` et un état précédent certifié. L'écart maximal de
PW entre bras est 0,132564 µs. Ces différences sur trois cycles ne permettent
pas de conclure à une amélioration; les temps dépassent aussi la période
de révolution de 1 s et ne constituent pas une preuve de temps réel.

L'audit de projection évalue neuf candidats en 0,397 s et retient
`[0.7377879465, 1.1066819197, 1.1066819197, 1.1066819197]` selon la marge
minimale du moment atteignable. La marge passe de 0,029713 à 0,030781 N·m.
Les neuf rollouts complètent les trois cycles demandés. Le replay Ding du
candidat sélectionné couvre les 90 intervalles : erreur maximale du moment
4,2774e-4 N·m, erreur maximale des états lents normalisés 9,5046e-6.
Les seuils 1e-3 N·m et 1e-3 sont satisfaits. Ce replay emploie RK4 avec
16 sous-pas et une cinématique prescrite; il valide le calcul conditionnel
musculaire, pas l'évolution mécanique libre du vélo.

## Diagnostic de l'écart Radau/DOP853

La réussite « physique » du benchmark concerne ici la trajectoire transcrite
et sa fermeture; elle ne remplace pas le replay continu indépendant.

| Trace RHO examinée | Maximum theta | Erreur theta terminale |
|---|---:|---:|
| Préparation, un cycle | 0,003218760 rad | non retenue dans cette comparaison |
| Troisième cycle, replay réinitialisé à son état RHO initial | 0,003578582 rad | -0,002749774 rad |
| Trois cycles sans réinitialisation | 0,056909449 rad | -0,047575965 rad |

Les 22 états archivés ont un saut exactement nul aux deux jonctions entre
cycles. Le troisième cycle diverge déjà en replay local; l'écart global plus
grand indique ensuite une amplification au fil de la dynamique couplée.
Les observations ne soutiennent donc pas une simple erreur de raccordement
des exports. Elles ne suffisent pas à isoler une erreur de transcription,
un effet de l'échantillonnage calcium ou une différence de RHS. À 30 Hz,
l'écart maximal calcium est 2,82e-5 et celui d'omega atteint 0,349235 rad/s
sur la trace globale. Pour le pilote à 30 Hz, la borne exploratoire est
maintenant 0,06 rad, à la demande explicite de l'utilisateur : elle accepte
l'écart observé sans le présenter comme une équivalence continue. Les
diagnostics DOP853 restent archivés, afin de pouvoir resserrer ce critère si
une utilisation clinique le requiert.

La prochaine vérification utile est une réintégration à PW et état initial
fixés avec plusieurs sous-pas Radau, puis une comparaison des RHS au même
état. Il faut conserver les 30 décisions PW et la fréquence physique, afin
de distinguer raffinement numérique et changement de problème. La prochaine
préparation du lanceur demande les diagnostics locaux de chaque cycle;
le run présenté ici a conservé le diagnostic local du dernier cycle fourni
par défaut. Aucune nouvelle campagne longue n'a été lancée.

Validation du code : 38 tests passent (`test_two_speed_30hz_pilot.py`,
`test_validate_multilevel_endurance.py`, `test_rho_pace.py`), dont les six
nouveaux contrôles de comparabilité. Les tests refusent notamment un état
initial incomplet, un replay limité à un préfixe, des états initiaux distincts
ou l'absence de mise à jour lente après le cycle zéro.
