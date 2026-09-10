# Campagne séquentielle à résistance constante

`scripts/build_resistance_comparison_campaign.py` construit un manifeste sans
lancer de solveur. Les trois bras sont, dans cet ordre : RHO uniforme d'un
cycle, RHO-PACE d'un cycle, puis FHO uniforme sur l'horizon fini déclaré.
Chaque campagne contient une seule résistance signée strictement positive,
avec dynamique directe réduite. Aucun mode isocinétique, balayage de charge
ou remplacement automatique de PACE par le RHO uniforme n'est disponible.

La limite de 100 cycles vient de l'adaptateur PACE actuel. Le FHO porte sur
le même nombre de cycles demandé aux deux RHO ; sa résolution ne certifie
pas un optimum global d'endurance. Il ne fournit aucune calibration à PACE.

## Déclaration

Dans l'environnement de calcul, fournir les fichiers effectifs de la
campagne :

```bash
python scripts/build_resistance_comparison_campaign.py \
  --campaign-id resistance-comparison-01 \
  --resistance-nm 0.22 --cycles 100 \
  --seed /absolute/path/certified-common-seed.npz \
  --reduced-profile /absolute/path/reduced-profile.npz \
  --pace-config /absolute/path/pace-config.json \
  --output-directory /absolute/path/new-campaign \
  --manifest /absolute/path/new-campaign-manifest.json
```

La valeur 0,22 est illustrative, pas une sélection automatique de résistance.
Le fichier PACE doit déclarer `initial_weight_basis` et, éventuellement, les
poids initiaux et paramètres de sa politique. Le contrôle uniforme fourni
dans `rho_pace_uniform_start.json` n'est pas une calibration physiologique.
La politique PACE et sa normalisation demeurent celles de son adaptateur ;
le manifeste ne les confond pas avec les poids min–max publiés.

Le manifeste fixe les commandes `argv`, l'état initial, le profil mécanique,
la résistance, la transcription Radau de degré 5, les 30 stimulations par
cycle et le solveur IPOPT/MA57. MUMPS peut être demandé explicitement pour
une campagne distincte. Les fichiers d'entrée et d'implémentation sont
empreintés. Changer ces fichiers ou une commande rend le manifeste invalide.
L'environnement numérique et les bibliothèques du solveur restent à
consigner dans les résultats effectifs.

## Gates et progression

L'API `next_command(manifest, reviews)` renvoie uniquement la prochaine
commande admissible. Elle ne lance rien et ne délivre aucun certificat
physique. Une personne ou un adaptateur d'audit doit fournir les reçus de
revue et les artefacts qui les justifient. Le reçu contient :

- `manifest_sha256`, au niveau global, identique à celui du manifeste ;
- une entrée `initial` avec tous les `initial_required_review_gates` passés
  et une liste `evidence` de fichiers `{path, sha256}` ;
- une entrée par bras terminé, avec ses `required_review_gates`, les
  artefacts du résultat (et du journal pour PACE), `outcome` et
  `certified_executed_cycles`.

Chaque entrée contient un objet `gates` associant les noms des contrôles à
`true`, et sa liste `evidence`. `artifact(path)` produit une référence de
fichier avec empreinte ; il ne certifie pas son contenu.

La première revue doit vérifier la faisabilité de l'état initial, l'identité
du modèle et des paramètres, la résistance constante et les seuils de
validation retenus. Les revues suivantes doivent vérifier le préfixe rejoué,
les bornes PW/force/états, les résidus et le statut du solveur, ainsi que la
classification de l'arrêt. PACE exige en plus les reçus de modification
effective du coût et un journal correspondant au résultat.

Seuls deux résultats sont admis ici : `completed`, lorsque tous les cycles
demandés sont certifiés, ou `numerical_or_unresolved_stop`, avec un préfixe
physique certifié non vide. Le second autorise la comparaison suivante mais
ne devient jamais « fatigue ». Un bras échoué avant toute trajectoire
certifiée bloque cette progression. Une analyse distincte serait nécessaire
pour attribuer un arrêt à une limite physiologique ou mécanique.

Des fichiers de résultat existants sans revue empêchent de relancer le bras.
Le générateur refuse aussi d'écraser un manifeste existant. Les commandes
sont déclaratives : lancer manuellement un `argv` contournerait les gates ;
ce fichier n'est pas un ordonnanceur ni un verrou de processus.

Comparer le nombre de cycles certifiés, le suivi, la perte de capacité non
pondérée, les PW/forces et les latences. Les coûts pondérés de PACE et les
coûts uniformes des deux références ne sont pas directement comparables.
Les mêmes seuils et critères physiques doivent être appliqués aux trois bras.

## Vérification sans résolution

```bash
python -m pytest -q tests/test_resistance_comparison_campaign.py
```

Ces tests vérifient les commandes existantes, les horizons, le verrouillage
de résistance, l'ordre des revues, les refus de changements et l'absence de
conversion d'un arrêt numérique en fatigue. Aucun solveur n'est importé.
