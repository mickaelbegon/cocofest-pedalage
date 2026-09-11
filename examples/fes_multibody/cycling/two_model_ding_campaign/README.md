# Deux variantes Ding, quatre conditions

Ces fichiers sont les entrées explicites de la comparaison `RHO`,
`RHO-Physio`, `RHO-PACE`, `FHO`. Les deux cas ne diffèrent que par
`Triceps.alpha_a` (`x0.5` et `x2` par rapport à la valeur du dépôt).

Les poids proviennent de la formule de l'annexe, recalculée indépendamment
pour chaque cas : calibration source de 45 cycles, rho 0.8, seuil 0.2 N m,
960 subdivisions, normalisation raw/max. Ils n'utilisent aucune donnée FHO.
Ils sont une calibration expérimentale : la géométrie source a un écart IK
maximal d'environ 9.7 mm et la convention temporelle pré-risque doit encore
être validée pour le RHO. Les bornes PACE `[0.001, 64]` préservent les rapports
initiaux après normalisation géométrique; les bornes par défaut les écraseraient.

Un seed frais et certifié est obligatoire pour chaque fichier Ding; il est
interdit de réemployer le seed nominal ou celui de l'autre variante.
