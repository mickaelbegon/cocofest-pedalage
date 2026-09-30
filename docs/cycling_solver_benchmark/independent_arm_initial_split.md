# Initialisation D/G capacité–fatigabilité

Cette initialisation s'applique exclusivement aux deux OCPs unilatéraux
isocinétiques indépendants. Elle ne représente ni une manivelle commune, ni
une prédiction FHO, ni un certificat de faisabilité.

Après le premier RHO certifié de chaque bras, le programme construit une
enveloppe Ding d'un tour à PW maximal. Pour le bras `s`, `c[s,m]` est le
travail positif opportuniste du muscle `m`, `C[s]` leur somme, et `q[s,m]` la
fraction `c[s,m]/C[s]`. L'enveloppe est seulement une opportunité mécanique :
elle ne borne pas une charge admissible.

Avec les forces réellement produites par ce RHO de référence, le défi Ding
calcule `d[s,m]`, la perte de capacité induite par la stimulation, séparée de
la récupération et normalisée par `A_rest`. La perte pondérée est
`D[s] = sum(q[s,m] d[s,m])`. À la référence réellement produite `Wref[s]`,
la règle normalise d'abord par l'utilisation relative `Wref[s]/C[s]`, puis
forme le score `E[s] = C[s] / (D[s]/(Wref[s]/C[s])) = Wref[s]/D[s]`.

Le prochain partage est `p_right = E[right]/(E[right]+E[left])`, puis est
projeté uniquement sur le couple minimal déclaré. Une mesure non certifiée,
un domaine Ding invalide, une opportunité nulle ou un dommage nul laisse le
partage manuel inchangé. Aucun epsilon ne concentre artificiellement la
charge sur un bras.

Le choix GUI `capacity_fatigability_after_first_cycle` déclenche cette étape
unique; `capacity_feedback` peut ensuite être activé ou non. Les décisions,
mesures et le fait explicite que l'enveloppe n'est pas un certificat sont
écrits dans `summary.json` et l'audit PACE.

## Lien avec les poids musculaires

L'objectif RHO actuel est déjà quadratique en fatigue normalisée. La variante
expérimentale `mechanical_sensitivity_squared_v1_experimental` propose donc
des poids proportionnels au carré du crédit mécanique Shapley normalisé. Elle
n'ajoute pas une seconde fatigabilité explicite : celle-ci agit par la
dynamique Ding de `A`. Cette variante reste un noyau d'ablation distinct de
Physio-U; aucun gain d'endurance n'est encore revendiqué.
