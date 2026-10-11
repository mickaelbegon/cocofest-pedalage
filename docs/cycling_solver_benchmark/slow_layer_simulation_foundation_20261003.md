# Couche lente en simulation — fondation et porte de validation

## Statut

Cette étape met en place la couche lente en **mode oracle d'état**. Elle ne
modifie pas le graphe du RHO, ne lance pas de FHO et n'active aucune décision
en ligne. Les variables RHO déjà paramétrées (travail terminal et poids de
fatigue) restent les seules écritures envisagées par la future intégration.

| Élément | Statut | Résultat |
| --- | --- | --- |
| Lecture oracle `A`, `Tau1`, `Km` | [Code + vérifié archive] | états lus uniquement depuis un témoin certifié ou une égalité initiale gelée; les warm starts sont refusés |
| Propagation lente de Ding | [Code + validé] | réutilise la convolution exacte conditionnelle à une force prescrite; aucun pronostic de force n'est déclaré |
| `Nhat` conditionnel | [Code] | franchissement de bornes lentes déclarées, avec censure explicite |
| Partage G–D | [Code] | conserve charge totale, bornes, asymétrie et pas de couple; maintien conservateur par défaut |
| Superviseur à 10 cycles | [Code] | reçu immuable et `feasibility=not_assessed`; non raccordé au coordinateur RHO |
| Carte `Phi` | [Code] | ne lit que des oracles RHO hachés, abstention hors support ou issue ambiguë |
| Pré-séance P1/P2/P7 | [Mesuré] | bundle réel à 228 cycles créé en `0,017 s`; couple passif non mesuré faute de profil réduit |
| A0/A1 | [Code] | constructeurs de manifestes seulement; aucun lancement ou résultat affirmé |

## Validation logicielle

La suite ciblée comporte **113 tests réussis** : contrats d'état, preuves
oracle, convolution, seuil conditionnel, contraintes de partage, superviseur,
carte `Phi`, manifeste A0/A1, pré-séance et Ding lent.

Le validateur `Nhat` a aussi relu une observation réelle : cycle 141 de
`left-c140-unit-d1`, quatre muscles, six artefacts vérifiés. Cette lecture
établit la provenance des données; elle ne valide pas encore une prédiction
d'endurance.

## Pré-séance mesurée

Le bundle
`slow-layer-sim-20261003/presession/presession-f77d182f5ad8840a` contient :

- le calendrier calcium à 30 Hz pour les huit muscles (`6 841` échantillons);
- copies hachées des modèles gauche et droit;
- manifeste des versions, affinité CPU, configuration et budget;
- une interface explicite de couple passif, marquée `not_measured` en absence
  de profil réduit ou de tour non stimulé.

Le calcium est précalculable ici car l'équation `Cn` utilise le calendrier des
stimulations et non les PW. La force, les PW et les états de fatigue restent
des variables de décision/observation distinctes.

## Ce qui bloque légitimement la campagne A0/A1

Le document fourni renvoie aux critères T1/T3 et aux bornes lentes du document
précédent, mais ils ne sont pas présents dans le dépôt ni dans le document
actuel. Ces éléments sont nécessaires pour qualifier `Nhat` :

1. des bornes lentes par muscle et état (`A`, `Tau1`, `Km`), avec leur source
   physiologique ou expérimentale;
2. les seuils T1/T3 préenregistrés (erreur, couverture et gestion de la
   censure) qui autorisent ou interdisent l'usage de `Nhat`;
3. les bornes de confort par bras `tau_min`/`tau_max` pour une vraie campagne
   A1. Le pas de `0,05 Nm` et l'asymétrie de `0,3 Nm` sont proposés dans le
   document, mais ils ne suffisent pas à définir les bornes individuelles.

Sans ces valeurs, le code maintient volontairement les cibles précédentes et
ne peut ni lancer A1 ni annoncer qu'un partage améliore l'endurance. Donner
ces choix au code par défaut serait une hypothèse clinique cachée.

## Reprise dès disponibilité des paramètres

1. Générer un manifeste `Nhat` T1/T3 avec les bornes et continuations
   certifiées; valider `Nhat` sur des familles séparées.
2. Construire la grille RHO `Phi` préenregistrée, puis passer son holdout.
3. Raccorder le superviseur au coordinateur existant, avec activation
   impossible tant que les deux portes ne sont pas vertes.
4. Lancer A0 (poids seuls, `0,96 Nm` par bras), puis A1 (partage seul,
   `1,92 Nm` total) et seulement ensuite les campagnes travail maximal,
   mismatch, BO lent et prior de population.
