# Géométrie bioMod dans l'identification Ding

`ReducedGeometryAdapter` utilise les mêmes séries de Fourier que l'OCP réduit.
Un profil `.npz` déjà généré avec `ReducedCyclingDynamics.save` se charge sans
biorbd, CasADi ni Bioptim. Sa construction et sa validation face au bioMod
restent réalisées par `cocofest.dynamics.reduced_cycling`.

```python
from cocofest.identification.cycling_geometry import ReducedGeometryAdapter
from cocofest.identification.synthetic_cycling import simple_mono_muscle_spec, generate_trial

adapter = ReducedGeometryAdapter.load(
    "reduced-cycling-fourier12.npz",
    muscle_map={"right_Biceps": "Biceps"},  # profil unilatéral
)
spec = simple_mono_muscle_spec("right_Biceps", duration_s=2,
                             cadence_rad_s=-2 * 3.141592653589793)
spec = adapter.apply(spec, omega_rad_s=-2 * 3.141592653589793)
trial = generate_trial(spec)
metadata = adapter.provenance
```

Pour un vrai profil bilatéral, les noms `right_Biceps`, `left_Biceps`, etc.
correspondent directement au bioMod. Aucun décalage de pi ni symétrie entre
côtés n'est supposé. Pour deux modèles unilatéraux indépendants, créer deux
adaptateurs avec leur mapping explicite. `angle_offset_rad` permet de convertir
le zéro de l'encodeur au zéro du profil. Fournir la vitesse angulaire mesurée
(tableau sur la grille temporelle), ou la vitesse constante isocinétique.

Deux coefficients distincts sont remplacés dans l'essai :

- `moment_arm_m = -dL/dtheta`, bras de levier signé issu de la projection
  mécanique du bioMod : `F_t = F * moment_arm_m / crank_radius_m`;
- `mechanical_gain = f_length * f_velocity + f_passive`, multiplicateur de
  la dérivée de force Ding utilisé par le code de pédalage.

Les inerties non positives, valeurs non finies, dimensions incohérentes et gains
Ding non positifs sont rejetés. Les bras signés ou nuls sont physiquement
possibles et conservés. Le terme passif légèrement négatif à certaines
longueurs est conservé pour reproduire exactement la loi du dépôt.

Le profil réduit décrit le couple musculaire généralisé. Pour le modèle
unilatéral à deux DDL de bras, `PlanarEndpointForceAdapter` complète ce profil
avec la Jacobienne du marqueur `hand` et l'inverse dynamique du bioMod. Il
résout `J_hand.T F = tau` et fournit donc directement la force tangentielle et
radiale appliquée par le bras au pédalier :

```python
from cocofest.identification.cycling_geometry import PlanarEndpointForceAdapter

endpoint = PlanarEndpointForceAdapter.from_paths(
    "reduced-cycling-fourier12.npz",
    "examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod",
    # nécessaire si le profil unilatéral emploie les noms sans préfixe
    muscle_map={"right_Biceps": "Biceps"},
)
spec_2d = endpoint.apply(spec, omega_rad_s=-2 * 3.141592653589793)
```

La résolution est refusée près d'une singularité du bras. Elle s'applique à
une chaîne de bras indépendante (deux DDL, deux axes). Pour la version
bilatérale Wu, utiliser les constructeurs non ambigus :

```python
right = PlanarEndpointForceAdapter.bilateral_arm_from_paths(profile, biomod, side="right")
left = PlanarEndpointForceAdapter.bilateral_arm_from_paths(profile, biomod, side="left")
right_audit = right.audit_unit_force_gains("right_Biceps", angles, omega_rad_s=omega)
```

Ils fixent respectivement `right_hand/(0, 1)` et `left_hand/(2, 3)` et
empêchent de mélanger un muscle, une main et des DDL de côtés opposés. Les
baselines de force totale restent à mesurer ou à calculer avec
`force_from_full_dynamics` à partir de **toutes** les forces musculaires et de
la cinématique connue; un couple résiduel inconnu ne doit pas être absorbé
artificiellement dans la force de contact.

L'audit compare `F_t * r` au bras de levier du profil réduit. Sur un profil
Fourier peu dense, un petit écart révèle l'erreur d'approximation de la
cinématique; il faut densifier et revalider ce profil avant l'identification.
Indépendamment, si les longueurs musculaires du bioMod produisent un gain Ding
non physique, `apply` bloque la campagne d'identification, tandis que
`unit_force_gains` et l'audit restent disponibles pour diagnostiquer la
géométrie mécanique sans contourner cette garde.

Avant toute campagne bilatérale, lancer le gate explicite suivant (avec un
profil réduit construit depuis le même bioMod) :

```bash
python scripts/audit_bilateral_biomod_geometry.py \
  --reduced-profile /chemin/vers/reduced-bilateral.npz \
  --biomod examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_Bilateral.bioMod \
  --output /tmp/bilateral-fibre-audit.json \
  --require-positive
```

Le code de sortie 2 signifie qu'au moins une fibre est non positive et bloque
la campagne.

## Via-points symétriques du modèle bilatéral Wu

Les raccourcis génériques au coude du bioMod bilatéral ont été remplacés par
les via-points détaillés du modèle Wu unilatéral pour le biceps et le triceps.
Les points du deltoïde antérieur et postérieur qui manquaient ont aussi été
restaurés. Le côté gauche est la réflexion exacte du côté droit par rapport au
plan sagittal : seule la coordonnée latérale `y` change de signe. Comme ce
bioMod bilatéral compact ne possède ni segment clavicule ni segment scapula,
les points correspondants du modèle unilatéral sont rattachés au thorax.

Avec 61 positions de manivelle, le gate est maintenant valide : les huit
longueurs de fibres sont strictement positives. À titre de contrôle, les
longueurs normalisées parcourent environ `0.70–0.71` (triceps), `1.38–1.49`
(biceps), `1.67–1.84` (deltoïde antérieur) et `1.71–1.96` (deltoïde
postérieur). Cette correction rend la géométrie exploitable par la garde Ding,
mais elle ne constitue pas une calibration anatomique individuelle : origines,
insertions, longueurs tendineuses et bras de levier doivent encore être validés
contre des données expérimentales avant toute interprétation physiologique.

Un profil Fourier ajuste nécessairement le cercle de pédalage avec une erreur
numérique très faible. `PlanarEndpointForceAdapter` accepte donc une variation
relative de rayon bornée (`1e-4` par défaut), tout en rejetant une variation
physique du rayon : le rayon moyen est utilisé pour conserver l'identité
`F_t r = couple musculaire`.

## Prototype bilatéral sagittal

`Modified_Wu_Shoulder_Model_Cycling_Bilateral_Sagittal.bioMod` est un nouveau
prototype à utiliser explicitement pendant la migration : il ne remplace pas
le bioMod bilatéral historique ni les profils et benchmarks déjà produits.
Son repère est `x` avant--arrière, `y` latéral et `z` vertical. Les deux bras
évoluent dans les plans parallèles `x-z` à `y = -0.10 m` et `y = +0.10 m`; les
épaules sont donc à la même hauteur. La manivelle, les épaules et les coudes
tournent autour de `y` (`wheel_rotation_RotY`).

La réduction reconnaît désormais automatiquement cet axe de manivelle et
projette le moment externe sur `y`. `PlanarEndpointForceAdapter` choisit de la
même manière les deux composantes de contact orthogonales à cet axe (`x-z`),
au lieu de supposer `x-y`. Le contact bilatéral ferme à moins de `1e-9 m`; les
ratios `L_fibre/L_opt` des huit muscles du prototype sont compris entre environ
`0.59` et `1.23` sur un tour, avec symétrie numérique droite/gauche.

Les origines, insertions et via-points de ce prototype sont une géométrie
fonctionnelle de transition : ils ont été réglés pour retrouver une plage de
longueurs comparable au Wu unilatéral. Ils doivent encore être confrontés à
une géométrie 3D anatomique et à des mesures sujet-spécifiques avant de devenir
le modèle de production pour l'identification ou les benchmarks.

La grille des coefficients est celle de l'essai; le simulateur les interpole
linéairement entre mesures. Vérifier la convergence en densifiant cette grille
avant une campagne d'identification. La provenance fournit le hash bioMod
enregistré dans le profil; elle ne certifie pas à elle seule la fidélité du
profil ni la calibration du capteur. Après adaptation, le même `spec` est
utilisable par `predict_trial`, l'estimation MAP et le plan d'expérience.
