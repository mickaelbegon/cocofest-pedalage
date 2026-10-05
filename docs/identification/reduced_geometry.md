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
bilatérale, instancier un adaptateur par bras en indiquant explicitement les
deux DDL et le marqueur de la main correspondants. Les baselines de force
totale restent à mesurer ou à calculer avec `force_from_full_dynamics` à partir
de **toutes** les forces musculaires et de la cinématique connue; un couple
résiduel inconnu ne doit pas être absorbé artificiellement dans la force de
contact.

La grille des coefficients est celle de l'essai; le simulateur les interpole
linéairement entre mesures. Vérifier la convergence en densifiant cette grille
avant une campagne d'identification. La provenance fournit le hash bioMod
enregistré dans le profil; elle ne certifie pas à elle seule la fidélité du
profil ni la calibration du capteur. Après adaptation, le même `spec` est
utilisable par `predict_trial`, l'estimation MAP et le plan d'expérience.
