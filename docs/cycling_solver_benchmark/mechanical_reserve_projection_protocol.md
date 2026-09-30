# RHO — Réserve mécanique projetée : protocole de revue et validation

Statut : proposition expérimentale et revue indépendante du code disponible le
28 septembre 2026. Ce document ne rapporte aucun gain d'endurance mesuré. Il ne
requiert ni solution FHO, ni allocation QP, ni apprentissage sur ces solutions.

## 1. Question testée et portée

Tester si des poids musculaires déduits de la sensibilité d'une réserve de
travail **projetée** améliorent le nombre de cycles isocinétiques certifiés à
demande mécanique identique. La projection est un replay d'une politique
déclarée ; elle ne prédit pas les décisions optimales des futurs RHO.

Trois nombres doivent rester distincts : l'horizon mécanique du NLP RHO, le
nombre `H` de cycles de projection physiologique et la période `K=20` cycles
certifiés entre mises à jour. Modifier H ne doit pas augmenter le nombre de
variables du NLP. Les valeurs H=1, 20 et 100 constituent des ablations, sans
changer K, la charge, le NLP ni les autres poids de coût.

Le cas bilatéral conserve la demande totale et sa répartition entre bras pour
la première comparaison. Une adaptation simultanée de cette répartition serait
une autre intervention et empêcherait d'attribuer le résultat aux seuls poids
musculaires.

## 2. Contrat mécanique et problèmes identifiés

À vitesse imposée `omega < 0`, noter `b_i(theta)` les valeurs
`muscle_effectiveness`, `e(theta)` l'efficacité du couple externe, `g(theta)`
la gravité projetée et `v(theta)` le coefficient quadratique de vitesse. Le
code impose :

```text
sum_i b_i F_i + e tau_load - g - v omega² = 0
tau_load = (g + v omega² - sum_i b_i F_i) / e
P_prod = -omega e tau_load
       = omega sum_i b_i F_i - omega (g + v omega²)
c_i(theta) = dP_prod/dF_i = omega b_i(theta)
P_common(theta) = -omega (g(theta) + v(theta) omega²)
T_cycle = 2 pi / abs(omega)
W_required = 2 pi * equivalent_mean_torque_nm
```

La cible déclarée au solveur est la référence de demande. Le travail mesuré
`E_prod[-1]-E_prod[0]` vérifie cette cible ; il ne doit pas faire dériver la
demande avec les erreurs de chaque cycle. La quadrature du travail utilise
des poids temporels en secondes. Le terme commun n'est ajouté qu'une fois.
Les relations longueur/vitesse/passives déjà incorporées dans l'évolution de
F ne doivent pas être ajoutées une seconde fois comme couple passif distinct.

Deux problèmes du bridge historique doivent être corrigés ou isolés avant
d'en réutiliser les sorties comme réserve :

1. `physio_update.py`, `build_isokinetic_max_pw_envelope`, calcule la puissance
   d'opportunité comme `max(0, -omega * moment_coefficients * F)`, alors que son
   bridge copie directement `muscle_effectiveness` dans `moment_coefficients`.
   Pour ces coefficients bruts, le signe cohérent avec `E_prod` est **+omega**.
   Par exemple, `omega=-2*pi`, `b=-0.02`, `F=10`, `g=v=0`, `e=1` donne un couple
   résistif `+0.2 Nm` et une puissance `+1.2566 W`. Le signe historique produit
   une contribution négative éliminée par le maximum. Les fixtures actuelles
   utilisent des coefficients positifs sans vérifier la dynamique inverse :
   elles ne détectent pas cette incompatibilité mécanique.
2. Le bridge définit `duration=1/stimulations_per_cycle`, ce qui suppose un
   tour par seconde. Utiliser `T_cycle/stimulations_per_cycle` et le véritable
   contrat d'impulsions, ou refuser explicitement les autres cadences. Changer
   omega en gardant ces durées modifierait simultanément la fatigue et le
   travail estimés de façon incohérente.

Les extrémités de phase et la géométrie constante par morceaux du bridge sont
une approximation déclarée. Elles ne constituent pas la quadrature Radau ni
un replay continu de la mécanique. Une comparaison en raffinement doit séparer
l'erreur de propagation Ding, celle de géométrie et celle de quadrature.

## 3. Définir Wmax, opportunité et marge sans fausse certification

La quantité théorique `W_max_star(x)` est le supremum du travail net sur un
cycle depuis l'état complet x, sous les mêmes bornes PW, transitions PW,
dynamique Ding, fréquence, contraintes de couple et contraintes de mouvement
que le RHO. On retire seulement l'égalité de travail cible pour cette
définition. Elle n'est pas calculée par l'enveloppe actuelle.

L'enveloppe existante fournit un replay cinq états à PW maximale depuis une
frontière x et des forces `F_maxpw_i(t;x)`. Construire l'indicateur explicite :

```text
W_opportunity(x) = sum_q dt_q [P_common_q
                    + sum_i max(c_iq, 0) * max(F_maxpw_iq(x), 0)]
M_joule(x) = W_opportunity(x) - W_required
M_relative(x) = M_joule(x) / W_required
```

Cette expression remplace à chaque phase les contributions défavorables par
une force nulle, même si la force résiduelle ne peut pas disparaître. C'est un
relâchement optimiste de l'usage de l'enveloppe. Ne pas plafonner
`W_opportunity` à `W_required` : cela annulerait la marge dès que la demande est
satisfaite. Une valeur de coalition Shapley plafonnée ne remplace donc pas
Wmax. Le clipping éventuel au plafond de couple doit être une variante nommée,
car il ne restaure ni la réalisabilité des forces ni les contraintes de slew.

Le nom de sortie conseillé est `work_opportunity_j` ou `wmax_proxy_j`, avec
`attainable_work_certified=false` et `global_upper_bound_proven=false`.
L'appellation « borne supérieure optimiste » n'est mathématiquement justifiée
que si l'on prouve que toutes les forces admissibles sont contenues dans la
boîte de forces utilisée, y compris l'historique et la fatigue. Une trajectoire
à PW maximale n'apporte pas cette preuve : davantage de stimulation peut
modifier A, Tau1 et Km, donc le futur maximum de force. Une enveloppe PW-max
peut même sous-estimer une force obtenue avec une autre histoire de PW.

Pour une véritable majoration conditionnelle, fournir des bornes de forces
`F_low_iq <= F_admissible_iq <= F_high_iq` démontrées pour le domaine considéré,
puis maximiser la forme linéaire dans chaque boîte : `c_i F_high` si `c_i>=0`,
`c_i F_low` sinon. Les contraintes oubliées élargissent alors explicitement
l'ensemble admissible. Une quadrature sur échantillons ne suffit toujours pas
à certifier une borne continue sans contrôle de son erreur.

À l'inverse, un replay d'une séquence PW respectant **toutes** les contraintes,
validé indépendamment, est un témoin réalisable et donne une borne inférieure
du maximum. Un cycle cible certifié prouve l'existence de ce cycle. Un échec
IPOPT local ou une marge proxy négative ne prouve pas une impossibilité
physiologique. Une marge proxy positive ne prouve pas la viabilité non plus.

## 4. Projection physiologique et sensibilités

Conserver à la frontière certifiée `x_k` les cinq états `(Cn,F,A,Tau1,Km)` de
chaque muscle, l'angle, l'historique PW requis par les contraintes de slew, le
calendrier d'impulsions et la configuration complète. Garder A_rest, alpha et
tau_fat comme paramètres physiologiques immuables.

Projection de référence recommandée : répéter la séquence PW du dernier cycle
certifié et intégrer les cinq états pendant H cycles avec les géométries
isocinétiques prescrites. Noter `Phi_H(x_k;u_ref)` cette propagation. Le replay
PW-max est réservé à la sonde d'opportunité après projection ; répéter PW-max
pendant H cycles testerait un scénario de stimulation maximale différent.
Si le raccord de la politique répétée viole le slew entre sa dernière et sa
première PW, le signaler et définir une politique de raccord explicite ; ne
pas prétendre que la politique est admissible parce que le cycle source l'est.

Définition sans ambiguïté :

```text
x_projected = Phi_H(x_k; u_ref)       # exactement H cycles complets
W_H(x_k) = W_opportunity(x_projected) # sonde additionnelle d'un cycle
M_H(x_k) = (W_H(x_k)-W_required)/W_required
```

Le label précise donc « opportunité du cycle H+1 après H cycles répétés ».
Une implémentation utilisant H-1 répétitions est possible mais doit porter une
autre convention, afin que H=1 ne soit pas interprété de deux façons.

La sensibilité mécanique d'intérêt pour l'état courant est
`s_i(H) = dM_H / d a_i`, avec `a_i=A_i/A_rest_i`. À défaut d'autodifférentiation
du propagateur, utiliser deux replays perturbés indépendants :

```text
s_i(H;eps) = [W_H(x_k + eps*A_rest_i*e_Ai)
              - W_H(x_k - eps*A_rest_i*e_Ai)] / (2 eps W_required)
```

Fixer les PW de référence, Cn, F, Tau1, Km, les paramètres biologiques et la
demande pendant la perturbation initiale ; propager ensuite **tous** les états
et leurs couplages. Cette dérivée partielle interroge la valeur marginale de A
à état courant fixé ; elle n'est pas la variation totale sur une famille
d'états fatigués physiquement atteints. Reporter aussi, comme ablation, une
perturbation cohérente de `(A,Tau1,Km)` selon une direction de fatigue déclarée.

Valeurs initiales recommandées `eps=1e-3`, puis `5e-4` et `2.5e-4` pour la
vérification. Près d'une limite de domaine, passer à une différence unilatérale
d'ordre deux et journaliser cette décision ; ne jamais écrêter silencieusement
l'état perturbé. Évaluer séparément la sensibilité terminale locale
`dW_opportunity(x_projected)/d(A_projected/A_rest)` : elle omet la dérivée de
Phi_H et n'est pas interchangeable avec s_i(H).

Un crédit de Shapley est une attribution de valeur de coalition ; son carré
normalisé, utilisé par `propose_mechanical_sensitivity_weights`, n'est pas
une différence finie de réserve. Conserver cette politique comme témoin
distinct et ne pas la renommer « sensibilité projetée ».

Une première intégration compatible avec le coût existant utilise
`raw_weight_i = s_i(H)^2`, puis la normalisation et les bornes existantes.
Elle correspond seulement à une approximation diagonale de
`(sum_i s_i delta a_i)^2` : les termes croisés sont omis, le coût existant porte
sur la fatigue totale depuis A_rest et non sur l'incrément `delta a_i`, et une
normalisation par moyenne géométrique élimine l'amplitude globale de la
réserve. Cette variante doit être nommée **pondération par sensibilités de
réserve projetée**, pas « maximisation de réserve ».

Une sensibilité négative significative révèle un mécanisme non monotone ou
une erreur. Ne pas la rendre positive par un carré sans audit. Une sensibilité
nulle crée un problème de logarithme : tenir les poids précédents et exposer
le motif, ou utiliser un plancher numérique explicite testé en ablation.
Avec le seuil minimum de poids 0.25, un muscle inutile conserve une pénalité
numérique mais ne reçoit pas pour autant un crédit mécanique positif.

## 5. Branchement avec un seul NLP compilé

Pour l'expérience minimale, conserver le résidu fatigue paramétré existant
`sqrt(w_i)*(1-A_i/A_rest_i)` et son même graphe. Construire une fois le
`ParametricFatigueWeightBinding`, puis écrire seulement les valeurs numériques
des paramètres fixes après les frontières 20, 40, 60, etc.

Flux requis à chaque frontière :

1. Le cycle courant est certifié et l'état de la frontière est copié, avec un
   index **physique** de cycle ; les nouvelles tentatives solveur ne comptent
   pas comme cycles accomplis.
2. Si la frontière n'est pas due, poursuivre avec les poids actuels. Si elle
   est due, calculer projection, marges et sensibilités depuis cette copie.
3. Refuser une mise à jour si l'entrée n'est pas certifiée, si le replay sort
   du domaine Ding, si une dérivée est instable ou si le contrat de modèle ne
   correspond plus. Conserver les poids actuels et le diagnostic.
4. Normaliser les poids positifs à moyenne géométrique un, borner à [0.25,4],
   lisser en logarithme avec 0.2 et limiter le changement à un facteur 1.1
   par mise à jour ; zone morte de 1 %. Conserver les valeurs brutes et les
   distorsions des ratios dues aux bornes.
5. Appliquer par le binding, vérifier le reçu `ocp_cost_updated=true`, puis
   laisser résoudre le cycle suivant. Un calcul proposé sans reçu n'est pas
   une mise à jour appliquée.

Ne pas appeler le fallback `update_bioptim_fatigue_cost` pour cette expérience
si le binding manque : refuser la configuration, car ce fallback remplace les
objectifs. Vérifier l'identité des callbacks compilés, dimensions, sparsité et
nombre de compilations après plusieurs mises à jour. Le contrôle d'identité
`nlp[0]` du binding est utile mais ne prouve pas seul la réutilisation des
callbacks générés par le backend.

La variante directe `penalty(M_H_affine(x_terminal))` nécessiterait un nouveau
résidu et un binding fixe contenant centre, valeur et gradient, créés une
seule fois avant compilation. Il faut décider son poids, son domaine local
et son interaction avec le coût fatigue. Le constructeur actuel refuse la
coexistence de plusieurs bindings ; ne pas contourner cette garde en écrasant
un `ParameterList`. Ce n'est pas nécessaire pour la première expérience de
pondération et ne doit pas être annoncé comme déjà implémenté.

La première étude est synchrone pour faciliter l'attribution. Inclure le coût
des replays dans la latence des frontières de mise à jour. Si une version
asynchrone devient nécessaire, enregistrer cycle source et cycle d'application,
âge du résultat, hash des paramètres et invalidation après changement de
charge ; utiliser le mécanisme de reçu existant. Un ancien résultat n'est pas
un résultat calculé depuis la frontière courante.

## 6. Ablations, métriques et critères fixés avant les campagnes

Comparer depuis les mêmes états et avec les mêmes solveurs, tolérances,
warm-starts, budgets CPU, bornes et charge :

| Variante | Projection | Mise à jour | Objet évalué |
|---|---:|---:|---|
| Unitaire | aucune | aucune | témoin principal |
| Physiologique fixe | aucune | aucune | témoin du coût existant |
| Shapley mécanique au carré | sonde courante | 20 | témoin attribution mécanique |
| Sensibilité figée | H=20 au départ | aucune | effet de réévaluation |
| Sensibilité projetée | H=1 | 20 | faible anticipation |
| Sensibilité projetée | H=20 | 20 | horizon égal à la cadence de mise à jour |
| Sensibilité projetée | H=100 | 20 | anticipation longue et erreur de modèle |

Avant endurance : checkpoints indépendants au repos, fatigue intermédiaire
et proximité d'arrêt ; au moins trois états de chaque catégorie, incluant
asymétrie des bras et muscles. Brancher de vrais RHO pendant 1, 5 puis 20 cycles
depuis ces mêmes états. Séparer les états utilisés pour régler les seuils des
états de validation finale. Faire trois répétitions d'exécution pour les
temps ; ces répétitions déterministes ne sont pas trois sujets biologiques.

Mesures à archiver :

- Nombre de cycles certifiés et travail total certifié ; arrêts numériques,
  échecs de replay, limites temporelles et essais censurés séparés.
- `W_required`, `W_opportunity`, marges actuelle et projetées, sensibilités
  signées, erreurs de dérivées, poids bruts/projetés/appliqués, saturation et
  proportion de propositions refusées.
- Travail par cycle, résidu de dynamique inverse, couple min/max, erreur de
  vitesse/angle, états Ding, violation PW et raccord slew.
- Erreur du replay par rapport à DOP853 à géométrie continue et erreur par
  rapport aux trajectoires RHO futures réellement observées. Leur différence
  inclut l'erreur de politique répétée, pas seulement l'erreur d'intégrateur.
- Corrélation/rang des marges avec l'échec futur, faux positifs et faux
  négatifs de viabilité, sans baptiser la corrélation « certification ».
- Temps construction/compilation, solveur, extraction, projection,
  différences finies, application et audit ; médiane/p95/max du cycle complet
  et des seules frontières mises à jour ; comptage des compilations.

Seuils initiaux proposés, à figer dans le manifeste avant comparaison :

| Porte | Succès requis | Action en cas d'échec |
|---|---|---|
| Identité mécanique | égalité puissance inverse/directe à 1e-10 relatif sur fixtures bien conditionnées ; aucun signe contradictoire | arrêt de l'intégration de la politique |
| Replay/quadrature | erreur de travail <= max(1e-4 J, 0.5 % de W_required) ; états lents normalisés <= 1e-3 vs DOP853 et raffinement | augmenter résolution ou refuser l'enveloppe |
| Dérivées | écart entre pas successifs <= max(1e-4, 2 % de la sensibilité) ; signes stables hors zone quasi nulle | tenir les poids et auditer |
| Validité des cycles | tolérances de certification existantes inchangées, aucun cycle déclaré réussi sans audit | arrêter au dernier cycle certifié |
| Compilation | une construction/compilation par configuration, zéro reconstruction aux mises à jour | rejeter l'allégation de NLP compilé réutilisé |
| Latence | moyenne complète < 1 s/cycle à cadence 1 Hz, surcoût amorti de politique <= 10 % du témoin | pas de promotion temps réel ; conserver comme expérience hors ligne |
| Endurance pilote | médiane du gain apparié >= 5 %, aucune régression > 5 % sur les conditions déclarées, aucun échec numérique supplémentaire attribuable à la politique | ne pas promouvoir ; examiner H et la validité du proxy |

Pour un contrat dur à 1 Hz, ajouter un plafond de 1 s **sur chaque cycle**, y
compris les mises à jour : une bonne moyenne amortie ne satisfait pas ce
contrat. Le p95 ne remplace pas non plus une garantie de pire cas.

Les tolérances existantes de transcription dans `isokinetic_cycling.py` sont
notamment 1e-9 rad/s, 1e-8 rad, 1e-6 pour l'équilibre, 1e-6 J pour le travail
et 1e-8 Nm pour les bornes. Le seuil de replay continu existant de 0.02 J est
un smoke test distinct ; il ne doit pas servir à valider des différences de
réserve plus petites que cette erreur. Les seuils de projection ci-dessus
sont des exigences proposées supplémentaires, pas des résultats acquis.

Fixer une limite administrative de 3000 cycles, comptée comme censure et non
comme endurance finale. Lors d'un arrêt, conserver l'état exact et utiliser
la sonde locale de viabilité existante à état gelé, mêmes contraintes et
historique. Un succès permet de diagnostiquer un arrêt numérique ; un échec
reste « aucun cycle admissible trouvé par cette sonde locale », jamais une
preuve d'épuisement global.

## 7. Checklist scientifique et numérique

- [ ] Signe `dE_prod_dot/dF_i=omega*b_i` vérifié depuis la dynamique du modèle.
- [ ] Aucune double attribution des termes gravité, vitesse ou passifs.
- [ ] Temps, cadence, fréquence, nombre de phases et quadrature cohérents.
- [ ] Demande prise du contrat de charge ; E_prod certifié utilisé comme contrôle.
- [ ] État complet et historique de contrôle préservés à la frontière certifiée.
- [ ] Enveloppe PW-max qualifiée de proxy, aucune propriété de majorant inventée.
- [ ] Politique future déclarée et H+1 distingué du nombre de cycles propagés.
- [ ] Projection sous PW de référence distincte de la sonde PW-max.
- [ ] Dérivées testées par raffinement ; A_rest reste immutable.
- [ ] Crédits Shapley et vraies dérivées présentés séparément.
- [ ] Signes négatifs, zéros et erreurs de domaine journalisés sans masquage.
- [ ] Approximation diagonale et perte des termes croisés explicites.
- [ ] Aucun reparamétrage artificiel de la fatigue à chaque mise à jour.
- [ ] Index des cycles physiques distinct de celui des tentatives solveur.
- [ ] Reçus d'application et identité des fonctions compilées vérifiés.
- [ ] Comparaison équitable de H=1/20/100 avec K=20, même NLP et même charge.
- [ ] Latence totale incluant audit/projection ; budgets matériel identiques.
- [ ] Validation indépendante avant comparaison d'endurance.
- [ ] Arrêt numérique, censure et impossibilité prouvée jamais confondus.
- [ ] Aucun résultat FHO/QP requis, utilisé pour réglage ou revendiqué comme référence.

## 8. Points de raccordement inspectés

Références valides lors de la revue ; les numéros de ligne peuvent évoluer
avec les implémentations parallèles.

| Fichier et fonction | Rôle / vigilance |
|---|---|
| `cocofest/optimization/physio_update.py:19`, `build_isokinetic_max_pw_envelope` | Replay PW-max cinq états ; signe de puissance à corriger pour les coefficients bruts. |
| `cocofest/optimization/physio_update.py:126`, `build_isokinetic_max_pw_envelope_from_discrete_cycle` | Extraction live, géométrie aux extrémités, durée actuellement fixée à un cycle d'une seconde. |
| `cocofest/optimization/physio_update.py:315`, `isokinetic_work_shapley` | Attribution plafonnée à la demande ; ne fournit ni Wmax ni sa dérivée. |
| `cocofest/optimization/physio_update.py:383`, `propose_mechanical_sensitivity_weights` | Ablation existante Shapley au carré ; garder un identifiant distinct. |
| `cocofest/optimization/adaptive_moment_rollout.py:261`, `propagate_ding_pulse_width_interval` | Propagateur cinq états, calcium exact et RK4 ; gains continus possibles. |
| `cocofest/optimization/adaptive_moment_rollout.py:665`, `rollout_fixed_pulse_width_policy` | Replay de politique répétée ; auditer séparément travail, couple et slew. |
| `cocofest/optimization/isokinetic_cycling.py:113`, `produced_mechanical_power` | Convention du travail contre la charge. |
| `cocofest/optimization/isokinetic_cycling.py:151`, `inverse_load_torque_from_coefficients` | Élimination analytique du couple de charge. |
| `cocofest/optimization/isokinetic_cycling.py:403`, `audit_isokinetic_trajectory` | Certification mécanique indépendante. |
| `cocofest/models/reduced_cycling_model.py:533` | Expression effective de E_prod dans le NLP. |
| `cocofest/custom_objectives.py:110`, `minimize_parameterized_overall_muscle_fatigue` | Résidu fatigue existant pondéré par paramètres fixes. |
| `cocofest/optimization/parametric_fatigue_weights.py:35`, `ParametricFatigueWeightBinding` | Construction unique et mises à jour numériques ; contrôle de dimension et d'identité NLP. |
| `examples/fes_multibody/cycling/cycling_pulse_width_mhe.py:1822` | Garde contre coexistence non validée de plusieurs bindings. |
| `cocofest/optimization/independent_arm_rho_pace.py:357` | Proposition, cadence, lissage et reçu d'application des poids. |
| `cocofest/simulation/independent_arms_process.py:691` | Frontière avant solveur, extraction et binding ; éviter son fallback de reconstruction. |
| `cocofest/optimization/endurance_viability.py:73` | Contrat de conclusion de la sonde gelée ; échec local non concluant. |
| `tests/test_physio_update.py:53` | Fixtures PW-max à compléter par une identité mécanique signée. |
| `tests/test_parametric_fatigue_weights.py` | Contrats de binding à compléter par vérification des callbacks du vrai backend. |
