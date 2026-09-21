# Deux bras isocinétiques indépendants

Cette architecture ne réutilise pas le modèle bilatéral couplé. Elle construit
un OCP unilatéral Ding/RHO pour la droite et un autre pour la gauche; états de
fatigue, commandes, solution chaude et solveur sont séparés. Les deux appels
peuvent être lancés en parallèle, mais partagent `omega < 0`, l'origine de
phase et le nombre de tours.

Le réglage propre à un bras est le travail demandé par cycle, représenté par
son couple moyen équivalent positif `tau_bar_i` :

`W_i = tau_bar_i * 2*pi`.

Ce n'est pas un couple externe instantané fixé. Comme dans la formulation
isocinétique existante, le couple instantané est inféré par l'équilibre
mécanique et vérifié contre ses bornes. Le réglage modifie uniquement la borne
terminale de l'état déjà existant `E_prod`.

`BioptimIndependentArmSolver` remplace les bornes terminales `E_prod` et le
guess numérique, sans reconstruire le graphe CasADi. Le variant ACADOS pousse
en plus `lbx/ubx` au dernier stage du solveur natif déjà généré. Les factories
`build_ipopt_independent_arms` et `build_acados_independent_arms` reçoivent
deux builders unilatéraux réels et les appellent une fois chacun.

L'orchestrateur public est `IndependentArmCoordinator`. Après une exécution,
`set_equivalent_mean_torques(right_nm=..., left_nm=...)` modifie les cibles
pour l'exécution suivante sans recréer les handles. Le CLI est :

```bash
python scripts/run_independent_isokinetic_arms.py \
  --config independent-arms-request.json --factory package.module:factory \
  --output-dir result/independent-arms --prefix "$CONDA_PREFIX"
```

La factory doit retourner le coordinateur avec deux handles préconstruits. Le
driver historique fournit désormais `build_unilateral_runtime(args)`, qui
retourne son `nmpc` et son solveur sans lancer la boucle RHO. Les helpers
`build_driver_ipopt_independent_arms(right_args, left_args, ...)` et
`build_driver_acados_independent_arms(...)` construisent donc les deux OCP
unilatéraux réels une seule fois. Aucun changement de formulation mécanique ni
de codegen n'est nécessaire pour les mises à jour ultérieures de `E_prod`.

Pour une séquence décidée à l'avance, le JSON peut contenir `adjustments`, par
exemple `[{"right_equivalent_mean_torque_nm": 0.25,
"left_equivalent_mean_torque_nm": 0.15}]`. Le CLI exécute alors ces ajustements
dans le même processus et écrit chaque résultat sous `adjustments/001/`; les
handles compilés sont conservés. Une modification décidée interactivement après
la fin du processus nécessite encore un futur protocole persistant (JSONL ou
service), car un nouveau lancement Python ne peut pas conserver la mémoire du
solveur.
