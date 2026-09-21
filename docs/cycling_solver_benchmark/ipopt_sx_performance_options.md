# Options expérimentales des évaluateurs SX IPOPT

La comparaison des solveurs et le runner RHO périodique acceptent les mêmes
options, désactivées par défaut :

```sh
--ipopt-function-transform \
--ipopt-c-compile \
--ipopt-c-compiler-flag=-O1 \
--ipopt-c-cache-dir /chemin/prive/cache \
--ipopt-c-cache-name rho
```

`--ipopt-function-transform` active la transformation des callbacks SX via
Bioptim (`set_function_transform(True)`). Cette option requiert la branche
Bioptim compatible et CasADi 3.8. Elle peut aussi être testée sans compilation.
Le pipeline est fixé par Bioptim : `cse`, `ref_count`, `const_folding`.

Les arguments du compilateur se répètent avec
`--ipopt-c-compiler-flag=-O1 --ipopt-c-compiler-flag=-g`.
Les options du compilateur et du cache requièrent explicitement un des deux
modes de compilation; elles ne sont pas des options natives d'IPOPT. En l'absence
d'arguments supplémentaires, l'appel historique `set_c_compile(bool)` reste
utilisé. Une version de Bioptim sans les extensions demandées produit une
erreur explicite.

La compilation séparée permet de conserver les grandes dérivées dans la VM.
Le premier sous-ensemble recommandé pour l'expérimentation est l'objectif,
les contraintes et le gradient de l'objectif :

```sh
--ipopt-function-transform \
--ipopt-c-compile-callback nlp_f \
--ipopt-c-compile-callback nlp_g \
--ipopt-c-compile-callback nlp_grad_f \
--ipopt-c-cache-dir /chemin/prive/cache \
--ipopt-c-cache-name rho
```

La présence de `--ipopt-c-compile-callback` active ce mode sans
`--ipopt-c-compile` : ces deux sélections sont incompatibles. Le répertoire de
cache est obligatoire. Chaque nom complet ne peut apparaître qu'une fois;
`nlp_jac_g` et `nlp_hess_l` sont également acceptés. Seuls les callbacks choisis
sont compilés, les autres restent dans la VM. Sans option explicite de
compilateur, ce mode conserve le défaut Bioptim `-O1`. Il requiert la méthode
Bioptim `set_c_compile_callbacks`; une ancienne version produit une erreur
explicite. La transformation est compatible avec ce mode.

Le cache de code compilé est distinct du seed physique. Ces contrôles sont
enregistrés dans les configurations du rapport de comparaison et dans les
diagnostics IPOPT; ils ne modifient pas les signatures scientifiques des seeds.
La comparaison résout un chemin de cache relatif depuis le répertoire
d'invocation. Utiliser un répertoire privé, car il contient des bibliothèques
natives exécutables.

Comparer les RHO à chaud en excluant la préparation, la transformation et la
compilation initiales. Vérifier également les certifications, les itérations et
les trajectoires : la seule présence de ces options ne garantit aucun gain.
