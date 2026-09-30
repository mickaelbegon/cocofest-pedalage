# Exploration : Hessienne exacte FHO par paquets

## Question

Le FHO dynamique réduit est transcrit par collocation directe Radau-5, avec
un graphe MX monolithique et IPOPT. Peut-on réduire le temps de
`nlp_hess_l` en évaluant exactement, en parallèle, des contributions de
Hessienne locales par paquets de cycles ?

La question porte sur l'évaluation de la Hessienne du Lagrangien, et non
sur la factorisation KKT (MA57, MA86 ou MA97), ni sur une approximation
L-BFGS.

## État de référence

Le FHO isorésistance actuel est en dynamique directe : les états mécaniques
sont `theta` et `omega`, avec `theta_dot = omega` et
`omega_dot = casadi_acceleration(theta, omega, muscle_forces, torque)`.
Le FHO reste MX et non compilé. La certification exige une phase IPOPT à
Hessienne exacte.

Mesures déjà disponibles avec MA57 :

| Cas | Temps solve IPOPT exact | `nlp_hess_l` | Part Hessienne |
| --- | ---: | ---: | ---: |
| FHO65 | 958.8 s | 724.8 s / 137 appels | 75.6 % |
| FHO91 | 2020.2 s | environ 1520 s / 242 appels | environ 75 % |

La réduction de temps recherchée est donc surtout dans l'évaluation AD de
la Hessienne ; un changement de solveur linéaire ne peut améliorer que la
part KKT restante.

## Ce que Bioptim parallélise déjà

Les pénalités multi-nœuds sont construites avec
`Function.map(..., "thread", n_threads)` dans Bioptim. Cela parallélise des
évaluations de dynamiques et contraintes répétées. La Hessienne fournie à
IPOPT est toutefois générée à partir du NLP global ; Bioptim n'expose pas
une option « Hessienne par paquets ».

CasADi accepte une callback `hess_lag` personnalisée. C'est le point
d'injection à étudier pour une Hessienne exacte assemblée par paquets.

## Formulation candidate

Découper les cycles en paquets de taille `P=10` au premier essai. Pour le
paquet `b`, définir `x_b` comme les variables des cycles du paquet et les
variables frontière nécessaires. Attribuer chaque contrainte de continuité
une seule fois au paquet de droite. Alors :

```
L(x, lambda) = sum_b [ sigma * f_b(x_b) + lambda_b' * g_b(x_b) ]
H(x, lambda) = sum_b P_b' * Hessian(L_b)(x_b) * P_b
```

`P_b` est un opérateur de placement creux vers le vecteur de décision global.
Les contributions `Hessian(L_b)` peuvent être construites par AD CasADi,
évaluées par `Function.map(..., "thread", n_threads)`, puis assemblées dans
la sparsité exacte attendue par IPOPT.

Les termes de frontière sont indispensables : une Hessienne strictement
bloc-diagonale omettrait les couplages de continuité et ne serait pas la
Hessienne exacte du problème.

## Préconditions techniques

- Le build CasADi doit réellement supporter `ThreadMap`; OpenMP seul ne
  suffit pas nécessairement. Le build de campagne constaté le 2026-09-25
  est CasADi 3.7.2 ; le test cible devra être fait avec CasADi 3.8 et son
  support de threads vérifié par microbenchmark.
- La callback `hess_lag` doit conserver exactement sa signature, ses valeurs
  et sa sparsité par rapport à `nlp_hess_l` de référence.
- Aucun code C du FHO n'est généré ni compilé. Le parallélisme concerne les
  fonctions CasADi évaluées en mémoire.
- Les threads de `map` et ceux d'un solveur HSL ne doivent pas se cumuler
  sans contrôle : le premier essai réserve 8 CPU, `ThreadMap=8` et
  `OMP/BLAS=1`.

## Protocole de décision

1. Microbenchmark synthétique : somme de fonctions MX indépendantes, avec
   Hessienne via `map` serial/thread/OpenMP. Vérifier le support effectif
   des threads, l'égalité numérique et la sparsité.
2. Prototype synthétique d'assemblage : variables locales recouvrantes,
   multiplicateurs, contraintes frontière. Comparer au `hessian` monolithique
   pour valeurs et non-zéros.
3. FHO65 : même seed, mêmes bornes, même affinité CPU, mêmes tolérances et
   MA57. Mesurer construction, `nlp_hess_l`, `nlp_jac_g`, temps IPOPT,
   itérations, mémoire et certificat.
4. Seulement si FHO65 est concluant, répéter sur FHO88 ou FHO91 et comparer
   MA57/MA86/MA97 à protocole identique.

Le prototype est retenu s'il conserve un certificat identique (contraintes
auditées <= 1e-5) et réduit `nlp_hess_l` d'au moins 20 %, sans augmenter le
temps total ou le pic mémoire. Sinon il est abandonné. Les valeurs 20 % et
1e-5 sont des seuils de décision expérimentaux, pas des propriétés
mathématiques.

## Risques à documenter dans chaque résultat

- overhead d'assemblage sparse et synchronisation ;
- perte de simplifications communes entre cycles ;
- absence de support de threads dans le binaire CasADi ;
- sur-allocation CPU avec MA86/MA97 ;
- différence de sparsité, qui invalide la comparaison ou la certification.

Toute sortie L-BFGS demeure une graine non certifiée. La comparaison de cette
expérience porte uniquement sur la phase IPOPT exacte.
