# FHO65 : coût réel de construction et d'évaluation de la Hessienne native

Le harnais `scripts/benchmark_fho_native_hessian.py` intercepte le NLP **après**
le dispatch et le shake Bioptim, construit son solveur IPOPT/MA57 interprété,
puis appelle directement `nlp_hess_l`. Il termine avant toute résolution.
La campagne active et ses fichiers ne sont pas modifiés.

Protocole : résistance signée +0.3 Nm, dynamique réduite, 65 cycles,
collocation Radau-5, graphe MX, CasADi 3.7.2, aucune compilation,
affinité CPU16–23 identique, OMP/OpenBLAS/MKL=1. Le seed et le préfixe FHO62
sont ceux du benchmark hybride FHO65, enregistrés dans chaque `command.json`.

Résultats bruts dans `local-results/fho65-native-hessian-audit-20260925/`.
Chaque répertoire contient le rapport complet et les valeurs creuses de H.

| Threads Bioptim | Préparation application/OCP | Dont dispatch/shake | Construction nlpsol | H froide | H chaude médiane (3) | Pic RSS |
|---|---:|---:|---:|---:|---:|---:|
| 1 (`threads-1-v2`) | 62.35 s | 11.40 s | 124.03 s | 10.65 s | 10.789 s | 26.51 GiB |
| 4 (`threads-4`) | 271.25 s | 25.57 s | 113.00 s | 7.84 s | 7.830 s | 22.35 GiB |

Dimensions identiques : 265223 variables, 259414 contraintes,
1833064 non-zéros de jacobienne, 1072504 non-zéros de Hessienne triangulaire
supérieure. À 4 threads, `find_functions` trouve **un ThreadMap dans la
Hessienne dérivée** (et 3970 MXFunction). À 1 thread : 5920 MXFunction,
4 SXFunction, aucun ThreadMap.

Cette observation corrige l'hypothèse selon laquelle les maps disparaîtraient
nécessairement lors de la dérivation globale : ici une partie de la Hessienne
native conserve effectivement le parallélisme.

## Garde de comparabilité

Ces deux premiers tests utilisent des multiplicateurs aléatoires avec les mêmes
indices et le même générateur. **Ils ne sont pas encore des tests d'égalité** :
Bioptim ordonne différemment les contraintes lorsque la pénalité devient
multi-thread (`interface_utils.py`, distribution `out[0]` versus `out[node_idx]`).
Les mêmes valeurs de lambda ne sont donc pas affectées aux mêmes contraintes
physiques. Les maxima de H diffèrent (32992.98 versus 56446.04).

Le harnais est complété par un test invariant à cette permutation : lambda=1,
puis lambda=0 pour isoler l'objectif; empreintes de x0 et de sparsité et stockage
des coefficients creux. Les répertoires suffixés `ones` contiennent ces mesures.
Le gain apparent de 27.4% ci-dessus reste indicatif tant que cet audit ne passe
pas. La construction totale est nettement plus coûteuse à 4 threads.

## Audit invariant final : 1 et 8 threads

| Threads | Préparation OCP | Dont dispatch/shake | Construction nlpsol | Total avant évaluations | H chaude médiane | Pic RSS |
|---|---:|---:|---:|---:|---:|---:|
| 1 (`threads-1-ones`) | 69.46 s | 12.98 s | 127.01 s | 196.47 s | 10.211 s | 26.52 GiB |
| 8 (`threads-8-ones`) | 267.91 s | 25.00 s | 102.65 s | 370.56 s | 6.100 s | 22.29 GiB |

Les vecteurs x0 sont exactement identiques; leur SHA256 vaut
`f5691e1e178aa7e73a8d4a71ac2d0b1e1f12419d97171d1c50355934cfc819eb`.
La sparsité supérieure est identique, empreinte
`22732eb2be9f24e96d472e9f8c474415bb605c4790f405fd8be071287a116f70`.
Différence absolue maximale des 1072504 coefficients : **0.0**, aussi bien
pour lambda=1 que pour lambda=0 (objectif seul). Le maximum absolu de H avec
lambda=1 est 222698.4487813366 dans les deux cas. Cela audite ces deux probes
au seed; cela ne remplace pas une preuve d'équivalence générale f/g/J/H, ni le
certificat d'une résolution.

La Hessienne native à 8 threads s'évalue ici **40.3% plus vite**, facteur 1.674.
Le surcoût de construction total est **174.09 s** et s'amortit en environ
**43 évaluations de Hessienne** à ce point. Pour 137 appels (compte historique
du FHO65), le modèle arithmétique « construction + H seulement » donne
1595.35 s à 1 thread et 1206.22 s à 8 threads, soit 389.13 s économisées.
**Ce calcul est une projection, pas un temps de résolution mesuré** : il omet
les autres callbacks, la factorisation et les variations d'itérations.

Le résultat montre que la Hessienne globale native profite déjà de ThreadMap;
le bon témoin d'un futur prototype paquets doit donc être ce témoin natif
threadé, et non le cas mono-thread. Aucun résultat ici n'établit un gain
supplémentaire dû aux paquets.

## Limite de conclusion

Aucun paquet local n'est encore extrait du NLP final. Aucune callback Hessienne
par paquets n'a été injectée dans ce vrai FHO65 et aucun temps de résolution
avec paquets n'a été mesuré. Le prototype synthétique ne permet pas de conclure
à une accélération de la campagne. Avant un solve expérimental il faut une
partition complète post-shake, un audit f/g/grad/J/H incluant les frontières et
multiplicateurs, puis une comparaison de certificat final.
