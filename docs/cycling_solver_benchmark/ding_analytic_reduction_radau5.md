# Réduire Ding en conservant Radau-5 : démarche et développements mathématiques

L'application visée est le cyclage à mécanique réduite, résolu par IPOPT avec
MA57 et une collocation Radau à cinq stages. L'idée est d'exploiter les parties
linéaires des équations musculaires pour diminuer le nombre d'inconnues que le
solveur traite. La référence reste le problème Radau-5 actuel.

Deux opérations doivent être distinguées. Une **réduction continue** reconstruit
certains états par des exponentielles exactes, puis discrétise les équations
restantes. Une **condensation des stages Radau** élimine algébriquement les
équations linéaires déjà discrétisées. La seconde conserve les solutions du
système Radau discret et constitue donc le premier candidat à intégrer dans
IPOPT/MA57. Les deux variantes peuvent être étudiées séparément.

Le [prototype de validation](../../scripts/validate_ding_radau5_condensation.py)
implémente ces variantes sur un muscle isolé. Il ne constitue pas encore une
intégration dans l'OCP de cyclage. Les [tests associés](../../tests/test_ding_radau5_condensation.py)
vérifient leurs propriétés numériques. Les temps de ce prototype Python ne
mesurent pas la performance d'IPOPT ou de MA57.

## 1. Modèle effectivement utilisé

Les équations suivent les implémentations du dépôt :

- [Ding 2007 avec fatigue](../../cocofest/models/ding2007/ding2007_with_fatigue.py) ;
- [forçage calcique périodique par intervalle](../../cocofest/models/ding2007/ding2007_with_fatigue_periodic_node.py) ;
- [équation de force](../../cocofest/models/ding2003/ding2003.py) ;
- [recrutement par largeur d'impulsion](../../cocofest/models/ding2007/ding2007.py).

Pour chaque muscle, les cinq états physiques sont

\[
x=(C,F,A,T,K),\qquad C=C_n,\quad T=\mathrm{Tau1},\quad K=K_m.
\]

La largeur d'impulsion est notée \(u\), avec les paramètres \(u_0\) et
\(u_t>0\). Posons

\[
\rho(u)=1-\exp\!\left(-\frac{u-u_0}{u_t}\right),\qquad
\sigma(C,K)=\frac{C}{K+C},
\]

\[
g(q,\dot q)=g_\ell(q)g_v(q,\dot q)+g_p(q).
\]

Dans l'implémentation actuelle, le facteur mécanique \(g\) multiplie toute
l'équation de force, y compris la relaxation :

\[
\begin{aligned}
\dot C &= \frac{H(t)-C}{\tau_c},\\
\dot F &=g(q,\dot q)\left[A\rho(u)\sigma(C,K)
            -\frac{F}{T+\tau_2\sigma(C,K)}\right],\\
\dot A &=-\frac{A-A_r}{\tau_f}+\alpha_A F,\\
\dot T &=-\frac{T-T_r}{\tau_f}+\alpha_T F,\\
\dot K &=-\frac{K-K_r}{\tau_f}+\alpha_K F.
\end{aligned}
\]

Les trois états de fatigue ont le même temps de récupération \(\tau_f\)
et sont forcés par la même force \(F\). Les paramètres peuvent différer entre
muscles ; la réduction s'applique muscle par muscle. Les formules supposent
qu'ils restent constants sur l'intervalle considéré.

Le prototype isolé impose \(g=1\). La réduction algébrique reste valable avec
la mécanique couplée, puisque celle-ci ne modifie pas les trois équations
linéaires de fatigue. Le coût et la précision du problème couplé doivent être
mesurés séparément.

## 2. Calcium : solution exacte sur un intervalle de stimulation

Considérons un intervalle \([t_k,t_k+h]\), sans stimulation intérieure, et
\(s=t-t_k\). Dans le modèle `periodic_node`, le forçage est

\[
H(t_k+s)=H_k e^{-s/\tau_c},\qquad 0\le s\le h.
\]

Le facteur intégrant de l'équation de calcium donne

\[
\frac{d}{ds}\left(e^{s/\tau_c}C(t_k+s)\right)=\frac{H_k}{\tau_c},
\]

d'où

\[
\boxed{C(t_k+s)=e^{-s/\tau_c}\left(C_k+\frac{H_k s}{\tau_c}\right).}
\]

Ce n'est pas une simple décroissance exponentielle : le facteur linéaire en
\(s\) provient du forçage qui décroît avec le même temps caractéristique.
À une stimulation, **le forçage \(H\) change, mais \(C\) reste continu** dans
ce modèle. Il faut transmettre \(C(t_k+h)\) à l'intervalle suivant, puis
utiliser sa nouvelle amplitude \(H_{k+1}\).

Pour un train périodique avec amplitude \(H_k=H\), le calcium à la frontière
des intervalles a le point fixe

\[
C_*=\frac{e^{-h/\tau_c}Hh/\tau_c}{1-e^{-h/\tau_c}}.
\]

Cette formule ne justifie pas de remplacer un état initial quelconque par
\(C_*\). Un état transmis depuis une fenêtre RHO peut contenir un transitoire
physique qui doit être conservé.

L'amplitude \(H_k\) doit correspondre exactement à l'historique et à sa règle
de troncature. Pour le train périodique du dépôt, avec \(N\) stimulations
retenues, \(d=e^{-h/\tau_c}\) et \(R=1+(r_0-1)d\),

\[
H=d^{N-1}+R\sum_{j=0}^{N-2}d^j\quad(N\ge2),\qquad H=1\quad(N=1).
\]

Le coefficient de la plus ancienne stimulation retenue vaut un dans cette
convention. Il ne faut pas remplacer cette somme par une histoire infinie
lors d'une comparaison numérique. Dans le modèle courant, \(r_0\) dépend de
\(K_r\), et non de l'état instantané \(K\).

La formule reste utilisable pour des durées variables si chaque intervalle
est découpé aux stimulations et si \(H_k\) est correctement recalculé. Si les
instants de stimulation, l'intensité ou les paramètres calciques deviennent
des décisions du NLP, leurs dépendances et leurs dérivées doivent être
conservées ; le calcium n'est alors plus une donnée numérique pré-calculable.

## 3. Fatigue : une convolution commune à trois états

Pour \(z\in\{A,T,K\}\), posons \(z_r\) sa valeur de repos et \(\alpha_z\)
son coefficient de fatigue. Le facteur intégrant donne

\[
z(t_k+s)=z_r+e^{-s/\tau_f}(z_k-z_r)
       +\alpha_z\int_0^s e^{-(s-v)/\tau_f}F(t_k+v)\,dv.
\]

En définissant

\[
J(s)=\int_0^s e^{v/\tau_f}F(t_k+v)\,dv,\qquad
J(0)=0,\qquad \dot J=e^{s/\tau_f}F,
\]

on obtient la reconstruction commune

\[
\boxed{z(t_k+s)=z_r+e^{-s/\tau_f}
                    \left[z_k-z_r+\alpha_zJ(s)\right].}
\]

Ainsi, au niveau continu, on peut intégrer seulement \((F,J)\), reconstruire
\(C\) par la formule précédente et reconstruire \((A,T,K)\) par cette
expression. **La convolution n'est connue exactement que si la force l'est**.
Lorsque \(F\) et \(J\) sont discrétisés par Radau-5, il subsiste une erreur
numérique sur la convolution.

Le prototype `analytic` emploie cette formulation \((F,J)\). Elle fonctionne
aussi lorsque \(\alpha_A=0\). Utiliser le temps local \(s\), puis remettre
\(J(0)=0\) à chaque intervalle en transmettant les états reconstruits,
évite les exponentielles croissantes d'un temps absolu très long.

### Formulation équivalente avec \(F\) et \(A\)

Si \(\alpha_A\ne0\), posons

\[
\beta_T=\frac{\alpha_T}{\alpha_A},\qquad
\beta_K=\frac{\alpha_K}{\alpha_A},
\]

\[
d_T=(T-T_r)-\beta_T(A-A_r),\qquad
d_K=(K-K_r)-\beta_K(A-A_r).
\]

En dérivant, les termes en force s'annulent :

\[
\dot d_T=-d_T/\tau_f,\qquad \dot d_K=-d_K/\tau_f.
\]

Il suffit alors d'intégrer \((F,A)\) et de reconstruire

\[
\begin{aligned}
T(t_k+s)&=T_r+\beta_T(A(t_k+s)-A_r)+d_{T,k}e^{-s/\tau_f},\\
K(t_k+s)&=K_r+\beta_K(A(t_k+s)-A_r)+d_{K,k}e^{-s/\tau_f}.
\end{aligned}
\]

Les offsets \(d_{T,k}\) et \(d_{K,k}\) se calculent à partir de l'état
physique complet reçu. Ils ne sont nuls que pour certaines conditions
initiales, notamment le repos suivi d'une propagation cohérente. Les imposer
à zéro pour un état fatigué arbitraire changerait le modèle initial.

Si \(\alpha_A\) est nul ou numériquement trop petit, on utilise \((F,J)\),
ou un autre état de fatigue dont le coefficient est non nul. La formule de
convolution reste valide si les trois coefficients sont nuls. Des temps de
récupération distincts ou des coefficients variant au cours de l'intervalle
invalident la réduction à cette convolution commune.

## 4. Priorité Radau-5 : condenser les équations déjà discrétisées

Ici, « Radau-5 » signifie cinq stages Radau IIA, ordre classique neuf pour une
solution suffisamment régulière sur l'intervalle. Ce nom ne désigne pas le
schéma Radau IIA à trois stages d'ordre cinq. Le tableau utilisé par le
prototype est dérivé des points `casadi.collocation_points(5, "radau")`.

Notons \(\mathcal A\in\mathbb R^{5\times5}\) la matrice du tableau
Runge–Kutta, \(c_i\) ses abscisses et \(\mathbf 1\) le vecteur de cinq uns.
On réserve \(A\), sans calligraphie, à l'état musculaire de capacité. Les
équations de stage sont

\[
\mathbf X=\mathbf 1 x_k+h\mathcal A\mathbf f(\mathbf X).
\]

Avec le forçage calcique évalué aux mêmes stages,
\(\mathbf H_i=H_k e^{-hc_i/\tau_c}\), les cinq stages de calcium vérifient

\[
\boxed{\mathbf C=
\left(I+\frac h{\tau_c}\mathcal A\right)^{-1}
\left(\mathbf1 C_k+\frac h{\tau_c}\mathcal A\mathbf H\right).}
\]

Pour chacun des trois états de fatigue, on obtient

\[
\boxed{\mathbf z=z_r\mathbf1+
\left(I+\frac h{\tau_f}\mathcal A\right)^{-1}
\left[\mathbf1(z_k-z_r)+h\alpha_z\mathcal A\mathbf F\right].}
\]

Le même système linéaire de taille cinq sert à \(A,T,K\). Pour un pas et des
paramètres fixes, ses facteurs peuvent être calculés une fois. En pratique,
on résout ces petits systèmes linéaires ; la notation inverse rend la
dérivation lisible sans imposer le calcul explicite d'une inverse.

Après substitution, seules les cinq forces de stage sont inconnues dans le
sous-problème musculaire isolé :

\[
\boxed{\mathbf F-\mathbf1F_k
 -h\mathcal A\,\mathbf f_F(\mathbf C,\mathbf F,\mathbf A,
                            \mathbf T,\mathbf K,\mathbf u)=0.}
\]

Dans le problème couplé, les états mécaniques de stage restent également
inconnus et interviennent dans \(\mathbf f_F\). Radau IIA est raide-précis
(`stiffly accurate`) : le dernier stage est l'état de fin d'intervalle. La
reconstruction fournit donc directement les cinq états physiques de sortie.

Cette élimination est **algébriquement exacte pour les équations Radau-5
discrètes**, sous réserve de l'inversibilité des petits systèmes linéaires.
Le prototype `full` résout 25 inconnues de stage par muscle ; `condensed`
en résout cinq et reconstruit les vingt autres. La correspondance porte sur
les stages et les extrémités, à la tolérance de résolution et aux arrondis
près. Elle ne suppose ni un état de repos ni des offsets nuls.

Dans une collocation directe pour IPOPT, on peut garder les forces de stage
comme décisions et substituer les expressions des autres stages. Cela
n'impose pas d'introduire un rootfinder interne dans la dynamique. Conserver
les cinq états physiques aux nœuds permet aussi de garder les liaisons locales
entre intervalles, les conditions initiales et les transferts RHO existants.

## 5. Exactitude continue et équivalence discrète ne se confondent pas

Les expressions exponentielles des sections 2 et 3 sont exactes pour les EDO
continues. Les expressions matricielles de la section 4 sont exactes pour la
discrétisation Radau-5. Les valeurs qu'elles produisent ne sont généralement
pas identiques pour un pas fini.

Par exemple, Radau appliqué à \(\dot d=-d/\tau_f\) utilise une fonction de
stabilité rationnelle \(R(-h/\tau_f)\), tandis que la reconstruction continue
utilise \(e^{-h/\tau_f}\). De même, remplacer les stages calciques Radau par
le calcium analytique modifie les valeurs utilisées dans l'équation de force.

| Variante du prototype | Inconnues non linéaires de stage par muscle | Garantie à vérifier |
|---|---:|---|
| `full` | 25 | Référence Radau-5 à cinq états |
| `condensed` | 5 | Même solution discrète que `full` |
| `analytic` avec \((F,J)\) | 10 | Modèle continu équivalent, erreur propre contre DOP853 |

La réduction continue à deux états par muscle conduirait formellement de
22 à 10 états pour quatre muscles et deux états mécaniques, lorsque les
données nécessaires aux reconstructions sont fixées. **Ce n'est pas le
comptage de la condensation locale des stages** : celle-ci peut conserver les
22 états aux nœuds et supprimer seulement les variables musculaires internes
qui sont éliminables. Le comptage effectif des variables et contraintes doit
être extrait du NLP construit, avec ses conventions de duplication des stages.

## 6. Contraintes, objectifs et sensibilités

Toute borne ou tout objectif faisant intervenir \(C,A,T,K\) doit être
réévalué sur leur reconstruction aux mêmes points que dans la référence.
Une borne simple sur un stage éliminé devient potentiellement une contrainte
sur les stages de force et les états d'entrée. Supprimer une variable ne
permet pas de supprimer sa contrainte physique.

Cette règle concerne notamment la positivité de \(A,T,K\), les critères de
fatigue, les bornes de force et les contraintes mécaniques aux stages. Les
singularités \(K+C=0\) et \(T+\tau_2 C/(K+C)=0\) doivent rester exclues par
le domaine physique et l'initialisation. Les trois transcriptions ont besoin
d'une vérification des états reconstruits, pas seulement d'un petit résidu.

Pour des paramètres constants, les reconstructions condensées sont affines
en \(\mathbf F\) et en l'état physique d'entrée. Elles se différencient
directement dans CasADi. Toutes les dépendances symboliques de \(h\), des
paramètres ou de l'historique doivent être conservées lorsqu'ils sont
optimisés ; les traiter comme des nombres changerait les sensibilités.

Si l'on construit ensuite une carte DMS, les forces internes doivent être
résolues. Pour son résidu \(r(\mathbf F,p)=0\), les sensibilités suivent

\[
\frac{\partial\mathbf F}{\partial p}
=-\left(\frac{\partial r}{\partial\mathbf F}\right)^{-1}
      \frac{\partial r}{\partial p},
\]

si le Jacobien est inversible. Ce calcul interne et ses dérivées ont un coût
à mesurer. La condensation de collocation directe offre une première étape
qui conserve les forces internes dans le NLP.

Éliminer aussi les états aux nœuds en les reconstruisant depuis le début de
l'horizon peut créer des dépendances longues entre intervalles. Il faut
distinguer ce choix de la condensation locale. Un gain de dimension peut être
annulé par une perte de parcimonie de la Jacobienne ou de la Hessienne.

## 7. Validation disponible et protocole à poursuivre

La sonde s'exécute depuis la racine du dépôt :

```bash
/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python \
  scripts/validate_ding_radau5_condensation.py \
  --intervals 30 \
  --output-json ding-radau5-condensation-validation/report.json
```

Elle compare `full`, `condensed` et `analytic` à DOP853 sur 18 cas :
30/50 Hz, trois PW constantes et trois états initiaux, dont des états fatigués
avec offsets non nuls. Les paramètres, les échelles et les tolérances sont
inscrits dans le JSON. DOP853 utilise `rtol=2e-13` et une tolérance absolue
par état `2e-14 * SCALE`. Cette référence numérique doit elle-même être
vérifiée par raffinement lorsque la précision visée l'exige.

Les indicateurs distinguent l'erreur aux extrémités contre DOP853, l'écart
entre stages `full` et `condensed`, et les résidus normalisés. Une équivalence
de condensation ne prouve pas que le pas Radau initial suffit à toutes les
conditions physiologiques. Inversement, un écart de `analytic` à `full` n'est
pas en soi un défaut : il faut comparer chacun à la référence indépendante.

Le [rapport du 12 septembre 2026](../../ding-radau5-condensation-20260912/validation.json)
contient 18 trajectoires de 30 intervalles. Sur ces cas, l'écart maximal
`full`–`condensed`, normalisé par les échelles physiques du rapport, vaut
\(1{,}48\times10^{-14}\) aux stages et \(1{,}34\times10^{-14}\) aux extrémités.
L'erreur maximale de force contre DOP853 est de 0,128023 N pour `full` et
`condensed`, et de 0,127982 N pour `analytic`. Ces valeurs étayent
l'équivalence discrète de la condensation ; elles ne montrent pas une
amélioration importante de la précision de force par la variante analytique
dans cet ensemble de cas. Aucun de ces chiffres ne mesure le gain IPOPT/MA57.

Une [exploration complémentaire de la formulation \((F,A)\)](../../local-results/ding-analytic-radau5-prototype-20260912/result.json)
est archivée séparément. Elle examine des cartes implicites locales et leurs
sensibilités avec un facteur mécanique prescrit. Son protocole diffère du
prototype ci-dessus ; elle ne remplace pas le test du cyclage couplé.

Avant un benchmark de production, compléter dans cet ordre :

1. Vérifier les stages et sorties reconstruits, les offsets non nuls, les cas
   sans recrutement, les limites physiologiques et les erreurs après plusieurs
   intervalles. Étendre les PW constantes à des suites variables et reprendre
   les paramètres exacts des muscles et des archives de la campagne.
2. Vérifier les Jacobiennes et Hessiennes des expressions condensées par
   différences finies avec plusieurs tailles de perturbation ; pour une carte
   implicite, comparer aussi aux équations variationnelles intégrées avec une
   référence précise. Inclure les dérivées par rapport aux états initiaux et
   aux PW, puis à la durée si elle est variable.
3. Rebrancher la mécanique réduite et toutes les contraintes/objectifs aux
   mêmes stages. Comparer les trajectoires à commandes imposées avant de
   laisser les solveurs choisir des commandes différentes.
4. Construire un OCP condensé IPOPT/MA57 sur une fenêtre. Vérifier la
   correspondance des objectifs, les résidus reconstruits du problème complet
   et la faisabilité physique avec le Radau-5 existant.
5. Comparer dix fenêtres RHO à warm-start contrôlé. Lancer 150 fenêtres
   seulement si l'équivalence et un gain de temps significatif sont établis.

Le benchmark doit utiliser la même machine, le même nombre de threads, les
mêmes bibliothèques, le même état initial physique, les mêmes commandes
initiales, paramètres, tolérances et conditions mécaniques. La transformation
du warm-start primal est à vérifier ; les multiplicateurs des contraintes
éliminées ne se transfèrent pas naïvement dans le nouveau NLP.

Rapporter séparément construction, compilation éventuelle, première fenêtre,
fenêtres chaudes, évaluations des fonctions/dérivées et résolution linéaire
MA57. Ajouter les nombres d'itérations, variables, contraintes et non-zéros,
ainsi que médiane et p90 des temps chauds. Une amélioration du nombre
d'instructions CasADi ou du temps d'une carte musculaire isolée ne suffit pas
à conclure à une accélération de l'OCP.

## 8. Pourquoi cela pourrait accélérer IPOPT/MA57, et ses limites

La condensation supprime des variables internes et des égalités linéaires
associées. Elle peut diminuer la taille du système KKT qu'IPOPT confie à MA57
et économiser des évaluations de contraintes. En contrepartie, une force de
stage influence plusieurs états reconstruits et plusieurs autres stages ;
les blocs locaux deviennent plus couplés. MA57 exploite déjà la parcimonie du
problème complet, et le gain réel dépend de ce remplissage ainsi que du coût
de reconstruction des contraintes et de leurs dérivées.

L'algèbre n'est pas spécifique à IPOPT : la même condensation peut alimenter
un autre solveur de NLP. Une carte discrète dérivée de ces équations pourrait
aussi être utilisée par FATROP ou ACADOS, après adaptation et vérification de
leurs interfaces. Les avantages de structure et les coûts des sensibilités
diffèrent selon le solveur ; la validation scientifique peut être partagée,
mais les performances doivent être mesurées pour chaque intégration.

La première décision expérimentale est donc précise : déterminer si
l'élimination locale des stages linéaires conserve les résultats Radau-5 et
réduit le coût total d'IPOPT/MA57 sur le modèle réduit. Le prototype fournit
la vérification musculaire initiale ; il ne démontre pas encore ce gain OCP.
