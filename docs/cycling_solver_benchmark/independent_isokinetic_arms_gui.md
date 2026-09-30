# Deux bras indépendants isocinétiques : flux GUI

Cette option ne correspond pas au modèle `bilateral_reduced`. Elle assemble deux problèmes unilatéraux distincts : un solveur Ding/RHO et son état de fatigue pour le bras droit, puis une autre instance pour le bras gauche. Les seules quantités communes sont l'horloge et la vitesse imposée \(\omega<0\). Les commandes, les états Ding, les états de fatigue et les trajectoires RHO ne sont jamais échangés entre les bras.

## Cible mécanique

Pour un tour, \(\Delta\theta=-2\pi\). L'opérateur choisit pour chaque bras soit le travail positif cible \(W\) (J/tour), soit le couple moyen résistant équivalent \(\bar\tau\) (N.m), avec

\[
W=2\pi\bar\tau.
\]

Lorsque les deux valeurs sont saisies, le GUI exige cette cohérence. Il ne s'agit **pas** d'imposer un couple externe instantané : la charge instantanée est toujours celle déduite du bilan isocinétique et bornée dans le problème unilatéral. Modifier \(W\) ou \(\bar\tau\) modifie donc le paramètre numérique de travail terminal, pas une dynamique de roue libre.

## Utilisation

1. Lancez le GUI avec l'environnement RHO, puis ouvrez l'onglet **Deux bras indépendants**.
2. Choisissez IPOPT ou ACADOS, l'horizon RHO, la cadence négative et les deux cibles mécaniques. Sélectionnez ou déclarez la fabrique (`module:fonction`) et les deux configurations propres aux modèles unilatéraux.
3. Utilisez **Prévisualiser les deux bras** : elle montre la requête traduite vers le coordinateur. Puis choisissez un dossier de sortie neuf et lancez les deux bras. Le lancement peut être concurrent, mais sans politique de répartition il n'existe pas de barrière cycle-par-cycle : les deux RHO restent temporellement indépendants.
4. Le résultat comprend `right/result.json`, `left/result.json` et `summary.json`. Le GUI ne déclare la réussite que si chaque worker l'a explicitement rapportée avec le nombre demandé de cycles.
5. Comparez travail, fatigue, réserve et temps dans la synthèse. Modifiez les cibles à gauche/droite et relancez. Si le coordinateur fournit une proposition explicite, **Reporter la proposition** copie les couples moyens proposés dans le formulaire; l'opérateur garde la décision de relancer.

## Répartition RHO synchronisée (expérimentale)

Le menu **Répartition de résistance** offre `capacity_feedback`. Il conserve le budget de travail des deux bras constant à chaque tour :

\[
\bar\tau_R^{(k)}+\bar\tau_L^{(k)}=\bar\tau_{\rm total},\qquad
W_R^{(k)}+W_L^{(k)}=2\pi\bar\tau_{\rm total}.
\]

Après les deux solutions certifiées du cycle \(k\), le coordinateur attend les deux résultats, lit la plus petite réserve terminale \(r_i=\min_m(A_{i,m}/A_{\mathrm{scale},i,m})\), puis propose pour le cycle suivant

\[
p_R^*=\frac{r_R^g}{r_R^g+r_L^g},\qquad
\bar\tau_R^{(k+1)}=\bar\tau_{\rm total}\,\Pi\!\left((1-\alpha)p_R^{(k)}+\alpha p_R^*\right),
\]

où \(g\) est le gain de capacité, \(\alpha\) le lissage, et \(\Pi\) applique le pas maximal et le couple minimal de chaque bras. Le gauche reçoit le complément. Les deux cibles terminales `E_prod` sont mises à jour *avant* de lancer la paire suivante; aucun état Ding n'est partagé et aucun graphe NLP n'est reconstruit.

Si une réserve n'est pas présente dans les métriques, la politique conserve exactement la dernière répartition : elle ne déduit jamais une réserve fictive. Cette intégration en mémoire est testée pour le protocole; les solveurs CasADi/IPOPT qui ne sont pas sûrs en threads doivent être exécutés avec un coordinateur à processus séparés et une barrière IPC avant de l'utiliser pour une campagne de référence.

Le champ **Mise à jour de répartition tous les N cycles** vaut 10 par défaut. C'est une limite stricte : les observations intermédiaires sont journalisées, mais ne changent aucune cible avant le dixième cycle certifié du bloc.

Le coordinateur `IndependentArmCoordinator` conserve deux handles compilés séparés. Son appel `set_equivalent_mean_torques(...)` met à jour le paramètre terminal dans ces handles sans recréer le graphe CasADi ni le code ACADOS. Le runner accepte aussi une liste JSON `adjustments` de paires droite/gauche : l'initial et chaque ajustement préplanifié sont alors résolus dans le même processus et écrits sous `adjustments/001/...`, avec `adjustments.json` comme historique. Cette voie garantit que les handles ne sont construits qu'une fois. Une décision humaine prise après la fin du sous-processus nécessite encore une nouvelle session (ou une future interface persistante); le GUI ne prétend donc pas garantir une réutilisation mémoire lors d'une relance manuelle.

## Limites et garde-fous

- Isocinétique uniquement; cette interface ne doit pas être utilisée pour la cadence libre.
- Les deux bras sont indépendants par construction : aucune compensation mécanique, synergie musculaire ou fatigue croisée n'est modélisée.
- Une fabrique doit construire deux handles unilatéraux distincts déjà préparés, implémentant `set_isokinetic_work_target(W, tau)` et `solve_rho(rho_state)`. Sans fabrique, le runner échoue explicitement plutôt que de générer un résultat fictif.
- Les résultats sont une comparaison de deux OCP unilatéraux; ils ne valident pas à eux seuls la géométrie biomécanique bilatérale.
- `capacity_feedback` est une politique causale de supervision, pas une preuve d'optimalité bilatérale ni un modèle de compensation mécanique.
