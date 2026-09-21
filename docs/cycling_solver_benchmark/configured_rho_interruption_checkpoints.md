# Protection des campagnes RHO longues contre les interruptions

Le runner configuré accepte désormais `--checkpoint-every 20` et
`--checkpoint-directory DOSSIER_NEUF`, avant le séparateur `--`. Ces options
concernent les conditions IPOPT `rho`, `rho-physio` et `rho-pace`.

Le benchmark sauvegarde le primal **après transfert d'un cycle certifié**, prêt
à initialiser le cycle suivant. Chaque archive contient les états, les commandes
et les métadonnées du problème ; le wrapper ajoute immédiatement l'empreinte et
les paramètres musculaires du modèle, sans attendre la fin de la campagne.
Ces options activent `--retry-failed-rho-without-advance` : une solution non
certifiée ne doit pas contaminer le cycle suivant. Cela peut modifier le nombre
de tentatives lors d'un échec, mais pas les cycles successivement certifiés.

Chaque jalon produit :

- `cycle-20.npz` : primal décalé, publié atomiquement après ajout de l'empreinte ;
- `cycle-20.receipt.json` : empreinte SHA256, nombre de cycles certifiés et
  photographie d'audit des poids PACE au moment de la sauvegarde ;
- `latest.json` : pointe uniquement sur une paire archive/reçu complètement
  publiée ;
- `manifest.json` : modèle effectif, arguments, politique PACE, PID et état du
  lancement.

Le dossier doit être neuf. Les anciennes archives ne sont jamais écrasées. Une
interruption entre publication de l'archive et publication du reçu peut laisser
une archive orpheline : elle n'est pas désignée par `latest.json`. Les JSON et
archives sont synchronisés avant publication. L'audit de configuration utilise
également une écriture atomique.

## Relancer une campagne fraîche avec protection

Conserver les arguments validés de la campagne à 0,15 Nm, en changeant tous les
chemins de sortie vers un dossier neuf. Ajouter avant `--` :

```bash
--checkpoint-every 20 --checkpoint-directory "$RUN/rho/checkpoints"
```

Pour l'autre bras :

```bash
--checkpoint-every 20 --checkpoint-directory "$RUN/rho-pace/checkpoints"
```

La limite demeure `--n-windows 2000`. Le dernier jalon est 1980 : le dernier
cycle n'a pas de fenêtre suivante à préparer et le résultat final décrit la
fin. Aucun FHO ni changement du rollout asynchrone PACE n'est impliqué.

## Ce que ces archives permettent, et ce qu'elles ne garantissent pas

Une archive est un point de récupération vérifiable pour un **nouveau segment**
RHO. Elle n'est pas encore une reprise complète de campagne certifiée. En
particulier, aucune option `--resume` de ce runner ne remet en place l'horloge
absolue PACE, le worker et sa proposition en attente, les multiplicateurs du
solveur et les métriques accumulées. La photographie des poids est une preuve
d'audit, pas une sérialisation complète du contrôleur. Un redémarrage avec ces
seuls poids changerait notamment la référence utilisée par certains calculs
de régularisation.

Avant d'annoncer une reprise équivalente il faudra comparer numériquement la
continuation ininterrompue et la continuation restaurée : états Ding, phase et
vitesse mécaniques, convention/historique de stimulation, premier cycle après
reprise, poids et calendrier PACE. Le manifeste et chaque reçu déclarent
explicitement `complete_campaign_resume_supported: false`.

## Interpréter un arrêt

`launcher_completed` veut dire que le code est revenu normalement ; consulter
le résultat pour les cycles certifiés et la raison d'arrêt. `exception` indique
une exception interceptée. Un manifeste demeuré `running` avec un processus
absent signale une terminaison non finalisée (signal, arrêt de session, arrêt
machine, etc.) ; il ne permet pas d'identifier le signal ni de conclure à une
fatigue physiologique. SIGKILL ne permet aucun traitement final Python : les
derniers reçus constituent alors la protection disponible.

Un échec du solveur, même à un état fatigué, ne constitue pas à lui seul une
preuve d'impossibilité physique. Le test de réalisation à état complet figé
reste nécessaire pour attribuer l'échec à la fatigue.
