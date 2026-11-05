# Premier lot de validation synthétique

Le [job HABS](https://github.com/mathiaschebbah/habs/actions/runs/42099628511)
du commit `fa4f356` a terminé 879 répétitions avant sa limite de calcul : 220
pour chacun des mécanismes contexte, style de réponse et stimulus partagé,
et 219 pour le signal individuel injecté. L'objectif de 1 000 répétitions
par mécanisme n'est pas atteint ; `complete` reste faux.

Ces résultats concernent des réponses synthétiques sur les covariables observées.
Ils ne sont pas des performances sur les autoévaluations réelles réservées.
Ils n'identifient pas davantage un état émotionnel dans les données réelles.
Le lot est conservé séparément pendant le réexamen de la question scientifique,
qui exige d'abord de vérifier les répétitions du même contenu chez une personne.

`simulation_replicates.csv` contient chaque répétition terminée avec sa graine,
ses contrastes et ses intervalles. `simulation_summary.json` conserve les
comptages et leur incertitude Monte-Carlo. `simulation_config.json` contient les
empreintes des données et de l'algorithme. Une reprise exacte nécessite ce
commit et les mêmes versions ; ces lignes ne sont pas fusionnées avec une
version différente du code.
