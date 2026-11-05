# Validation du protocole par simulation

Statut : préparation de la validation prévue au protocole, avant évaluation des notes réservées. Date : 5 novembre 2026.

## Question

Dans quelle mesure la procédure peut-elle produire un gain prédictif lorsque les réponses synthétiques dépendent du contenu ou d'un style personnel, sans terme de réponse individuelle courante ? Peut-elle détecter un terme individuel injecté ? Les intervalles annoncés sont-ils raisonnablement calibrés dans un cas où le gain vrai est exactement nul ?

## Plan observé et données synthétiques

Les covariables, vidéos, sessions, absences de modalités et partitions de la cohorte 2 restent fixes. La composante commune des notes est obtenue par une référence ajustée seulement sur les personnes de développement et de référence. Toutes les notes, y compris celles de calibration, sont ensuite remplacées par des valeurs synthétiques avant de rejouer l'apprentissage.

Les réponses comportent un décalage personnel, un gain personnel positif, un décalage de session, un bruit hétéroscédastique et une autocorrélation de 0,4 entre essais successifs. Elles sont bornées entre 1 et 9. Les quatre mécanismes sont :

1. Contexte, style aléatoire et bruit, sans relation individuelle courante avec les signaux.
2. Style de notation lié au niveau EEG de la première session, sans terme d'état courant.
3. Composante physiologique commune à la vidéo, identique pour tous ses spectateurs.
4. Terme individuel injecté à partir d'une caractéristique EEG normalisée sur la calibration, après retrait de la composante commune au stimulus.

Le coefficient du terme ajouté vaut 0,5 dans les deux derniers mécanismes. Il ne correspond pas à un effet réel estimé ni à une réduction connue d'erreur de 0,5. Ces mécanismes sont des tests de comportement du protocole, pas des modèles biologiques validés.

## Répétitions et analyse

Objectif : 1 000 répétitions par mécanisme. Chaque répétition reconstruit la référence et la calibration, resélectionne leurs pénalités puis celles des corrections multimodale, moyenne vidéo et disponibilité. Les séparations des participants, les critères de support et la construction hors pli des résidus sont identiques à l'étude principale. Les modèles unimodaux exploratoires ne sont pas nécessaires à cette validation du contraste principal.

Conserver pour chaque répétition la graine, les gains, les intervalles et la durée. Rapporter les taux de gains positifs par rapport à la référence et après comparaison aux contrôles, avec leur incertitude Monte-Carlo. Dans les trois mécanismes sans terme individuel courant, ces taux décrivent le risque d'une interprétation excessive du gain. Ils ne sont pas automatiquement des taux d'erreur de type I pour l'hypothèse prédictive : une correction peut réellement améliorer une référence imparfaite sans mesurer un état individuel.

Une validation supplémentaire, peu coûteuse, utilise des réponses placées au milieu des deux prédictions centrées plus un bruit de moyenne nulle. Le gain d'erreur centré est alors exactement nul en espérance. Mille tirages par condition de bruit, indépendant ou autocorrélé à 0,4, servent à examiner la couverture à 95 % et la probabilité de rejet positif. Une borne inférieure d'intervalle bilatéral à 95 % supérieure à zéro correspond à une queue nominale de 2,5 %, et non à 5 %.

## Reproductibilité et limites

Les exécutions sont bornées dans le temps et enregistrent chaque répétition terminée. Une reprise vérifie les empreintes des covariables et de l'algorithme ; elle ne mélange pas des variantes différentes. Les tableaux publiables ne contiennent que les statistiques synthétiques, sans signaux ni notes individuelles réelles.

La sélection initiale du jeu livré reste fixe dans ces simulations. Les réponses originales exclues par les organisateurs ne sont pas disponibles ; le mécanisme de sélection correspondant n'est donc pas validé. La puissance observée sous le terme injecté ne garantit pas une puissance identique pour les mécanismes émotionnels réels. Aucun résultat de cette validation n'est encore revendiqué.

## Réduction des calculs répétés

Le profilage avant l'évaluation finale a identifié deux répétitions inutiles :
filtrer plusieurs fois une table de calibration partagée par les essais, et
reconstruire les résidus hors pli pour chacune des pénalités de correction.
Ces calculs sont désormais réutilisés au sein d'un même ajustement. La grille,
les plis, les graines, les modèles candidats et les critères restent identiques.

Sur un plan artificiel de 113 identifiants, les paramètres retenus, les scores de
tous les candidats et les prédictions sont exactement égaux avant et après ce
changement. Le temps mesuré passe de 26,7 à 6,3 secondes. Les répétitions du
premier lot conservent leur empreinte initiale et sont archivées séparément ;
la validation complète utilise une nouvelle empreinte. Les résultats communs
aux mêmes graines seront comparés sur les covariables réelles.
