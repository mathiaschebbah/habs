# Journal de recherche

## État au 5 novembre 2026

La question centrale est réexaminée après l'[audit des répétitions](../results/repetitions.json). La [comparaison des questions possibles](../docs/research/QUESTION_REASSESSMENT.md) distingue personnalisation des prédictions et valeur des capteurs. Aucune nouvelle orientation n'est encore retenue. Le développement de l'ancien protocole est terminé, mais aucune performance réelle réservée n'a été évaluée. Son premier lot de simulation est archivé comme validation partielle. Le dossier de lecture commence dans [la préparation de l'étude](../docs/research/README.md).

## Compréhension des données

L'extraction exhaustive relie 22 972 essais sur 314 identifiants et vérifie tous les liens dans les marqueurs BDF. Elle conserve 173 identifiants avec deux séances suffisamment observées, en utilisant les signaux disponibles séparément. Après vérification du support des vidéos, les modèles emploient 267 identifiants dans trois cohortes, dont 50 réservés au test. La livraison ne contient aucun essai non masqué marqué comme exclu. Les limites et les unités à vérifier figurent dans [la note de faisabilité](../data/FEASIBILITY.md).

## Décisions

- Cible principale : `rating_arousal`, jamais le label final dérivé.
- Cohorte 2 principale, cohortes 1 et 4 en réplication exploratoire, avec catalogue connu et calibration sur la première séance suffisante.
- Critère principal centré au sein des personnes, insensible à un décalage constant.
- EDA, EEG, ECG et combinaison sur la même grille ; contrôles par moyennes physiologiques des vidéos et disponibilité des capteurs.
- Paramètres sélectionnés seulement dans le développement, puis figés ; les sorties finales doivent reproduire exactement les empreintes des données et de la partition.
- Code sous les conventions du dépôt, dépendances épinglées et sorties individuelles conservées dans HABS.

## Historique

| Date | Travail | Résultat |
|---|---|---|
| 2026-11-05 | Recherche bibliographique | 529 notices, 483 notices après rapprochement, 54 fiches retenues ; couverture et lecture documentées |
| 2026-11-05 | Inventaire CI `e1b7b4c` | Jointures et couverture des sessions mesurées sur le montage réel |
| 2026-11-05 | Contrôle de signal CI `a5f68ab` | 60 fenêtres/canal sur 10 personnes ; bornage GSR et échelle ECG à vérifier |
| 2026-11-05 | Inventaire consolidé `979634b` | Résultats confirmés, médianes conventionnelles ; quatre tests locaux réussis |
| 2026-11-05 | Protocole et précision | Hypothèses, références, critères et sensibilités rédigés ; simulation complète et entraînement non exécutés |
| 2026-11-05 | Réconciliation exhaustive | 2 690 annotations récupérées par correction ou restitution du jour ; 314 identifiants alignés |
| 2026-11-05 | Qualité des trois signaux | 173 identifiants avec deux séances ; 396 essais EDA récupérés dans les copies prétraitées |
| 2026-11-05 | Développement CI `a332226` | Paramètres figés pour les cohortes 1, 2 et 4 ; effectifs de test 10, 26 et 14 ; cohorte 3 exclue pour support vidéo insuffisant |
| 2026-11-05 | Validation CI `fa4f356` | Premier lot de simulation lancé, avant les performances réelles réservées |

## Points ouverts

Confirmer le filtrage des annotations, la provenance des notes des métadonnées, l'unicité des personnes et les unités/calibration périphériques. La liste de questions est prête dans la note de faisabilité ; aucune réponse des organisateurs n'est supposée.

## Exécution de l'étude

L'examen de toutes les cohortes récupère 2 690 annotations par correction ou restitution du jour, puis vérifie 22 972 liens dans les marqueurs BDF. Les données alignées couvrent 314 identifiants. En utilisant les modalités disponibles plutôt qu'en exigeant la GSR, 173 identifiants disposent de deux sessions d'au moins 20 essais. Les 4 206 vidéos absentes des enregistrements disponibles et les sept correspondances restantes incertaines ne sont pas reconstruites.

Le développement des modèles est terminé après des tests de séparation des participants, de masquage des notes futures, de conservation de la grille de prédiction et de comparaison à calibration identique. Les contrôles de prédiction ont également réussi sur les données réelles de développement. L'essai complet sur données synthétiques fonctionne ; il ne constitue pas un résultat HABS. Les performances des personnes réservées au test ne sont pas encore évaluées. Une analyse indépendante par cohorte évite les fuites entre éventuelles réinscriptions.

Les calculs utilisent un environnement pip/venv, pandas, scikit-learn, skrub et des rapports skore locaux. Les versions sont épinglées. Les tables et rapports individuels restent dans le stockage privé HABS ; seuls les agrégats et le code sont versionnés.

## Réexamen de la question avant les résultats réservés

Le protocole actuel porte sur des vidéos différentes entre calibration et test.
Il ne permet pas de distinguer un état momentané, des préférences stables pour
certains contenus et un style de réponse résiduel. Une question portant sur le
changement de réponse au même stimulus demande des répétitions exactes chez
la même personne. L'absence de telles répétitions dans l'inventaire initial de
cohorte 2 doit être vérifiée sur toutes les correspondances corrigées.

Le lot de simulation déjà lancé sera conservé, sans lancement automatique de
nouveaux lots. `data/repetition_audit.py` inventorie les événements et les notes
appariables, sans sélectionner les personnes selon le changement de leur note.
La portée scientifique sera redéfinie à partir de ces observations avant toute
évaluation des performances réelles réservées.

L'audit CI `1d54164` trouve neuf paires de notes répétées sur six identifiants,
dont huit paires en cohorte 2 et une en cohorte 4. Même l'inventaire des événements
sans exigence de deux notes appariables ne trouve que seize paires sur dix
identifiants. Le changement de réponse à contenu constant n'offre donc pas un
échantillon adapté à une analyse prédictive indépendante dans cette livraison.
La question recommandée porte désormais sur l'utilité transférable d'une séance
de réponses personnelles, avec MAE brute et exclusion des rares stimuli déjà vus.
Cette recommandation reste à retenir explicitement avant une nouvelle analyse.

Le lot de validation antérieur a terminé 879 répétitions synthétiques sur les
4 000 visées. Il est conservé dans `results/validation_pilot/`, avec son empreinte
et son statut incomplet. Il ne constitue pas un résultat sur les réponses réelles.
