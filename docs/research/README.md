# Préparation de l'étude de l'excitation individuelle

L'étude proposée demande si les signaux électrodermaux prédisent les variations d'excitation déclarée lors d'une seconde session, au-delà de la vidéo et d'une calibration de la façon habituelle de noter. Le critère principal ne récompense pas une simple correction constante du niveau personnel.

La recherche bibliographique a récupéré 529 notices, rapprochées en 483 notices après rapprochement, dont 54 sont retenus et annotés. Elle est documentée, mais sa couverture et la profondeur de lecture restent limitées : 30 références ont été examinées dans leurs sections accessibles et 24 au niveau du résumé ou de la notice. Aucun modèle HABS n'a été entraîné dans cette préparation.

## Parcours de lecture

1. [Synthèse bibliographique et apport possible](LITERATURE_REVIEW.md) : ce qui est établi, discuté et déjà proche de notre idée.
2. [Faisabilité constatée dans la CI](../../data/FEASIBILITY.md) : données disponibles, jointures, sélection et qualité des signaux.
3. [Protocole détaillé](../../journal/01_individual_arousal.md) : question, hypothèses, cible, modèles de référence, calibration, test et règles de conclusion.
4. [Précision et sensibilité à l'effectif](../../data/PRECISION.md) : limites de 25 personnes de test et simulation complète à prévoir.

## Ce que les données permettent aujourd'hui

La cohorte 2 contient 75 identifiants avec au moins 20 essais reliés dans chacun des deux jours, avant contrôle qualité. Elle offre le meilleur point de départ parmi les cohortes examinées. Le schéma proposé est 50 personnes de développement et 25 de test, avec calibration J1 et évaluation J2.

Deux limites empêchent de traiter le protocole comme définitivement gelé : le filtrage amont des annotations n'est pas entièrement documenté, et la conversion GSR atteint ses bornes dans une partie du petit échantillon inspecté. Les questions précises à poser aux organisateurs sont dans la note de faisabilité. Le projet reste une étude prédictive du jeu livré ; il ne prétend pas mesurer une émotion indépendante des autoévaluations.

## Dossier de preuves

| Fichier | Contenu |
|---|---|
| [SEARCH_PROTOCOL.md](SEARCH_PROTOCOL.md) | Périmètre, critères et limites de la recherche |
| [search_log.json](search_log.json) | Sources, dates et requêtes exactes |
| [search_counts.json](search_counts.json) | Comptages de sélection |
| [screening.csv](screening.csv) | Toutes les notices dédupliquées, décisions et motifs |
| [evidence.csv](evidence.csv) | 54 fiches d'extraction, limites et implications |
| [references.bib](references.bib) | Notices bibliographiques importables, avec identifiants Rxxx |
| [inventory.json](../../data/inventory.json) | Résultats agrégés des sondages, empreintes des tables, exécutions CI |
| [precision_scenarios.csv](../../data/precision_scenarios.csv) | Scénarios théoriques de précision, sans résultats prédictifs réels |

Les copies intégrales de textes consultés et les journaux techniques temporaires ne sont pas versionnés. Les DOI, URLs, requêtes et niveaux de lecture conservés permettent de retrouver les sources sans redistribuer leurs textes.

## Reproduire la préparation technique

```bash
# Inventaire des tables/métadonnées, sans dépendance externe
python3 data/inventory.py /chemin/vers/DATASET out/inventory

# Contrôles de préparation et calcul de précision
python3 -m unittest discover -s tests -v
python3 data/planning.py data/precision_scenarios.csv
```

`process.py` appelle le même inventaire, complété par le petit sondage GSR/ECG, sur la plateforme HABS. L'exécution du workflow HABS est manuelle. Les packages du contrôle de signal sont ceux déjà épinglés dans `requirements.txt`. Le sondage local sur `sample/` confirme seulement le format des métadonnées : cet échantillon n'inclut pas les tables de notes ni les fichiers vidéo nécessaires à l'étude.

## Étape suivante

Documenter les réponses sur le filtrage, les unités et les identités, puis finaliser la qualité du signal et le manifeste. L'implémentation de B0–B3 et la simulation du protocole viendront ensuite. Les performances obtenues plus tard devront être séparées de cette préparation prospective.
