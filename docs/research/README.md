# Préparation de l'étude de l'excitation individuelle

L'étude demande si l'EDA, l'EEG et l'ECG prédisent les variations d'excitation déclarée lors d'une seconde session, au-delà de la vidéo et d'une calibration personnelle. Le critère principal ne récompense pas une simple correction constante du niveau personnel.

La recherche bibliographique a récupéré 529 notices, soit 483 après rapprochement, dont 54 sont retenues et annotées. Elle est documentée, mais sa couverture et la profondeur de lecture restent limitées : 30 références ont été examinées dans leurs sections accessibles et 24 au niveau du résumé ou de la notice. Ce dossier distingue la préparation initiale des analyses exécutées ensuite.

## Parcours de lecture

1. [Synthèse bibliographique et apport possible](LITERATURE_REVIEW.md) : ce qui est établi, discuté et déjà proche de notre idée.
2. [Faisabilité constatée dans la CI](../../data/FEASIBILITY.md) : données disponibles, jointures, sélection et qualité des signaux.
3. [Protocole détaillé](../../journal/01_individual_arousal.md) : question, hypothèses, cible, modèles de référence, calibration, test et règles de conclusion.
4. [Précision et sensibilité à l'effectif](../../data/PRECISION.md) : limites de 25 personnes de test et simulation complète à prévoir.

## Ce que les données permettent aujourd'hui

La réconciliation exhaustive conserve 22 972 essais vérifiés dans les BDF, répartis entre 314 identifiants. L'usage de capteurs complémentaires permet de conserver 173 identifiants avec deux séances suffisamment observées. Après les contraintes de support par vidéo, les cohortes 1, 2 et 4 comprennent respectivement 10, 26 et 14 identifiants réservés au test. La référence et le développement utilisent 217 autres identifiants. Les analyses restent séparées par cohorte, car l'unicité des personnes entre cohortes n'est pas vérifiée.

Les paramètres et partitions sont maintenant figés dans [development.json](../../results/development.json). Le filtrage amont des annotations, les unités périphériques et les identités entre cohortes ne sont toujours pas entièrement documentés. Les fenêtres invalides restent manquantes ; aucune note n'est reconstruite. Le projet reste une étude prédictive du jeu livré ; il ne prétend pas mesurer une émotion indépendante des autoévaluations.

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

Ces commandes reproduisent la préparation initiale. `process.py` effectue désormais une extraction exhaustive et une validation prédictive sur la plateforme HABS. L'exécution du workflow reste manuelle. Les dépendances sont épinglées dans `requirements.txt`. Le dossier `sample/` ne contient pas les tables et les enregistrements vidéo nécessaires à l'étude complète.

## Étape suivante

Achever la [validation par simulation](../../journal/02_method_validation.md), puis évaluer les modèles figés sur les identifiants réservés. Les performances réelles seront distinguées des mécanismes synthétiques et des contrôles de préparation.
