# Journal de recherche

## État au 5 novembre 2026

La question retenue concerne l'information physiologique sur les variations d'arousal déclaré au-delà de la vidéo et de la notation personnelle. Le [protocole proposé](01_individual_arousal.md) est rédigé ; aucun modèle prédictif n'est ajusté. Le dossier de lecture commence dans [la préparation de l'étude](../docs/research/README.md).

## Compréhension des données

Deux sondages CI réussis, confirmés par une troisième exécution, ont vérifié les tables, les correspondances temporelles et un petit échantillon GSR/ECG. La cohorte 2 offre 75 identifiants avec deux jours et au moins 20 essais reliés par jour, avant qualité du signal. La livraison ne contient aucun essai non masqué marqué comme exclu. Les résultats détaillés, les limites et les unités à vérifier figurent dans [la note de faisabilité](../data/FEASIBILITY.md).

## Décisions

- Cible principale : `rating_arousal`, jamais le label final dérivé.
- Étude limitée à la cohorte 2 et au catalogue connu, avec calibration J1 et prédiction J2.
- Critère principal centré au sein des personnes, insensible à un décalage constant.
- EDA principale ; ECG exploratoire après vérification des unités ; pas de recherche d'architecture profonde.
- Les conventions existantes sont conservées : `process.py`, `requirements.txt` et workflow HABS. Les calculs de préparation locaux utilisent la bibliothèque standard Python ; aucune nouvelle dépendance n'est imposée.

## Historique

| Date | Travail | Résultat |
|---|---|---|
| 2026-11-05 | Recherche bibliographique | 529 notices, 483 notices après rapprochement, 54 fiches retenues ; couverture et lecture documentées |
| 2026-11-05 | Inventaire CI `e1b7b4c` | Jointures et couverture des sessions mesurées sur le montage réel |
| 2026-11-05 | Contrôle de signal CI `a5f68ab` | 60 fenêtres/canal sur 10 personnes ; bornage GSR et échelle ECG à vérifier |
| 2026-11-05 | Inventaire consolidé `979634b` | Résultats confirmés, médianes conventionnelles ; quatre tests locaux réussis |
| 2026-11-05 | Protocole et précision | Hypothèses, références, critères et sensibilités rédigés ; simulation complète et entraînement non exécutés |

## Points ouverts

Confirmer le filtrage des annotations, la provenance des notes des métadonnées, l'unicité des personnes et les unités/calibration périphériques. La liste de questions est prête dans la note de faisabilité ; aucune réponse des organisateurs n'est supposée.
