# Faisabilité de l'étude de l'excitation individuelle

État au 5 novembre 2026. Les constats ci-dessous proviennent de deux sondages réels, confirmés par une troisième exécution de l'inventaire sur les données montées par HABS. Aucun modèle prédictif n'a été ajusté.

## Provenance

- [Inventaire des tables et des événements](https://github.com/mathiaschebbah/habs/actions/runs/42029637910), commit `e1b7b4c`.
- [Contrôle complémentaire GSR et ECG](https://github.com/mathiaschebbah/habs/actions/runs/42030909882), commit `a5f68ab`.
- [Inventaire consolidé avec calcul conventionnel des médianes](https://github.com/mathiaschebbah/habs/actions/runs/42034455933), commit `979634b`.
- Résultats agrégés et empreintes SHA-256 des trois tables : [inventory.json](inventory.json).
- Code : [inventory.py](inventory.py). Les entrées sont lues sans modification. Le sondage complet lit les tables et métadonnées ; le contrôle de signal porte seulement sur 60 fenêtres par canal.

Les descriptions du kickoff, du manuscrit et de la livraison ne désignent pas exactement les mêmes effectifs. Le manuscrit annonce 383 personnes uniques ; la table des participants contient 383 lignes, ce qui ne suffit pas à vérifier l'unicité biologique. L'analyse doit utiliser les observations effectivement disponibles et reliées aux enregistrements.

## Couverture constatée

`labels_train.csv` contient 29 514 lignes, dont 2 266 comportent `TESTSUBJECT`. Il reste 27 221 notes d'arousal numériques sur 317 identifiants. Aucune ligne avec arousal numérique n'a de label final manquant dans cette livraison. Cela n'autorise pas à utiliser ce label comme cible ou prédicteur.

La clé participant–jour–marqueur de départ n'a pas de doublon dans la table. La jointure stricte avec un événement unique de même vidéo, un fichier présent et une fenêtre entièrement comprise dans le segment conserve 20 231 essais. Elle ne trouve pas d'événement pour 6 960 notes et écarte 30 autres essais pour fichier absent ou fenêtre hors segment. Ce sont des résultats de jointure, pas la preuve que toutes ces observations sont irrécupérables. Aucune réaffectation de jour ni reconstruction de note n'a été faite.

| Cohorte | Notes numériques | Essais reliés strictement | Identifiants reliés | Deux jours disponibles | ≥20 essais par jour sur J1 et J2 |
|---|---:|---:|---:|---:|---:|
| 1 | 8 345 | 5 659 | 102 | 34 | 31 |
| 2 | 11 032 | 9 661 | 113 | 75 | 75 |
| 3 | 2 725 | 2 495 | 30 | 18 | 18 |
| 4 | 4 811 | 2 109 | 47 | 0 | 0 |
| 5 | 308 | 307 | 14 | 0 | 0 |

La cohorte 2 constitue le choix principal proposé : 75 identifiants admissibles avant contrôle des signaux, 3 855 essais de calibration et 3 854 essais futurs. Ses notes sont majoritairement continues, contrairement à la cohorte 1 dont toutes les notes sont entières. Le manuscrit décrit un ordre entièrement randomisé dans les cohortes 1 et 2, et des blocs émotionnels dans les cohortes ultérieures.

Sur les 113 identifiants reliés de cohorte 2, les 252 vidéos sont représentées ; 246 ont au moins dix spectateurs. Ce chiffre ne décrit pas encore le sous-ensemble de développement de 50 personnes proposé dans le protocole. Le support devra être recompté après partition et contrôle qualité. Aucun des 75 participants ayant J1 et J2 n'a la même vidéo dans ces deux jours selon les jointures retenues. Le transfert concerne donc des vidéos nouvelles pour cette personne, éventuellement connues chez d'autres participants.

## Sélection et provenance des annotations

Les 27 248 lignes non masquées ont toutes `flagged_outliers=0`. Aucun essai marqué 1 n'est disponible pour une analyse avec/sans exclusions. Le manuscrit décrit des règles d'exclusion fondées notamment sur les notes et la cohérence des labels. L'observation du seul zéro est compatible avec une livraison déjà filtrée, mais ne suffit pas à reconstituer exactement le filtrage amont. Il faut demander sa confirmation aux organisateurs.

Cette limite interdit de conclure sur la distribution complète des réactions atypiques ou sur tous les essais collectés. Le protocole vise explicitement les essais annotés et livrés. Une pondération ne récupère pas automatiquement les notes qui n'ont pas été livrées.

Dans les événements reliés, la différence absolue entre `arousal_score` des métadonnées et `rating_arousal` du CSV atteint 4,416, malgré une médiane nulle. Les champs `rating_tool` indiquent surtout `emotinder`, puis `default`. L'égalité fréquente des nombres n'établit pas une provenance identique. La cible reste le CSV ; les métadonnées ne doivent ni compléter ni remplacer ses notes sans documentation.

Les stimuli ont été sélectionnés en fonction d'associations émotionnelles préalables. La matrice personne–vidéo n'est pas un plan croisé aléatoire complet. L'estimation décrit le protocole de sélection réellement utilisé ; elle ne permet pas d'estimer la réaction à toutes les vidéos chez toutes les personnes.

## Vérification limitée des signaux

Le second sondage examine les dix premiers identifiants admissibles de cohorte 2, les deux jours, et les trois premiers essais reliés de chaque jour. Cet échantillon de commodité n'est pas représentatif du taux de défaut de tout le jeu.

| Mesure | GSR | ECG |
|---|---:|---:|
| Fenêtres de dix secondes | 60 | 60 |
| Échantillons lus | 600 000 | 600 000 |
| Fréquence déclarée | 1 000 Hz | 1 000 Hz |
| Unité de l'en-tête | µV | mV |
| Valeurs finies | 100 % | 100 % |
| Fenêtres plates après traitement appliqué | 9 | 0 |

La GSR est convertie en µS par la fonction livrée `uV_to_uS`. Cette conversion borne les résistances à la table de calibration ; 20,305 % des valeurs converties atteignent 1 ou 40 µS. Ce résultat ne prouve pas à lui seul une saturation physique du capteur : il peut refléter la conversion ou ses unités. Les neuf fenêtres plates sont détectées après cette conversion.

Pour l'ECG, l'étendue médiane de fenêtre vaut environ 1 149 dans l'unité physique retournée par le lecteur alors que l'en-tête indique mV. L'échelle doit être vérifiée avant toute interprétation physiologique ou seuil d'amplitude. La présence de valeurs finies ne démontre pas la qualité des battements détectés.

Aucun audit exhaustif EEG, ECG, respiration ou pupille n'a été réalisé. La pupille ne sera pas utilisée dans l'analyse principale : les vidéos ne sont pas fournies, ce qui empêche un contrôle direct de leur luminance.

## Conséquences pour l'étude

La question est étudiable sous une forme prédictive limitée au jeu livré. La cohorte 2 offre un plan de calibration J1 et test J2. Les effectifs annoncés sont des maxima avant contrôle des signaux, pas une taille finale garantie. La validité de la conversion GSR est une condition préalable au test principal ; l'effectif utilisable et la précision seront réévalués sans consulter le gain physiologique de l'ensemble de test.

Questions à transmettre au mentor avant le gel final :

1. Quels filtres ont produit `labels_train.csv`, et les notes originales des essais exclus peuvent-elles être fournies ?
2. Quelle est la provenance de `arousal_score` dans les métadonnées, et pourquoi diffère-t-il parfois du CSV ?
3. Pourquoi certains couples jour–marqueur du CSV ne figurent-ils pas dans les segments correspondants ?
4. Les identifiants de cohorte 2 correspondent-ils chacun à une personne unique ? Existe-t-il une table d'identité entre cohortes ?
5. Quelles unités, calibration et bornes utiliser pour la GSR et l'ECG de cette livraison ?

Les réponses pourront modifier le protocole par un amendement daté. Elles ne doivent pas servir à sélectionner rétrospectivement la variante qui obtient le meilleur score.
