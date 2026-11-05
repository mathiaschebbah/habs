# Faisabilité de l'étude de l'excitation individuelle

Cette note conserve les premiers sondages du 5 novembre 2026. Leurs effectifs ont été révisés par la réconciliation exhaustive ci-dessous. Les valeurs des sections suivantes décrivent l'inventaire initial, avant ces corrections.

## Mise à jour après réconciliation exhaustive

Les 27 221 notes numériques concernent 317 identifiants. La réconciliation conserve 22 972 essais pour 314 identifiants, tous vérifiés dans les annotations temporelles BDF. Elle corrige 2 517 jours erronés et restitue 173 jours absents, seulement lorsque l'événement est unique. Parmi les correspondances encore absentes, 4 206 concernent une vidéo absente des enregistrements disponibles ; sept sont répétées ou contradictoires. Les 36 fenêtres hors segment restent exclues.

| Cohorte | Identifiants alignés | Deux séances suffisantes, EEG | Deux séances suffisantes, EDA | Développement | Test |
|---|---:|---:|---:|---:|---:|
| 1 | 105 | 32 | 21 | 19 | 10 |
| 2 | 113 | 76 | 47 | 50 | 26 |
| 3 | 30 | 25 | 7 | non retenue | non retenue |
| 4 | 52 | 40 | 40 | 26 | 14 |
| 5 | 14 | 0 | 0 | non retenue | non retenue |

Une séance suffisante contient au moins vingt essais. Le recours aux signaux complémentaires conserve 173 identifiants avec deux séances. Les copies prétraitées récupèrent aussi 396 essais EDA. Ces contrôles établissent une lisibilité technique, pas l'absence de tout artefact physiologique.

Après vérification du nombre de spectateurs sources par vidéo, les cohortes 1, 2 et 4 apportent 95 identifiants de développement et 50 de test. La référence utilise aussi 122 identifiants distincts de ces candidats inter-session, soit 267 identifiants contribuant à l'analyse. La cohorte 3 ne satisfait pas le support vidéo requis dans le développement ; la cohorte 5 ne possède pas deux séances. Aucune fusion entre cohortes ne suppose leur indépendance biologique.

Le seuil de 60 personnes du protocole initial est un repère de prudence, pas l'effectif du jeu. Le nombre de personnes de test reste distinct du nombre total utilisé pour construire les modèles. L'élargissement ne justifie pas à lui seul une conclusion confirmatoire.

Sources : [réconciliation](../results/alignment.json), [qualité des trois signaux](../results/multimodal_quality.json), [partition et paramètres figés](../results/development.json). Les résultats prédictifs réservés ne sont pas contenus dans ces fichiers de préparation.

### Répétitions exactes après correction des sessions

L'[audit exhaustif des répétitions](../results/repetitions.json), exécuté par le
[job 42108589456](https://github.com/mathiaschebbah/habs/actions/runs/42108589456),
trouve neuf paires de notes appariables pour une même vidéo et un même identifiant
à des sessions enregistrées différentes. Elles concernent six identifiants :
huit paires sur cinq identifiants de cohorte 2, une paire sur un identifiant de
cohorte 4. Les autres cohortes n'en contiennent aucune.

Les six identifiants ont au moins une variation de note d'au moins 0,5 point,
dont cinq d'au moins un point. Les notes stables n'ont pas été exclues de
l'inventaire. Ces comptages ne permettent pas d'estimer une instabilité générale
des réponses, et ne constituent pas un échantillon suffisant pour entraîner puis
tester un modèle de changements à contenu constant.

L'identité du stimulus repose sur le nom de fichier fourni et normalisé. Les
vidéos ne sont pas disponibles pour une comparaison de leur contenu. En comptant
tous les événements enregistrés des identifiants ayant des notes numériques,
même sans deux notes appariables, l'audit trouve seize paires sur dix identifiants.
Les 176 identifiants avec plusieurs sessions alignées quelconques restent
distincts des 173 ayant deux sessions d'au moins vingt essais exploitables.

Le constat initial de zéro répétition entre J1 et J2 dans la cohorte 2 ne doit
donc pas être étendu aux données corrigées. Une analyse de transfert à d'autres
vidéos devra exclure explicitement ces rares répétitions de la cible future.

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
