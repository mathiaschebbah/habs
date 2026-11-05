# Prédire les variations individuelles d'excitation déclarée

Version 1.0 du 5 novembre 2026. Protocole prospectif préparé après revue bibliographique et examen de faisabilité. Aucun résultat de modèle HABS n'a été utilisé pour choisir ces analyses. Ce document n'est pas un préenregistrement externe. Les conditions de validité des unités et de la sélection des annotations restent à documenter avant le test final.

## 1. Problème scientifique

Une bonne prédiction de l'arousal peut provenir de la réaction habituelle à une vidéo ou de la tendance stable d'une personne à employer certaines notes. Il faut mesurer ce que les signaux de l'essai ajoutent à ces informations. La littérature possède déjà des modèles combinant un prior de population et une correction individuelle ; notre apport concerne la comparaison après calibration, le transfert inter-session et la maîtrise des limites de HABS. Voir la [revue](../docs/research/LITERATURE_REVIEW.md).

## 2. Question et population cible

Chez les participants de cohorte 2 disposant d'une calibration J1 et d'essais J2 exploitables, les caractéristiques électrodermales mesurées pendant une vidéo améliorent-elles la prédiction des variations d'excitation déclarée à J2, au-delà d'une référence utilisant la vidéo, le contexte de présentation et une calibration affine de la notation personnelle ?

La population cible est celle des participants et essais annotés dans la livraison HABS, après les exclusions documentées et le contrôle qualité sans connaissance des performances. Le catalogue de vidéos et le mécanisme de sélection personnalisée sont fixés. Les vidéos de J2 sont nouvelles pour la personne, mais connues chez les participants de développement. Les personnes de test ne contribuent au modèle de population par aucune de leurs sessions. Leur J1 sert exclusivement à une calibration individuelle autorisée par le scénario.

Ce scénario n'est ni une prédiction sans calibration, ni une validation sur des vidéos inconnues, ni une estimation causale de l'effet de la physiologie sur l'émotion.

## 3. Hypothèses et estimand

**H1 principale.** L'ajout des caractéristiques électrodermales diminue l'erreur quadratique centrée au sein de chaque personne à J2, en moyenne entre personnes, par rapport à la référence calibrée.

Pour un modèle m, une personne i et ses essais admissibles j, définir l'erreur `e_mij = y_ij - prediction_mij`, puis :

`L_mi = moyenne_j[(e_mij - moyenne_j(e_mij))²]`.

L'effet principal est `Δ = moyenne_i(L_B2,i - L_B3,i)`. Une valeur positive favorise la physiologie. Les personnes ont le même poids, quel que soit leur nombre d'essais.

Ce critère compare les fluctuations, pas le niveau moyen. Si B3 se contente d'ajouter une constante propre à chaque personne aux prédictions de B2, les erreurs centrées et le score restent exactement identiques. Les moyennes des erreurs de J2 sont calculées seulement lors du score, après gel des prédictions. Elles ne corrigent jamais une prédiction et n'entrent jamais dans l'entraînement. Un score ainsi centré ne mesure pas à lui seul la qualité de prédictions absolues utilisables en temps réel.

**H2 secondaire.** L'ajout des signaux améliore l'ordre des notes à l'intérieur des personnes. Comparer les corrélations de Spearman par personne, puis leurs différences moyennes. Cette analyse est descriptive ; les notes constantes sont comptées et produisent une corrélation non définie, jamais remplacée par zéro.

**H3 exploratoire.** L'ECG apporte une information supplémentaire à l'EDA. Cette hypothèse ne sera examinée que si ses unités et l'extraction des battements sont validées. Elle ne conditionne pas la réussite de H1.

Le gain pratique secondaire sera la différence de MAE brute, en points de l'échelle 1–9. Une baisse de MAE sans amélioration du critère centré sera interprétée comme une amélioration possible du niveau moyen, pas comme une preuve de suivi des fluctuations.

## 4. Faisabilité et critères d'admissibilité

L'[inventaire exécuté](../data/FEASIBILITY.md) trouve 75 identifiants de cohorte 2 ayant au moins 20 essais reliés dans chaque jour, avant contrôle des signaux. Ce sont 3 855 essais J1 et 3 854 essais J2. Aucune vidéo n'est répétée entre leurs deux sessions dans le sous-ensemble relié.

Conserver un essai si :

1. La note `rating_arousal` du CSV est numérique, entre 1 et 9, et non masquée par `TESTSUBJECT`.
2. La clé participant–jour–marqueur est unique ; elle correspond à un seul événement de même vidéo dans la métadonnée.
3. La fenêtre de dix secondes est entièrement dans le segment et le fichier existe.
4. Le contrôle du canal principal passe les règles ci-dessous.
5. La vidéo possède au moins dix participants dans la partie source utilisée pour estimer sa référence. Ce support est recalculé dans chaque pli, jamais sur les personnes de validation/test.

Pour J1, conserver au moins 20 essais de calibration admissibles ; pour J2, au moins 20 essais de test après les mêmes vérifications. Rapporter les exclusions à chaque étape, par personne et par vidéo. Utiliser exactement les mêmes essais pour B2 et B3.

Ne pas éliminer un essai parce que sa note paraît atypique, parce qu'il contredit l'émotion attendue ou parce qu'il augmente l'erreur du modèle. Ne pas utiliser `final_emotion_label`, `flagged_outliers`, les autres autoévaluations ou le temps de réponse comme prédicteurs principaux. Le filtrage amont reste une limite : tous les indicateurs non masqués livrés valent zéro.

Les notes issues des métadonnées ne complètent pas les notes absentes du CSV. Ne pas reconstruire un jour ambigu par simple similarité des notes. Toute réparation de correspondance doit être justifiée indépendamment de la cible, enregistrée et suivie d'un amendement avant évaluation.

## 5. Contrôle du signal et caractéristiques

L'analyse principale utilise l'EDA seule pour garder une question et une chaîne de traitement vérifiables. L'EEG, la respiration et la pupille sont hors de cette première analyse. La pupille demanderait notamment un contrôle de luminance que les vidéos absentes ne permettent pas directement.

Le contrôle déjà effectué a trouvé 9 fenêtres GSR plates sur 60 et 20,305 % des valeurs aux bornes de conversion. Avant toute extraction définitive, vérifier les unités et la calibration avec les organisateurs et comparer les signaux avant/après conversion. Ne pas traiter automatiquement l'atteinte des bornes comme une non-réactivité physiologique.

Après validation de la conversion, appliquer les règles opérationnelles suivantes, sans utiliser les notes : fenêtre complète, au moins 99 % de valeurs finies, étendue non nulle et moins de 5 % d'échantillons aux bornes de la conversion. Le seuil de 5 % est un choix prudent proposé pour cette étude, pas une norme clinique. La sensibilité à 1 % et 10 % sera décrite sur le développement ; le test principal conserve le seuil gelé. Si le contrôle montre que la conversion elle-même est inadéquate, corriger son origine avant de modifier le seuil.

Fenêtre principale : [0,10[ secondes depuis l'apparition de la vidéo. Ne pas inclure les réponses au questionnaire. Les latences électrodermales et la superposition des réponses limitent l'interprétation d'une fenêtre courte ; cette restriction est explicitée plutôt que compensée par un délai optimisé sur le test. Les recommandations de mesure et la littérature sur les latences sont répertoriées dans la revue.

Trois caractéristiques EDA initiales : niveau médian ; différence entre les médianes de [7,10[ et [0,3[ ; écart interquartile sur [0,10[. Elles décrivent des variations du signal, pas des biomarqueurs spécifiques d'une émotion. Éviter une grande recherche de caractéristiques. Une décomposition tonique/phasique constitue une sensibilité de développement, pas une nouvelle tentative sur le test.

La médiane et l'échelle de chaque caractéristique personnelle sont estimées sur J1 seulement. Les coefficients de standardisation globaux et tout modèle imputant une donnée utilisent seulement la partie d'entraînement. Aucun z-score ne doit être calculé sur toutes les sessions d'une personne avant partition.

Pour l'ECG exploratoire : vérifier les unités puis détecter les pics sur le signal continu, extraire la médiane des intervalles RR dans la fenêtre et le changement de fréquence cardiaque. Ne pas annoncer une HRV fréquentielle fiable à partir d'une seule fenêtre de dix secondes. L'extraction et ses contrôles devront être documentés avant H3.

## 6. Références et modèle physiologique

| Modèle | Information autorisée | Rôle |
|---|---|---|
| B0 | Vidéo et informations de présentation disponibles avant la réponse | Mesurer la prédictibilité commune au contenu |
| B1 | B0 et décalage personnel estimé sur J1 | Contrôler une tendance constante à noter haut/bas |
| B2 | B0 et calibration affine personnelle sur J1 | Référence principale, niveau et amplitude personnels |
| B3 | B2 et trois caractéristiques EDA de l'essai | Test principal de l'information supplémentaire |
| B4 | B3 et caractéristiques ECG validées | Exploration multimodale séparée |

B0 est une régression régularisée avec effets de vidéo, catégorie attendue, source `emotinder/default` et ordre relatif dans la session. Les informations attendues ne sont admises que si leur provenance antérieure au visionnage est confirmée ; sinon les retirer de tous les modèles et documenter cette modification avant le gel. L'ordre relatif est dérivé de la chronologie des événements, car les conventions du CSV varient. Les modalités et interactions supplémentaires ne sont pas ajoutées après inspection du test.

Pour une personne nouvelle, B2 prédit `a_i + b_i × B0`, avec `b_i ≥ 0`. Les paramètres sont ajustés sur J1 par moindres carrés pénalisés, avec rappel de `a_i` vers 0 et de `b_i` vers 1. Si la variance des prédictions B0 de calibration est inférieure à 0,1 point², fixer `b_i=1` pour éviter une pente non identifiable. Cette condition est fixée avant le test.

B3 prédit une correction résiduelle par régression linéaire ridge. Les cibles résiduelles d'entraînement sont produites hors pli au niveau des participants : une personne ne contribue pas au modèle B0 qui génère sa propre référence. Sa calibration J1 est autorisée exactement comme elle le sera chez une personne de test. Les caractéristiques EDA peuvent être recentrées par rapport à la moyenne source de la même vidéo ; cette moyenne ne contient jamais les observations de la personne tenue à l'écart.

Dans chaque comparaison, conserver B2 fixé et apprendre sa correction ; ne pas favoriser B3 en lui donnant une calibration ou des métadonnées absentes de B2. Les pénalités candidates sont `{0.1, 1, 10, 100, 1000}` pour les régressions et `{1, 10, 100}` pour les deux pénalités de calibration. Les combinaisons sont choisies seulement dans le développement, sur le critère centré. En cas d'égalité à 10⁻⁶ près, retenir la pénalisation la plus forte. Prévoir un maximum de 25 configurations de correction après sélection de B2, pas une recherche ouverte d'architectures.

## 7. Partition et prévention des fuites

Une seule cohorte évite le problème documenté de réinscriptions sous plusieurs identifiants entre cohortes, sans démontrer à elle seule l'unicité des personnes à l'intérieur de la cohorte 2. Cette unicité doit être confirmée.

Après contrôle qualité aveugle aux gains prédictifs, trier les identifiants admissibles par SHA-256 de la chaîne `habs-arousal-v1|participant_id`. Affecter le premier tiers arrondi au supérieur au test, les autres au développement. Avec 75 personnes, cela donne 25 test et 50 développement. Conserver les deux journées d'une personne dans son groupe et écrire le manifeste avec empreinte avant l'ajustement des modèles.

Utiliser cinq plis internes par participant pour sélectionner B0/B2/B3 dans le développement. À l'intérieur de chaque entraînement interne, construire à nouveau les références et résidus hors pli ; les résidus calculés une fois sur tout le développement ne doivent pas être réutilisés pour valider un hyperparamètre. Les unités, critères de qualité et traitements déterministes restent les mêmes dans les deux groupes.

Après sélection, réajuster les modèles sur le développement. Pour chaque personne de test, estimer seulement les paramètres de calibration prévus à partir de J1, puis produire et sauvegarder toutes les prédictions J2 avant d'en calculer les scores. Les labels J2 du test ne doivent pas servir à choisir les caractéristiques, les fenêtres, le niveau de normalisation ou les pénalités.

L'examen de faisabilité a déjà utilisé la disponibilité des notes de l'ensemble livré et des résumés globaux. Le test n'est donc pas un jeu complètement vierge de toute inspection descriptive. Aucun score physiologie–cible n'a été calculé ici. Cette distinction devra être conservée dans le compte rendu.

## 8. Contrôles d'interprétation

- **Décalage constant.** Vérifier sur données synthétiques que l'ajout de n'importe quelle constante par personne ne change pas le critère principal.
- **Notation affine.** Comparer B1 à B2 et rapporter l'amélioration brute et centrée. Une correction d'échelle peut améliorer les amplitudes sans réordonner les réactions.
- **Information commune à la vidéo.** Remplacer les caractéristiques de l'essai par les moyennes physiologiques de cette vidéo dans le développement. Réentraîner une correction de même capacité. Si elle suffit à expliquer le gain de B3, ne pas attribuer ce gain à une information individuelle.
- **Information stable de la personne.** Remplacer chaque caractéristique d'essai par son résumé J1 propre à la personne, avec la même référence B2. Le critère centré devrait empêcher une simple correction constante de gagner.
- **Ordre et dérive.** Rapporter le gain par tiers chronologique et vérifier la sensibilité à l'ordre et à la vidéo précédente, représentée seulement par des informations déjà connues.
- **Décalage des signaux.** Décalages circulaires des caractéristiques au sein des sessions comme diagnostic de mauvaise correspondance temporelle. Ne pas les traiter comme un test de permutation exact : l'ordre des vidéos, les dépendances temporelles et la qualité peuvent violer l'échangeabilité.

Ces contrôles sont définis avant le test, mais restent secondaires. Ils encadrent l'interprétation ; ils ne permettent pas de rechercher après coup un sous-groupe favorable. Les contrôles avec des caractéristiques substituées doivent être évalués sur les mêmes essais et avec le même budget d'ajustement.

## 9. Incertitude et précision

Calculer les différences `Δ_i` appariées entre modèles, une par personne de test. L'intervalle principal à 95 % est obtenu par 5 000 rééchantillonnages de personnes de test entières, après gel du modèle et du catalogue, avec graine 20261105. Ne pas rééchantillonner les fenêtres comme indépendantes. Fournir aussi la distribution des `Δ_i`, leur moyenne, médiane et un intervalle de Student en sensibilité. Ne pas interpréter la dispersion des cinq scores internes comme un intervalle de confiance final.

Cette incertitude est conditionnelle au modèle ajusté, au catalogue et au protocole de sélection. Elle ne couvre pas la variabilité d'un nouvel entraînement, de nouveaux stimuli ou d'une autre cohorte. Une extension exigeant ces généralisations demandera un rééchantillonnage adapté et un réajustement complet.

Seuil d'intérêt proposé : `δ*=0,10 point²` de réduction de l'erreur centrée. Il s'agit d'un seuil opérationnel pour rendre les conclusions interprétables, sans validation clinique. Le confirmer ou le modifier par un amendement avant consultation du test.

L'[analyse de précision](../data/PRECISION.md) fournit une sensibilité à plusieurs dispersions supposées de `Δ_i`. Elle n'est pas une puissance garantie. Avec 25 personnes de test, l'effet détectable à 80 % de puissance dans une approximation normale bilatérale correspond à environ 0,56 écart-type interpersonnel des gains. Les milliers d'essais ne remplacent pas les 25 personnes.

Avant le gel final, simuler le protocole complet sur la matrice personne–vidéo observée, avec des données synthétiques comprenant effets vidéo/personne, style de niveau et de pente, dépendance de session, sélection des stimuli et taux de perte observé. Scénarios : absence d'information physiologique ; information stable de la personne seulement ; information du stimulus seulement ; gains progressifs de variation individuelle. Vérifier le taux de faux positifs à 5 %, la couverture et la puissance. Au moins 1 000 réplications par scénario, avec erreur Monte-Carlo rapportée. Ce test complet dépend de la chaîne de features et de modèles qui n'est pas encore implémentée ; aucun résultat de cette simulation n'est revendiqué ici.

Si moins de 60 personnes restent admissibles après qualité, ou si la précision simulée est insuffisante pour `δ*`, maintenir l'étude comme estimation exploratoire. Ne pas lancer une recherche de modèles destinée à obtenir une significativité.

## 10. Règles de conclusion

| Observation | Conclusion permise |
|---|---|
| Borne inférieure de l'IC de Δ > 0 | Information prédictive supplémentaire sur les fluctuations dans ce scénario |
| Borne inférieure > δ* et contrôles compatibles | Gain au moins aussi grand que le seuil opérationnel proposé |
| Gain centré positif mais aucun gain de rang, ou contrôle commun à la vidéo comparable | Interprétation individuelle limitée ; étendue de notation ou meilleure référence stimulus encore plausibles |
| MAE brute améliorée, critère centré non amélioré | Correction du niveau possible, sans preuve de meilleure prédiction des fluctuations |
| IC recouvrant zéro et des gains utiles | Résultat indéterminé, pas preuve d'absence de signal |
| Borne supérieure de l'IC < δ* | Gains supérieurs au seuil non soutenus pour la méthode et le scénario étudiés |
| Échec d'unités, de jointure, de séparation ou de calibration des tests | Résultat méthodologiquement non interprétable ; corriger avant toute conclusion |

Ne pas conclure que les participants se trompent sur leurs émotions, que la physiologie est une vérité terrain indépendante, ou qu'aucune autre méthode ne pourrait réussir.

## 11. Exécution et livrables

1. Faire confirmer les points de provenance, d'identité et d'unités de la note de faisabilité. Archiver les réponses dans un amendement.
2. Valider les jointures et la conversion sur des cas contrôlés ; produire le manifeste et le diagramme d'inclusion sans regarder le gain final.
3. Extraire les caractéristiques et figer les versions de dépendances, unités, fenêtres et règles qualité.
4. Implémenter B0–B3, leurs séparations et contrôles ; vérifier les invariances et les simulations nulles.
5. Réaliser la sélection interne, documenter toutes les configurations, puis geler code et paramètres.
6. Exécuter une seule évaluation finale, conserver les prédictions, les scores et les intervalles. Les analyses ultérieures seront clairement exploratoires.
7. Rédiger un rapport court avec deux figures principales : distribution des gains individuels et comparaison B0–B3 avec incertitude. Ajouter un schéma de calibration/test et le diagramme d'inclusion en annexe.

Les sorties reproductibles attendues sont un manifeste d'inclusion/partition, une table de caractéristiques par essai, les prédictions hors échantillon, les différences par personne et un rapport d'exclusions. Les données et signaux individuels restent dans l'espace autorisé ; publier seulement ce que les conditions HABS permettent. Les jobs durent moins d'une heure et sauvegardent leurs résultats par étape. Le code de préparation actuel ne lance pas automatiquement cette étude.

## Mise en œuvre du 5 novembre 2026

L'étude est autorisée à l'exécution par la demande de mener les recherches jusqu'aux résultats et au papier. Le premier traitement exhaustif porte sur la qualité et les caractéristiques déterministes de cohorte 2, sans évaluation de modèle. Les versions de dépendances sont épinglées. Les tables individuelles restent dans le stockage HABS ; les journaux publics ne contiennent que des agrégats.

La conversion GSR utilisée est celle documentée et livrée dans le dépôt. Son bornage fait l'objet d'une analyse explicite. Sans vérification externe des unités, les conclusions porteront sur cette mesure opérationnelle et ses caractéristiques, sans validation physiologique absolue. La sensibilité sur le voltage non converti sera évaluée sur le même ensemble d'essais que l'analyse principale ; elle ne remplacera pas un résultat défavorable de celle-ci.

### Extension de la recherche des observations exploitables

À la demande d'élargir l'échantillon, l'examen de qualité s'étend à toutes les cohortes avant toute performance prédictive réelle. Une annotation dont le jour ne concorde pas avec les fichiers peut être reliée à un autre jour seulement si le triplet participant–marqueur–vidéo désigne un unique événement enregistré. Les valeurs des notes ne servent jamais à choisir une correspondance. Les duplications d'une même mesure après réparation sont exclues et comptées. Les copies CURATED et PREPROCESSED d'un événement sont rapprochées pour ne pas créer de fausse ambiguïté.

Les données possédant une seule session exploitable pourront contribuer à la référence de population si elles sont disjointes de tous les participants candidats au test inter-session. Cette adaptation, motivée par la disponibilité des mesures et décidée avant tout score de modèle, sera rapportée séparément du protocole initial. Les modalités manquantes et les différences entre cohortes seront étudiées avant de fixer l'analyse élargie.

### Contrôle multimodal après réconciliation

Le contrôle d'identité des événements a récupéré 2 517 jours incorrects et 173 jours absents. La cohorte 4 retrouve notamment des essais de jour 2 précédemment annoncés au jour 1. L'extraction élargie vérifie aussi les marqueurs temporels dans les annotations BDF, avec une tolérance de 10 ms, puis calcule séparément la disponibilité EDA, EEG et ECG. Une modalité absente ne rendra pas automatiquement les autres inutilisables.

Les caractéristiques EEG proposées sont les puissances relatives thêta, alpha et bêta moyennées dans deux groupes de quatre électrodes, sur les signaux prétraités quand ils existent. Les caractéristiques ECG sont une fréquence cardiaque médiane et sa variation, calculées à partir des intervalles entre pics pour éviter une dépendance à l'unité d'amplitude non confirmée. Ces mesures ne sont pas des indices validés de HRV sur dix secondes. Leur intérêt prédictif et les contrôles de données manquantes restent à évaluer ; aucune performance réelle n'a encore été observée à ce stade.

### Analyse élargie avant observation des performances

L'audit multimodal du 5 novembre a relié et vérifié temporellement 22 972 essais, sur 314 identifiants. Les critères de lisibilité de l'EEG permettent de conserver 173 identifiants ayant deux sessions d'au moins 20 essais, contre un effectif beaucoup plus réduit avec la seule GSR. Les sessions sont les deux premières sessions suffisamment observées, dans l'ordre enregistré ; leur numéro d'origine reste conservé dans les sorties.

L'analyse élargie compare EDA, EEG, ECG et leur combinaison sur la même grille de test. Les valeurs d'une modalité invalide restent manquantes. Leur remplacement utilise la calibration personnelle lorsqu'au moins cinq valeurs sont disponibles, puis uniquement les statistiques de la population d'entraînement. Les caractéristiques standardisées sont bornées à ±10 pour limiter les mesures extrêmes. Des indicateurs de disponibilité sont inclus et font l'objet d'un modèle de contrôle spécifique. Le contrôle « vidéo » remplace les mesures individuelles par les moyennes de cette vidéo chez les participants sources, avec les mêmes indicateurs de disponibilité.

Les cohortes sont ajustées et évaluées séparément. Cela évite de placer une éventuelle réinscription sous un autre identifiant dans une cohorte d'entraînement opposée à une cohorte de test. Les intervalles sont calculés à l'intérieur de chaque cohorte ; les effectifs ne seront pas présentés comme autant de personnes biologiquement distinctes entre cohortes sans table de correspondance. La cohorte 2 reste l'analyse principale, les autres cohortes des réplications exploratoires. La cohorte 5 n'a qu'une session et ne permet pas le test inter-session.

Les personnes sans deux sessions suffisantes peuvent contribuer à la référence par vidéo, dans leur propre cohorte, à condition d'être disjointes de tous les candidats à l'évaluation inter-session. Les analyses conservent le seuil de dix spectateurs sources par vidéo et les vingt essais par session ; les pertes à cette étape seront publiées. Aucune cohorte ne sera remplacée discrètement par une variante plus favorable après consultation de ses performances.

Les traitements de signal utilisent des enregistrements complets, notamment l'EEG prétraité fourni et le seuil de détection ECG. Il s'agit d'une analyse hors ligne, sans revendication de fonctionnement causal en temps réel. Les notes des sessions de test n'interviennent ni dans ces traitements ni dans la sélection des modèles. Les scripts ont été vérifiés sur un cas synthétique à effet connu ; ces résultats synthétiques ne constituent pas des résultats HABS.

Le contrôle par moyenne physiologique de vidéo conserve exactement la calibration physiologique personnelle de la première session. Seules les mesures courantes sont remplacées par les moyennes sources. Il contrôle ainsi une explication par réponse commune au contenu et mise à l'échelle personnelle, et pas seulement une réponse commune non personnalisée. Cette comparaison est fixée avant la première analyse réelle de performance.

La sensibilité au voltage GSR non converti utilisera le même masque de validité et la même pénalisation que le modèle EDA. Cette comparaison modifie seulement la représentation des mesures admises, sans recherche supplémentaire de paramètres ni réintroduction de fenêtres saturées. Les corrélations de rang non définies ne sont pas remplacées par zéro ; leur effectif sera affiché.
