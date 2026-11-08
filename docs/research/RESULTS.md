# Résultats de l'étude HABS

Analyse effectuée à partir du projet choisi le 8 novembre 2026. Les résultats
ci-dessous proviennent des observations humaines HABS, après gel des paramètres.
Les jeux synthétiques des tests logiciels ne servent à aucune conclusion scientifique.

[Papier de quatre pages](manuscript.pdf) · [Supplément de deux pages](supplement.pdf)
· [Résultats numériques](results/study_summary.json) · [Paramètres figés](results/frozen.json)
· [Audit exécuté](results/audit.md).

## Résultat principal

L'ajout EEG + ECG + EDA n'améliore pas de façon démontrée la prédiction du proxy
d'activation aversive par rapport à la référence stimulus/contexte et calibration J1.

| Évaluation réservée, cohorte 2 | Valeur |
|---|---:|
| Identifiants évalués | 22 |
| Essais J2 nouveaux pour la personne | 997 |
| MAE moyenne par identifiant, M1 | 0,112443 |
| MAE moyenne par identifiant, M4 multimodal | 0,112826 |
| Gain M1 − M4 | −0,000383 |
| IC à 95 %, bootstrap apparié de 10 000 tirages d'identifiants | [−0,002877 ; +0,001822] |
| Identifiants aidés par M4 | 10/22 |

L'intervalle contient zéro. Il exclut un gain de 0,01, seuil d'utilité technique
fixé avant le test sur l'échelle 0–1. Ce seuil n'a pas de validation clinique.
La conclusion concerne ces caractéristiques et cette famille de modèles, dans
le catalogue de vidéos observé. Elle ne démontre pas une absence d'information
émotionnelle dans les signaux bruts.

Les 26 identifiants initialement réservés n'ont pas été réassignés : 22 satisfont
les critères finaux de source, de qualité et de support de calibration. Le modèle
de population utilise 72 identifiants et 5 099 essais ; 48 identifiants permettent
la validation interne par groupes. Aucune des 997 vidéos cibles n'est absente de
l'apprentissage de population. Il s'agit donc de contenus nouveaux pour la
personne, connus chez d'autres personnes. Le test officiel masqué de HABS n'est
pas évalué ici.

## Ablation et contrôles

Toutes ces lignes utilisent les mêmes 22 identifiants et 997 essais futurs.

| Modèle | MAE moyenne par identifiant |
|---|---:|
| Vidéo et contexte M0 | 0,115680 |
| Stratégie de calibration sélectionnée M1 | 0,112443 |
| EEG M2 | 0,111879 |
| ECG + EDA M3 | 0,113070 |
| Multimodal M4 | 0,112826 |
| Disponibilité des capteurs seule | 0,112259 |
| Physiologie de J1 seule | 0,112694 |
| EEG + ECG, sans EDA | 0,112597 |
| Splines physiologiques | 0,112631 |
| M5 avec fréquences de clusters J1 | 0,112825 |
| Historique continu par PCA | 0,112814 |
| Prédiction constante zéro | 0,174672 |

Le point EEG seul est légèrement meilleur que M1, mais ne remplace pas le
contraste principal prédéfini et ne suffit pas à établir une découverte.
M4 n'apporte pas de gain démontré face au contrôle de disponibilité des capteurs
ou à la physiologie historique seule. Dans les 495 observations complètes de
15 identifiants, le gain M1 − M4 reste incertain : −0,000926
[−0,004420 ; +0,002259].

Les splines donnent un gain M4 linéaire − splines de +0,000195
[−0,000736 ; +0,001110]. Le contrôle de 100 décalages circulaires conjoints dans
chaque identifiant/séance produit une MAE moyenne de 0,113425. Il s'agit d'un
diagnostic de correspondance essai–physiologie, pas d'un test exact de permutation.

## Spécificité et dose-réponse

Le modèle de valence négative conditionné par l'arousal effectivement déclaré
donne un gain physiologique de −0,000444 [−0,002658 ; +0,001722]. Cette analyse
est explicative ; l'arousal courant n'entre jamais dans le prédicteur prospectif.

Le rapprochement des réponses positives et négatives à arousal élevé forme
123 paires sans réutilisation, à écart d'arousal ≤0,5 point dans une même personne.
L'écart moyen effectif est de 0,092 point. Les différences pour alpha frontal,
fréquence cardiaque et conductance ne présentent pas d'intervalle excluant zéro ;
les supports sont respectivement 20, 13 et 12 identifiants. Les associations de
rang intraindividuelles de ces trois marqueurs avec le proxy sont également
incertaines. Aucun seuil physiologique de stress n'est établi.

Le score entraîné sur les extrêmes utilise 3 321 observations de développement,
avec un seuil haut fixé à 0,56908. Sur 380 essais intermédiaires réservés,
18 identifiants contribuent à la corrélation de rang :

| Score sur les réponses intermédiaires | Corrélation moyenne et IC à 95 % |
|---|---|
| Stimulus/contexte seul | 0,295 [0,132 ; 0,441] |
| Avec physiologie | 0,340 [0,180 ; 0,484] |

Un gradient existe donc déjà sans physiologie. Ces intervalles descriptifs ne
démontrent pas un avantage apparié du score physiologique. Le score binaire
entraîné aux extrêmes n'est pas une quantité calibrée de stress.

## Profils individuels

Deux clusters sont sélectionnés sur 1 596 essais J1 complets de 48 identifiants
de développement, sans utiliser les réponses émotionnelles. Leur silhouette
est de 0,174, ce qui n'établit pas une séparation biologique forte.

Dix identifiants réservés ont des descripteurs complets aux deux séances.
La similarité moyenne de Hellinger est de 0,889 pour la même identité, contre
0,775 entre identités. La différence est +0,113 [0,014 ; 0,210]. Cependant,
0,889 reste dans la référence à 95 % des permutations d'identité par strates
d'effectifs [0,712 ; 0,893]. La reproductibilité n'est donc pas démontrée contre
tous les contrôles de composition et de précision.

L'utilité prédictive n'est pas démontrée : le gain M4 − M5 est de
+0,0000006 [−0,0000162 ; +0,0000179]. M5 n'améliore pas non plus de façon établie
le descripteur continu par PCA. Aucune étiquette « fight », « flight », « freeze »
ou clinique n'est attribuée aux clusters.

## Sensibilités et autres cohortes

Les variantes valence, arousal et composite alternatif ne montrent pas de gain
physiologique clair dans leurs intervalles. L'analyse à cinq secondes utilise
son propre critère de qualité à cinq secondes, sa référence et sa sélection sur
développement : 23 identifiants et 1 060 essais, gain +0,000491
[−0,001025 ; +0,002024]. Ses erreurs absolues ne se comparent pas comme des
mesures appariées à celles de l'analyse complète, car les grilles diffèrent.
Elle prédit un rapport ultérieur, pas un instant annoté de survenue du stress.

Avec les paramètres de cohorte 2 conservés, les analyses exploratoires séparées
donnent :

| Cohorte | Identifiants / essais de test | Gain M1 − M4 et IC à 95 % |
|---|---:|---|
| 1 | 8 / 287 | −0,011687 [−0,032859 ; +0,004981] |
| 4 | 14 / 578 | −0,002293 [−0,006549 ; +0,002776] |

L'identité biologique entre cohortes n'est pas reliée. Ces analyses ne constituent
pas des réplications indépendantes de personnes, ni un test de transfert entre cohortes.
Les critères secondaires restent exploratoires, sans correction de multiplicité.

## Reproductibilité et limites

- Source scientifique évaluée : `29379d851ff817dcccbeaf371b32c9647ee8a759`.
- [Calcul HABS terminé](https://github.com/mathiaschebbah/habs/actions/runs/42533466083) : le calcul et l'enregistrement réussissent ; l'ancien transfert Azure de dossiers échoue ensuite. Les résultats sont récupérés sans réentraînement.
- [Audit CPU réussi](https://github.com/mathiaschebbah/habs/actions/runs/42535323126) : relecture des 11 rapports primaires, concordance des métriques, empreintes de code/données et reconstruction indépendante de l'intervalle principal.
- Empreinte du gel : `0839097862a4fa70675df57fdc25ba77712c5a4bc0ff356147f6832915714d4d`.
- Les annotations individuelles, jeux de données des rapports et manifestes d'essais restent dans le stockage HABS de l'équipe. Seuls les agrégats, les figures et l'audit sont versionnés ici.

Les contrôles automatisés génériques de skore sont majoritairement non applicables
au rapport préajusté sans données d'entraînement. `default_score` échoue sur le
routage de la calibration ; les prédictions sauvegardées, R² et la MAE par personne
enregistrée sont relus correctement. L'audit ne transforme pas ces contrôles
non applicables en preuves d'absence de fuite. Les invariants de découpage et de
disponibilité reposent aussi sur le code, les tests ciblés et le smoke test réel.

L'indicateur reste un proxy d'activation aversive. Les résultats ne valident pas
une mesure clinique du stress, ne datent pas sa survenue et ne prouvent pas
l'absence de signal dans d'autres représentations ou familles de modèles.
