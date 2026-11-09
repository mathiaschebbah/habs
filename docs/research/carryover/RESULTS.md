# Résultats : décalage ressenti–EEG entre J1 et J2

Étude 02, calculée sur les observations humaines HABS et relue par un audit
en lecture seule des rapports sauvegardés. Le [papier](manuscript.pdf) compte cinq
pages ; le [supplément](supplement.pdf) trois. Les simulations et tests logiciels
ne servent pas de preuves d'un effet émotionnel.

## Réponse à la question

**Nous n'établissons pas de décalage individuel ressenti–EEG reproductible entre
J1 et J2 avec cette mesure et ce protocole. L'ajout de sa correction personnelle
n'améliore pas de façon démontrée la prévision en J2.**

Cela ne démontre ni une absence de corrélats EEG de l'émotion, ni une absence de
persistance émotionnelle, ni une propriété des qualia. La question testée porte
sur la stabilité d'une différence précise de coefficients d'histoire, après
contrôle des vidéos, des délais et des échelles personnelles.

## Faisabilité réellement mesurée

| Étape | Observations |
|---|---:|
| Essais évalués de cohorte 2 | 9 726, 113 identifiants |
| Transitions consécutives avec notes et EEG CURATED valides | 6 759 |
| Personnes avec ≥20 transitions par séance avant restriction de nouveauté | 54 |
| Panel final, vidéos actuelles nouvelles pour la personne | 53 identifiants |
| Transitions finales | 4 554 : 2 294 J1 et 2 260 J2 |
| Vidéos actuelles dans l'audit de transitions | 252 |
| Transitions distinctes entre identités de vidéos | 6 222 |
| Délai de référence, médiane J1 finale | 15,957 s |

Les prédécesseurs sont déterminés dans l'inventaire complet avant d'exclure les
essais sans note ou sans EEG exploitable. Aucun essai invalide n'est sauté pour
fabriquer une paire. Les jours sont ceux des enregistrements réconciliés.
Les 106 colonnes d'histoire personnelle restent identifiables dans chaque séance
après les contrôles de contexte ; leurs conditionnements sont 5,17 et 19,13.

Cette faisabilité structurelle ne garantit pas une bonne précision sur les
différences individuelles. Les 4 554 transitions ne constituent pas autant de
personnes indépendantes.

## Ce que mesure le « décalage »

Le ressenti est ici l'arousal déclaré après la vidéo. Le marqueur EEG est
l'opposé du logarithme de la puissance alpha relative postérieure, 8–13 Hz
relativement à 1–40 Hz. Les deux mesures sont normalisées par l'information J1.

Pour chaque personne et chaque séance, on estime l'influence de l'arousal
précédent sur l'arousal actuel et sur l'EEG actuel, en contrôlant aussi l'EEG
précédent, les identités de vidéos actuelle/précédente, les descripteurs attendus,
l'ordre, le temps et le délai. Leur différence est le contraste individuel delta.
Il ne s'agit pas d'un retard mesuré en secondes du ressenti par rapport au cerveau.

La reproductibilité de delta est estimée avec un modèle à composantes de variance
qui conserve la covariance complète de ses erreurs d'estimation, les dépendances
temporelles et les contrôles de longueur de séance et de rapport d'échelles J1.

## Résultat principal

| Paramètre | Estimation |
|---|---:|
| Covariance interséance des contrastes individuels ajustés | +0,003496 |
| IC à 95 %, bootstrap paramétrique, 999 tirages | [−0,010277 ; +0,015055] |
| Test sous covariance nulle, bootstrap paramétrique bilatéral | p = 0,253 |
| Écart-type interindividuel estimé, J1 / J2 | 0,0450 / 0,0781 |
| Plage de profil approximative de corrélation | [−0,995 ; +0,995] |

La covariance est compatible avec zéro. La corrélation est **mal identifiée** :
son maximum numérique proche de 0,995 ne signifie pas une stabilité presque
parfaite. Le profil de vraisemblance accepte pratiquement toute sa plage.

Les erreurs-types médianes des contrastes estimés valent environ 0,246 en J1 et
0,297 en J2, contre des écarts-types interindividuels estimés de 0,045 et 0,078.
Dans ce modèle, le contraste personnel est donc difficile à séparer du bruit
d'estimation. La corrélation brute des pentes, 0,149, reste descriptive et ne
remplace pas cette analyse.

Les intervalles sont conditionnels au modèle normal, à la covariance d'erreur
estimée et au catalogue de vidéos. Ils ne corrigent pas toute erreur de mesure
dans les prédicteurs retardés. Les petits effets stables restent possibles.

## Vérification prédictive en J2

La cible prédite est **D = arousal standardisé − EEG standardisé**, et non le
stress ou l'arousal seul. Les valeurs d'erreur ci-dessous ne se comparent donc
pas directement à la MAE 0,112 de l'étude 01, qui portait sur une autre cible.

Cinq plis de personnes séparent l'apprentissage de population des personnes
évaluées. Seule leur J1 sert à la calibration personnelle ; les réponses J2
précédentes sont autorisées comme histoire déjà observée à un pas. Les paramètres
sont fixés avant les ajustements : ridge de population 10, pénalité personnelle 20.

| Prévision figée, mêmes 53 personnes / 2 260 transitions J2 | MAE moyenne par personne |
|---|---:|
| Ablation exacte de la correction personnelle d'arousal précédent | 1,212366 |
| Correction personnelle complète | 1,217905 |
| Calibration restreinte réajustée | 1,213779 |

- Gain complet contre ablation : **−0,005539**, IC participant à 95 %
  **[−0,012251 ; +0,000938]** ; 21/53 personnes aidées.
- Gain complet contre restriction réajustée : **−0,004125**,
  IC **[−0,011034 ; +0,002534]** ; 22/53 personnes aidées.

Les deux intervalles contiennent zéro. Le point moyen est moins bon avec la
correction personnelle, sans démonstration d'une dégradation statistique dans
ces intervalles. Aucun avantage prédictif n'est établi pour cette correction.
Les intervalles rééchantillonnent 10 000 fois les personnes, pas les lignes.

La moyenne globale par personne et la moyenne des cinq métriques de pli ne sont
pas identiques lorsque les plis contiennent des nombres de personnes différents.
Les résultats ci-dessus utilisent la moyenne globale à poids égal par personne,
reconstruite depuis les prédictions hors personne.

## Sensibilités et simulation de précision

Les profils de corrélation restent compatibles avec zéro dans les variantes
prédéfinies : covariance d'erreur divisée/multipliée par deux, trente paires
minimum par jour, suppression du modérateur de longueur, autre covariance
temporelle et échelles J1 communes. La restriction aux délais dans l'intervalle
interquartile de J1 ne conserve que 17 personnes / 969 transitions, avec un profil
très large. Le filtre d'échelles J1 plus stables ne retire aucune personne et
reproduit donc la définition principale.

Les simulations conduites **avant les effets empiriques** montrent pourquoi la
prudence est nécessaire. Dans un scénario où seules les échelles communes à J1
et J2 créent une structure, la corrélation brute médiane atteint environ 0,56
malgré une covariance personnelle résiduelle génératrice nulle. Le modèle ajusté
recentre cette covariance près de zéro. Ces chiffres simulés décrivent un piège
d'estimation, pas une découverte sur nos participants.

## Audit et reproduction

- [Calcul scientifique réussi](https://github.com/mathiaschebbah/habs/actions/runs/42618363214),
  backend `20261109T123726-87e44e`, source `5b079a8`.
- Gel à **12:37:36 UTC le 9 novembre 2026**, avant les ajustements scientifiques.
- Les deux tests sur vraies données passent : **2 passed in 15.59s**.
- [Audit en lecture seule réussi](https://github.com/mathiaschebbah/habs/actions/runs/42657666606) :
  trois rapports CV / quinze plis relus, 2 260 prédictions finies par modèle,
  concordance des métriques et reconstruction des intervalles appariés ;
  vraisemblance sauvegardée, sources et empreintes vérifiées sans réoptimisation.
- Récupération sélective de **309 fichiers, 21,62 Mo**, sous la limite de 25 Mo,
  en écartant des miroirs `latest__` et une sérialisation de modèle inutilisée.
  Aucun modèle n'a été réentraîné pour récupérer les résultats.
- [Compte rendu exécuté](results/audit.md) et
  [validation structurée liée aux empreintes exactes](results/audit_completion.json).

Les contrôles génériques de skore ne sont pas présentés comme réussis : ils
renvoient ici une erreur de dimension sur les métadonnées de requête et un
avertissement d'entrée constante. Les prédictions sauvegardées, métriques
enregistrées, score par défaut et invariants ciblés restent vérifiés. La MAPE
générique n'est pas interprétée pour cette cible signée proche de zéro.

L'étude est exploratoire après ouverture des résultats de 01. Les vidéos sont
nouvelles pour chaque personne mais appartiennent au catalogue déjà disponible
chez d'autres personnes. Les notes finales ne permettent pas de séparer
complètement émotion persistante, habitudes de réponse et bruit. Les résultats
ne justifient pas de profils cliniques, de diagnostic de stress ni de conclusion
sur les qualia.
