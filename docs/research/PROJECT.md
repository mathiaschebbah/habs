# Projet de recherche — HABS Neuro-AI Hackathon 2026

## Au-delà du stimulus : prédiction de la réactivité émotionnelle aversive et identification de signatures neurophysiologiques individuelles

*Beyond the Stimulus: Predicting Aversive Emotional Reactivity and Identifying Individual Neurophysiological Signatures*

**Domaine :** neurosciences computationnelles, psychophysiologie, apprentissage automatique multimodal et affective computing  
**Cadre :** HABS Neuro-AI Hackathon — Université Paris-Saclay  
**Statut :** proposition de protocole, avant validation expérimentale  
**Date :** 8 novembre 2026

---

# Résumé scientifique

La détection automatique du stress à partir de signaux cérébraux et physiologiques constitue un domaine actif de recherche. De nombreux travaux ont identifié des associations entre stress, activité électroencéphalographique (EEG), activité cardiovasculaire (ECG), conductance électrodermale (EDA) et respiration. Néanmoins, la spécificité, la stabilité et la capacité de généralisation de ces corrélats restent discutées.

Une difficulté particulièrement importante concerne la distinction entre la réponse neurophysiologique au stimulus et l'expérience émotionnelle subjective de l'individu.

Gerster et al. (2026), dans une réanalyse du dataset FACED, montrent que des classificateurs EEG entraînés à reconnaître des émotions peuvent exploiter principalement l'identité des vidéos et les dépendances temporelles des signaux, plutôt que les émotions effectivement déclarées par les participants.

Ces résultats interrogent l'interprétation des performances obtenues dans les systèmes automatiques de reconnaissance émotionnelle.

Notre projet propose d'étudier cette problématique dans le contexte du stress émotionnel. Nous chercherons à déterminer si les signaux cérébraux et physiologiques contiennent une information prédictive sur la réponse émotionnelle aversive d'un individu qui ne peut être expliquée par les caractéristiques du stimulus, son niveau général d'activation émotionnelle ou ses différences individuelles stables.

Nous proposons trois contributions complémentaires :

1. **Évaluer la prédictibilité individuelle :** mesurer l'information supplémentaire apportée par la physiologie au-delà d'une référence connaissant les stimuli et d'une calibration personnelle.

2. **Caractériser les relations dose-réponse :** déterminer si les corrélats physiologiques varient de manière continue, linéaire ou non linéaire, avec l'intensité d'activation émotionnelle aversive déclarée.

3. **Rechercher des profils neurophysiologiques :** identifier d'éventuelles configurations multimodales de réponse, puis évaluer leur reproductibilité et leur utilité prédictive lors de séances ultérieures.

Le dataset HABS contient des enregistrements synchronisés d'EEG, d'ECG, d'activité électrodermale, de respiration et de mouvements oculaires, ainsi que des autoévaluations émotionnelles réalisées après le visionnage de vidéos.

En l'absence de mesure directe validée du stress, l'étude utilisera un indicateur composite expérimental d'activation émotionnelle négative. Cet indicateur sera explicitement traité comme un proxy du stress émotionnel aigu, sans prétention diagnostique.

L'objectif n'est donc pas de démontrer l'existence présupposée d'une signature universelle du stress, mais de tester si une réponse individuelle, quantifiable et partiellement reproductible peut être identifiée dans les interactions entre cerveau et physiologie périphérique.

**Mots-clés :** stress émotionnel, EEG, ECG, EDA, activité autonome, réactivité individuelle, multimodalité, calibration, apprentissage supervisé, clustering, généralisation, dose-réponse.

---

# 1. Introduction et problématique

## 1.1. Le stress comme processus psychophysiologique

Le stress est un processus complexe faisant intervenir l'évaluation subjective d'une situation, l'activité du système nerveux central, les réponses autonomes et, selon la nature du stresseur, les systèmes neuroendocriniens.

Il ne peut être réduit à la simple activation d'un circuit cérébral ou à l'augmentation d'une variable physiologique.

Les réponses couramment étudiées comprennent notamment :

- les variations de fréquence cardiaque et de variabilité cardiaque ;
- les modifications de la conductance cutanée ;
- les changements de fréquence et de rythme respiratoires ;
- les variations de puissance spectrale de l'EEG ;
- les modifications du ressenti émotionnel et du contrôle perçu.

Ces manifestations ne sont toutefois pas propres au stress.

Une augmentation de l'activité électrodermale peut accompagner la peur, l'excitation positive, l'attention ou la surprise. De même, une variation de fréquence cardiaque peut résulter d'une réponse émotionnelle, d'une modification respiratoire ou d'autres processus autonomes.

Cette absence de correspondance univoque constitue un problème fondamental pour la construction de systèmes automatiques de détection du stress.

## 1.2. Le problème de la mesure subjective

Le stress comporte une dimension vécue qui n'est pas directement observable.

Deux personnes exposées au même stimulus peuvent déclarer des expériences différentes, alors même que leurs mesures physiologiques présentent certaines similitudes.

Inversement, deux participants déclarant une intensité émotionnelle comparable peuvent présenter des réponses cardiovasculaires ou cérébrales très différentes.

Cette difficulté renvoie à la distinction entre expérience phénoménale et rapport subjectif, discutée notamment dans le cours *Réseaux de la conscience* de Bertrand Hermann (2024).

Notre étude ne cherchera pas à résoudre le problème philosophique de l'accès aux *qualia*. Elle adoptera une définition opérationnelle : les autoévaluations sont des observations du ressenti déclaré, et les signaux physiologiques sont des observations des réponses corporelles et cérébrales.

Leur concordance constitue une question empirique, non une hypothèse à considérer comme acquise.

## 1.3. Le problème du stimulus

Les recherches en reconnaissance émotionnelle sont exposées à un biais spécifique : les stimuli employés pour induire les émotions peuvent eux-mêmes être reconnus par les modèles.

Un classificateur capable de reconnaître une vidéo réputée susciter la peur peut prédire une étiquette « peur » sans identifier si le spectateur a effectivement ressenti cette émotion.

Dans leur étude de FACED, Gerster et al. (2026) mettent en évidence ce problème avec des classificateurs EEG classiques et profonds.

Leur résultat impose une distinction entre deux tâches :

**Reconnaissance du stimulus :** prédire l'émotion associée à la vidéo.

**Prédiction de l'expérience individuelle :** prédire l'émotion effectivement déclarée par le participant, notamment lorsqu'elle diffère de celle attendue.

Notre projet se concentre sur la seconde tâche.

---

# 2. Question de recherche

## 2.1. Question principale

**Les signaux cérébraux et physiologiques permettent-ils de prédire la réactivité individuelle au stress émotionnel au-delà de l'information fournie par le stimulus et l'historique personnel, et cette information présente-t-elle des signatures neurophysiologiques reproductibles entre séances ?**

## 2.2. Questions secondaires

**Q1 — Mesure.** Peut-on construire, à partir des annotations émotionnelles disponibles, un proxy continu et interprétable du stress émotionnel aigu ?

**Q2 — Prédiction.** Les signaux EEG, ECG et EDA apportent-ils une information prédictive supplémentaire par rapport à une référence fondée sur la vidéo, son contexte de sélection et les réponses antérieures du participant ?

**Q3 — Spécificité.** À intensité générale d'activation émotionnelle comparable, existe-t-il des caractéristiques neurophysiologiques associées plus spécifiquement à la valence négative ?

**Q4 — Intensité.** Les caractéristiques physiologiques suivent-elles une relation graduelle avec le proxy de stress, ou observe-t-on des effets de seuil, de saturation ou d'autres non-linéarités ?

**Q5 — Individualité.** Les réponses physiologiques présentent-elles des configurations distinctes selon les participants, même lorsque le niveau d'activation négative déclaré est comparable ?

**Q6 — Reproductibilité.** Ces configurations persistent-elles lors d'une séance ultérieure face à des stimuli différents ?

**Q7 — Utilité des profils.** L'identification de profils individuels améliore-t-elle réellement la prédiction par rapport à un modèle multimodal continu sans clustering ?

**Q8 — Dynamique temporelle.** Les caractéristiques physiologiques précoces permettent-elles d'anticiper la réponse émotionnelle déclarée en fin d'essai ?

La dernière question doit être distinguée de la prédiction d'un instant exact de survenue du stress, lequel n'est pas directement annoté dans les données actuellement livrées.

---

# 3. Positionnement par rapport à l'état de l'art

## 3.1. Classification multimodale du stress

Pei et al. (2024) ont étudié la classification de plusieurs types et niveaux de stress en combinant EEG et mesures physiologiques.

Leurs travaux montrent la faisabilité expérimentale d'une classification multimodale, mais concernent un effectif restreint et ne résolvent pas entièrement le problème de généralisation interindividuelle.

Notre projet ne revendiquera donc pas la première classification multimodale du stress.

## 3.2. Variabilité des signatures émotionnelles

La méta-analyse de Siegel et al. (2018), portant sur 202 études, remet en question l'hypothèse d'une signature autonome universelle et spécifique de chaque catégorie émotionnelle.

Ce résultat suggère que les réponses physiologiques peuvent présenter une variabilité importante à l'intérieur d'une même catégorie.

Nous considérerons donc deux modèles concurrents :

- **Hypothèse dimensionnelle :** les réponses physiologiques sont principalement organisées selon une ou plusieurs dimensions continues d'activation.
- **Hypothèse de profils :** certaines configurations physiologiques distinctes présentent une structure reproductible qui n'est pas entièrement expliquée par ces dimensions.

Le clustering ne sera utilisé que pour confronter ces hypothèses.

## 3.3. Variabilité individuelle

Brown et al. (2019) ont étudié la concordance entre expérience émotionnelle subjective et réponses physiologiques, en montrant que cette relation varie entre individus.

Les travaux sur le stress aigu utilisant des analyses de profils latents montrent également que les réponses autonomes ne sont pas homogènes dans la population.

Notre étude cherchera à dépasser la simple mise en évidence de cette hétérogénéité.

La question sera de déterminer si une organisation individuelle apprise à partir d'une première séance conserve une valeur prédictive lors d'une séance ultérieure.

## 3.4. Validité du stress induit par vidéo

Roy et al. (2025) ont montré, dans un protocole expérimental contrôlé, que la présentation de vidéos aversives pouvait entraîner une augmentation de l'anxiété déclarée, des modifications cardiovasculaires et une réponse cortisolique.

Cette étude justifie l'utilisation de stimuli audiovisuels dans la recherche sur le stress émotionnel.

Elle ne démontre cependant pas que toute vidéo de valence négative induit un stress mesurable, ni que la valence et l'arousal peuvent remplacer une mesure validée du stress.

## 3.5. La contribution envisagée

L'originalité ne sera pas recherchée dans l'utilisation d'un modèle plus complexe ou dans la création de nouveaux clusters.

**La contribution envisagée est une évaluation conjointe de la spécificité, de la généralisation et de la stabilité des informations physiologiques individuelles associées à un proxy de stress émotionnel.**

Cette combinaison constitue une piste de contribution méthodologique à tester. Elle ne doit pas être présentée comme une première mondiale établie sans recherche bibliographique exhaustive supplémentaire.

---

# 4. Hypothèses scientifiques

## H1 — Information physiologique supplémentaire

À stimulus et calibration individuelle comparables, l'ajout de caractéristiques cérébrales et physiologiques réduit l'erreur de prédiction du proxy de stress émotionnel.

**Hypothèse nulle :** l'ajout des mesures physiologiques n'améliore pas la prédiction sur les personnes réservées au test.

## H2 — Spécificité de l'activation négative

Certains corrélats cérébraux ou physiologiques différencient une activation émotionnelle négative d'une activation positive d'intensité comparable.

**Hypothèse nulle :** après contrôle de l'arousal et des autres facteurs de confusion observables, les caractéristiques testées n'apportent pas d'information supplémentaire sur la valence négative.

## H3 — Relation dose-réponse

Certaines caractéristiques neurophysiologiques présentent une relation continue, linéaire ou non linéaire, avec l'intensité du proxy de stress.

**Hypothèse nulle :** aucune association robuste ne subsiste après contrôle des facteurs individuels et des stimuli.

## H4 — Hétérogénéité physiologique

Des observations présentant des scores émotionnels similaires peuvent correspondre à des configurations physiologiques différentes.

Cette hypothèse est descriptive : la présence de plusieurs clusters ne suffit pas à démontrer qu'il existe plusieurs types biologiques de stress.

## H5 — Stabilité inter-séances

Certaines propriétés des profils physiologiques individuels observés lors d'une première séance sont reproductibles lors d'une seconde séance.

**Hypothèse nulle :** la ressemblance entre les profils d'une même personne sur différentes séances n'est pas supérieure à celle attendue compte tenu des stimuli, de la distribution globale des réponses et de la variabilité des mesures.

## H6 — Utilité prédictive des profils

L'intégration de profils neurophysiologiques individuels améliore la prédiction sur une séance ultérieure par rapport à un modèle continu utilisant les mêmes caractéristiques physiologiques.

Cette hypothèse est plus exigeante que la simple identification de clusters.

---

# 5. Présentation des données HABS

## 5.1. Structure du dataset

Le dataset HABS contient des données acquises lors de plusieurs sessions expérimentales.

Les tâches distribuées comprennent :

- le visionnage de vidéos émotionnelles ;
- des périodes de repos ;
- une tâche de calibration d'artefacts.

Les enregistrements comprennent 64 canaux EEG, des mesures cardiovasculaires, électrodermales et respiratoires, ainsi que des informations de regard et de pupille.

Des annotations émotionnelles sont disponibles au niveau des essais vidéo.

Les vidéos originales ne sont pas fournies dans la livraison actuelle.

## 5.2. Annotations disponibles

Le fichier `labels_train.csv` comprend notamment :

| Variable | Utilisation proposée |
|---|---|
| `participant_id` | Regroupement et calibration individuelle |
| `cohort` | Contrôle des différences de protocole |
| `day` | Séparation des séances |
| `video` | Identification du stimulus |
| `rating_valence` | Construction du proxy |
| `rating_arousal` | Construction du proxy et contrôle de l'activation |
| `rating_dominance` | Analyse secondaire du contrôle perçu |
| `self_reported_emotion_1/2/3` | Caractérisation des réponses déclarées |
| `video_theme` | Analyse exploratoire des familles de stimuli |
| `expected_emotion_label` | Contexte préalable à la réponse, sous réserve de provenance |
| `video_presentation_order` | Contrôle de l'ordre de présentation |

Le champ `final_emotion_label` est dérivé de plusieurs sources d'annotation. Il ne sera pas utilisé comme preuve indépendante de l'état émotionnel.

Les annotations associées à chaque essai seront distinguées des informations réellement accessibles avant le visionnage.

## 5.3. Résultats de préparation déjà disponibles

L'audit du dépôt HABS conserve 22 972 essais alignés et 314 identifiants.

La cohorte 2 constitue le point de départ privilégié, en raison de sa couverture et de la disponibilité de deux séances chez une partie des participants.

L'ancien protocole retenait 50 identifiants de développement, 26 identifiants de test et 1 306 essais de test dans cette cohorte après ses propres critères de sélection.

Ces effectifs ne doivent pas être considérés comme ceux de notre nouvelle étude : l'ajout des critères de valence, du proxy composite et des comparaisons multimodales exigera une nouvelle vérification.

Les cohortes 1 et 4 pourront servir de réplications exploratoires séparées si leur structure et leurs effectifs le permettent.

Les différentes cohortes ne seront pas fusionnées sans vérification de l'unicité biologique des participants.

---

# 6. Construction de l'indicateur composite

## 6.1. Principe

La variable d'intérêt doit être construite indépendamment des signaux physiologiques utilisés pour sa prédiction.

Nous retenons donc une première mesure fondée uniquement sur les autoévaluations.

Le proxy de stress émotionnel représente une **activation subjective de valence négative**.

Il ne mesure directement ni le stress clinique, ni la charge cognitive, ni la réponse endocrinienne.

## 6.2. Composantes

Soit, pour un essai \(i\) :

- \(V_i\) : valence déclarée, comprise entre 1 et 9 ;
- \(A_i\) : arousal déclaré, compris entre 1 et 9.

Nous définissons une composante d'affect négatif :

\[
N_i=\max\left(0,\frac{5-V_i}{4}\right)
\]

et une composante d'activation :

\[
E_i=\frac{A_i-1}{8}.
\]

Le composite principal exploratoire est :

\[
\boxed{S_i=N_iE_i}
\]

avec :

\[
S_i\in[0,1].
\]

Le produit impose que l'activation et la valence négative soient simultanément présentes pour obtenir une valeur élevée.

Ainsi, une forte excitation positive ne reçoit pas automatiquement un score élevé.

**Le choix de cette formule est une décision opérationnelle proposée, et non un indice validé par la littérature.**

## 6.3. Analyses de sensibilité

Trois variantes seront étudiées :

1. Une formulation alternative de l'interaction entre valence négative et arousal, avec des poids ou exposants définis avant l'évaluation finale.
2. Une analyse utilisant séparément valence et arousal, sans les réduire à un score unique.
3. Une analyse intégrant la dominance comme dimension explicative supplémentaire lorsque cette variable est disponible.

Les catégories émotionnelles directement déclarées serviront à étudier la cohérence du score.

Nous examinerons notamment si les essais déclarés comme associés à la peur, à la colère ou au dégoût présentent des distributions de scores différentes.

Une association avec ces catégories constituerait un indice de validité convergente, mais non une validation indépendante du stress.

## 6.4. Seuils et groupes extrêmes

Les seuils seront déterminés exclusivement à partir des données de développement.

À titre de protocole candidat, nous pourrons définir :

- un groupe bas : premier quartile du score ;
- un groupe haut : dernier quartile ;
- un groupe intermédiaire : observations situées entre ces deux seuils.

Il faudra préalablement vérifier le nombre de scores nuls, les ex æquo et les effectifs réellement disponibles.

Si la distribution est fortement concentrée à zéro, les quantiles ne seront pas nécessairement adaptés.

L'analyse continue sera privilégiée comme référence principale.

Les comparaisons sur les extrêmes seront considérées comme des expériences complémentaires.

## 6.5. Validité du construit

La validité sera examinée selon trois dimensions :

**Validité de contenu :** les composantes correspondent-elles à notre définition explicite de l'activation émotionnelle aversive ?

**Validité convergente :** le score est-il associé aux émotions négatives effectivement déclarées ?

**Validité discriminante :** permet-il de distinguer une activation négative d'une activation positive d'intensité comparable ?

Il ne sera pas possible de conclure à la validité clinique du stress sans mesure indépendante.

---

# 7. Extraction des caractéristiques physiologiques

## 7.1. EEG

Le pipeline actuel du dépôt extrait déjà des puissances spectrales relatives sur huit électrodes :

F3, F4, Fz, Cz, P3, P4, O1 et O2.

Les caractéristiques sont regroupées en régions frontales et postérieures :

- theta : 4–8 Hz ;
- alpha : 8–13 Hz ;
- bêta : 13–30 Hz.

Six caractéristiques régionales sont actuellement produites.

Nous utiliserons ces mesures comme première représentation.

Des extensions exploratoires pourront inclure les puissances absolues logarithmiques, des ratios spectraux ou des caractéristiques de connectivité, à condition de documenter les choix et de contrôler les artefacts.

Aucune direction universelle de variation du theta, de l'alpha ou du bêta ne sera présupposée comme un seuil de stress.

## 7.2. ECG

Les caractéristiques existantes comprennent :

- la fréquence cardiaque médiane ;
- sa variation au cours de la fenêtre observée.

Une extension pourra examiner certains indices de variabilité cardiaque.

Cependant, les indices de HRV exigent des fenêtres de durée appropriée et une détection fiable des intervalles RR. Les indices fréquentiels classiques ne seront pas calculés sur dix secondes comme s'ils étaient équivalents à des mesures de plusieurs minutes.

L'ECG sera également utilisé pour étudier les réponses cardiaques accélératrices ou décélératrices.

## 7.3. Activité électrodermale

Le pipeline extrait déjà :

- le niveau de conductance ;
- son évolution au cours de l'essai ;
- sa dispersion.

L'EDA pourra constituer une modalité particulièrement pertinente pour l'étude de l'activation sympathique.

Toutefois, les audits existants signalent des difficultés possibles de conversion et d'écrêtage de certains signaux.

Les caractéristiques EDA ne seront intégrées au modèle principal qu'après vérification des unités et du traitement.

## 7.4. Respiration

La respiration est une modalité potentiellement intéressante pour étudier les interactions cerveau–corps.

Nous pourrons rechercher :

- la fréquence respiratoire ;
- sa variation au cours de l'essai ;
- les changements de régularité ;
- certaines caractéristiques de couplage cardiorespiratoire.

Ces analyses nécessiteront une extraction supplémentaire et une vérification de la qualité des canaux.

La respiration ne sera pas considérée comme une mesure spécifique du stress.

## 7.5. Référence individuelle

Lorsque la qualité et la chronologie des mesures le permettent, les caractéristiques seront exprimées relativement à une référence personnelle estimée sur la séance de calibration.

Les enregistrements de repos pourront être étudiés séparément.

Ils ne seront pas assimilés automatiquement à un état physiologique immédiatement antérieur à chaque vidéo.

La normalisation individuelle devra utiliser exclusivement les données disponibles avant l'essai cible.

---

# 8. Protocole de prédiction

## 8.1. Définition de la tâche principale

Pour chaque participant \(p\) et essai vidéo \(i\), on observe :

\[
Y_{pi}=S_{pi}
\]

où \(S_{pi}\) est le proxy d'activation aversive déclaré.

Nous définissons :

- \(C_{pi}\) : informations préalables ou contextuelles sur le stimulus ;
- \(H_p\) : historique personnel issu de la première séance ;
- \(X_{pi}\) : caractéristiques physiologiques de l'essai.

La question est de déterminer si :

\[
P(Y\mid C,H,X)
\]

permet une meilleure prédiction que :

\[
P(Y\mid C,H).
\]

Le contraste principal porte sur l'information incrémentale de la physiologie.

## 8.2. Modèles de référence

**M0 — Référence stimulus et contexte**

Modèle utilisant les informations disponibles sur les vidéos, leur catégorie attendue et leur ordre de présentation.

Il représente ce qu'un système peut prédire sans connaître les signaux physiologiques du spectateur.

**M1 — Référence avec calibration individuelle**

Même modèle, auquel on ajoute une correction personnalisée estimée sur les déclarations de la première séance.

La correction pourra inclure un décalage de niveau et, si les données le permettent, un facteur d'échelle régularisé.

Ce modèle constitue la référence principale.

**M2 — Modèle EEG**

M1 enrichi par les caractéristiques EEG.

**M3 — Modèle physiologique périphérique**

M1 enrichi par les caractéristiques ECG et EDA.

**M4 — Modèle multimodal**

M1 enrichi par EEG, ECG et EDA.

**M5 — Modèle avec profils physiologiques**

M4 enrichi d'une représentation des profils physiologiques apprise sans utiliser les scores des essais de test.

L'objectif est de déterminer si les profils ajoutent une information utile au-delà des caractéristiques physiologiques continues.

## 8.3. Méthodes d'apprentissage

Nous privilégierons d'abord des méthodes simples et interprétables :

- régression ridge ;
- régression Elastic Net ;
- modèles additifs généralisés ;
- éventuellement arbres de gradient boosting pour les non-linéarités.

Les méthodes complexes seront secondaires et devront apporter un gain supérieur à des références correctement contrôlées.

L'utilisation d'architectures profondes ne sera pas considérée comme une contribution scientifique en soi.

## 8.4. Contrôle de la vidéo

Le modèle de référence devra connaître le stimulus lorsque notre scénario porte sur des vidéos déjà présentes dans le catalogue d'apprentissage.

Ce choix évite d'attribuer à l'EEG une information que l'identité de la vidéo suffit à fournir.

Nous distinguerons deux scénarios.

**Scénario A — Nouvelle vidéo pour la personne.**

La vidéo n'a pas encore été vue par la personne testée, mais peut être connue chez d'autres participants.

Il s'agit du scénario principal compatible avec le protocole de calibration actuel.

**Scénario B — Nouvelle vidéo pour la population d'apprentissage.**

La vidéo est entièrement absente de l'apprentissage et de la calibration des modèles de population.

Ce scénario est plus exigeant. Il impose un modèle capable de fonctionner sans effet spécifique appris pour cette vidéo et nécessitera une nouvelle étude de faisabilité.

Les deux scénarios ne seront jamais présentés comme équivalents.

---

# 9. Mesure de la spécificité au stress émotionnel

## 9.1. Le problème de l'arousal

Si un modèle prédit correctement notre composite, il peut simplement avoir appris à identifier une activation physiologique élevée.

Il faut donc examiner si les signaux contiennent une information supplémentaire concernant le caractère négatif du ressenti.

## 9.2. Comparaison à activation comparable

Nous chercherons à comparer des essais présentant des niveaux voisins d'arousal déclaré, mais des valences différentes.

À titre d'exemple :

- activation élevée et valence positive ;
- activation élevée et valence négative.

La comparaison utilisera les réponses individuelles et non uniquement l'étiquette de la vidéo.

Les différences d'effectifs, de participants et de stimuli devront être prises en compte.

## 9.3. Analyse conditionnelle

Nous étudierons la relation entre caractéristiques physiologiques et valence en contrôlant statistiquement l'arousal déclaré.

Ce contrôle est une analyse explicative réalisée après collecte des déclarations.

**L'arousal de l'essai cible ne sera pas utilisé comme prédicteur dans le modèle prospectif**, puisqu'il ne serait pas encore connu au moment de la prédiction.

Cette distinction évite une fuite de cible.

## 9.4. Interprétation

Si une information physiologique sur la valence négative subsiste après contrôle de l'arousal, cela soutiendra l'existence d'une information liée à la dimension aversive du ressenti.

Cela ne démontrera pas automatiquement qu'il s'agit d'un marqueur spécifique du stress.

Une interprétation plus forte nécessiterait une mesure indépendante du stress perçu.

---

# 10. Étude des relations dose-réponse

## 10.1. Hypothèse d'une relation graduelle

Nous voulons déterminer si une augmentation du proxy de stress s'accompagne de variations progressives de certains marqueurs.

Les relations étudiées pourront être :

- approximativement linéaires ;
- monotones mais non linéaires ;
- caractérisées par une saturation ;
- non monotones ;
- différentes selon les personnes.

## 10.2. Modèles statistiques

Une première analyse utilisera des régressions linéaires avec effets individuels et facteurs liés aux vidéos.

Une analyse complémentaire utilisera des fonctions splines ou un GAM afin d'étudier les non-linéarités.

Les modèles devront tenir compte des observations répétées pour chaque participant et du partage de stimuli entre individus.

Les relations intraindividuelles et interindividuelles seront rapportées séparément.

## 10.3. Expérience sur les extrêmes

Une expérience secondaire consistera à entraîner un modèle sur les observations appartenant aux catégories basses et hautes du score composite.

Le modèle sera ensuite appliqué aux observations intermédiaires non utilisées dans son apprentissage.

Nous examinerons si son score de décision évolue progressivement avec la valeur déclarée.

Une progression serait compatible avec l'existence d'un gradient physiologique.

Toutefois, un score de classificateur entraîné sur les extrêmes ne peut pas être directement interprété comme une quantité validée de stress.

La calibration et les performances devront être évaluées séparément.

## 10.4. Comparaison entre modèles

Nous comparerons notamment :

- un modèle linéaire ;
- un modèle non linéaire ;
- un modèle tenant compte des effets individuels.

L'objectif sera de déterminer si la complexité supplémentaire est justifiée par une amélioration reproductible de la prédiction.

---

# 11. Clustering des signatures neurophysiologiques

## 11.1. Objectif

Le clustering cherchera à identifier des configurations physiologiques récurrentes sans utiliser les scores de stress pour construire les groupes.

Chaque essai sera représenté par un vecteur de caractéristiques cérébrales et périphériques standardisées.

## 11.2. Méthodes candidates

Nous comparerons :

- K-means comme référence simple ;
- un modèle de mélanges gaussiens (GMM) pour autoriser des appartenances probabilistes ;
- un modèle factoriel ou une représentation continue pour vérifier si des clusters discrets sont réellement nécessaires.

Le nombre de clusters sera sélectionné sur les données de développement.

Les scores de silhouette et autres mesures géométriques seront utilisés avec prudence : ils n'établissent pas une signification neurophysiologique.

## 11.3. Caractérisation des clusters

Après construction, nous examinerons les différences de :

- puissance EEG ;
- réactivité cardiaque ;
- réponse électrodermale ;
- intensité d'activation aversive déclarée ;
- émotions rapportées ;
- type de stimulus ;
- participant et séance.

Les clusters seront d'abord désignés de manière neutre : A, B, C, etc.

Les étiquettes comme « fight », « flight », « freeze » ou « résistant au stress » ne seront pas attribuées sans validation physiologique et comportementale appropriée.

## 11.4. Profils individuels

Deux niveaux de clustering seront distingués.

**Niveau essai :** une configuration physiologique observée pendant une vidéo.

**Niveau individu :** un résumé de la distribution des réponses physiologiques d'une personne sur plusieurs essais.

Par exemple, un profil individuel pourra être représenté par la fréquence d'appartenance aux différents clusters, après contrôle de la distribution des stimuli.

## 11.5. Test de stabilité

Les profils seront construits à partir de la première séance.

Nous examinerons ensuite la ressemblance entre le profil obtenu pour une personne en J1 et celui observé en J2.

La stabilité sera évaluée par rapport à des références tenant compte du hasard, de la composition des stimuli et de la variabilité entre personnes.

Nous comparerons la similarité intraindividuelle et la similarité entre individus.

## 11.6. Test prédictif

Le critère le plus important sera l'utilité des profils.

Nous comparerons :

\[
M4:\quad \hat S=f(X,C,H)
\]

à :

\[
M5:\quad \hat S=f(X,C,H,Z)
\]

où \(Z\) représente l'information de profil estimée sans utiliser les réponses de test.

Si M5 n'améliore pas M4, nous conclurons que le clustering n'apporte pas d'information prédictive démontrée, même si des groupes descriptifs semblent intéressants.

---

# 12. Analyses complémentaires des interactions cerveau–corps

## 12.1. Couplage EEG–respiration

Une piste exploratoire consiste à déterminer si les variations de certaines caractéristiques EEG permettent de prédire des variations respiratoires futures.

Nous comparerons :

1. un modèle fondé uniquement sur l'historique respiratoire ;
2. un modèle utilisant cet historique et les caractéristiques EEG passées.

Les fenêtres d'entrée et de sortie seront temporellement distinctes.

Les prétraitements devront éviter toute contamination des entrées par des observations futures.

## 12.2. Interprétation

Un gain prédictif indiquerait une information temporelle complémentaire entre les deux modalités.

Il ne prouverait pas un lien causal direct entre l'activité cérébrale mesurée et la respiration.

Il ne démontrerait pas non plus que ce couplage est spécifique au stress.

Une extension intéressante serait d'examiner si les propriétés de couplage diffèrent selon les niveaux de notre proxy d'activation aversive.

## 12.3. Réponse et récupération

Si les périodes expérimentales sont identifiables avec une précision suffisante, nous étudierons :

- la réactivité initiale ;
- l'évolution des signaux pendant le visionnage ;
- la réponse après l'exposition ;
- les éventuels indicateurs de récupération.

Cette analyse dépendra de la disponibilité effective des segments temporels, et non d'une hypothèse sur l'organisation des sessions.

Les termes de stress diffus, réaction soudaine et récupération ne seront utilisés que pour des phénomènes réellement mesurables dans le protocole.

---

# 13. Stratégie de validation

## 13.1. Séparation des participants

L'apprentissage et la sélection des paramètres seront réalisés exclusivement sur les identifiants de développement.

Les participants réservés au test ne contribueront pas à l'apprentissage de population.

Pour ces participants, seule leur première séance pourra être utilisée pour la calibration individuelle prévue par le protocole.

Les réponses de leur seconde séance resteront masquées jusqu'à l'évaluation finale.

## 13.2. Séparation temporelle

Les fenêtres d'un même essai ne seront pas réparties aléatoirement entre entraînement et test.

Il faudra éviter qu'un modèle exploite l'autocorrélation de segments EEG proches appartenant au même événement.

Les transformations apprises à partir des données, comme les normalisations de population, les imputations et les sélections de caractéristiques, seront ajustées uniquement sur les données d'apprentissage appropriées.

Un filtrage non causal pourra être utilisé pour des analyses hors ligne lorsque sa portée est explicitement décrite, mais pas pour soutenir une démonstration de prédiction anticipée en temps réel.

## 13.3. Contrôles négatifs

Nous prévoyons plusieurs contrôles.

**Contrôle stimulus seul :** vérifier ce que la vidéo permet de prédire sans physiologie.

**Contrôle de disponibilité des capteurs :** vérifier si les indicateurs de valeurs manquantes prédisent la cible.

**Contrôle de permutation :** détruire certaines correspondances physiologie–réponse dans des permutations respectant autant que possible la structure des sessions et des stimuli.

**Contrôle de valence :** comparer les réponses négatives aux réponses positives d'arousal similaire.

**Contrôle de calibration :** comparer les mêmes modèles sur les mêmes essais, avec un historique personnel identique.

Les permutations exactes seront définies selon la structure observée des données, afin de ne pas détruire arbitrairement des dépendances nécessaires au test statistique.

## 13.4. Sélection des paramètres

La sélection des modèles, hyperparamètres, caractéristiques et nombres de clusters s'effectuera sur le développement uniquement.

Les résultats des personnes réservées au test ne serviront pas à choisir rétrospectivement une variante.

La nouvelle cible nécessitera un nouveau protocole versionné ; les paramètres figés pour l'ancienne étude d'arousal ne seront pas supposés optimaux pour le proxy de stress.

---

# 14. Critères d'évaluation

## 14.1. Critère principal

Le critère principal sera l'amélioration de l'erreur absolue moyenne (MAE) de prédiction du composite continu.

Nous calculerons d'abord la MAE pour chaque personne, puis la moyenne des MAE individuelles.

L'effet principal sera :

\[
\Delta MAE =
MAE(M1)-MAE(M4)
\]

Une valeur positive indiquera que le modèle multimodal réduit l'erreur par rapport à la référence calibrée.

Ce critère mesure une amélioration de prédiction du proxy, non une validité clinique du stress.

## 14.2. Critères secondaires

Nous examinerons également :

- la corrélation de rang entre prédictions et déclarations au sein des personnes ;
- la part d'erreur liée à un décalage de niveau moyen ;
- la capacité à prédire les différences entre essais d'une même personne ;
- la performance sur les observations intermédiaires ;
- la performance conditionnelle aux niveaux d'arousal ;
- la stabilité des profils entre séances ;
- le gain de M5 par rapport à M4.

## 14.3. Incertitude statistique

Les intervalles d'incertitude du contraste prédictif principal seront calculés en respectant la dépendance entre essais d'un même participant.

Une approche candidate consiste à rééchantillonner les participants de test et à calculer la différence appariée des erreurs individuelles.

Pour les conclusions portant sur des vidéos entièrement nouvelles, il faudra également prendre en compte l'incertitude liée à l'échantillonnage des stimuli.

Le nombre d'essais ne sera pas confondu avec le nombre d'unités statistiques indépendantes.

## 14.4. Importance pratique

Un seuil minimal de gain utile devra être fixé avant la consultation des performances réservées.

Ce seuil sera choisi dans l'unité du score, avec justification de sa signification pratique et de la précision attendue.

Aucun seuil universel de gain ne sera présenté comme cliniquement validé.

## 14.5. Interprétation des résultats

Nous distinguerons quatre situations :

1. **Gain positif et suffisamment précis :** information prédictive supplémentaire soutenue par les données.
2. **Gain positif mais incertain :** résultat prometteur mais non concluant.
3. **Absence de gain avec précision insuffisante :** impossibilité de conclure à l'absence d'information.
4. **Gain utile raisonnablement exclu pour les modèles étudiés :** résultat négatif informatif, limité au protocole et à la famille de modèles testés.

---

# 15. Risques méthodologiques et limitations

## 15.1. Absence de vérité terrain indépendante

Le dataset ne fournit pas directement de score de stress d'état validé.

Notre indicateur ne pourra pas être présenté comme une mesure du stress au sens clinique.

Il faudra distinguer les conclusions sur le proxy émotionnel de celles, hypothétiques, sur le stress lui-même.

## 15.2. Biais de sélection des stimuli

Les vidéos ont notamment été sélectionnées en fonction d'associations émotionnelles individuelles préalables.

Le plan n'est donc pas un croisement complet et aléatoire entre toutes les personnes et toutes les vidéos.

Les conclusions porteront sur le protocole de sélection observé.

## 15.3. Confusion avec l'identité du stimulus

Les modèles peuvent exploiter des caractéristiques propres aux vidéos.

Une référence stimulus doit donc être intégrée aux comparaisons.

Les analyses de transfert à de nouvelles vidéos nécessiteront une séparation spécifique.

## 15.4. Qualité des signaux

Les résultats existants signalent des limitations de mesure, notamment sur la conversion EDA.

Une analyse de qualité indépendante des réponses émotionnelles sera nécessaire avant l'évaluation finale.

## 15.5. Différences entre cohortes

Les cohortes peuvent différer en matière de protocoles, d'annotations et d'organisation des stimuli.

Elles seront traitées séparément tant que leur comparabilité n'est pas démontrée.

## 15.6. Stabilité des profils

Le fait de retrouver plusieurs clusters sur un jeu de données ne démontre pas qu'ils correspondent à des catégories physiologiques naturelles.

La stabilité sera examinée par des comparaisons hors échantillon et des contrôles de la variabilité attendue.

## 15.7. Interprétation causale

Une association entre activité EEG, respiration, EDA et stress déclaré ne démontre pas de causalité.

Une capacité prédictive ne permet pas, à elle seule, d'identifier les mécanismes neuronaux responsables du ressenti.

## 15.8. Biais de sélection des observations

Le filtrage amont des annotations HABS n'est pas totalement documenté dans les fichiers disponibles.

Les observations livrées peuvent ne pas être représentatives de l'ensemble des essais collectés.

Les résultats seront explicitement conditionnés à la population d'observations exploitables.

---

# 16. Plan de réalisation pour le hackathon

## Phase 1 — Audit et faisabilité

**Objectif :** vérifier que la nouvelle question peut être étudiée avec des effectifs suffisants.

Travaux :

- recalculer les observations disposant de valence et d'arousal ;
- calculer la distribution du proxy ;
- examiner les effectifs par participant et par séance ;
- identifier les vidéos partagées et répétées ;
- vérifier les intersections de disponibilité EEG, ECG et EDA ;
- étudier la disponibilité de cas positifs et négatifs à arousal comparable.

**Livrable :** un rapport de faisabilité et un manifeste des essais retenus.

## Phase 2 — Définition du protocole

**Objectif :** figer les hypothèses et les critères avant l'évaluation finale.

Travaux :

- fixer la définition principale du proxy ;
- documenter les analyses de sensibilité ;
- définir les partitions par participant et séance ;
- identifier les contrôles négatifs ;
- fixer le contraste principal et les critères d'interprétation.

**Livrable :** protocole versionné, avec graines et empreintes des partitions.

## Phase 3 — Modèles prédictifs

**Objectif :** mesurer la valeur supplémentaire des signaux physiologiques.

Travaux :

- entraîner M0 et M1 ;
- ajouter EEG, ECG et EDA ;
- comparer les modèles sur une grille identique ;
- analyser les performances sur le développement ;
- vérifier les invariants de séparation.

**Livrable :** tableau d'ablation multimodale.

## Phase 4 — Dose-réponse

**Objectif :** caractériser les relations entre les signaux et l'intensité déclarée.

Travaux :

- régressions linéaires et non linéaires ;
- comparaisons des extrêmes ;
- tests sur les observations intermédiaires ;
- analyses conditionnelles à l'arousal ;
- exploration des effets individuels.

**Livrable :** courbes dose-réponse avec intervalles d'incertitude.

## Phase 5 — Profils physiologiques

**Objectif :** rechercher des structures reproductibles de réponse.

Travaux :

- clustering des observations de développement ;
- caractérisation des signatures ;
- construction de résumés individuels ;
- estimation de la stabilité entre séances ;
- comparaison M4 versus M5.

**Livrable :** représentation des profils et résultats de leur utilité prédictive.

## Phase 6 — Évaluation finale

**Objectif :** tester les hypothèses sélectionnées sans ajustement rétrospectif.

Travaux :

- exécuter les modèles figés sur les participants réservés ;
- calculer les critères et intervalles ;
- documenter les résultats positifs et négatifs ;
- examiner les limites et analyses de sensibilité prévues ;
- produire les figures et la synthèse scientifique.

**Livrable :** rapport expérimental reproductible.

## Priorités en cas de ressources limitées

Le projet peut être réduit sans perdre sa question fondamentale.

Le noyau minimal sera :

1. Proxy émotionnel et contrôle de sa distribution.
2. Référence stimulus + calibration contre référence multimodale.
3. Analyse de spécificité à arousal comparable.
4. Évaluation J2 réservée.

Le clustering, l'analyse temporelle fine et les nouveaux indices respiratoires constituent des extensions, non des prérequis au résultat principal.

---

# 17. Résultats attendus et scénarios scientifiques

## Scénario A — Signal individuel prédictif

Les mesures physiologiques améliorent la prédiction au-delà du stimulus et de la calibration.

Ce résultat soutiendrait l'existence d'une information individuelle exploitable dans les signaux cérébraux et périphériques.

Il faudrait ensuite déterminer si ce gain provient de caractéristiques spécifiques à la dimension aversive ou d'une activation émotionnelle générale.

## Scénario B — Information essentiellement liée au stimulus

Le modèle multimodal n'améliore pas la référence calibrée.

Ce résultat serait cohérent avec l'hypothèse selon laquelle une grande partie de la prédiction apparente provient du stimulus et de différences individuelles stables.

Il prolongerait les questions soulevées par Gerster et al. dans le contexte particulier des données HABS.

## Scénario C — Hétérogénéité sans profils discrets

Les réponses physiologiques varient selon les individus, mais une représentation continue rend aussi bien compte de ces différences qu'un clustering.

Ce résultat soutiendrait une approche dimensionnelle plutôt qu'une typologie de personnes.

## Scénario D — Profils reproductibles et utiles

Des profils physiologiques appris sur J1 se retrouvent sur J2 et améliorent la prédiction de réponses individuelles à de nouveaux stimuli.

Ce serait le résultat le plus intéressant pour notre hypothèse sur les signatures individuelles.

Il constituerait toutefois une preuve limitée aux participants, aux stimuli, aux mesures et au proxy employés.

## Scénario E — Absence de conclusion suffisamment précise

Les effectifs disponibles, la qualité des signaux ou la variabilité entre personnes ne permettent pas de distinguer clairement les hypothèses.

Dans ce cas, l'apport scientifique sera de préciser les limites du dataset et les modifications nécessaires à un protocole expérimental confirmatoire.

---

# 18. Contributions scientifiques potentielles

La première contribution serait **méthodologique** : proposer une évaluation qui sépare les informations liées au stimulus, les réponses subjectives et les signaux physiologiques.

La deuxième serait **prédictive** : mesurer la valeur ajoutée réelle de l'EEG et des mesures périphériques pour estimer une réponse émotionnelle aversive individualisée.

La troisième serait **psychophysiologique** : étudier les relations entre intensité déclarée et caractéristiques du système nerveux central et autonome, y compris leurs non-linéarités.

La quatrième serait **interindividuelle** : déterminer si les différences de réactivité peuvent être décrites par des signatures relativement stables ou principalement par des dimensions continues.

La cinquième serait **épistémologique** : examiner ce que signifie réellement « prédire une émotion » ou « détecter le stress » lorsque les labels sont subjectifs, les stimuli partagés et les mesures physiologiques non spécifiques.

Ces contributions sont conditionnelles aux résultats. Le projet ne présuppose aucune amélioration prédictive ni aucune existence de clusters stables.

---

# 19. Visualisations prévues

Le rapport final comportera, au minimum :

**Figure 1 — Distribution du proxy de stress.** Histogrammes et densités par cohorte et séance, avec effectifs aux extrêmes.

**Figure 2 — Effet de la physiologie sur la prédiction.** Comparaison appariée des erreurs individuelles entre référence calibrée et modèle multimodal.

**Figure 3 — Spécificité physiologique.** Relations entre caractéristiques EEG/ECG/EDA et valence négative à niveaux comparables d'arousal.

**Figure 4 — Dose-réponse.** Courbes estimées des principales caractéristiques physiologiques en fonction du proxy continu, avec incertitude.

**Figure 5 — Structure des réponses.** Projection descriptive des observations physiologiques, colorée après apprentissage par cluster, score émotionnel et séance.

**Figure 6 — Stabilité individuelle.** Comparaison des profils J1/J2 et de leur similarité par rapport aux autres participants.

**Figure 7 — Ablation des modalités.** Apport relatif EEG, ECG, EDA, multimodal et information de profil.

Aucune projection bidimensionnelle ne sera utilisée seule comme preuve de clusters distincts.

---

# 20. Questions à transmettre à HABS

Plusieurs informations pourraient modifier la portée scientifique du projet.

**Mesure du stress :** existe-t-il des autoévaluations directes de stress, d'anxiété d'état, de menace perçue ou de contrôle qui ne figurent pas dans la livraison ?

**Chronologie :** les vidéos et leurs événements internes peuvent-ils être reliés à des annotations temporelles de peur, de surprise ou de stress ?

**Protocole :** quelles tâches supplémentaires ont été réalisées pendant les séances et pourquoi ne sont-elles pas distribuées ?

**Participants :** les identifiants correspondent-ils à des personnes uniques entre cohortes ?

**Qualité :** quelles calibrations et unités doivent être utilisées pour les signaux EDA, ECG et respiratoires ?

**Annotations :** quels filtres ont été appliqués en amont et quelle est la provenance exacte des différentes notes ?

**Stimuli :** existe-t-il un référentiel de thèmes ou d'annotations de menace, ainsi qu'un accès contrôlé aux vidéos ?

Une réponse positive sur les mesures directes de stress améliorerait fortement la validité du projet.

---

# 21. Éthique, reproductibilité et portée des conclusions

Les données individuelles seront pseudonymisées et les résultats publiés sous forme agrégée lorsque les conditions de la livraison l'exigent.

Les profils identifiés ne seront pas employés pour attribuer des diagnostics, des traits de personnalité ou des vulnérabilités psychologiques aux participants.

Les variables d'âge et de genre pourront être examinées de manière secondaire, avec prudence quant aux effectifs et aux différences de protocole. Elles ne seront pas utilisées pour interpréter automatiquement les profils comme des catégories biologiques.

Le code, les paramètres, les choix de prétraitement, les partitions et les analyses d'incertitude seront versionnés.

Les évaluations finales seront séparées des simulations et des analyses de développement.

Toute modification du protocole après consultation de résultats réservés sera explicitement documentée comme exploratoire.

Les conclusions seront limitées à la validité empirique du proxy, aux scénarios de généralisation effectivement testés et aux modalités physiologiques disponibles.

---

# 22. Bibliographie de référence

### Article fondateur pour notre démarche

Gerster, M., Sirotina, E., Orlovskii, A., Hertz, A., Champaud, J., Guarino, D., & Tulli, S. (2026). *Stimulus identity rather than emotion drives EEG classification on the FACED dataset*. bioRxiv, prépublication.  
https://doi.org/10.64898/2026.06.12.731889

### Mesure et classification du stress

Pei, D., Tirumala, S., Tun, K. T., Ajendla, A., & Vinjamuri, R. (2024). *Identifying neurophysiological correlates of stress*. Frontiers in Medical Engineering.  
https://doi.org/10.3389/fmede.2024.1434753

Roy, S., et al. (2025). *Multimodal assessment of acute stress dynamics using an aversive video paradigm (AVP)*. International Journal of Clinical and Health Psychology.  
https://doi.org/10.1016/j.ijchp.2025.100607

Ernst, H., et al. (2023). *Assessment of the human response to acute mental stress—An overview and a multimodal study*. PLOS ONE.  
https://doi.org/10.1371/journal.pone.0294069

Ding, Y., et al. (2022). *Dynamic Tracking of State Anxiety via Multi-Modal Data and Machine Learning*. Frontiers in Psychiatry.  
https://doi.org/10.3389/fpsyt.2022.757961

### Spécificité et variabilité physiologique

Siegel, E. H., et al. (2018). *Emotion Fingerprints or Emotion Populations? A Meta-Analytic Investigation of Autonomic Features of Emotion Categories*. Psychological Bulletin.  
https://doi.org/10.1037/bul0000128

Mauss, I. B., Levenson, R. W., McCarter, L., Wilhelm, F. H., & Gross, J. J. (2005). *The Tie That Binds? Coherence Among Emotion Experience, Behavior, and Physiology*. Emotion.  
https://doi.org/10.1037/1528-3542.5.2.175

Brown, C. L., et al. (2019). *Coherence Between Subjective Experience and Physiology in Emotion: Individual Differences and Implications for Well-being*. Emotion.  
https://doi.org/10.1037/emo0000579

### Méthodes et données

Brouwer, A.-M., Zander, T. O., van Erp, J. B. F., Korteling, J. E., & Bronkhorst, A. W. (2015). *Using Neurophysiological Signals that Reflect Cognitive or Affective State: Six Recommendations to Avoid Common Pitfalls*. Frontiers in Neuroscience.  
https://doi.org/10.3389/fnins.2015.00136

Chatzaki, C., & Tsiknakis, M. (2025). *An Overview of Stress Analysis Based on Physiological Signals: Systematic Review of Open Datasets and Current Trends*. Sensors.  
https://doi.org/10.3390/s25237108

Hermann, B. (2024). *Réseaux de la conscience*. DIU Morphologie et anatomie du système nerveux central, Université Paris Cité, support de cours du 29 mars 2024.

### Documentation du projet

HABS Neuro-AI Hackathon — [Guide du dataset](https://github.com/mathiaschebbah/habs/blob/main/DATASET.md)

HABS — [Dossier de recherche et revue bibliographique](https://github.com/mathiaschebbah/habs/tree/main/docs/research)

HABS — [Audit de faisabilité](https://github.com/mathiaschebbah/habs/blob/main/data/FEASIBILITY.md)

---

# 23. Conclusion générale

La recherche en reconnaissance automatique du stress a montré que les réponses cérébrales et physiologiques peuvent contenir une information discriminante sur certaines conditions expérimentales de stress.

Toutefois, la nature exacte de cette information reste difficile à déterminer.

Une classification performante peut refléter des différences entre stimuli, des caractéristiques stables des participants ou des processus physiologiques non spécifiques du stress.

Les résultats de Gerster et al. (2026) rappellent particulièrement la nécessité de distinguer la reconnaissance des stimuli de la prédiction des expériences émotionnelles individuelles.

Dans ce contexte, notre projet propose de déplacer la question scientifique.

Il ne s'agit plus seulement de demander si un algorithme peut reconnaître une condition stressante à partir d'un EEG.

Nous chercherons à déterminer si les signaux du cerveau et du corps permettent d'expliquer et de prédire des différences individuelles de réactivité émotionnelle aversive au-delà de l'information déjà disponible sur les stimuli et les participants.

Nous étudierons ensuite si cette information correspond à une dimension physiologique continue, à des relations dose-réponse non linéaires ou à des configurations individuelles reproductibles.

**L'hypothèse centrale est que la réactivité associée au stress émotionnel pourrait être mieux décrite par des signatures multimodales individualisées que par un seuil physiologique universel.**

La valeur scientifique du projet reposera sur la qualité de sa validation : comparaison à des références solides, contrôle des stimuli, séparation des participants et des séances, analyse des relations dose-réponse et test indépendant de la stabilité des profils.

Un résultat positif soutiendrait l'intérêt d'une modélisation neurophysiologique personnalisée.

Un résultat négatif permettrait de préciser les limites de l'information contenue dans les signaux et les conditions nécessaires pour progresser vers une mesure plus fiable du stress.

Dans les deux cas, l'étude vise à contribuer à une question fondamentale des neurosciences computationnelles :

**Peut-on identifier dans les réponses du cerveau et du corps une information reproductible sur ce qu'un individu déclare ressentir, indépendamment de ce que l'on sait déjà du stimulus auquel il est exposé ?**
