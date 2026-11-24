# 09_eeg_predictive_physiology : l'EEG précoce apporte-t-il une information prédictive additionnelle sur la physiologie future au-delà du passé physiologique ?

**Protocole pré-enregistré et audit méthodologique préalable**, rédigé le 2026-11-24 **avant** toute nouvelle extraction de signaux, tout entraînement de modèle ou toute inspection de résultats de l'Étude 09.

---

## 1. Audit rigoureux des Études 07 et 08 (avant toute expérimentation)

L'audit complet du code (`decoding/common.py`, `decoding/timecourse.py`, `decoding/timecourse_extract.py`, `decoding/revision.py`, `decoding/rich_extract.py`, `decoding/ssl_eval.py`, `decoding/ssl_fm_extract.py`, `decoding/ssl_data.py`, `decoding/ssl_train.py`, `decoding/ssl_study.py`) et des agrégats publiés (`docs/research/revision/results/`, `docs/research/ssl/results/`) a mis en évidence **six défauts méthodologiques majeurs** qui rendent non conclusives les affirmations de l'Étude 08 sur la prévision cerveau→corps et la comparaison entre protocoles.

### 1.1 Tableau d'audit code par code

| # | Fichier & ligne | Défaut identifié | Mécanisme de biais / fuite | Correction Étude 09 |
|---|---|---|---|---|
| **A1** | `decoding/ssl_eval.py:279–283, 343–344` ; `decoding/revision.py:1603–1604, 1703, 1772` ; `decoding/timecourse.py:147` ; `decoding/ssl_study.py:58, 284, 378` ; `decoding/ssl_train.py:1114, 1201, 1310, 1333` | **Normalisation transductive globale (`person_center_matrix` / `_person_centre`)** appliquée sur toute la table avant découpage | (1) En CV `dev` 3×3, la moyenne d'un sujet `dev` agrège ses 3 folds de thèmes (y compris le fold de thème retenu `t`) ET ses essais de `test_new_video` ; (2) pour les sujets inédits (`test_new_both`, `test_new_person`), soustraire la moyenne de session sur tous leurs essais de test (futurs compris) est **transductif** et confond inférence *cold-start* et personnalisation avec historique non labellisé. | Séparation stricte **Protocole A (`cold_start`, zéro historique du sujet, invariance par essai isolé)** vs **Protocole B (`unlabeled_history` : B0 repos seul, B1 historique causal strict en ligne avec rétrécissement empirique de Bayes, B2 session isolée par fold explicitement étiquetée)**. Garde-fous unitaires anti-fuite dans `ssl_eval.py` et `revision.py`. |
| **A2** | `decoding/ssl_fm_extract.py:282–292, 345–355, 415–435, 572–586` | **Fuite temporelle bidirectionnelle future dans l'extraction `REVE-base`, `CBraMod`, `LaBraM`, `Signal-JEPA`** | `REVE-base` utilise une auto-attention spatiotemporelle **bidirectionnelle** (`causal=False` dans `modeling_reve.py`). En passant la fenêtre entière `[-3, +10) s` en une seule passe puis en découpant les jetons `3:8` comme `z_early`, chaque jeton `z_early` a directement reçu l'information EEG future de **`[5, 10) s`** (contemporaine de `Corps [5, 10) s`). À l'inverse, `S07`, `PhysioJEPA` (`extract_jepa_eeg_only_early`) et `EEGPT` (fenêtres isolées 4 s) n'avaient pas accès à `t >= 5 s` — expliquant l'écart suspect `0,255 / 0,136 / 0,213` vs `0,159–0,177 / 0,051–0,089 / 0,053–0,099` ! | **Mur causal dur à $t = 5{,}000\text{ s}$** : aucun échantillon $t \ge 5{,}0\text{ s}$ n'entre dans le filtrage, la normalisation ni l'encodeur `REVE-base` (`[-3{,}2, +5{,}0) s` et `[0{,}0, +5{,}0) s` passés isolément). Quantifier exactement l'inflation due à la fuite bidirectionnelle en comparant `fm_reve.parquet` (Étude 08) à `study09_reve.parquet` (strictement causal). |
| **A3** | `decoding/ssl_eval.py:673–716` ; `decoding/ssl_study.py:210, 317, 389` | **Absence de baseline autorégressive `M0 : Corps[0:5] → Corps[5:10]`** et **raccourci stimulus (`test_new_person`)** | L'Étude 08 n'a évalué que `M1` (`EEG[0:5] → Corps[5:10]`), sans jamais mesurer le gain résiduel `M2 − M0` au-delà de l'inertie autonome `Corps[0:5]`. De plus, l'évaluation mêlait `test_new_both` (1 481 essais, 18 thèmes inédits) et `test_new_person` (4 934 essais, **66 thèmes déjà vus dans `dev`**, soit 77 % des essais), permettant un raccourci de reconnaissance de stimulus familier. | Question principale recentrée sur le **gain résiduel apparié `M2 − M0`** (`M0` linéaire riche ET `M0` non linéaire GBDT) évalué exclusivement sur blocs **sujets inédits × thèmes inédits** dans `dev` (CV imbriquée bloquée 3×3 + sensibilité 5×5) avec décomposition intra-vidéo vs inter-vidéos et contrôle négatif par permutation intra-vidéo. |
| **A4** | `decoding/ssl_fm_extract.py:121–125, 575–577` ; `decoding/timecourse_extract.py:57, 97, 101` ; `decoding/ssl_data.py:111, 158, 220, 228` | **Filtrage acausal (`sosfiltfilt`), Hilbert global, interpolation linéaire inter-battements et z-score sur l'enregistrement entier** | `sosfiltfilt`, la transformée de Hilbert sur tout le fichier, l'interpolation `np.interp` entre un pic R avant 5 s et un pic R après 5 s, et `rec_mean/rec_std` sur 22 min violent à la fois le *cold-start* et la séparation stricte à $t = 5\text{ s}$. | Filtrage causal avant (`sosfilt` avec `sosfilt_zi`) coupé strictement à $t < 5{,}0\text{ s}$, pics R et conductance strictement internes à `[-3, 5) s` (zéro battement $\ge 5{,}0\text{ s}$), amplitude respiratoire RMS causale sans Hilbert global, fenêtres cibles isolées avec **écarts de garde (`[7, 10) s` : +2 s ; `[10, 12) s` : +5 s)**, et normalisation `REVE-base` strictement causale par essai (Protocole A) ou sur l'historique passé strict (Protocole B1). |
| **A5** | `decoding/ssl_eval.py:306, 321, 323` ; `decoding/revision.py:250, 1806` ; `decoding/ssl_train.py:290, 779` | **Mise à l'échelle transductive de `pa/std(pa) - pv/std(pv)` sur le lot d'évaluation** et **CV interne non groupée** | Diviser `pa` et `pv` par leur écart-type calculé sur `x_ev` fuit les statistiques du lot de test et casse l'inférence essai-par-essai. De plus, `LogisticRegressionCV(cv=3)` et `RidgeCV` (GCV) sélectionnent les hyperparamètres par CV aléatoire non groupée (fuites sujet/thème dans la boucle interne). | Écarts-types `sd_pa_tr`, `sd_pv_tr` calculés exclusivement sur les prédictions OOF internes du fold d'entraînement. Sélection de tous les hyperparamètres (`alpha`, `C`, `n_pca`, poids de mélange) par **CV interne imbriquée bloquée sujets × thèmes** strictement dans le fold d'entraînement. |
| **A6** | `docs/research/ssl/RESULTS.md:17, 225` ; `decoding/extract.py:7–8` | **Revendication non prouvée d'un « plafond physiologique intrinsèque »** ; **données `pupil/gaze` jamais auditées quantitativement** ; **`test_new_both` Étude 08 désormais exposé** | Conclure à un plafond biologique à partir de sondes linéaires ou de 53 h de SSL est épistémiquement injustifié. Les canaux oculométriques (16 canaux à ~166 Hz dans `.record_map`) ont été écartés sur déclaration informelle sans audit chiffré. Enfin, `test_new_both` de `habs-ssl-2026-11-24-b` a déjà été consulté dans l'Étude 08. | (1) Distinction stricte prédiction / association / causalité ; aucune revendication de plafond biologique ; (2) audit quantitatif complet de `pupil/gaze` sur les 508 enregistrements avec porte d'exploitabilité préspécifiée ; (3) **`test_new_both` Étude 08 classé comme EXPOSÉ** — toute sélection et inférence principale se fait exclusivement dans `dev` par CV imbriquée bloquée sujets × thèmes. |

---

## 2. Séparation stricte des protocoles d'évaluation

### 2.1 Espace d'inférence principal (`dev` uniquement) et statut du split exposé

- **Corpus de développement (`dev` de `STUDY08_SALT = "habs-ssl-2026-11-24-b"`) :** 12 693 essais, **228 participants** (208 éligibles stress, 124 multi-jours), **66 thèmes vidéo**.
- **Schéma de validation croisée imbriquée bloquée sujets × thèmes :**
  - **Boucle externe (inférence principale) :** 3 folds sujets × 3 folds thèmes = **9 blocs externes** `(p, t)`. Chaque bloc externe `(p, t)` est prédit exclusivement par un pipeline ajusté sur `(person_fold != p) & (theme_fold != t)` (sujets strictement inédits ET thèmes vidéo strictement inédits). Tous les prétraitements (imputation, standardisation, PCA, rétrécissement trait/état, sélection d'hyperparamètres, adaptation LoRA, méta-apprenant éventuel) sont **strictement isolés par fold externe**.
  - **Boucle interne (sélection des hyperparamètres et arrêt précoce) :** à l'intérieur de chaque ensemble d'entraînement externe `(p' != p, t' != t)`, une CV interne bloquée `3×3` sujets × thèmes (hachage déterministe indépendant du split externe) sélectionne `alpha`, `C`, la dimension PCA, les hyperparamètres de factorisation trait/état et les poids de stacking éventuels ; pour les modèles PyTorch/LoRA, l'arrêt précoce s'effectue sur un sous-ensemble interne **disjoint à la fois en sujets (15 %) et en thèmes (15 %)**.
  - **Contrôle de sensibilité de granularité :** sur la question principale `M0 / M1 / M2` (modèles gelés), une vérification supplémentaire en CV externe bloquée **5×5** (25 blocs) confirme que les conclusions ne dépendent pas du découpage 3×3.
- **Statut de `test_new_both` (et `test_new_person`, `test_new_video`) de l'Étude 08 :** formellement déclarés **EXPOSÉS**. Ils ne sont utilisés pour aucune sélection ni aucune validation indépendante ; ils servent uniquement à un **audit rétrospectif transparent** quantifiant exactement la baisse de performance lorsque l'on corrige les fuites A1–A3 de l'Étude 08.

### 2.2 Protocole A (*Cold-Start*) vs Protocole B (*Personnalisation avec historique non labellisé*)

1. **Protocole A — `cold_start` (zéro historique du sujet) :**
   - Lors de l'inférence sur l'essai $i$ d'un sujet inédit, le modèle ne dispose **que** des signaux causaux de l'essai $i$ (`[-3, +5) s` pour la prévision ; `[-3, +10) s` pour l'affect).
   - **Invariance testée unitairement :** prédire l'essai $i$ seul (`batch_size = 1`) produit exactement le même résultat (à $10^{-5}$ près) que prédire l'essai $i$ dans un lot de 100 essais.
   - Normalisation EEG d'entrée `REVE-base` : z-score par canal calculé **uniquement sur la fenêtre causale de l'essai $i$** (`[-3{,}2, +5{,}0) s` pour la prévision ; `[-3{,}0, +10{,}0) s` pour l'affect), écrêté à $\pm 15\sigma$. Aucune statistique de session, aucun autre essai.
2. **Protocole B — `unlabeled_history` (historique physiologique non labellisé autorisé, zéro label du sujet) :**
   - **B0 (`rest_only`) :** calibration individuelle uniquement à partir de l'enregistrement `resting-state` non labellisé du sujet (disponible avant tout clip émotionnel).
   - **B1 (`causal_online_history`, référence principale Protocole B) :** moyenne individuelle estimée **strictement sur le passé causal non labellisé** du sujet (essais strictement antérieurs $1, \dots, i-1$ de la session + `resting-state` si disponible), stabilisée par **rétrécissement empirique de Bayes** vers la moyenne populationnelle du fold d'entraînement :
     $$\hat{\mu}_{p, i} = \frac{n_{p,<i}\,\bar{x}_{p,<i} + \kappa\,\mu_{\text{train}}}{n_{p,<i} + \kappa}$$
     avec $\kappa$ sélectionné par CV interne groupée. L'essai $i=1$ démarre donc continûment au prior populationnel et aucun essai futur $j > i$ n'est jamais lu.
   - **B2 (`session_isolated`, audit comparatif transductif explicite) :** centrage sur la session non labellisée du sujet **strictement isolée à l'intérieur de son bloc d'évaluation `(p, t)`** (sans jamais traverser vers le train ni un autre fold), permettant la comparaison directe avec la pratique de l'Étude 08 tout en éliminant la fuite inter-splits.

---

## 3. Question principale (Priorité absolue) : prévision physiologique résiduelle `M2 − M0`

### 3.1 Formulation des trois modèles emboîtés

Pour un essai $i$, avec un **mur causal strict à $t = 5{,}000\text{ s}$** (zéro échantillon $t \ge 5{,}0\text{ s}$ en entrée) :

- **$M_0$ (`Corps[0:5] → Corps[5:10]`) :** régression régularisée (Ridge CV imbriquée groupée, plus contrôle non linéaire `HistGradientBoostingRegressor` `M0_gbdt`) à partir des descripteurs physiologiques périphériques causaux riches sur `[-3, 5) s` (trajectoire 1 s ECG HR/RR, 2 derniers intervalles RR avant 5 s, pente/accélération/RMSSD `[0, 5)`, niveau/dérivée phasique `[3, 5)` et pics SCR de conductance EDA calibrée, amplitude RMS causale 1 s, phase/vitesse à $t = 4\text{--}5\text{ s}$ et fréquence respiratoire, ainsi que le vecteur complet des 3 modalités périphériques conjointement afin de ne pas affaiblir artificiellement $M_0$).
- **$M_1$ (`EEG[0:5] → Corps[5:10]`) :** prédiction à partir de l'EEG précoce causal seul (`[-3{,}2, +5{,}0) s` ou `[0, 5) s`), évalué pour :
  - `M1_handcrafted` : caractéristiques spectrales/spatiales/volatilité EEG causales manuelles ;
  - `M1_reve_frozen` : **référence principale EEG — `REVE-base` gelé** (`brain-bzh/reve-base`, 69,2 M params, 62/62 voies), inférence causale stricte sur `[-3{,}2, +5{,}0) s` (`h = 9` patches se terminant exactement à $+5{,}000\text{ s}$), représentation structurée (`early_1`, `early_2`, `early_mean`, `early_delta`, `attn_pool`) réduite par PCA/Ridge dans chaque fold interne.
- **$M_2$ (`(EEG + Corps)[0:5] → Corps[5:10]`) :** modèle combiné emboîté ajoutant les représentations EEG précoces à $M_0$ via deux paramétrisations complémentaires :
  - `M2_joint_ridge` : régression Ridge à pénalités séparées $(\lambda_{\text{body}}, \lambda_{\text{eeg}})$ sélectionnées par CV interne groupée (garantissant qu'une haute dimensionnalité EEG ne dégrade pas les poids de $M_0$ par sur-régularisation partagée) ;
  - `M2_residual_reve_frozen` : apprentissage en deux étapes strictement dans le fold d'entraînement — $M_0$ prédit $\hat{y}_{M_0}$ (OOF interne), puis l'EEG prédit le résidu $e = y - \hat{y}_{M_0}$ avec facteur de mélange $\eta \in [0, 1]$ choisi par CV interne groupée ($\hat{y}_{M_2} = \hat{y}_{M_0} + \eta\,\hat{e}_{\text{EEG}}$), garantissant structurellement une variance contrôlée ;
  - `M2_gbdt_reve` : contrôle non linéaire (GBDT sur `Corps[0:5]` + PCs causales `REVE-base`) ;
  - `M2_reve_lora` : **adaptation LoRA légère sur `REVE-base`** (`layers[16:22]`, `to_qkv` + `to_out`, rang $r = 8$, $\alpha = 16$, ~147 k paramètres LoRA = 0,21 % du backbone, 3 seeds `0, 1, 2`) entraînée dans `dev` (CV externe 3×3) sur la prévision conjointe/résiduelle de `Corps[5:10]` à partir de `EEG[-3{,}2, 5{,}0)` + `Corps[0:5]`, avec contrôle apparié sans LoRA (`M2_reve_frozen_head`).

### 3.2 Cibles physiologiques et délais préspécifiés

| Modalité cible | Cible primaire `[5, 10) s` | Cible incrémentale de trajectoire | Délais physiologiques préspécifiés (justification) |
|---|---|---|---|
| **ECG** | $\Delta\text{HR}_{[5,10)} = \text{HR}_{[5,10)} - \text{HR}_{[-3,0)}$ (et $\Delta\text{RR}_{[5,10)}$) | $\text{HR}_{[5,10)} - \text{HR}_{[0,5)}$ | `[5, 8) s` (latence vagale rapide 1–3 s), **`[7, 10) s`** *(écart de garde +2 s)*, `[10, 12) s` *(récupération post-clip, +5 s)* |
| **EDA** | $\Delta\text{SC}_{[5,10)} = \text{SC}_{[5,10)} - \text{SC}_{[-3,0)}$ et $\text{SCR\_rise\_new}_{[5,10)} = \max \text{SC}_{[5,10)} - \text{SC}_{[4,5)}$ | $\text{SC}_{[5,10)} - \text{SC}_{[0,5)}$ | **`[6, 9) s`** *(pic sudomoteur canonique +1 à +4 s)*, **`[7, 10) s`** *(écart de garde +2 s)*, `[10, 12) s` *(queue sudomotrice, +5 s)* |
| **Respiration** | $\Delta\text{RespAmp}_{[5,10)} = \text{Amp}_{[5,10)} - \text{Amp}_{[-3,0)}$ (et $\Delta\text{RespRate}_{[5,10)}$) | $\text{Amp}_{[5,10)} - \text{Amp}_{[0,5)}$ | `[5, 8) s` (cycle courant), **`[7, 10) s`** *(cycle suivant, écart de garde +2 s)*, `[10, 12) s` *(cycle post-clip, +5 s)* |

### 3.3 Métriques appariées, hypothèses préspécifiées et critères de succès / réfutation

- **Métriques par participant inédit** (pour chaque sujet $p$ ayant $\ge 5$ essais valides sur la cible) :
  1. Corrélation de Pearson intra-sujet $r_p(y, \hat{y})$ ;
  2. Variance expliquée intra-sujet $R^2_{\text{within}, p} = 1 - \frac{\sum_i ((y_{p,i}-\bar{y}_p) - (\hat{y}_{p,i}-\bar{\hat{y}}_p))^2}{\sum_i (y_{p,i}-\bar{y}_p)^2}$ (centrée intra-sujet) et $R^2_{\text{raw}, p} = 1 - \frac{\sum_i (y_{p,i} - \hat{y}_{p,i})^2}{\sum_i (y_{p,i} - \bar{y}_p)^2}$ (sensible au biais d'échelle en cold-start !) ;
  3. Réduction relative d'erreur $\Delta\text{RMSE}_p = \text{RMSE}_p(M_0) - \text{RMSE}_p(M_2)$ (positif = gain de $M_2$).
- **Inférence statistique :** moyenne sur les 228 sujets `dev`, **IC 95 % bootstrap apparié par sujet** (2 000 tirages), test bilatéral de Wilcoxon signé-rang ($p$), correction Benjamini-Hochberg FDR ($q$) sur la famille des 3 cibles primaires (puis sur la grille complète des délais), ainsi que la distribution complète inter-sujets (médiane, IQR, déciles P10/P90, proportion de sujets améliorés $P(M_{2,p} > M_{0,p})$).

#### Hypothèses et critères de décision préspécifiés (H1–H5)

- **$H_1$ (Question principale — Information EEG prédictive additionnelle sur la physiologie future) :**
  - **Critère de confirmation forte sur une modalité (ECG, EDA ou Respiration) :**
    1. Gain apparié positif et statistiquement fiable en CV imbriquée `dev` : $\Delta r(M_2 - M_0) > 0$ avec **IC 95 % apparié excluant 0** ET $q_{\text{FDR}} < 0{,}05$ ET $\Delta R^2_{\text{within}} > 0$ (IC 95 % > 0) ;
    2. **Seuil de pertinence pratique minimale :** $\Delta r(M_2 - M_0) \ge +0{,}010$ (ou $\Delta R^2_{\text{within}} \ge +0{,}005$) ;
    3. **Survie aux contrôles :** le gain demeure significativement positif (IC 95 % > 0) après : (a) ajout des covariables d'artefacts (`M0 + artefacts`), (b) exclusion des canaux frontopolaires/périphériques (`interior_only`) ou résidualisation artefacts, (c) contrôle d'ordre/essai précédent (`M0 + prev_trial`), et (d) écart de garde de 2 s (`[7, 10) s`) ;
    4. **Passage de tous les contrôles négatifs** (voir § 3.4).
  - **Critère de confirmation partielle (médiée par le stimulus) :** $H_1$ vérifie les points 1–4 sur thèmes inédits, mais s'annule lors du contrôle intra-vidéo (`M2_real − M2_vid_shuf` ou corrélation intra-vidéo nulle) $\Rightarrow$ information EEG additionnelle **réelle et généralisable à des vidéos inédites**, mais **médiée par la réponse normative commune au stimulus** plutôt que par un couplage idiosyncratique spécifique à l'essai.
  - **Critère de réfutation (résultat nul établi) :** sur les trois modalités (ECG, EDA, Resp) et tous les délais préspécifiés, les gains appariés `M2 − M0` (gelé et LoRA) ont des IC 95 % incluant zéro, ou disparaissent intégralement sous les contrôles d'artefacts / d'écart de garde / `M0_gbdt` $\Rightarrow$ **réfutation de l'hypothèse d'une information EEG prédictive additionnelle exploitable au-delà de `Corps[0:5]` dans HABS** (et diagnostic que le résultat de l'Étude 08 provenait des fuites A2/A3).

- **$H_2$ (Secondaire — `REVE-base` gelé vs LoRA multitâche + ranking intra-sujet) :**
  - Comparaison en CV imbriquée `dev` 3×3 (3 seeds `0, 1, 2`) : `REVE_Frozen_Linear` vs `REVE_Frozen_MT_Rank` vs `REVE_LoRA_MT_Rank` (objectif : BCE pondérée stress + Smooth-L1 valence + Smooth-L1 arousal + perte de ranking logistique par paires intra-sujet).
  - **Critère de succès LoRA :** gain apparié `REVE_LoRA_MT_Rank − REVE_Frozen_MT_Rank` avec IC 95 % > 0 sur l'AUC stress intra-sujet ou $\bar{\rho}$ valence/arousal (et gain positif face à `REVE_Frozen_Linear`).

- **$H_3$ (Factorisation trait individuel / état transitoire) :**
  - Comparaison de 5 représentations : `raw_unfactored`, `blind_ref_sub` ($z_{\text{stim}} - z_{\text{ref}}$), `blind_person_center`, `state_only_shrinkage` ($z_{\text{stim}} - \mathbf{W}_{\text{shrink}} z_{\text{base}}$), et `factored_trait_plus_state` ($[\hat{z}_{\text{state}}, \hat{z}_{\text{trait}}]$) sous Protocole A et Protocole B.
  - **Critère de succès :** `state_only_shrinkage` bat `blind_ref_sub` sur les métriques intra-sujet (réduction de variance) ET `factored_trait_plus_state` préserve/améliore les métriques poolées (AUC poolée / Brier / prédiction de sensibilité-trait inter-jours) sans dégrader les métriques intra-sujet.

- **$H_4$ (Audit `pupil/gaze` et porte d'exploitabilité préspécifiée) :**
  - **Porte d'exploitabilité (vérifiée AVANT tout modèle sur les cibles) :**
    1. Couverture essais valides ($\ge 50\%$ d'échantillons non nuls dans $[1{,}0, 10{,}0]\text{ mm}$) $\ge 50\%$ de tous les essais (ou $\ge 60\%$ dans un sous-ensemble de cohortes clairement identifié) ;
    2. Corrélation binoculaire médiane gauche-droite $r(\text{pupil}_L, \text{pupil}_R) \ge 0{,}50$ ;
    3. **Preuve objective d'alignement temporel :** déflexion pupillaire moyenne évoquée à l'onset vidéo ($t \in [0{,}25, 2{,}5)\text{ s}$ vs pré-stimulus $[-2, 0)\text{ s}$) cohérente et statistiquement significative ($|t| > 4$, $p < 10^{-4}$), sans décalage temporel aberrant avant $t = 0\text{ s}$.
  - Si et seulement si cette porte passe, des baselines simples causales `pupil/gaze` sont ajoutées (affect + contrôle `M0 + oculo`). En cas d'échec, abstention explicite et rapport détaillé des défauts mesurés.

- **$H_5$ (Stacking OOF conditionnel à une complémentarité démontrable) :**
  - **Porte de complémentarité (calculée sur les prédictions OOF `dev` avant tout ajustement de méta-modèle) :**
    1. Au moins deux familles distinctes fiablement au-dessus du hasard (IC 95 % > 0,50 ou > 0) ;
    2. Corrélation intra-sujet entre leurs prédictions $r(\hat{y}_a, \hat{y}_b) < 0{,}85$ ;
    3. Contribution conditionnelle unique (corrélation partielle intra-sujet $r(y, \hat{y}_b \mid \hat{y}_a)$) strictement positive avec **IC 95 % > 0** pour au moins un modèle secondaire au-delà du meilleur modèle unique.
  - Si la porte échoue : **aucun stacking OOF n'est ajusté** ; la table de complémentarité documente formellement l'abstention.

### 3.4 Contrôles négatifs et contrôles de confusion préspécifiés

1. **NC1 — Permutation intra-sujet de `EEG[0:5]` (`eeg_within_person_shuffle`) :** permute les vecteurs `EEG[0:5]` entre essais d'un *même* participant en gardant `Corps[0:5]` intact. Attendu impératif : $\Delta r(M_{2,\text{NC1}} - M_0) \le 0$ (IC 95 % couvrant $\le 0$). Tout gain positif invalide immédiatement le pipeline (détecteur de bug/fuite).
2. **NC2 — Permutation intra-vidéo inter-sujets de `EEG[0:5]` (`eeg_within_video_shuffle`) :** remplace l'EEG précoce du participant $p$ sur la vidéo $v$ par l'EEG précoce d'un *autre* participant $q \neq p$ regardant la *même* vidéo $v$ (en gardant `Corps[0:5]` de $p$ intact). Permet de décomposer exactement :
   - Part **médiée par le stimulus partagé** : $M_{2,\text{NC2}} - M_0$ ;
   - Part **spécifique à l'essai / idiosyncratique** : $M_{2,\text{réel}} - M_{2,\text{NC2}}$ (complétée par l'évaluation directe intra-vidéo `r_within_video`).
3. **NC3 — EEG pré-stimulus seul (`EEG[-3:0] + Corps[0:5] → Corps[5:10]`) :** vérifie si un gain provient de l'état tonique pré-essai ou spécifiquement de la réponse corticale évoquée post-onset `[0, 5) s`.
4. **NC4 — Permutation temporelle interne à `[0, 5) s` (`token_time_shuffle`) :** permute les 5 secondes de `EEG[0:5]` afin de distinguer la trajectoire temporelle de la puissance moyenne statique.
5. **Contrôles d'artefacts (`AC1`–`AC3`) :**
   - `AC1` : `M0 + Artefacts[0:5]` (proxy clignement `CURATED` Fp1/Fp2 ptp & 0,5–4 Hz, EMG périphérique `rim` 30–45 Hz & 45–80 Hz vs `interior`, RMS `CURATED − PREPROCESSED` / proxy champ cardiaque) vs `M2` ;
   - `AC2` : `REVE-base` évalué avec **masquage des 16 canaux périphériques/frontopolaires (`interior_only`, 46 voies centrales/pariétales/frontales internes)** et/ou résidualisation linéaire des covariables d'artefacts dans chaque fold ;
   - `AC3` : sous-ensemble haute qualité (0 canal EEG défaillant, >95 % battements QRS propres, EDA dans la plage de calibration).
6. **Contrôles de dépendance temporelle inter-essais (`TC1`) :** `M0 + Corps[essai t-1] + event_order + onset_s` pour écarter toute dérive lente de session.

---

## 4. Plan d'exécution et gouvernance du calcul

| Étape | Module | Support | Budget estimé | Sorties |
|---|---|---|---|---|
| 1. Extraction causale + `REVE-base` causal + Audit `pupil/gaze` | `decoding/study09_extract.py` (`stage="study09_extract"`) | 1 job H100 (`submit.yml`) | ~6–10 min H100 | `features/study09_causal.parquet`, `study09_reve.parquet`, `study09_oculo.parquet` ; `public/study09_extract_audit.json` |
| 2. `REVE-base` LoRA (Prévision `M2_LoRA` + Secondaire Multitâche & Ranking intra-sujet, 3 seeds, CV 3×3 `dev`) | `decoding/study09_lora.py` (`stage="study09_lora"`) | 1 job H100 (`submit.yml`) | ~25–40 min H100 | `features/study09_lora_preds.parquet` ; `public/study09_lora.json` |
| 3. Analyse complète CV imbriquée `dev` (audit É07–08, M0/M1/M2, délais, contrôles, trait/état, oculo conditionnel, stacking conditionnel) | `decoding/study09_analysis.py` (`--study study09`) | 1–2 jobs CPU (`decoding.yml`) | ~15–25 min CPU (0 GPU) | `docs/research/study09/results/*.json` |
| 4. Figures vérifiables + Rapport FR + Article LaTeX compilé | `docs/research/study09/make_figures.py`, `docs/paper/study09_fr.tex` | Local CPU + `pdflatex` | ~5 min | `docs/research/study09/figures/*.png`, `RESULTS.md`, `docs/paper/study09_fr.pdf` |

**Discipline épistémique finale :** tout résultat (positif, partiel ou négatif) sera rapporté fidèlement sans retuning a posteriori sur le test exposé ni interprétation causale abusive.
