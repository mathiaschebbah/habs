# Synthèse complète de l'étude et guide de soumission (Équipe 12)

**Équipe 12 (Multimodal AI), Hackathon Neuro-AI Université Paris-Saclay x HABS 2026**
Membres actifs : Mathias Chebbah, Aurélien Saint-Grégoire, Ashley Russo (mentor HABS : Andrew Jeyathasan).

Ce document rassemble l'ensemble des échanges de l'équipe (2 au 26 novembre 2026), les consignes officielles du hackathon (`docs/Hackathon_Guidelines.pdf`, kick-off du 2 novembre, réunion mentor du 9 novembre, annonce Slack du 11 novembre), les apports cliniques et méthodologiques d'Aurélien, la revue de littérature et le plan de présentation d'Ashley, ainsi que tous les résultats vérifiés des études 04 à 10. Il sert de base commune pour la relecture finale, le dépôt GitHub nettoyé, le papier soumis (`paper/submission_paper.pdf`) et la présentation enregistrée de 10 minutes.

---

## 1. Livrables officiels et critères du jury HABS

Selon `docs/Hackathon_Guidelines.pdf` et le briefing du kick-off, trois livrables sont évalués par les mentors (sélection de 10 finalistes le 10 décembre, finale le 14 décembre avec pitch de 10 min + 5 min de questions) :

1. **Dépôt GitHub minimal et reproductible** : code réduit au strict nécessaire (Moritz : *"hopefully not AI-bloated with one hundred thousand files, but as small and compact as possible"*), environnement documenté, scripts de reproduction et 1 à 2 fichiers `README` / notebook ciblés.
2. **Papier court (PDF, 2 à 5 pages de texte + au moins 2 figures)** : résumé dense, introduction centrée sur la question de recherche sans revue de littérature exhaustive, méthodes exactes, résultats illustrés, et discussion répondant à *"What do you think it means?"*. Notre soumission en anglais (`paper/submission_paper.pdf`, 5 pages exactement avec 4 figures double-colonne et 2 tableaux) respecte strictement ce format et la limite de 5 pages.
3. **Présentation PowerPoint enregistrée (~10 minutes, date limite dimanche 29 / lundi 30 novembre)** : doit couvrir les résultats ET six questions réflexives obligatoires du jury (parcours, difficultés, ce qui a pris plus de temps que prévu, limites des outils IA, apprentissage majeur, plan à 4 semaines).

**Pondération officielle du jury :**
- 15 % : Question de recherche et périmètre (question précise, adaptée aux données et au temps disponible).
- 35 % : Méthode et preuves (comparaisons et baselines adaptées, conclusions soutenues par les chiffres).
- 25 % : Apport scientifique et apprentissage (un résultat négatif rigoureux vaut autant qu'un modèle performant).
- 25 % : Reproductibilité et clarté de communication.

---

## 2. Fil directeur et intégration des contributions de l'équipe

### Question de recherche finale
> *Comment se déploient les réponses corticales (EEG) et périphériques (ECG, EDA, respiration) lors d'un stress aigu induit par des vidéos de 10 s, l'EEG précoce prédit-il causalement la physiologie future au-delà du passé corporel, et quelles sont les limites réelles d'un suivi passif multimodal chez des personnes, vidéos, jours et cohortes jamais vus ?*

### Chronologie et apports intégrés
- **2 novembre (Kick-off)** : présentation des 384 participants initiaux, des trois thèmes et des enjeux cliniques de HABS (notamment la mesure objective de la douleur chez les patients non verbaux).
- **8 novembre (Ashley, pistes initiales)** : cadrage autour du stress aigu, de la charge cognitive et de la dynamique attentionnelle.
- **9 novembre (Réunion mentor avec Andrew + cours Benjamin Hermann par Aurélien)** : décision stratégique confirmée par Aurélien après l'appel mentor : privilégier une question nette, des preuves propres et une histoire défendable à l'oral plutôt que la course au score. Introduction de l'hypothèse HEP (potentiels évoqués cardiaques) et de son lien potentiel avec le nettoyage ICA.
- **11 novembre (Annonce Slack de Moritz, épinglée par Kévin)** : les données d'oculométrie et de pupillométrie sont désalignées par rapport aux autres modalités et aux horodatages vidéo, avec consigne explicite de ne pas les utiliser dans les analyses principales. Notre audit indépendant (Étude 09) a confirmé une corrélation binoculaire élevée (`r = 0,855`) mais un réflexe pupillaire nul à `t = 0 s` (`p = 0,999`) et un décodage du stress au hasard (`AUC = 0,506`). Nous excluons donc l'oculométrie des modèles principaux et citons explicitement cet avis dans le papier.
- **20 au 23 novembre (Études 04 à 06 + critique structurée en 6 points d'Aurélien)** : Aurélien réoriente le récit hors du bloc opératoire vers la physiologie générale du stress aigu et le suivi clinique non verbal (pédiatrie pré-verbale, coma, locked-in syndrome, anesthésie générale). Ses six points déclenchent directement l'Étude 07 :
  1. Vérification du critère `V <= 4 et A >= 6` sur les émotions brutes de rang 1 à 3.
  2. Modèles directs d'émotions et vérification non supervisée (`ARI < 0,009`).
  3. Découpage en 4 phases (`ref`, `early`, `late`, `post`) et délinéation ECG battement par battement (notamment le segment PR pour tester le tonus adrénergique du noeud AV, et la correction d'une inversion de polarité de dérivation I présente sur 96,6 % des enregistrements).
  4. Test du HEP sur signal `CURATED` brut sans ICA pour exclure un artefact de suppression par ICA.
  5. Réanalyse à vidéo fixe en retirant la dilution des essais neutres/intermédiaires (`stressed` vs `very calm`) et séparation entre effet automatique et évaluation subjective.
  6. Analyse fine des différences individuelles et de la topographie EEG.
- **24 novembre (Études 08 et 09 par Mathias)** : comparaison systématique de 5 modèles de fondation EEG (`S-JEPA`, `CBraMod`, `LaBraM`, `EEGPT`, `REVE-base`) et d'un `PhysioJEPA` multimodal. Audit complet des fuites méthodologiques (centrage transductif `+0,0141`, encodeur bidirectionnel + thèmes déjà vus), mise en place du mur causal strict à `t = 5,000 s` vérifié au bit près, et démonstration du gain causal cerveau vers coeur (`M2 - M0`) avec 9 covariables d'artefacts et contrôles négatifs (`NC1` à `NC4`).
- **25 novembre (Revue de 39 articles et plan de présentation par Ashley)** : structuration de la littérature et de l'exposé en 5 axes exacts : cinétique rapide vagale vs sympathique, généralisation et délais EEG/SNA, attention alpha et limites de déploiement, cohérence subjectif-autonome, et traduction clinique non verbale.
- **25 et 26 novembre (Étude 10 finale, 100 % CPU)** : test empirique direct de chacun des 5 axes d'Ashley (échelle de généralisation `R1` à `R4`, dérive longitudinale `Jour 1 -> Jour 2`, `CORAL`, personnalisation chronologique `k`-shot `EB-MAP`, réplication `LOCO-CV` sur 5/5 cohortes, décomposition Varimax à 3 facteurs, suivi clinique passif de `10 s` à `100 s`, et comparaison intra-clip Peur vs Dégoût).

---

## 3. Synthèse consolidée des résultats vérifiés (314 sujets, 22 974 essais)

Cohorte analysée : 314 adultes sains, 22 974 essais alignés, 508 enregistrements, 252 vidéos de 10 s (84 thèmes sémantiques), 5 cohortes, 169 participants mesurés deux fois à une semaine d'intervalle, 0 essai marqué outlier.

| Axe | Résultat principal vérifié | Valeur et incertitude | Portée / résultat négatif associé |
| --- | --- | --- | --- |
| **1. Critère & cascade** | `V <= 4 et A >= 6` couvre 89,2 % d'émotions négatives au rang 1 (97,2 % rangs 1 à 3). Cascade : EEG delta `<1 s`, FC `1 s`, Resp `2-6 s`, EDA `4-10 s`. | Durcir le seuil (`2/8`) double l'EDA (`+0,143 -> +0,342`) et le delta (`+0,126 -> +0,302`), et réduit de moitié la baisse de FC (`-0,106 -> -0,041`). | Sur 10 s, la FC suit 60 s (`r = 0,942`), mais le RMSSD est peu fiable (`r = 0,425`, `0,224` sur 5 s). Clustering non supervisé nul (`ARI < 0,009`). |
| **1b. ECG & HEP** | 96,6 % des ECG Lead I sont inversés (corrigés automatiquement). Allongement RR (`+5,06 ms`). | `d(RR) = +0,116 [+0,071, +0,163]`. `d(PR seg) = -0,005`. QTc Bazett passe de `-0,088` à `-0,007` après contrôle de `dRR`. | Profil compatible avec une décélération vagale sans variation mesurable du PR ni du QTc ajusté sur 10 s. HEP brut `CURATED` (`+0,038`) = HEP ICA (`+0,039`), tous deux nuls. |
| **1c. Vidéo fixe & émotions** | À vidéo identique (48 clips), `stressed` vs `very calm` monte l'AUC de `0,525` à `0,564` et déplace la FC de `-2,97 bpm` (`d = -0,41`). | Les sujets calmes face aux clips aversifs ont une FC et une EDA nulles ; seule la baisse alpha occipitale reste automatique (`d = -0,33 à -0,41`). | Peur vs Dégoût intra-clip (`N = 209`) : Dégoût mène en delta précoce (`-0,098`) et gamma frontal tardif (`-0,178`, prudence EMG) ; Peur mène en alpha (`-0,137`) et EDA tardif (`+0,193`). |
| **2. Audit de fuites** | Le centrage global gonfle LogReg de `+0,0141`. Un `REVE` bidirectionnel 13 s sur 77 % de thèmes vus créait de faux gains croisés. | Fuites corrigées : `+0,059` FC, `+0,042` EDA, `+0,069` Resp disparaissaient sous frontière temporelle `t = 5,000 s` et thèmes non vus. | Toute validation sans double séparation `personne x thème` ni frontière temporelle surestime le décodage multimodal. Note : l'ICA amont de la release `PREPROCESSED` reste une limite documentée. |
| **2b. Prévision prospective EEG -> Corps** | Avec frontière temporelle stricte à `t = 5,000 s`, l'EEG précoce (`REVE-base`) améliore la prévision de la FC future (`[5, 10) s`) au-delà du passé corporel `M0` (`r = 0,6849`). | `M2_resid +0,0089`, `M2_joint +0,0110` (71,4 % améliorés, `q < 1e-6`), `GBDT +0,0063`, `5x5 +0,0120`, `LOCO 5/5 cohortes +0,0107`. | Survit aux 9 artefacts (`+0,0085`), aux 46 électrodes intérieures (`+0,0070`), aux essais propres (`+0,0110`), au contrôle autre spectateur sur même vidéo (`+0,0242`, 79 %), et tombe à zéro sous `NC1` (`-0,0003`) et `NC3` (`-0,0010`). **Aucun gain incrémental détecté pour EDA (`-0,0097`) ni respiration (`+0,0008`).** |
| **3. Modèles & généralisation** | Une régression Ridge 4 phases (0 GPU) atteint `0,5709` (`[0,5565, 0,5849]`) sur le protocole Cold-Start audité dev `3x3`. | Égale `EEGNet` (`0,5722`) et dépasse 5 modèles de fondation EEG (`0,526-0,560`), `PhysioJEPA` (`0,557`), `LoRA` (`0,550`, 21,7x GPU) et le stacking OOF (`-0,0084`). | 70 % de l'optimisme `R1 -> R4` (`0,5887 -> 0,5688`) vient des thèmes déjà vus (`R3 - R4 = +0,0139`). En `R4`, EEG (`-0,0284`) et EDA (`-0,0177`) sont les deux piliers (`ECG_only = 0,5075`). |
| **3b. Dérive & personnalisation** | Transfert `Jour 1 -> Jour 2` (1 semaine d'écart) et calibration chronologique `k`-shot (`k = 0..40`). | `EDA_only` chute de `-0,0392` (`0,560 -> 0,523`) et `Resp_only` de `-0,0299` (`0,501`) ; EEG et multimodal tiennent. | Le réentraînement personnel naïf s'effondre (`-0,081` à `k = 5`, `-0,053` à `k = 40`). `CORAL` est neutre en LOCO (`0,589 -> 0,588`). `EB-MAP` empêche tout effondrement. |
| **4. Cohérence & suivi passif simulé** | Analyse Varimax (314 sujets, 16 276 essais, 53,3 % variance) : 3 axes orthogonaux (`F1` EDA, `F2` EEG-Cardio, `F3` Subjectif-CardioResp). | 79,0 % de sujets cohérents (`rho` médian `+0,111`), 21,0 % discordants. Aucune association significative observée entre `M2 - M0` et la cohérence subjective (`rho = +0,091, p = 0,18`). | Agrégation passive Cold-Start de `K = 1` à `K = 10` clips discrets (`10-100 s` cumulées, contrôlée par blocs disjoints sans remise jusqu'à `K = 5`) : AUC `0,680` (standard), `0,692` (menace aiguë), `0,698` (strict `3/7` vs détendu) ; sensibilité à 90 % de spécificité doublée (`21 % -> 42-53 %`). Ne constitue pas une validation sur 100 s cliniques continues. |

---

## 4. Plan de la présentation enregistrée (10 minutes, aligné sur Ashley + jury)

Le plan ci-dessous fusionne les 5 axes du document `Presentation HABS.pdf` d'Ashley avec les **6 questions réflexives obligatoires** de `Hackathon_Guidelines.pdf`. Langue recommandée pour les slides et l'enregistrement : **anglais** (cohérent avec le jury HABS, les slides d'Ashley et `paper/submission_paper.pdf`).

| Temps | Intervenant suggéré | Slide / section | Contenu clé et visuel à afficher |
| --- | --- | --- | --- |
| **0:00 - 1:00** | Aurélien | 1. Clinical motivation & research question | Besoin clinique non verbal (pédiatrie pré-verbale, réanimation, anesthésie) + question unique sur 314 sujets / 22 974 essais. Mention claire de l'exclusion de l'oculométrie suite à l'avis de désalignement. |
| **1:00 - 2:45** | Aurélien | 2. Fast physiology, ECG morphology & raw HEP | **Figure 1 (`fig1_physiology_cascade.png`)** : cascade 15 s (delta -> FC -> Resp -> EDA), découverte des 96,6 % d'ECG inversés, freinage vagal sinusal pur (PR segment nul, QTc expliqué par `dRR`), HEP brut vs ICA nul, et effet d'évaluation subjective à vidéo fixe vs alpha automatique. |
| **2:45 - 4:45** | Mathias | 3. Leak audit & causal brain-to-heart wall (`t = 5.000 s`) | **Figure 2 (`fig2_causal_brain_heart.png`)** : comment nous avons détecté deux fuites majeures (centrage global, encodeur bidirectionnel + thèmes vus), mur causal strict, gain cardiaque spécifique et idiosyncratique (`+0,0110`, 5/5 cohortes LOCO, survie aux 9 artefacts et `NC1-NC4`), et réfutation honnête sur EDA et respiration. |
| **4:45 - 6:30** | Mathias | 4. Foundation models vs. 4-phase linear, drift & `k`-shot | **Figure 3 (`fig3_generalization_drift.png`)** : Ridge 4 phases (0 GPU) égale ou bat 5 modèles de fondation EEG, `PhysioJEPA` et LoRA. Echelle `R1 -> R4` (70 % d'optimisme dû aux thèmes vus), dérive EDA à 1 semaine (`-0,039`), effondrement du `k`-shot naïf (`-0,081`) vs stabilité `EB-MAP`. |
| **6:30 - 8:00** | Ashley | 5. Coherence, divergence & passive clinical monitoring | **Figure 4 (`fig4_coherence_monitoring.png`)** : 3 axes Varimax, 21 % de discordance subjectif-physiologie, indépendance du couplage cerveau-coeur, et courbe opérationnelle `10 s -> 100 s` (`AUC ~ 0,70`, sensibilité doublée à 90 % de spécificité sans calibration). |
| **8:00 - 10:00** | Tous (ou relais court) | 6. Journey, pitfalls, AI limits & next 4 weeks (obligatoire jury) | Réponses directes aux 6 questions de `Hackathon_Guidelines.pdf` (voir scripts ci-dessous). |

### Réponses prêtes à l'emploi aux 6 questions obligatoires du jury
1. **How we got there:** We started from broad affective decoding and an initial OR/surgery framing. Combining Aurélien's clinical critique, Ashley's 39-paper literature synthesis, and Mathias's leakage audits narrowed the project to one falsifiable question: causal brain-body kinetics, cross-day generalization, and passive monitoring.
2. **What confused us:** Three surprises forced course corrections. First, 96.6% of Lead I ECG traces were inverted, breaking naive QRS/P-T delineation until we added automatic template polarity alignment. Second, Bazett QTc initially looked like a stress signal (`d = -0.088`) until within-person RR control proved it was a pure heart rate artefact (`d = -0.007`). Third, early high cross-modal correlations came from a bidirectional encoder and seen video themes rather than true causal forecasting.
3. **What took longer than expected:** Building strict zero-leakage protocols (`person x theme` nested CV, bit-verified forward-only filtering at `t = 5.000 s`, and 9-artefact / same-video falsification controls) took far more engineering than training deep models, and refactoring 10 studies into 6 clean Python modules required disciplined pruning.
4. **Where AI tools helped vs. where we could not rely on them:** AI coding assistants accelerated boilerplate extraction, unit testing, and figure scripting. However, we could not rely on AI for scientific skepticism: AI-generated code initially introduced global per-person centering and bidirectional windowing that silently leaked future information, and AI tended to bloat the repository across exploratory scripts. Detecting ECG inversion, interpreting null PR/QTc/HEP physiology, and enforcing clinical caution came strictly from human medical and methodological review.
5. **Our biggest learning:** In multimodal Neuro-AI, evaluation rigor matters more than parameter count. Once seen-theme leakage and transductive normalization are removed, a transparent 4-phase Ridge model beats five EEG foundation models and a custom JEPA at zero GPU cost, and honest negative results (null HEP, EDA drift, naive `k`-shot collapse, 21% subjective-autonomic discordance) define what can actually work at the bedside.
6. **What we would do with 4 more weeks:** First, validate the 100-s passive EEG+EDA monitoring curve on a clinical pediatric pain or ICU sedation cohort where autonomic signs shift from orienting bradycardia to sympathetic pain arousal. Second, re-run with repaired synchronized pupillometry to test locus coeruleus coupling. Third, test lightweight real-time streaming of the 4-phase Cold-Start model with online `EB-MAP` adaptation and sensor-dropout fallback.

---

## 5. Guide de vérification minimale pour l'équipe (relecture rapide)

Le dépôt a été réduit de **85 fichiers Python (35 798 lignes)** à **6 modules centraux (1 825 lignes)** plus 1 script de reproduction, afin qu'Aurélien et Ashley puissent relire l'intégralité du code utile en moins de 30 minutes. L'archive exploratoire complète des études 01 à 10 reste intacte dans l'historique Git (commit `501d24b`).

### Carte exacte des fichiers à vérifier
- `src/habs_stress/signals.py` : calibration GSR, correction automatique de polarité ECG (96,6 %) et délinéation P-QRS-T, filtrage causal strict à `t = 5,000 s`.
- `src/habs_stress/protocol.py` : splits disjoints `personne x thème`, CV imbriquée, séparation Cold-Start vs centrage causal en ligne, bootstrap et BH-FDR.
- `src/habs_stress/physiology.py` : définition du stress, tailles d'effet intra-sujet, contrôle RR de l'ECG, contraste à vidéo fixe.
- `src/habs_stress/forecasting.py` : prévision causale `M0` vs `M1` vs `M2`, résidualisation des artefacts, test idiosyncratique sur même vidéo.
- `src/habs_stress/generalization.py` : `CORAL`, personnalisation `EB-MAP`, rotation Varimax et courbes de suivi passif `10 s -> 100 s`.
- `src/habs_stress/figures.py` + `scripts/reproduce.py` : régénération déterministe des tableaux et des 4 figures du papier depuis `results/*.json`.
- `paper/submission_paper.pdf` (anglais) et `paper/submission_paper_fr.pdf` (français) : manuscrit final de 5 pages exactes prêt à déposer.
- `notebooks/results_walkthrough.ipynb` : notebook de synthèse interactif pour le jury.

### Commandes de vérification locale (moins de 15 secondes)
```bash
pip install -r requirements.txt
pip install --no-deps -e .
ruff check src scripts tests process.py
pytest -q
python scripts/reproduce.py
```
