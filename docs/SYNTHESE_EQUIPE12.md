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
> *Comment se déploient les signaux corticaux (EEG) et périphériques (ECG, EDA, respiration) lors d'un stress aigu, comment les contrôles de fuite et la dérive inter-jours contraignent-ils le décodage multimodal, et l'EEG précoce apporte-t-il une information prospective sur la réactivité autonome future au-delà du passé corporel ?*

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

## 3. Synthèse consolidée des résultats vérifiés (314 IDs, 22 974 essais)

Corpus analysé : 314 identifiants d'adultes sains (sur 317 étiquetés dans la release hackathon ; corpus source HABS 383 personnes, 39,6 +/- 12 ans, 271 femmes ; une même personne physique a pu recevoir plusieurs `participant_id` entre cohortes), 22 974 essais alignés, 508 enregistrements, 252 clips de 10 s présentés (sur 256 prévus, 84 thèmes), 5 cohortes, 169 identifiants multi-jours à une semaine, 0 essai marqué outlier.

| Axe | Résultat principal vérifié | Valeur et incertitude | Portée / résultat négatif associé |
| --- | --- | --- | --- |
| **1. Critère & chronologie** | `V <= 4 et A >= 6` (24,0 %, 5 510 essais) : sur les 3 938 avec rang 1 renseigné, 89,2 % choisissent colère, peur, dégoût, tristesse ou culpabilité (97,2 % rangs 1-3). Chronologie descriptive (`dev N = 235`) : EEG delta `<1 s`, FC `1 s`, Resp `2-6 s`, hausse positive tardive EDA `4-10 s`. | Durcir le seuil (`2/8`) accroît l'EDA (`+0,14 -> +0,34`) et le delta (`+0,13 -> +0,30`), et réduit la baisse de FC (`-0,11 -> -0,04`). Sur `N = 52` hors-dev historique, seuls delta (`1 s`) et SC (`4 s`) passent le critère d'onset. | Sur 10 263 segments de 10 s imbriqués dans des fenêtres de 60 s d'enregistrements émotion-vidéo : corrélation globale `r = 0,942` (FC) / `0,745` (RMSSD), mais intra-enregistrement `0,553` (FC) / `0,425` (RMSSD ; `0,274` sur 5 s). Clustering nul (`ARI < 0,01`). |
| **1b. ECG & HEP** | Polarité Lead I alignée par enregistrement (96,6 % des lignes d'essais signalées pour inversion ; médiane pondérée par essai 99,2 % battements retenus). Allongement RR (`+5,1 ms`). | `d(RR) = +0,12 [+0,07, +0,16]`. `d(PR seg) = -0,005`. QT brut `beta = +0,05` (ajusté `+0,03`). QTc Bazett passe de `beta = -0,09` à `-0,007` après contrôle de `dRR`. | Compatible avec une décélération vagale sans variation mesurable du PR ni du QT indépendant de RR sur 10 s. HEP brut `CURATED` (`+0,038 [-0,001, +0,077]`) et ICA (`+0,039 [-0,002, +0,078]`, `N = 287`) couvrent tous deux zéro. |
| **1c. Vidéo fixe & émotions** | À côté du R4 intra-ID (`0,569`), l'appariement distinct inter-spectateurs à vidéo fixe donne `0,525 [0,511, 0,540]` pondéré (`0,519` non pondéré) sur 117 vidéos, et `0,492 [0,456, 0,534]` pondéré (`0,524` non pondéré) sur 37 vidéos vs calmes. | Spectateurs calmes face aux clips aversifs : faible variation non significative de FC (`-0,49 bpm`) et EDA (`+0,01 uS`) ; baisse alpha occipitale observée (`d = -0,33 à -0,41`). | Peur vs Dégoût intra-ID (`0-5 s` vs `5-10 s`, potentiellement clips différents, `N = 116-160`) : Dégoût delta précoce `-0,10` et gamma frontal tardif `-0,11` (tardif moins précoce `-0,18`, prudence EMG) ; Peur alpha tardif `-0,14` et EDA tardive `+0,19`. |
| **2. Audit de fuites** | Le centrage global gonflait LogReg de `+0,014` (`0,571` vs `0,556`). Un `REVE` bidirectionnel 13 s sur 77 % de thèmes vus créait de faux gains croisés. | Fuites corrigées : `+0,059` FC, `+0,042` EDA, `+0,069` Resp s'effondrent sous coupure post-release `t = 5,000 s` et thèmes inédits. | Note amont (`Dataset_Paper.pdf`) : `PREPROCESSED` a subi un FIR phase nulle `0,5-48 Hz` et une ICA Picard continue (rejet auto ICLabel oeil/muscle/ligne/canal, sans retrait systématique ECG). |
| **2b. Prévision exploratoire EEG -> Corps** | Coupure post-release `t = 5,000 s` ; cible `FC[5,10) - FC[-3,5,0)` mesurée sur tous les essais (`M0` 57 variables, `r = 0,685`, RMSE `5,540 bpm` ; `M1` EEG seul `r = 0,20`). | `M2_hc +0,007`, `M2_resid +0,009` (`-0,077 [0,052, 0,103] bpm`), `M2_joint +0,011` (`-0,086 [0,053, 0,120] bpm`, 71,4 % avec `dr > 0`), `GBDT +0,006`, `5x5 +0,012`, `LOCO` interne 5 cohortes (sans masque de thème) `+0,011` (5/5 positifs, 3/5 IC > 0). | Persiste aux 9 artefacts (`+0,009`, `n = 224`), 46 électrodes intérieures (`+0,007`), essais propres (`+0,011`, `n = 207`), intra-vidéo (`+0,007`, `n = 198`) ; s'annule sous `NC1` (`0,000`) et `NC3` (`-0,001`) ; permuter l'EEG entre essais de la même vidéo dégrade le score (`NC2 -0,015`, contraste `+0,024`). **Dégrade l'EDA (`-0,010` résiduel `q = 0,016`, `-0,030` conjoint) ; aucun effet détectable sur respiration (`+0,001`).** |
| **3. Modèles & généralisation** | En Cold-Start `3x3` dev (`N = 208`, 12 693 essais, 66 thèmes), les ROC-AUC restent modestes et dépendent des fenêtres nominales (`[-3,12) s` inclut `[10,12) s` pouvant chevaucher l'écran de notation). | EEG seul : `S-JEPA` `0,526`, `REVE LoRA` `0,550` (`-0,007 [-0,021, +0,007]` vs gelé, 21,7x GPU), Ridge EEG `0,551`, `EEGPT` `0,560 [0,547, 0,573]`, `REVE` gelé (`[0,10) s`) `0,561`, `EEGNet` (`[0,10) s`, ens. 3 graines) `0,572`. Multimodal (`[-3,12) s`) : `PhysioJEPA` `0,557`, Ridge 4 phases `0,569 [0,553, 0,584]`, Ridge+REVE `0,570`. | Le pas masquant les thèmes représente 70 % (via R3 3 plis ID inédits `0,583`) à 79 % (via R2 `0,573`) de la baisse R1-R4 (`0,589 -> 0,569`, volume d'entraînement variable). En `R4`, EEG (`-0,028`) et EDA (`-0,018`) dominent (`ECG_only = 0,508`). |
| **3b. Dérive & adaptation** | Transfert `J1 -> J2` (1 semaine ; pas de référence `J2 -> J2`) et calibration chronologique (`t >= 40`, 118 IDs éligibles, 104 AUC en queue). | `EDA_only` chute de `0,560` (J1, `N = 162`) à `0,523` (J2, `N = 149` ; apparié `-0,039 [-0,073, -0,009]`, `N = 103`), `Resp_only` tombe à `0,501`, aucune baisse détectée pour `EEG_only` (`+0,025 [-0,006, +0,055]`) ni multimodal (`+0,004 [-0,028, +0,037]`). | Le réentraînement personnel Ridge VA reste sous le Cold-Start jusqu'à `k = 40` (`-0,053 [-0,087, -0,021]`). `CORAL` est neutre en LOCO (`0,589 -> 0,588`). `EB-MAP` reste plat (`-0,001`, fort retrait possible). |
| **4. ACP Varimax, cohérence & moyennage hors ligne** | ACP sur corrélations avec rotation Varimax (`314` IDs, `16 276` essais complets, `53,3 %` de variance expliquée) : 3 axes orthogonaux (`F1` EDA, `F2` EEG-CardioResp, `F3` Subjectif-CardioResp). | Cohérence neuro-autonome : médiane `+0,111` (`21,0 % <= 0` ; autonome seul `+0,074`, `28,7 % <= 0`), stabilité à 1 semaine `r = 0,066 [-0,087, +0,211], p = 0,39, N = 169` (compatible avec un bruit d'échantillonnage substantiel). Lien avec gain EEG-coeur non concluant (`rho = +0,09, p = 0,18`). | Exercice hors ligne de moyennage de `K = 10` clips R4 de **classe connue** (blocs disjoints vérifiés jusqu'à `K = 5`) : AUC `0,680` (standard ; sensibilité oracle par ID à 90 % spec `21,0 -> 41,8 %`), `0,692` (menace ; `27,0 -> 45,5 %`), `0,698` (`+0,096`, strict 3/7 vs détendu ; `31,5 -> 52,8 %`). Ne valide pas une alarme temps réel. |

---

## 4. Plan de la présentation enregistrée (10 minutes, aligné sur Ashley + jury)

Le plan ci-dessous fusionne les 5 axes du document `Presentation HABS.pdf` d'Ashley avec les **6 questions réflexives obligatoires** de `Hackathon_Guidelines.pdf`. Langue recommandée pour les slides et l'enregistrement : **anglais** (cohérent avec le jury HABS, les slides d'Ashley et `paper/submission_paper.pdf`).

| Temps | Intervenant suggéré | Slide / section | Contenu clé et visuel à afficher |
| --- | --- | --- | --- |
| **0:00 - 1:00** | Aurélien | 1. Clinical motivation & research question | Besoin clinique non verbal (pédiatrie pré-verbale, réanimation, anesthésie) + question unique sur 314 sujets / 22 974 essais. Mention claire de l'exclusion de l'oculométrie suite à l'avis de désalignement. |
| **1:00 - 2:45** | Aurélien | 2. Fast physiology, ECG morphology & raw HEP | **Figure 1 (`fig1_physiology_cascade.png`)** : chronologie 15 s (`dev N = 235`), alignement de polarité ECG (96,6 % des essais signalés), décélération compatible avec une modulation vagale (PR segment nul, QTc Bazett annulé par `dRR`), HEP brut vs ICA couvrant zéro, et contraste à vidéo fixe (`0,525`) vs baisse alpha occipitale. |
| **2:45 - 4:45** | Mathias | 3. Leak audit & exploratory brain-to-heart cutoff (`t = 5.000 s`) | **Figure 2 (`fig2_causal_brain_heart.png`)** : détection des deux fuites majeures (centrage global, encodeur bidirectionnel + thèmes vus), coupure post-release `t = 5,000 s`, faible gain cardiaque (`+0,009` à `+0,011`, `-0,08` bpm RMSE, 5/5 cohortes LOCO sans masque de thème, survie aux 9 artefacts et `NC1-NC4`, dégradation sur EDA), limite transparente du prétraitement amont `PREPROCESSED`. |
| **4:45 - 6:30** | Mathias | 4. Architecture comparison, drift & `k`-shot | **Figure 3 (`fig3_generalization_drift.png`)** : scores modestes en Cold-Start (`0,526-0,572` en EEG seul, `0,569` Ridge multimodal 4 phases avec fenêtre `[10,12) s` explicitée). Échelle `R1 -> R4` (70-79 % au pas masquant les thèmes), dérive EDA à 1 semaine (`-0,039`), retard du `k`-shot naïf jusqu'à `k = 40` (`-0,053`) vs `EB-MAP` plat. |
| **6:30 - 8:00** | Ashley | 5. Varimax PCA, coherence & known-class averaging | **Figure 4 (`fig4_coherence_monitoring.png`)** : 3 axes ACP Varimax (53,3 % variance), faible cohérence instable à 1 semaine (`r = 0,07, p = 0,39`), et exercice hors ligne de moyennage de `K = 1..10` clips de classe connue (`AUC 0,680-0,698`, seuil oracle par ID pour la sensibilité). |
| **8:00 - 10:00** | Tous (ou relais court) | 6. Journey, pitfalls, AI limits & next 4 weeks (obligatoire jury) | Réponses directes aux 6 questions de `Hackathon_Guidelines.pdf` (voir scripts ci-dessous). |

### Réponses prêtes à l'emploi aux 6 questions obligatoires du jury
1. **How we got there:** We started from broad affective decoding and an initial OR/surgery framing. Combining Aurélien's clinical critique, Ashley's 39-paper literature synthesis, and Mathias's leakage audits narrowed the project to one falsifiable question: acute distress physiology, leakage and cross-day limits, and exploratory brain-heart forecasting.
2. **What confused us:** Three surprises forced course corrections. First, recording-level Lead I ECG polarity was inverted on 96.6% of trial rows, breaking naive QRS/P-T delineation until we added template polarity alignment. Second, Bazett QTc initially looked like a stress signal (`beta = -0.088`) until within-ID RR adjustment showed it tracked heart rate (`beta = -0.007`). Third, early high cross-modal correlations came from a bidirectional encoder and seen video themes rather than prospective forecasting.
3. **What took longer than expected:** Building strict leakage controls (`ID x theme` nested CV, bit-verified forward-only post-release filtering at `t = 5.000 s`, and 9-artefact / permuted same-video falsification controls) took far more engineering than training deep models, and refactoring 10 studies into 6 clean Python modules required disciplined pruning.
4. **Where AI tools helped vs. where we could not rely on them:** AI coding assistants accelerated boilerplate extraction, unit testing, and figure scripting. However, we could not rely on AI for scientific skepticism: AI-generated code initially introduced global per-ID centering and bidirectional windowing that silently leaked future information, and AI tended to bloat the repository across exploratory scripts. Detecting ECG inversion, interpreting null PR/QTc/HEP physiology, auditing upstream preprocessing, and enforcing clinical caution required human medical and methodological review.
5. **Our biggest learning:** In multimodal Neuro-AI, evaluation rigor matters more than parameter count. Once seen-theme shortcuts and transductive normalization are removed, single-trial decoding stays modest (`0.526-0.572` across EEG encoders, `0.569` multimodal 4-phase Ridge), and honest negative results (null HEP, one-week EDA drift, naive personal retraining lag, unstable individual coherence) define what actually constrains bedside transfer.
6. **What we would do with 4 more weeks:** First, rerun the exploratory `EEG -> HR` forecast end-to-end from raw `CURATED` EEG without upstream zero-phase filtering or full-recording ICA, adding explicit ECG-artifact regression and a beat-by-beat RR baseline. Second, run strictly sensor-matched `[0, 10) s` benchmarks with paired-difference tests. Third, evaluate chronological state-change alarms with fixed thresholds on a clinical cohort where autonomic signs shift from orienting bradycardia to sympathetic pain arousal.

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
