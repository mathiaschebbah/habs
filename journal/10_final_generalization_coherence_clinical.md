# Étude 10 (Expérience finale) — Généralisation réaliste, personnalisation, stabilité longitudinale, cohérence autonome–subjective et monitoring clinique passif sur 314 personnes

**Statut :** Exécuté, vérifié et complété (2026-11-25 ; protocole pré-enregistré avant calcul au commit `62bc955`).  
**Base bibliographique :** Revue de littérature complète de l'Équipe 12 (Ashley Russo, 39 publications couvrant réponses autonomes rapides, décalage EEG/SNA, généralisation inter-sujets et adaptation de domaine, cohérence subjective–physiologique et monitoring clinique non verbal) + acquis vérifiés des Études 04 à 09.  
**Objectif :** Clôturer les six angles empiriques ouverts par la revue de littérature sur les 314 participants / 22 974 essais de HABS, sans nouvelle dépense GPU (en exploitant les tables causales et représentations `REVE-base` déjà extraites et vérifiées), afin d'étayer l'article scientifique final complet de l'Équipe 12 (`docs/paper/final_paper_fr.pdf`).

---

## 1. Hypothèses pré-enregistrées (H10.1 à H10.6)

Toutes les analyses prédictives respectent les garde-fous établis en Étude 09 : aucun centrage transductif (`person_center_matrix` interdit), prétraitement strictement isolé par pli, mur causal à `t = 5,000 s` pour la prévision, et `test_new_both` du sel Étude 08 exclu de toute sélection.

### H10.1 — Échelle de généralisation (R1 → R4) et ablation de modalités
*(Réf. : Alshamrani 2026 ; Hinduja et al. ACII 2023 ; Coelli et al. 2026 ; Li et al. 2026)*
- **Protocole :** Évaluer le même détecteur multimodal (`S07_enriched` Ridge V&A et LogReg) et EEG (`REVE-base` gelé) dans `dev` (`habs-ssl-2026-11-24-b`, 12 693 essais, 228 sujets, 66 thèmes) sur 4 régimes strictement ordonnés :
  - **R1 (Split aléatoire 5-fold naïf)** : fuite conjointe sujet + vidéo/thème.
  - **R2 (Intra-sujet, thèmes inédits)** : bloqué par thèmes uniquement (sujets vus, thèmes nouveaux).
  - **R3 (Inter-sujets, thèmes vus — LOSO naïf)** : bloqué par sujets uniquement (sujets nouveaux, thèmes vus).
  - **R4 (Double inédit Cold-Start, $3 \times 3$ sujets × thèmes)** : standard strict sans historique.
- **Ablation de modalités sur R4 :** `Complet (All)`, `−ECG`, `−EDA`, `−Resp`, `−EEG`, et capteurs isolés (`ECG`, `EDA`, `Resp`, `EEG manuel`, `REVE-base`).
- **Hypothèse :** R1 et R3 surestiment systématiquement la généralisation réelle R4 ; l'ablation multimodale révèle la contribution propre de chaque capteur (en particulier l'ECG et l'EEG) et vérifie si le multimodal bat chaque capteur isolé.

### H10.2 — Stabilité longitudinale inter-jours (`J1 → J2`) et habituation intra-session
*(Réf. : Sakimura et al. ACII 2025 ; Chen et al. 2014)*
- **Protocole :**
  1. Sur les participants enregistrés sur deux jours distincts (`recorded_day` 1 et 2, $N \ge 120$ dans `dev` et $N \ge 180$ sur HABS), calculer séparément à J1 et J2 la pente individuelle de réactivité au stress ($d$ intra-sujet) et à l'arousal ($\rho$ intra-sujet) pour ECG (FC/RR), EDA (SC et SCR), respiration, EEG ($\alpha$ occipital, $\delta$ centro-pariétal, $\gamma$ frontal) et oculométrie, puis mesurer la répétabilité inter-jours $r(J_1, J_2)$ (IC 95 % bootstrap 2 000 tirages).
  2. Mesurer le transfert d'un décodeur entraîné sur J1 et testé sur J2 (mêmes sujets sur thèmes inédits vs sujets inédits sur thèmes inédits), capteur par capteur.
  3. Tester l'habituation intra-session (terciles chronologiques d'essais T1 vs T2 vs T3) de la décélération cardiaque vagale, de la réponse électrodermale sympathique et de la suppression alpha occipitale.
- **Hypothèse :** Si le constat de Sakimura et al. (2025) s'étend à HABS, la réactivité cardiaque et EEG présentera une répétabilité $J_1 \to J_2$ supérieure à la conductance cutanée, et le transfert inter-jours chutera davantage pour l'EDA que pour l'ECG.

### H10.3 — Adaptation de domaine non supervisée (CORAL), personnalisation supervisée peu d'essais ($k = 0 \dots 40$) et transfert inter-cohortes (LOCO)
*(Réf. : Mihirette et al. 2025 ; Khan & Ishikawa 2026 ; Engemann et al. 2018)*
- **Protocole :**
  1. **Courbe $k$-shot strictement chronologique dans `dev`** : pour chaque participant test en CV externe $3 \times 3$ possédant $\ge 55$ essais, ordonner ses essais chronologiquement et fixer une queue d'évaluation commune stricte (`essais [40:]`, identique pour tout $k$) afin de garantir une comparaison appariée exacte pour $k \in \{0, 5, 10, 15, 20, 30, 40\}$ essais passés :
     - `Cold-Start` ($k = 0$) ;
     - `Centrage non supervisé causal (B1)` sur $k$ essais non labellisés ;
     - `CORAL non supervisé` (alignement de moyenne et covariance régularisée sur $k$ essais non labellisés, Mihirette et al. 2025) ;
     - `Personnalisation supervisée naïve` (modèle/calibration locale sur $k$ essais labellisés) ;
     - `Personnalisation supervisée régularisée (shrinkage bayésien empirique)` combinant le modèle populationnel et la mise à jour personnelle sur $k$ essais labellisés.
  2. **Calibration inter-jours complète (`J1 → J2`)** : zéro calibration vs B1/CORAL non labellisés sur J1 vs calibration supervisée sur J1, évalués sur J2 (thèmes inédits).
  3. **Généralisation inter-cohortes (`Leave-One-Cohort-Out`, LOCO-CV sur les 5 cohortes HABS)** : évaluer le transfert hors-cohorte sans vs avec **CORAL non supervisé** (alignement de covariance de la cohorte cible sans aucune étiquette) pour la détection de stress (`S07_enriched`, `REVE_frozen`) et pour le gain prédictif cardiaque `M2 − M0` de l'Étude 09.
- **Hypothèse :** Tester directement l'affirmation de Khan & Ishikawa (2026) selon laquelle une personnalisation supervisée naïve avec $k < 20$ essais labellisés est contre-productive, vérifier si le *shrinkage* bayésien empirique élimine cette dégradation, et mesurer le gain réel de CORAL en transfert inter-cohortes (LOCO).

### H10.4 — Cohérence et divergence autonome–subjective et lien avec l'anticipation cerveau–cœur
*(Réf. : Brown et al. 2019 ; Cuve et al. 2023 ; Nguyen et al. 2026 ; Saviola et al. 2020)*
- **Protocole :**
  1. **Espace multidimensionnel autonome–subjectif (Cuve et al. 2023)** : ACP / SVD sur les 9 variables intra-sujets standardisées `[Valence, Arousal, ΔFC, ΔSC, ΔSCR, ΔResp, Δα_occ, Δδ_cp, Δγ_fr]` (plus version avec `ΔPupille`), en comparant l'espace total et la part idiosyncrasique résidualisée sur la vidéo.
  2. **Distribution individuelle de cohérence (Brown et al. 2019)** : pour chaque participant, calculer la cohérence intra-sujet $\rho_i$ entre détresse subjective ($A - V$) et chaque signal physiologique (`−ΔFC`, `+ΔSC`, `−Δα`, indice multimodal), sa variation selon l'intensité émotionnelle (essais extrêmes vs modérés), sa stabilité-trait inter-jours $r(\text{coh}_{J1}, \text{coh}_{J2})$, et sa corrélation avec le gain individuel d'anticipation EEG→Cœur $\Delta r_i = r_i(\mathrm{M2}) - r_i(\mathrm{M0})$ issu de l'Étude 09.
- **Hypothèse :** Confirmer à grande échelle ($N = 314$) la structure orthogonale de Cuve et al. (2023) (un axe partagé subjectif–autonome distinct d'un axe de covariance purement autonome) et quantifier la fraction exacte d'individus « cohérents » vs « discordants » (Brown et al. 2019).

### H10.5 — Simulation opérationnelle de monitoring clinique passif non verbal (agrégation multi-épisodes)
*(Réf. : Danel et al. 2025 ; Roué et al. 2025 ; Adama & Bogdan 2025 ; Paniagua-Gómez 2025 ; Engemann et al. 2018)*
- **Protocole :** Sur les prédictions hors-pli strictement Cold-Start dans `dev` ($3 \times 3$ sujets × thèmes), simuler l'agrégation de $K \in \{1, 2, 3, 5, 8, 10\}$ épisodes brefs de 10 s (soit 10 s à 100 s d'observation passive) par tirages appariés intra-sujet (200 réplicats déterministes par sujet) pour trois contrastes cliniquement gradués :
  1. `Stress standard` ($V \le 4, A \ge 6$) vs `Non-stress` ;
  2. `Stress aigu franc` ($V \le 3, A \ge 7$) vs `Calme / Relaxé` ($V \ge 6, A \le 4$) ;
  3. `Menace aiguë` (peur/colère/dégoût) vs `Calme / Neutre`.
- Rapporter l'AUC intra-sujet, la sensibilité à 80 % et 90 % de spécificité, et comparer `S07_enriched` (multimodal), `REVE-base` gelé (EEG seul), `ECG seul` et `EDA seul`.

### H10.6 — Dynamique intra-clip précoce vs tardive par émotion déclarée
*(Réf. : Si et al. 2026 ; Alasfour & AlSabah 2025 ; Stegmann et al. 2024 ; Qin et al. 2023)*
- **Protocole :** Sur les 209 participants des cohortes 2 à 5 disposant du premier choix d'émotion brute (`self_reported_emotion_1`, 17 142 essais), quantifier séparément la phase précoce `[0 ; 5) s` et la phase tardive `[5 ; 10) s` (et la transition `tardif − précoce`) sur les 5 bandes EEG ($\delta, \theta, \alpha, \beta, \gamma$) et les 3 capteurs périphériques (`ΔFC`, `ΔSC`, `ΔResp`) pour peur, dégoût, colère, tristesse, joie et relaxation (contre neutre, IC 95 % bootstrap par sujet).

---

## 2. Discipline épistémique

Tout résultat (positif, nuancé ou négatif) sera rapporté intégralement avec IC 95 %, tailles d'effet appariées et effectifs exacts, sans retuning sur un split exposé ni extrapolation clinique au-delà des limites d'un paradigme vidéo passif chez l'adulte sain.

---

## 3. Journal d'exécution et verdicts vérifiés (2026-11-25)

- **Protocole pré-enregistré :** commit `62bc955` (strictement antérieur au code et à l'exécution).
- **Code & tests unitaires :** `decoding/study10_final.py`, `tests/test_study10_final.py` (commit `51394bc`, lint `ruff` vert, 4/4 tests locaux passés).
- **Run CPU reproductible :** GitHub Actions `decoding.yml` run `45604993423` (`study=study10_final`, `split=all`, `2 145,67 s` CPU, **0,00 H100-h GPU**), agrégat sauvé dans `docs/research/study10/results/study10_metrics.json`.
- **Verdicts H10.1–H10.6 :**
  - **H10.1 (Confirmée avec nuance organe) :** `S07` `R1=0,5887 → R3=0,5827 → R2=0,5729 → R4=0,5688` (70 % de l'inflation vient des thèmes vus ; `+0,0408` d'inflation poolée biométrique sur `REVE` LogReg). En R4, `−EEG` (`−0,0284*`) et `−EDA` (`−0,0177*`) sont les deux chutes significatives ; `ECG_only = 0,5075`, `All_plus_REVE = 0,5699`.
  - **H10.2 (Confirmée) :** stabilité tonique `J1→J2` la plus faible pour l'EDA (`0,420` vs `0,569–0,724`) ; décodeur `EDA_only` chute significativement entre J1 et J2 (`0,5597 → 0,5228`, `−0,0392 [−0,0728 ; −0,0085]`), comme `Resp_only` (`−0,0299*`), tandis qu'EEG (`0,543 → 0,552`, REVE `0,540 → 0,559`) et multimodal (`0,572 → 0,559`) tiennent. Habituation intra-session sur `SCR` (`−0,0322*`), absence sur `RR` (`+5,66 ms*`), accentuation alpha (`−0,0710*`).
  - **H10.3 (Confirmée sur la chute naïve ; réplication LOCO 5/5 sur ECG `M2−M0`) :** `Naive` s'effondre de `−0,081` à `k=5` jusqu'à `−0,053` à `k=40` (et `−0,040*` / `−0,069*` en `J1→J2`) ; `EB-MAP` supprime la chute (`0,564` S07, `+0,010` à `k=15` REVE) ; `LOCO` stress = `0,5893` (CORAL neutre) ; `LOCO` prévision cardiaque `M2 − M0 = +0,0107 [+0,0071 ; +0,0145]` (68,8 % améliorés, **5/5 cohortes positives**).
  - **H10.4 (Répliquée sur $N = 314$, $n = 16\,276$) :** 3 facteurs Varimax orthogonaux (`F1 EDA 20,9 %`, `F2 Neuro-Auto 18,9 %` orthogonal au subjectif, `F3 Subj.-Cardio 13,6 %`) ; composite neuro-autonome `méd. +0,111`, `79,0 % > 0`, `21,0 % < 0`, effet d'intensité `+0,1008*` ; pont `M2−M0` indépendant du trait ($\rho = +0,091$, $p = 0,18$).
  - **H10.5 (Confirmée avec prudence clinique) :** agrégation `K=1→10` (`10 s → 100 s`) en Cold-Start R4 porte l'AUC à `0,680` (std), `0,692` (menace) et `0,698` (3/7 vs relaxé, `Sens@Spec90` `31 % → 53 %`, `Sens@Spec80` `59 %`), confirmé sans remise (`K=3, 5`).
  - **H10.6 (Confirmée et nuancée) :** dégoût domine `δ CP` précoce (`−0,098*`) et `γ` frontal tardif (`L−E −0,178*`, prudence micro-EMG) ; peur domine `α` soutenu (`−0,137*`), bradycardie tardive (`−0,104*`) et explosion `EDA` tardive (`+0,193*`, `L−E +0,303*`).
- **Livrables finaux :** `docs/research/study10/RESULTS.md`, 4 figures vérifiées dans `docs/research/study10/figures/`, et article final consolidé `docs/paper/final_paper_fr.pdf`.
