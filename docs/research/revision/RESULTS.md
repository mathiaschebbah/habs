# Étude 07 : révision complète — critère de stress, émotions brutes, morphologie ECG P-QRS-T, EEG enrichi, HEP brut sans ICA, vidéo fixe et architectures séquentielles

[Protocole figé et provenance](../../../journal/07_comprehensive_revision.md) ·
[agrégats JSON](results) · [figures EN/FR](figures) ·
[papier FR](../../paper/paper_fr.pdf) · [papier EN](../../paper/paper.pdf) ·
[synthèse 2 pages FR](../../paper/synthese_fr.pdf)

## Contexte et recadrage

Cette étude reprend de fond en comble les études 04 à 06 à partir d'une revue médico-scientifique en six points. Le cadrage est généralisé au-delà du bloc opératoire : il s'agit de (1) caractériser neurophysiologiquement une réaction de stress aigu (ce qui varie, dans quel ordre, avec quelle spatialisation et sous le contrôle de quelle branche autonome) et (2) auditer rigoureusement les dispositifs de détection non verbale --- du neurofeedback aux **populations non communicantes** (douleur aiguë en pédiatrie pré-verbale, *locked-in syndrome*, coma, anesthésie générale).

## Ce qui a été extrait et calculé

1. **Extraction enrichie (`decoding/rich_extract.py`, job Azure `20261123T004720-2ea585`)** : 650 colonnes sur les 22 974 essais (314 sujets), organisées en **4 phases** (Référence `[-3, 0)`, Précoce `[0, 5)`, Tardif `[5, 10)`, Post `[10, 12)` s) :
   - Vérifications matériel/données : **1 seule dérivation ECG bipolaire sous-claviculaire** (comparaison multi-dérivations impossible) ; `flagged_outliers == 0` sur 100 % des 22 974 essais audités ; **polarité ECG inversée sur 96,6 % des essais**, corrigée automatiquement par gabarit P/T avant délinéation battement par battement (99,2 % de battements QRS propres).
   - Morphologie ECG P-QRS-T (HR, RR, RMSSD, pic PR, PR clinique, segment PR pur, QRS, QT, QTc Bazett/Fridericia, pic RT, ratio T/R).
   - Homogénéité spatiale EEG (écart-type, IQR, entropie spatiale normalisée, écart inter-régional), gradients/asymétries (AP, FAA, asymétrie temporale), volatilité temporelle intra-phase (SD 1 s, pente, saut max) et proxy oculaire pré-ICA sur Fp1/Fp2 (`CURATED`).
   - **HEP brut `CURATED` (CAR 62 voies, sans ICA)** calé sur exactement les mêmes pics R que `PREPROCESSED`.
2. **Huit volets analytiques (`decoding/revision.py`, commit `b909c69`)** : `criterion`, `criterion_full`, `emotion_models`, `ecg_morphology`, `hep_raw`, `fixed_video`, `sensitivity`, `sequence_models`.

## Résultats principaux

| Question | Résultat principal | Chiffre clé |
|---|---|---|
| **Q1 — Critère $V/A$** | Pureté validée sur `self_reported_emotion_1..3` ; durcir purifie, double peau et $\delta$, mais atténue la bradycardie à arousal extrême | 89,2 % nég. rang 1 (97,2 % rangs 1–3) ; incoh. 5,3 % ; opt. $3/7$ (AUC 0,549) ; peau $+0{,}14 \to +0{,}34$ |
| **Q1 — Émotions n°1 & non sup.** | Peur (FC+peau+resp.+$\alpha$), dégoût (FC+$\delta$, sans peau), tristesse plate ; aucun sous-groupe caché ; clustering non supervisé nul | Peur SC $+0{,}25$, FC $-0{,}20$ ; multiclasse 22,3 % vs 20 % ; $\mathrm{ARI} < 0{,}009$ |
| **Q2 — ECG P-QRS-T** | RR s'allonge mais **PR (nœud AV) est strictement nul** et QTc Bazett s'annule totalement après contrôle $\Delta\mathrm{RR}$ → freinage vagal sinusal | RR $+5{,}1$ ms ($d = +0{,}116$) ; PR $-0{,}05$ ms ; QTc ctrl $-0{,}007$ |
| **Q2 — EEG enrichi** | Forte volatilité temporelle précoce ; réorganisation spatiale biphasique (homogénéisation précoce puis focalisation post) | Saut $\delta$ : $+0{,}151$ ; SD temp. $+0{,}09$–$+0{,}10$ ; entropie $\alpha$ post : $-0{,}076$ |
| **Q3 — HEP brut vs ICA** | L'ICA ne détruit aucun signal HEP caché ; `CURATED` brut et `PREPROCESSED` sont superposables | Brut $+0{,}038$ vs ICA $+0{,}039$ ; diff. retirée $+0{,}012$ ; vs relaxé $+0{,}076^*$ |
| **Q4 — Vidéo fixe & 2×2** | L'ancien contraste diluait 93 % d'essais limites. Contre les calmes (« oklm »), la séparation saute et les calmes ne décélèrent pas (seule $\alpha$ est automatique) | AUC vidéo fixe $0{,}525 \to \mathbf{0{,}564}$ ; apparié FC $-2{,}97$ bpm ($d = -0{,}41^*$), $\delta +0{,}45^*$ |
| **Q5 — Topographie sensibilité** | Essai : $\beta/\gamma$ frontaux + $\delta$ centro-pariétal (survit à 93 % au proxy oculaire). Trait : FC et $\gamma$ de repos élevés | $\gamma$ front. $-0{,}039^*$ ; trait FC $+0{,}140^*$, $\gamma +0{,}165^*$ |
| **Q6 — 4 phases & séquences** | 4 phases + EEG riche + Ridge V&A / conv. 1D surclassent les bins 1 s sur le découpage inédit ; PR et MLP/GBDT n'apportent rien sur thèmes inédits | Test P&T $0{,}526 \to \mathbf{0{,}589}$ (lin.) / **0,586** (Ridge : dév. 0,576, nouv. pers. 0,607, vs relaxé 0,656) |

## Benchmark des 12 modèles (`habs-revision-2026-11-23`)

| Configuration | $p$ | Dév. WP | Dév. SV | Test P&T | Nouv. Pers. | Nouv. Th. | vs Relaxé (P) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_1s_logreg` | 340 | 0,546 | 0,519 | 0,526 | 0,600 | 0,527 | 0,632 |
| `phase4_logreg` | 114 | 0,560 | 0,524 | 0,570 | 0,599 | 0,546 | 0,632 |
| `phase4_deltas_logreg` | 190 | 0,561 | 0,520 | 0,565 | 0,605 | 0,548 | 0,650 |
| `ablation_plus_ecg_pqrst` | 243 | 0,559 | 0,523 | 0,562 | 0,608 | 0,540 | 0,649 |
| `ablation_plus_eeg_spatial` | 365 | 0,562 | 0,521 | 0,582 | 0,600 | 0,555 | 0,640 |
| `ablation_plus_eeg_tvar` | 220 | 0,567 | 0,539 | 0,580 | 0,599 | 0,547 | 0,640 |
| `enriched_full_logreg` | 448 | 0,565 | 0,533 | **0,589** | 0,602 | 0,552 | 0,643 |
| `enriched_full_ridge_va` | 448 | **0,576** | 0,527 | 0,586 | 0,607 | **0,558** | **0,656** |
| `enriched_full_gbdt` | 448 | 0,562 | 0,514 | 0,580 | **0,608** | 0,532 | 0,644 |
| `sequence_multiscale_conv_logreg` | 620 | 0,569 | **0,541** | 0,561 | 0,607 | 0,540 | 0,639 |
| `sequence_plus_enriched_logreg` | 1 068 | 0,573 | 0,539 | 0,573 | 0,607 | 0,549 | 0,654 |
| `sequence_plus_enriched_mlp` | 1 068 | 0,552 | 0,527 | 0,570 | 0,608 | 0,531 | 0,648 |

## Régénérer les figures

```bash
.venv-agent/bin/python docs/research/revision/make_figures.py
.venv-agent/bin/python docs/research/revision/make_figures.py fr
```
