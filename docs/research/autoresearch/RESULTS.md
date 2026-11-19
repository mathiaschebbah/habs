# Étude 03 : autoresearch sur le décodage EEG des émotions

[Papier (4 pages)](paper.pdf) · [journal des 72 expériences](../../../autoresearch/results.tsv) ·
[protocole figé](../../../journal/03_autoresearch.md) · [métriques scellées](results/sealed_test.json)

## Question

La physiologie enregistrée pendant une vidéo de 10 s prédit-elle la note de valence et
d'arousal d'un spectateur quand la personne **et** le thème de la vidéo sont absents de
l'entraînement ? Le guide du dataset prévient que les vidéos se répètent d'une personne à
l'autre. Un découpage par personne seule laisse donc un modèle marquer des points en
reconnaissant la vidéo.

## Méthode

- Boucle de [karpathy/autoresearch](https://github.com/karpathy/autoresearch) : un seul
  fichier modifiable (`autoresearch/train.py`), un score fixe (`autoresearch/prepare.py`),
  garder une modification seulement si elle bat le meilleur score d'au moins 0,005.
- La CI prend `train.py` depuis le commit d'expérience et tout le reste depuis `main`. Une
  expérience ne peut donc pas modifier sa propre notation.
- 22 974 essais, 314 IDs, 84 thèmes. Développement : 13 388 essais (259 personnes,
  63 thèmes), évalués par validation croisée 3 × 3 personnes × thèmes. Test scellé :
  55 personnes nouvelles × 21 thèmes nouveaux, évalué une seule fois.
- Score : corrélation de Spearman intra-personne entre prédiction et note, moyennée sur
  valence et arousal. Le hasard vaut 0.

## Résultats

| | Dev | Test scellé (personne et thème nouveaux) |
|---|---:|---:|
| Modèle de départ (ridge, puissance par canal + périphérique) | 0,030 | 0,083 [0,031 ; 0,136] |
| Modèle final (deux ridges moyennées) | 0,105 | **0,139 [0,083 ; 0,196]** |
| Gain apparié | +0,075 | **+0,056 [+0,012 ; +0,101]** |

- Quatre modifications gardées sur 72 essais : cibles centrées par personne (0,030 → 0,085),
  moyenne par six régions (0,090), puissance relative (0,097), moyenne d'une ridge régionale
  et d'une ridge par canal (0,105). Tout ce qui est plus flexible (boosting, noyaux,
  quadratique) a perdu.
- Le score de dev ne surestime pas le test. Le gain recule d'environ un quart entre le dev
  et le test, dans l'intervalle.
- L'EEG porte la valence (ρ 0,108 en dev, comme le modèle complet) et le périphérique porte
  surtout l'arousal (0,081 contre 0,053 pour l'EEG seul).
- Un modèle entraîné sur la physiologie **moyennée par vidéo** tombe au niveau du hasard
  (0,010). Ce qui se transfère à des vidéos nouvelles, c'est la réaction individuelle de
  l'essai, pas une signature commune du stimulus.
- Le gamma seul ne vaut que 0,024. L'artefact musculaire n'explique donc pas le décodage.
  La position dans la session seule vaut 0,009 en dev.
- L'effet reste faible : AUC poolées entre 0,52 et 0,56.

## Limites

Un seul découpage de test (55 personnes, 21 thèmes, intervalles larges d'environ 0,11).
Les features d'extraction sont figées. Le modèle final normalise ses deux sorties sur le lot
prédit, ce qui suppose un lot d'essais non labellisés de la nouvelle personne. Les variantes
ont été écrites par lots parallèles, pas une à une comme dans la boucle d'origine.
