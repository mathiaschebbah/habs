# Étude 03 : prédire les notes d'une personne nouvelle devant une vidéo nouvelle

[Papier (4 pages)](paper.pdf) · [les 72 expériences](experiments.tsv) ·
[protocole](../../../journal/03_affect_decoding.md) · [chiffres du test scellé](results/sealed_test.json)

## La question

On voulait savoir si l'EEG, le cœur, la peau et la respiration enregistrés pendant une vidéo
de 10 s permettent de prédire la note de valence et d'arousal du spectateur, quand **ni la
personne ni le thème de la vidéo** n'ont été vus à l'entraînement. Le guide du dataset prévient
que les mêmes vidéos reviennent chez beaucoup de participants. Si on découpe seulement par
personne, un modèle peut avoir l'air bon juste en reconnaissant la vidéo.

## Ce qu'on a fait

- 22 974 essais, 314 participants, 84 thèmes. Développement : 13 388 essais (259 personnes,
  63 thèmes), évalués en validation croisée 3 × 3 personnes × thèmes. Test scellé :
  55 personnes nouvelles × 21 thèmes nouveaux, ouvert une seule fois à la fin.
- Score : corrélation de Spearman entre note prédite et note donnée, calculée à l'intérieur de
  chaque personne, moyennée sur valence et arousal. Le hasard vaut 0.
- On a amélioré le modèle pas à pas : 72 variantes, toutes notées par le même code figé
  (la CI prend le fichier du modèle dans le commit et le reste dans `main`). On ne gardait un
  changement que s'il battait le meilleur score d'au moins 0,005.

## Résultats

| | Dev | Test scellé (personne et thème nouveaux) |
|---|---:|---:|
| Modèle de départ | 0,030 | 0,083 [0,031 ; 0,136] |
| Modèle final | 0,105 | **0,139 [0,083 ; 0,196]** |
| Gain apparié | +0,075 | **+0,056 [+0,012 ; +0,101]** |

- Quatre changements gardés : centrer les notes par personne (0,030 → 0,085), moyenner l'EEG
  sur six régions (0,090), ajouter la puissance relative (0,097), moyenner une ridge régionale et
  une ridge par canal (0,105). Les modèles plus flexibles (boosting, noyaux, termes
  quadratiques) ont tous perdu.
- Le score de dev ne surestime pas le test. Le gain recule d'environ un quart entre les deux,
  sans sortir de l'intervalle.
- L'EEG porte la valence, le périphérique surtout l'arousal.
- Un modèle entraîné sur la physiologie moyennée par vidéo tombe au hasard (0,010). Ce qui se
  transfère, c'est la réaction propre à chaque personne.
- Le gamma seul ne vaut que 0,024 : les muscles du visage n'expliquent pas le décodage.
- L'effet reste faible : AUC poolées entre 0,52 et 0,56.

## Limites

Un seul découpage de test (intervalles larges d'environ 0,11). Features figées avant la
recherche. Le modèle final remet ses sorties à l'échelle sur le lot prédit, ce qui suppose
d'avoir quelques essais non notés de la nouvelle personne.
