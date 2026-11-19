# Étude 04 : peut-on détecter le stress à partir de la physiologie ?

[Papier (5 pages)](paper.pdf) · [les 43 expériences](experiments.tsv) ·
[protocole figé](../../../journal/04_stress_detection.md) ·
[test scellé](results/sealed_test.json) · [corrélats](results/correlates.json)

## La question

Pendant une vidéo de 10 s, l'EEG, le cœur, la peau et la respiration permettent-ils de dire si
la personne vit une réaction de stress ? Et qu'est-ce qui change dans le corps et le cerveau ?

Le dataset n'a aucune mesure de stress (ni questionnaire, ni cortisol, ni tâche stressante).
On a donc défini le stress à partir des notes de chaque essai : vidéo jugée **négative**
(valence ≤ 4) et **activante** (arousal ≥ 6), comme Giannakakis et al. (2015). C'est une réaction
aiguë auto-déclarée chez des volontaires sains, pas un stress clinique.

## Ce qu'on a fait

- Mêmes essais que l'étude 03 (22 974), nouveau découpage personnes × thèmes, donc un nouveau
  test scellé (60 personnes nouvelles × 18 thèmes nouveaux).
- Nouvelles features sympathiques, sur l'idée d'Aurélien et la piste des mentors (VFC sur des
  fenêtres de moins de 30 s) : VFC sur 10 s (RR moyen, RMSSD, SDNN, variation de fréquence
  cardiaque), réponse de conductance 1 à 6 s après le début, respiration (fréquence, profondeur,
  irrégularité, pauses).
- Score : AUC calculée à l'intérieur de chaque personne (hasard = 0,5), plus trois contrastes
  de contrôle fixés à l'avance : stress contre excitation (même arousal), stress contre négatif
  calme (même valence), stress contre relaxé.
- 43 variantes de modèles, dont 11 familles scikit-learn (PLS, noyau RBF, SVR, MLP, Huber,
  SVM, forêt aléatoire, LDA, extra-trees, gradient boosting, k-NN).

## Résultats

| Test scellé (AUC intra-personne) | Personne et thème nouveaux | Personne nouvelle, thème vu | Personne vue, thème nouveau |
|---|---:|---:|---:|
| Modèle de départ (logistique) | 0,634 [0,570 ; 0,699] | 0,588 | 0,555 |
| Modèle final (ridge arousal − valence) | 0,605 [0,544 ; 0,668] | 0,592 | 0,561 |
| EEG seul | 0,603 | 0,586 | 0,560 |
| Corps seul | 0,541 | 0,545 | 0,545 |

- **On détecte le stress au-dessus du hasard**, chez des personnes et sur des vidéos nouvelles,
  mais faiblement (AUC 0,56 à 0,60).
- **La recherche de modèle n'a rien apporté au test.** +0,048 en dev, rien au test scellé. Une
  régression logistique simple fait aussi bien. Sans test scellé, on aurait annoncé un progrès
  qui n'existe pas. Aucun modèle scikit-learn plus flexible n'a battu la ridge.
- **Stress contre excitation : peu fiable** (0,45 à 0,59 selon le test). Stress contre relaxé :
  solide (0,61 à 0,63). Le corps seul capte surtout l'activation. Ce qui distingue le stress de
  l'excitation passe par l'EEG, et c'est faible.

## Ce qui change physiologiquement (dev, taille d'effet intra-personne)

- **Le cœur ralentit** (fréquence cardiaque −0,12, baisse par rapport aux 5 s d'avant −0,11).
  C'est la décélération cardiaque connue devant des images désagréables, pas une accélération.
  Elle tient même contre l'excitation (−0,17).
- **La peau réagit plus** (montée de conductance +0,20, pente +0,19).
- **La respiration est plus superficielle** (−0,12).
- **L'alpha relatif baisse sur tout le scalp** (−0,20 à −0,25), signe d'activation corticale ; le
  delta frontal monte (+0,21), en partie peut-être à cause des mouvements oculaires (pas d'EOG).
- La VFC sur 10 s colle à l'ancienne fréquence cardiaque (r = 0,94) et couvre 93 % des essais,
  contre 65 % auparavant.

## Limites

Label auto-déclaré sur des vidéos de 10 s. Seulement 35 personnes utilisables dans le test
principal (et 10 à 20 pour les contrastes de contrôle). Delta frontal peut-être oculaire.
Features figées avant la recherche.
