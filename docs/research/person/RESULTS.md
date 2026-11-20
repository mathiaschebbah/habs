# Étude 06 : la physiologie lit-elle la personne ou la vidéo ?

[Protocole figé et bilan](../../../journal/06_person_or_video.md) ·
[agrégats](results) · [figures](figures) · [papier final](../../paper/paper.pdf)

## La question

Les études 04 et 05 montrent que le stress ralentit le cœur, fait monter la peau et rend la
respiration plus superficielle. Mais les mêmes vidéos passent devant beaucoup de gens : chacune des
252 vidéos a été vue par 13 à 234 personnes (médiane 83). Une vidéo qui fait peur à presque tout le
monde change le corps de tout le monde, en partie par ses images et son son, quoi que chacun ressente.
Quand un détecteur dit « stress », lit-il la personne ou la vidéo ? Pour un moniteur de stress, c'est
toute la question.

## Ce qu'on a fait

- Pour chaque note et chaque réponse physiologique d'un essai, on sépare deux parts. La **part
  partagée** est ce que les autres spectateurs de la même vidéo ont montré, centré sur chaque
  personne. La **part propre** est le reste : ce que cette personne a ressenti ou montré en plus
  ou en moins que les autres.
- Dans chaque personne, on régresse la réponse (cœur 1–8 s, peau 4–10 s, respiration 2–6 s,
  delta 0–2 s, alpha 0–10 s) sur les deux parts de la valence ou de l'arousal. On fait ensuite la
  moyenne sur les personnes, avec un intervalle à 95 % par bootstrap des personnes.
- On calcule aussi une AUC « dans la même salle ». Elle porte sur toutes les paires d'un spectateur
  stressé et d'un non stressé devant la même vidéo : 140 vidéos et 265 019 paires. La vidéo est la
  même dans chaque paire, donc cette AUC ne peut venir que de la personne.
- Le protocole a été figé et commité avant de calculer quoi que ce soit, après un audit qui ne
  comptait que les effectifs.

## Ce qu'on a trouvé

| Réponse | Note | Part partagée (vidéo) | Part propre (personne) |
|---|---|---|---|
| Cœur | valence | 0,049 [0,029 ; 0,068] | **0,004 [−0,015 ; 0,021]** |
| Respiration | valence | 0,049 [0,032 ; 0,066] | 0,001 [−0,017 ; 0,018] |
| Peau | arousal | 0,106 [0,083 ; 0,129] | 0,023 [0,003 ; 0,043] |
| Alpha | arousal | −0,062 [−0,081 ; −0,044] | −0,007 [−0,024 ; 0,011] |
| Delta | arousal | 0,010 [−0,008 ; 0,027] | 0,029 [0,010 ; 0,047] |

- **Le cœur et la respiration ne suivent que la vidéo.** Quand quelqu'un trouve une vidéo plus
  désagréable que les autres spectateurs, son cœur ne le montre pas.
- **La peau suit un peu la personne**, mais cinq fois moins que la vidéo, et seulement dans la
  cohorte 1 quand on regarde les cohortes séparément.
- **Le delta précoce est le seul signal qui suit plus la personne que la vidéo.** Sans EOG, on ne
  peut pas exclure un sursaut ou un mouvement.
- **Dans la même salle, la physiologie distingue à peine les spectateurs stressés.** Le modèle figé
  obtient une AUC de 0,517 [0,503 ; 0,531], contre 0,549 [0,537 ; 0,561] d'une vidéo à l'autre chez
  une même personne. En post hoc, le même modèle entraîné uniquement sur la part propre des signaux
  donne 0,505, soit le hasard.
- **Savoir ce que les autres ont ressenti devant la vidéo prédit le stress d'une personne avec une
  AUC de 0,917.**
- **Post hoc, la foule distingue les vidéos stressantes.** La réponse moyenne de la peau chez une
  moitié des gens corrèle à r = 0,26 avec le taux de stress de l'autre moitié, celle du delta aussi à
  0,26, et celle du cœur à −0,19.
- **Exploratoire, sur 25 personnes seulement.** Le cœur ralentit plus quand la personne dit garder
  le contrôle, à l'inverse de ce que prédit l'immobilité de peur.

## Est-ce que c'est bien ?

C'est notre résultat le plus solide et le plus utile. Il est net, il se reproduit dans les trois
groupes de cohortes pour la part partagée, et il est mesuré avec un outil simple que d'autres
peuvent reprendre. Il dit aussi quelque chose de dérangeant pour les études de détection d'émotions,
y compris nos études 03 à 05. Tenir à l'écart les personnes et les vidéos empêche le modèle de
mémoriser une vidéo, mais il apprend quand même ce que les vidéos stressantes font à tout le monde.
Son AUC mesure alors surtout le stimulus.

Limite principale : la part propre d'une note contient aussi le bruit de notation, qui tire tous
les coefficients propres vers zéro. Les coefficients non nuls de la peau et du delta montrent que ce
n'est pas que du bruit, mais on ne peut pas mesurer la fiabilité des notes (seulement 9 paires
personne–vidéo répétées).

## Volet B : la réponse du cerveau à chaque battement cardiaque

Idée tirée du cours de B. Hermann sur les réseaux de la conscience, partagé dans le groupe.
On moyenne l'EEG autour de chaque pic R (de −200 à +600 ms), sur les battements de chaque vidéo,
et on compare les réactions de stress aux autres essais.

- **Pas d'effet établi** : d = 0,039 [−0,002 ; 0,078] sur 287 personnes, aux sites frontaux et
  centraux, entre 250 et 600 ms. Le contrôle par l'amplitude de l'ECG et l'intervalle RR ne change
  rien (0,039 [−0,002 ; 0,081]).
- Le cœur ralentit bien (RR, d = +0,087 [0,042 ; 0,132]) et la forme de l'ECG ne diffère pas, donc ce
  n'est pas un problème de mesure. Le cœur ralentit, mais la réponse corticale à chaque battement ne
  suit pas de façon mesurable.
- On peut exclure un effet plus grand que d ≈ 0,08. Les sites frontaux seuls penchent vers le positif
  (d jusqu'à 0,05 vers 300–500 ms) : c'est une observation post hoc, à tester dans une étude figée à
  l'avance.
