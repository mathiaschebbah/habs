# Revue de littérature pour la prédiction de l'excitation individuelle

## Conclusion de la recherche

La question générale n'est pas nouvelle. La contribution défendable serait d'évaluer, dans HABS, si la physiologie prédit des variations d'arousal déclaré après une calibration explicite de la personne, au-delà d'une référence qui connaît la vidéo et le contexte de sélection. La séparation développement/test doit porter sur les personnes, et le test se déroule sur leur seconde session. Une correction constante de la prédiction personnelle ne doit pas pouvoir améliorer le critère principal.

La littérature ne justifie ni l'affirmation que les capteurs lisent directement le ressenti, ni celle que les autoévaluations sont essentiellement du bruit. Elle appelle à distinguer les niveaux d'analyse, la provenance des labels et la généralisation effectivement testée.

## Méthode et couverture

Recherche arrêtée au 5 novembre 2026. [Protocole de recherche](SEARCH_PROTOCOL.md), [requêtes exactes](search_log.json), [registre de sélection](screening.csv), [matrice d'extraction et de lecture critique](evidence.csv), [bibliographie importable](references.bib).

Neuf requêtes Exa ont demandé 101 résultats ; quatre recherches booléennes Europe PMC ont renvoyé 428 notices, toutes récupérées par l'API. La recherche Europe PMC n'est pas une interrogation indépendante de PubMed : ces contenus se recouvrent largement. Les recherches Exa fournissent notamment des actes de conférence et des prépublications peu couverts par Europe PMC.

| Étape | Nombre |
|---|---:|
| Notices récupérées, avant déduplication | 529 |
| Doublons DOI, URL ou titre normalisé fusionnés | 39 |
| Notices après cette première fusion | 490 |
| Autres notices du même travail repérées | 7 |
| Notices après les deux étapes de rapprochement | 483 |
| Références retenues et annotées | 54 |
| Références exclues du périmètre de synthèse | 429 |

Les inclusions portent sur les comparaisons sujet/stimulus, la cohérence physiologie-expérience au niveau individuel, la mesure des styles de notation, les méthodes de validation et quelques jeux de données ou recommandations directement utiles. Les travaux sur les « response styles » entendus comme rumination ou coping ne sont pas des travaux sur l'utilisation d'une échelle et ont été distingués. Les architectures de classification sans comparaison pertinente ne sont pas systématiquement extraites.

Les 54 références n'ont pas toutes été lues intégralement : 30 ont fait l'objet d'une lecture de sections accessibles et 24 d'une extraction au niveau du résumé ou de la notice. Le niveau exact est indiqué par référence. Certaines récupérations de texte sont partielles. Une conclusion négative sur l'absence d'un contrôle ne doit donc pas être déduite du silence d'un résumé.

Ce travail est une recherche bibliographique systématisée dans un périmètre documenté, suivie d'une synthèse critique. Ce n'est pas une revue exhaustive à double sélection indépendante. Les limites comprennent l'absence d'interrogation directe de Scopus, PsycINFO, Web of Science et de l'index intégral IEEE Xplore, le classement opaque d'Exa, la langue et l'accessibilité des textes. Les dates correspondent autant que possible à la première mise en ligne, parfois antérieure à l'année du fascicule. PRISMA-S guide le compte rendu, sans constituer un label de qualité de cette synthèse [R068](https://pmc.ncbi.nlm.nih.gov/articles/PMC7839230/).

## 1. Le partage des stimuli change ce que mesure un score

Hu et ses collègues étudiaient déjà en 2021 des partitions tenant compte des vidéos sur DEAP et plusieurs jeux SEED [R003](https://bcmi.sjtu.edu.cn/home/blu/papers/2021/2021-18.pdf). Moreno-Alcayde et ses collègues distinguent les scénarios sujets, vidéos et temps ; les performances changent fortement avec le découpage [R095](https://doi.org/10.1007/s13534-023-00316-5).

Deux prépublications de 2026 rendent le problème particulièrement explicite. Gerster et ses collègues réanalysent FACED avec des essais où l'émotion assignée concorde ou non avec l'autoévaluation [R002](https://doi.org/10.64898/2026.06.12.731889). Xu et Plataniotis examinent une exclusion conjointe des sujets et stimuli [R001](https://arxiv.org/abs/2610.28422). Ces résultats concernent leurs modèles et jeux de données ; ils ne démontrent pas une impossibilité générale du décodage émotionnel. Le travail FACED provient notamment de HABS et doit être présenté comme tel, sans le compter comme une validation indépendante du jeu HABS.

Pandilova et ses collègues montrent aussi l'importance de l'identité des personnes dans les caractéristiques qEEG de DEAP et DREAMER [R093](https://doi.org/10.3390/s26175327). Il faut donc regrouper les sessions et fenêtres d'une personne, mais cette précaution seule ne règle pas l'effet vidéo.

**Conséquence.** Notre scénario principal est la prédiction d'une session future chez une personne calibrée, sur un catalogue déjà observé chez d'autres personnes. Une extension aux vidéos inconnues constituerait une autre étude. Le modèle de référence doit connaître les vidéos et disposer de la même calibration personnelle que le modèle physiologique.

## 2. Prédire une correction individuelle a déjà des précédents

AffectRoute utilise un prior de population puis des corrections tirées de l'EEG et de la physiologie périphérique, avec des résidus construits hors pli [R084](https://doi.org/10.64898/2026.08.24.746535). Wang et ses collègues combinent une trajectoire commune au stimulus avec une mise à l'échelle individuelle issue de l'EEG [R043](https://arxiv.org/abs/2610.02796). Les deux sont des prépublications de 2026. Leur pertinence suffit à écarter une revendication de nouveauté portant simplement sur « moyenne de la vidéo + correction personnelle ».

Le transfert entre journées est également étudié par Lin [R035](https://doi.org/10.1109/jbhi.2019.2934172). Des méthodes plus complexes, comme les autoencodeurs masqués inter-sessions, existent déjà [R037](https://doi.org/10.1109/tnsre.2024.3389037). Une nouvelle architecture n'est donc pas le premier besoin scientifique de HABS.

**Conséquence.** La contribution proposée concerne la sévérité de la comparaison : référence calibrée sur le niveau et l'étendue de notation, évaluation insensible à un simple décalage constant, contrôles de stimulus et analyse des limitations de la livraison. Il s'agit d'une application méthodologique à HABS, pas d'une première mondiale établie.

## 3. Cohérence physiologique et expérience subjective ne sont pas interchangeables

Les travaux de Mauss, Brown, Cuve et leurs collègues décrivent une cohérence variable selon le système, le contexte et la personne [R016](https://doi.org/10.1037/1528-3542.5.2.175), [R014](https://doi.org/10.1037/emo0000579), [R020](https://doi.org/10.1111/psyp.14262). Zawadzki et ses collègues illustrent qu'une association intraindividuelle peut exister alors qu'une corrélation entre personnes est nulle [R390](https://doi.org/10.1037/hea0000466). Ces résultats d'association ne garantissent pas une prédiction hors échantillon.

McVeigh et ses collègues soulignent la dépendance à la situation pour la peur [R098](https://doi.org/10.1037/emo0001265). Koppold et ses collègues montrent que la prise en compte de la variabilité individuelle change l'interprétation des ressemblances psychophysiologiques [R018](https://doi.org/10.1111/psyp.70224). Les résultats très détaillés chez quatre personnes d'un préprint récent ne fournissent pas une précision attendue pour une population nouvelle [R009](https://doi.org/10.64898/2026.10.23.761434).

EMAP dispose d'annotations continues, contrairement à notre cible principale HABS [R083](https://doi.org/10.1111/psyp.14446). Une note donnée après dix secondes ne permet pas de valider une trajectoire émotionnelle à chaque milliseconde.

**Conséquence.** La cible est une autoévaluation par essai. Un résultat positif établira une information prédictive supplémentaire, sans démontrer que le signal est la cause de l'émotion ou une mesure indépendante de sa « vérité ».

## 4. Les styles de réponse sont une hypothèse de mesure à contrôler

Baird et ses collègues, puis Deng et ses collègues, montrent des contributions possibles des styles de réponse à la variabilité calculée à partir de notes répétées [R026](https://doi.org/10.1016/j.jrp.2016.06.015), [R023](https://doi.org/10.1080/00273171.2017.1413636). Les études contrôlées et longitudinales de Henninger et ses collègues distinguent la variabilité de l'inertie ou de la réactivité aux stimuli [R167](https://doi.org/10.1037/pas0001370), [R119](https://doi.org/10.1080/00273171.2026.2721527).

Les résultats ne sont pas unanimes. Anvari et ses collègues trouvent peu d'éléments en faveur de l'explication par les styles dans leur étude [R230](https://doi.org/10.1177/01461672231208499). Schimmack et ses collègues contestaient déjà une interprétation systématiquement dominée par ces biais [R440](https://doi.org/10.1207/s15327752jpa7803_06). Nestler et ses collègues trouvent des associations avec la variabilité sans bouleversement systématique des résultats après contrôle [R030](https://doi.org/10.1525/collabra.29929).

Une calibration affine décrit des tendances observées ; elle ne sépare pas parfaitement un style de réponse et une différence réelle de sensibilité. Les modèles psychométriques plus complets supposent davantage d'items et d'information que la note unique disponible ici.

**Conséquence.** Comparer une référence vidéo, une référence avec décalage personnel et une référence avec décalage plus pente. Ajouter un critère intraindividuel et une analyse par rangs. Ne pas appeler le résidu « émotion pure ».

## 5. La validation doit correspondre à la question et au plan réel

Curran et Bauer, ainsi que Wang et Maxwell, distinguent les effets intra/interindividuels et les tendances temporelles [R400](https://doi.org/10.1146/annurev.psych.093008.100356), [R376](https://doi.org/10.1037/met0000030). Le choix du centrage doit correspondre à l'estimand ; une normalisation globale n'est pas automatiquement une estimation de la relation moyenne au sein des personnes [R034](https://pmc.ncbi.nlm.nih.gov/articles/PMC6696920/).

Les travaux sur les sujets et stimuli croisés rappellent que les essais ne sont pas des individus indépendants [R044](https://doi.org/10.1037/a0028347), [R046](https://doi.org/10.1037/xge0000014). Pour notre question conditionnelle à un catalogue connu et à un modèle fixé, l'unité d'incertitude principale sera la personne de test. Une revendication sur de nouveaux stimuli demanderait un autre échantillonnage et une incertitude adaptée.

Enfin, des exclusions guidées par les réponses peuvent changer les conclusions [R076](https://doi.org/10.7554/eLife.52465). Les données HABS livrées ne permettent pas une sensibilité incluant les essais marqués 1. Cette restriction est une limite de population cible, pas un détail de nettoyage.

## Décisions issues de la revue

Le protocole proposé retient une seule cible principale, une seule cohorte, une calibration antérieure au test et un petit nombre de caractéristiques physiologiques. Il distingue explicitement l'amélioration des variations individuelles de la correction du niveau moyen. Il conserve les résultats négatifs possibles et prévoit une analyse de précision avant de prétendre exclure un effet utile. Les choix détaillés et les conditions de lancement figurent dans le [protocole de l'étude](../../journal/01_individual_arousal.md).
