# Émotion déclarée, EEG et physiologie : revue de portée systématisée

Recherche du 9 novembre 2026, Europe/Paris. Moteur principal : Exa, complété
par Europe PMC. Objectif : trouver une question originale et testable dans HABS.

La piste recommandée est d'étudier **la reproductibilité entre séances de la
différence entre les effets de l'histoire émotionnelle sur le ressenti déclaré
et sur l'EEG**. L'effet de l'essai précédent et la discordance entre réponses
émotionnelles sont déjà connus. La lacune ciblée concerne leur combinaison
conditionnelle et leur stabilité individuelle, sur de nouveaux clips pour la
personne. C'est une proposition issue de la revue, pas un nouveau résultat HABS.

[Question et protocole proposés](QUESTION.md) · [Protocole de recherche](PROTOCOL.md)
· [Journal des requêtes](SEARCH_LOG.json) · [Sélection de toutes les notices](SCREENING.csv)
· [Extraction des 57 publications](EVIDENCE.csv).

## Méthode et périmètre

La recherche couvre cinq axes : spécificité stress/effort/émotion, concordance
subjective–physiologique, contexte temporel, représentations EEG, et interactions
cerveau–cœur. Quinze requêtes initiales Exa sont suivies de six recherches ciblées,
dont des requêtes explicitement destinées à trouver des antécédents invalidant la
nouveauté de la proposition. Les requêtes et objectifs exacts figurent dans le
journal. Les publications postérieures au 09/11/2026 sont hors périmètre.

Europe PMC complète le repérage sémantique par cinq requêtes booléennes dans les
champs titre/résumé. Elles portent sur concordance émotionnelle, effets
séquentiels, activité apériodique, relations cerveau–cœur et confusions liées au
stimulus. Tous leurs résultats ont été récupérés. Quatre essais pilotes de
recherche dans tout le texte, trop larges et incomplets, sont conservés mais
exclus du flux formel.

Le tri porte sur les titres, puis sur les résumés et textes accessibles lorsque
nécessaire. Le noyau privilégie les expériences humaines pertinentes pour une
émotion induite visuellement. Des travaux ambulatoires, cliniques, perceptifs,
d'IRMf et de méthode sont distingués comme contexte. Les revues et articles de
données ont un rôle séparé. La colonne `access` distingue les résumés des textes
partiels : un rendu de page ou un DOI consulté ne vaut pas lecture intégrale.

| Étape | Compte vérifiable |
|---|---:|
| Requêtes Exa | 21 |
| Résultats demandés / obtenus avec Exa | 242 / 242 |
| Requêtes formelles Europe PMC | 5 |
| Notices Europe PMC obtenues | 689 |
| Occurrences avant déduplication | 931 |
| Doublons fusionnés automatiquement | 154 |
| Versions/pages liées fusionnées après vérification | 19 |
| Notices distinctes selon ces règles | 758 |
| Exclusions sur titre/résumé | 675 |
| Exclusion après consultation étendue | 1 |
| Publications du noyau de synthèse | 28 |
| Publications empiriques/méthodologiques de contexte | 29 |
| Ressources de données / revues repérées / protocole planifié | 7 / 17 / 1 |

Les 45 demandes de pages par Exa ont produit 41 rendus, dont une documentation
d'API. Certaines pages ne contiennent que le résumé, d'autres sont tronquées.
Ces nombres ne sont donc ni un nombre de textes intégraux lus, ni un nombre
d'échantillons indépendants. La réutilisation de DEAP, SEED et FACED est fréquente.
Les deux travaux de Deodato cités plus bas pourraient également partager leur
recrutement ; ils ne sont pas présentés comme deux réplications indépendantes.

Il s'agit d'une **revue de portée systématisée exploratoire**. La sélection est
réalisée par un seul assistant IA, sans double lecture indépendante. Le périmètre
a été précisé pendant le repérage et n'est pas préenregistré. La couverture
n'inclut pas une interrogation exhaustive de Scopus, Web of Science ou PsycINFO.
Une méta-analyse et une certification de conformité PRISMA ne sont pas revendiquées.
L'appréciation des limites est qualitative, publication par publication ; aucun
score formel de risque de biais ni chaînage exhaustif des citations n'est annoncé.

## Ce que les travaux permettent de dire

### Les variations EEG ne sont pas automatiquement spécifiques au stress

Ehrhardt et collègues distinguent effort mental, pression temporelle et menace
d'évaluation sociale. Les variations alpha/bêta observées entre tâche et repos
ne discriminent pas clairement les composantes supplémentaires du stress.
Cela fragilise l'interprétation d'une bande EEG comme quantité universelle de
stress. [Ehrhardt et al.](https://doi.org/10.1111/ejn.15211).

À l'inverse, le protocole vidéo aversif de Roy et collègues associe questionnaires,
cortisol, cardiovasculaire et EEG chez 78 hommes. Il montre qu'un protocole vidéo
peut étudier le stress avec plusieurs critères indépendants. Il ne valide pas le
produit valence négative × arousal employé dans notre première expérience.
[Roy et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12271860/).

Conséquence pour HABS : parler d'arousal, de valence et de réponses à des stimuli
aversifs. Une question clinique sur le stress, ses causes ou son apparition
requiert des critères supplémentaires absents de la livraison actuelle.

### La concordance entre ressenti et corps est partielle et contextuelle

Les travaux de Mauss montrent que la cohérence dépend de l'intensité et des
systèmes comparés. Une analyse multivariée plus récente identifie à la fois une
dimension commune et une structure autonome qui ne covarie pas avec le ressenti.
Une mesure physiologique peut donc varier sans déplacement proportionnel des
notes déclarées. [Mauss et al., 2005](https://doi.org/10.1037/1528-3542.5.2.175),
[cohérence et divergence, 2023](https://doi.org/10.1111/psyp.14262).

Les résultats ne sont pas uniformes. Une étude de films trouve une association
entre cohérence et bien-être ; le suivi ambulatoire n'en trouve pas de lien
crédible dans ses analyses exploratoires. Les différences de tâche, de mesure et
d'agrégation empêchent de conclure qu'une forte cohérence serait toujours
bénéfique. [Brown et al.](https://doi.org/10.1037/emo0000579),
[suivi ambulatoire](https://doi.org/10.1016/j.biopsycho.2021.108074).

L'étude des phases avant/pendant/après les films soutient surtout la cohérence
expérience–expression, avec des résultats plus limités pour la physiologie.
La réanalyse de Koppold souligne aussi que préserver la variabilité
intraindividuelle change la représentation du lien affect–physiologie.
[Étude des phases émotionnelles, 2023](https://doi.org/10.1111/psyp.14359),
[Koppold et al., 2026](https://doi.org/10.1111/psyp.70224).

Conséquence : chercher simplement une « dissociation corps–ressenti » ne serait
pas nouveau. HABS peut plutôt étudier **dans quelles circonstances** cette
dissociation varie et **si sa structure est reproductible**. Elle ne doit pas être
étiquetée comme déficit d'interoception ou psychopathologie sans mesure adaptée.

### Les effets de l'expérience précédente sont connus, mais diffèrent selon la mesure

L'antécédent le plus direct est Fujimura et collègues. Chez 43 adultes, les
images précédentes modifient différemment les sentiments et certaines réponses
périphériques, avec des images cibles identiques entre blocs. Notre projet ne
peut donc pas revendiquer la découverte générale d'un décalage séquentiel entre
ressenti et corps. [Fujimura et al., 2013](https://doi.org/10.3389/fpsyg.2013.00212).

Les notes affectives présentent aussi des dépendances sérielles à travers
images et sons. Des expériences avec essais sans réponse montrent toutefois que
la perception et l'acte de répondre peuvent produire des effets distincts,
voire opposés. Une autocorrélation des notes n'est donc pas une preuve suffisante
de persistance de l'émotion vécue.
[Van der Burg et al., 2021](https://doi.org/10.1163/22134808-bja10064),
[effets séquentiels opposés, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11088209/),
[contrôle des stimuli et réponses précédents](https://doi.org/10.1080/02699931.2019.1696752).

Pour l'EEG, des études trouvent des réponses principalement liées au stimulus
actuel, sans effet net du contexte précédent sur EPN/LPP ou sur certaines bandes
oscillatoires. D'autres trouvent des modifications après l'émotion, notamment
dans les microétats et l'activité apériodique au repos. Ces protocoles et
représentations diffèrent ; leurs résultats ne sont pas une contradiction directe.
[Schupp et al., 2012](https://doi.org/10.1016/j.biopsycho.2012.04.006),
[réponses liées aux événements, 2023](https://doi.org/10.3389/fnhum.2022.868549),
[Deodato et al., 2024](https://doi.org/10.1093/scan/nsae012),
[Deodato et al., 2025](https://doi.org/10.1111/psyp.70103).

Conséquence : comparer explicitement histoire et événement présent est motivé.
Les notes HABS après chaque vidéo permettent un niveau **entre essais**, pas une
reconstruction seconde par seconde du vécu pendant une vidéo.

### Changer de représentation ne suffit pas à créer une question nouvelle

L'activité apériodique et les oscillations ont déjà été séparées dans une étude
de vidéos émotionnelles utilisant DEAP. Les microétats ont également été reliés
aux émotions et au contenu audiovisuel. Tester « l'exposant 1/f prédit-il
l'arousal ? » ou « peut-on classifier l'émotion par microétats ? » serait
principalement une réplication ou un changement de représentation.
[Activité apériodique et arousal, 2025](https://doi.org/10.1162/imag_a_00451),
[Hu et al., 2023](https://doi.org/10.1093/cercor/bhac082).

La comparaison sur HABS pourrait rester utile pour la robustesse et la
généralisation. Elle ne justifie pas de déduire directement un équilibre
excitation–inhibition, une causalité thalamique ou une mesure de stress d'un
exposant spectral isolé.

### Les interactions cerveau–cœur sont prometteuses, mais déjà étudiées

Des travaux relient le couplage EEG–ECG à l'arousal et à la familiarité. Une étude
en VR observe plusieurs corrélats mais ne confirme pas le flux directionnel
principal attendu. Des expériences distinguent aussi émotion ressentie et
émotion attribuée à autrui avec des réponses évoquées par le cœur.
[Mishra et al., 2022](https://doi.org/10.3390/brainsci12060702),
[VR, 2024](https://doi.org/10.1111/psyp.14696),
[Engelen et al., 2023](https://doi.org/10.1016/j.neuroimage.2023.119867).

Une étude de flux EEG–ECG rapporte une dynamique précoce cœur–cerveau, mais sa
direction dépend d'un modèle observationnel. Elle ne fournit pas une intervention
prouvant une causalité biologique. De plus, les HEP sont sensibles aux artefacts
cardiaques et à des choix de traitement encore hétérogènes.
[Candia-Rivera et al., 2022](https://doi.org/10.1073/pnas.2119599119),
[revue des méthodes HEP, 2026](https://doi.org/10.1093/scan/nsag057).

Conséquence : un projet HEP sur nos clips de dix secondes demanderait un audit
ECG/EEG, une stratégie d'agrégation et un contrôle des artefacts. Ni « couplage
cerveau–cœur » ni « HEP et émotion » ne constituent seuls une nouveauté.

### Stabilité de groupe et signature personnelle sont deux résultats différents

Des configurations EEG relativement stables entre séances et une fiabilité de
réponses périphériques ont déjà été étudiées. Mais une étude de peur conditionnée
sur 120 personnes montre qu'une meilleure fiabilité de groupe peut coexister
avec une fiabilité longitudinale individuelle limitée. Les analyses de LPP
distinguent aussi fiabilité des agrégats et des différences intraindividuelles.
[Zheng et al.](https://arxiv.org/abs/1601.02197),
[pilote sur dix semaines](https://doi.org/10.1186/s40101-019-0209-y),
[Klingelhöfer-Jens et al., 2022](https://doi.org/10.7554/eLife.78717),
[Hill et al., 2023](https://doi.org/10.1111/psyp.14185).

Conséquence : l'originalité ne serait pas simplement d'avoir deux séances.
Il faut vérifier la stabilité d'un **paramètre individuel précisément défini**,
avec son erreur de mesure, des stimuli contrôlés et une comparaison à une
référence sans information personnelle provenant de J1.

### Le stimulus reste un concurrent sérieux de toute interprétation émotionnelle

Les audits récents de FACED et d'encodeurs EEG montrent que l'identité de la
vidéo et les procédures de découpage peuvent soutenir les performances sous
étiquettes émotionnelles. Ce sont des prépublications, à distinguer d'un consensus
définitif. Elles renforcent néanmoins un contrôle déjà pertinent pour HABS :
comparer les réponses de personnes à des contenus identifiés et ne pas traiter
la reconnaissance du film comme une lecture du ressenti.
[Gerster et al., 2026](https://doi.org/10.64898/2026.06.12.731889),
[audit cross-sujet/cross-stimulus, 2026](https://arxiv.org/html/2610.28422),
[confusion par stimuli répétés](https://arxiv.org/html/2508.00531).

## Questions comparées

| Piste | État de la nouveauté | Faisabilité HABS | Décision |
|---|---|---|---|
| Alpha/bêta comme marqueur universel de stress | Très étudié, spécificité problématique | Pas de critère indépendant de stress | Écartée |
| Apériodique 1/f et arousal | Antécédent direct sur DEAP | Réextraction nécessaire, dix secondes à auditer | Réplication utile, pas question principale inédite |
| HEP/couplage EEG–ECG et ressenti | Plusieurs antécédents directs | Signaux présents; validation des battements et artefacts nécessaire | Extension technique, nouveauté insuffisante seule |
| Concordance ou profils individuels stables | Idée déjà étudiée; notre expérience 01 n'établit pas d'utilité de clusters | Deux séances, mais dépendance au stimulus et erreur de mesure | Trop général |
| Effet séquentiel ressenti–EEG et sa reproductibilité individuelle | Antécédents séparés; combinaison précise non retrouvée | Notes, EEG, événements et séances présents; support des transitions à compter | **Recommandée, sous contrôle de faisabilité** |
| Mesurer directement des qualia ou une interoception consciente | La question ne se réduit pas aux notes disponibles | Ni protocole de conscience ni tâche d'interoception dans la livraison | Non identifiable ici |

## Lacune retenue et niveau de confiance

La question recommandée est : **l'influence de l'arousal déclaré à l'essai
précédent diffère-t-elle entre le ressenti actuel et l'EEG actuel, et cette
différence se retrouve-t-elle chez la même personne entre J1 et J2 ?**

Les recherches contradictoires Q16–Q21 ont retrouvé les travaux de contexte,
d'inertie, de fiabilité et de persistance neurale décrits ci-dessus. Aucun
document retenu n'établit exactement la reproductibilité personnelle de ce
contraste, avec contrôle des vidéos actuelle/précédente, du délai réel et de la
fiabilité, sur de nouveaux contenus pour la personne. Cette absence dans le
corpus vérifié est la base de la proposition. Elle ne prouve pas qu'aucun article,
préprint ou projet non indexé n'a jamais posé cette question.

La confiance dans la pertinence scientifique est bonne : plusieurs antécédents
donnent des prédictions distinctes et des résultats négatifs plausibles. La
confiance dans une priorité mondiale est modérée. La faisabilité est établie au
niveau des variables et du volume brut ; l'identifiabilité et la précision ne
sont pas encore établies au niveau des transitions réellement utilisables.

## Confrontation aux données HABS

L'[audit existant](../../../../data/eda.md) réconcilie 22 974 essais évalués et
314 identifiants dans l'ensemble des cohortes. La cohorte 2 contient 9 726 essais,
113 identifiants et 76 identifiants avec au moins vingt essais évalués par séance
sur deux séances. La préparation existante compte 7 795 essais EEG utilisables
dans cette cohorte, avant les restrictions finales de l'expérience 01. **Ces
comptes ne sont pas des nombres de transitions admissibles** et ne constituent
pas une analyse de puissance pour la nouvelle question.

Les [métadonnées de la livraison](../../../../DATASET.md) donnent événements,
onsets, origines de segments et identifiants de vidéos. L'ordre doit être
reconstruit à partir des événements enregistrés, car l'ordre déclaré manque dans
28,7 % de la vue de développement. Il ne faut jamais relier artificiellement deux
essais séparés par un événement exclu ou une interruption non documentée.

La faisabilité fine exige de compter les vraies transitions, les délais, les
vidéos communes et la diversité des contextes précédents. Les vidéos ne sont pas
fournies : on peut contrôler leur identité, pas reconstruire ici leurs propriétés
visuelles ou sonores. Les notes sont finales, non continues. Les questionnaires
de personnalité évoqués au kick-off ne sont pas dans la table de participants
livrée. Ces limites rendent plusieurs explications causales non identifiables.

Enfin, les réponses réservées de l'expérience 01 ont déjà été examinées. Elles
ne redeviennent pas un test vierge pour la nouvelle question. Toute analyse sur
ce corpus doit être présentée comme exploratoire ou comme validation interne
d'une hypothèse formulée après observation des résultats précédents. Une
confirmation indépendante demanderait des observations nouvelles ou un jeu dont
l'indépendance et le non-usage sont réellement vérifiés.

## Livrables et arrêt de cette phase

La [fiche de question](QUESTION.md) précise les variables, les contrôles, les
critères d'abandon et le résultat qui réfuterait l'hypothèse. La revue n'a lancé
aucun nouveau modèle, aucun réentraînement sur le test ouvert et aucune analyse
d'association supplémentaire des réponses HABS. Elle ne transforme pas le
résultat négatif précédent en preuve sur les qualia, ni en absence de corrélats
neuraux de l'émotion.
