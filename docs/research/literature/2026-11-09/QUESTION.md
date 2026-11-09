# Une dissociation temporelle entre ressenti et EEG est-elle reproductible ?

Version 0.1, 9 novembre 2026. Proposition issue de la revue, sans nouvelle analyse
des réponses HABS. Statut : question recommandée, faisabilité fine à vérifier.

## Question précise

**L'influence de l'arousal déclaré à l'essai précédent diffère-t-elle entre le
ressenti actuel et l'EEG actuel, et cette différence se retrouve-t-elle chez la
même personne entre J1 et J2, après contrôle des vidéos et du délai entre essais ?**

Exemple : après une vidéo ressentie comme très intense, la personne déclare-t-elle
encore une activation élevée au clip suivant ? Son EEG conserve-t-il également
une association avec cette activation précédente, une fois considérés le clip
actuel et son propre EEG précédent ? Et retrouve-t-on ce décalage au deuxième jour ?

Cela porte sur des **mesures de persistance entre essais**. Une réponse lente du
capteur, une habitude de notation et une émotion qui persiste sont des mécanismes
différents. L'analyse doit préserver cette distinction.

## En quoi la question pourrait être originale

| Antécédent vérifié | Ce qui est déjà établi ou étudié | Différence recherchée ici |
|---|---|---|
| [Fujimura et al., 2013](https://doi.org/10.3389/fpsyg.2013.00212) | Contexte précédent, ressenti et périphérie peuvent changer différemment | Ajouter l'EEG et tester la reproductibilité personnelle entre jours |
| [Koval et al., 2013](https://doi.org/10.1037/a0033579), [2016](https://doi.org/10.3389/fpsyg.2015.01997) | Inertie affective individuelle, distincte de moyenne et variabilité | Contraste conjoint notes–EEG plutôt qu'inertie des notes seule |
| [Schupp et al., 2012](https://doi.org/10.1016/j.biopsycho.2012.04.006), [étude oscillatoire, 2023](https://doi.org/10.3389/fnhum.2022.868549) | Certaines réponses EEG restent liées au stimulus actuel | Tester un effet individuel conditionnel du passé plutôt qu'un effet moyen de bloc |
| [Deodato et al., 2024](https://doi.org/10.1093/scan/nsae012), [2025](https://doi.org/10.1111/psyp.70103) | Persistance neurale après des films aversifs | Nouveaux clips pour la personne, autre séance et contrôle du stimulus actuel |
| [Zheng et al.](https://arxiv.org/abs/1601.02197), [pilote de fiabilité](https://doi.org/10.1186/s40101-019-0209-y) | Stabilité de caractéristiques ou de réponses agrégées | Stabilité d'un contraste de pentes séquentielles avec erreur de mesure |

Les recherches Q16–Q21 n'ont pas retrouvé d'antécédent réunissant tous ces
éléments. La nouveauté revendicable est **cette lacune précise du corpus
consulté**, pas la découverte des effets de contexte, de l'inertie ou de la
discordance. Un antécédent exact, s'il est retrouvé, ramènerait le projet à une
réplication ou à une extension de généralisation.

## Données disponibles et limites de faisabilité

| Besoin | État vérifié dans HABS | Ce qui reste à compter |
|---|---|---|
| Notes d'arousal et de valence par essai | Présentes, échelles 1–9 | Paires consécutives avec deux notes valides |
| Deux séances par personne | 76 identifiants de cohorte 2 ont au moins vingt essais évalués dans chacune | Effectif restant après qualité et source CURATED |
| EEG pendant les vidéos | 7 795 essais EEG utilisables en cohorte 2 dans la préparation existante | Paires dont les deux fenêtres sont utilisables et disjointes |
| Identité des vidéos | Présente | Vidéos partagées, diversité des contextes précédents, rang du plan d'analyse |
| Ordre et délai | Onsets et événements enregistrés présents | Distribution des délais, interruptions et événements intermédiaires |
| Physiologie périphérique | ECG/EDA présents, qualité variable | Support pour des analyses secondaires sur les mêmes paires |

Ces chiffres proviennent de l'[audit de données](../../../../data/eda.md), pas
d'une nouvelle analyse de transitions. Les 76 identifiants ne sont donc pas un
effectif final garanti ni une preuve de puissance. Le fichier d'ordre déclaré
incomplet ne suffit pas ; les événements enregistrés sont la référence.

La cohorte 2 est le premier périmètre, sans fusion avec d'autres cohortes dont
l'identité biologique commune n'est pas reliée. Les vidéos ne sont pas fournies :
leur identité est contrôlable, leurs propriétés audiovisuelles détaillées ne le
sont pas dans cette étude. Les notes finales ne permettent pas d'estimer la
chronologie consciente à l'intérieur des dix secondes.

## Variables fixées avant l'analyse d'association

- Mesure subjective principale : arousal déclaré, conservé comme dimension
  continue. La valence est secondaire. Le proxy de stress expérimental de 01
  n'est pas la cible principale de cette question.
- Mesure EEG principale : opposé du logarithme de la puissance alpha relative
  postérieure, 8–13 Hz relativement à 1–40 Hz, sur P3/P4/O1/O2 disponibles et
  valides. Son signe est fixé pour qu'une valeur élevée désigne une alpha relative
  plus faible ; cela ne l'identifie pas à une quantité de stress.
- Fenêtres principales : dix secondes du clip, sans recouvrement entre essais.
  Utiliser CURATED pour les deux séances, avec traitement confiné aux fenêtres.
  Un filtrage de toute la séance qui mélange le futur et le passé pourrait créer
  artificiellement une dépendance temporelle.
- Histoire principale : arousal et EEG de la vidéo précédente dans l'inventaire
  complet des événements. Une vidéo sans note ou sans signal ne doit pas être
  supprimée avant de déterminer qui précède qui.
- Délai : temps entre la fin de la vidéo précédente et le début de l'actuelle,
  comprenant l'activité de notation. Documenter les événements intermédiaires.
- Contrôles : identités actuelle et précédente des vidéos, personne, séance,
  position et temps dans la séance, source du label attendu et qualité des signaux.
  Le label attendu n'est pas une attente consciente mesurée.

Les jours sont ceux des enregistrements et des événements réconciliés. L'audit
initial a corrigé 2 690 correspondances de jour ; repartir directement du jour
brut des annotations risquerait de fabriquer une histoire temporelle incorrecte.

Les sources de normalisation et les paramètres sont estimés sur J1 ou sur le
développement prévu, puis conservés pour l'évaluation J2. Ne pas normaliser J2
en utilisant la distribution de ses réponses futures.

## Analyse principale proposée

Pour chaque mesure actuelle, arousal et EEG, ajuster un modèle hiérarchique qui
sépare l'association avec l'arousal précédent de l'autocorrélation propre de
l'EEG. En notation schématique :

```text
mesure actuelle
  = personne + vidéo actuelle + vidéo précédente + séance + temps/ordre
  + fonction du délai
  + effet de l'arousal précédent
  + effet de l'EEG précédent
  + interactions de ces effets avec le délai
  + erreur
```

Les coefficients d'histoire varient entre personnes et entre séances avec
estimation partiellement mutualisée. Les deux mesures sont mises sur des
échelles standardisées à partir de l'information antérieure disponible. Le
contraste individuel est la différence entre leurs effets standardisés de
l'arousal précédent, évalués à un délai de référence fixé à partir des seuls
temps de J1. Il s'agit d'un contraste statistique de mesures différentes, pas
d'une différence de durées intrinsèques de l'émotion.

Le critère principal est la **reproductibilité de ce contraste individuel** :
son estimation J1 doit informer celle de J2 au-delà du niveau moyen, de la
variabilité de chaque mesure et du contexte. Estimer l'incertitude des pentes et
de leur covariance entre séances. Une corrélation naïve de deux séries de pentes
bruitées ou régularisées ne suffit pas. Vérifier également si utiliser les effets
individuels de J1 améliore la prévision J2 par rapport à une référence qui possède
les mêmes contrôles mais seulement les effets d'histoire de population.

L'incertitude doit tenir compte des personnes, de la structure temporelle et des
stimuli partagés. Les essais ne sont pas autant de réplications indépendantes.
Le contrôle de moyenne/variabilité et la sensibilité à l'erreur de mesure sont
nécessaires avant de parler de caractéristique personnelle.

L'analyse est falsifiable : des contrastes J1/J2 incompatibles, une variance
individuelle non distinguable du bruit, ou aucun apport de J1 à J2 réfuteraient
la proposition de reproductibilité dans ce protocole. Un simple effet moyen de
l'essai précédent ne satisferait pas le critère principal.

## Contrôles qui peuvent invalider l'interprétation

1. **Identifiabilité.** Vérifier que l'ordre des clips et les contextes précédents
   ne sont pas déterministes au point d'empêcher leur séparation. Examiner le
   rang du plan et le support des vidéos communes avant de tester les associations.
2. **Biais de réponse.** Distinguer histoire des contenus et histoire des notes.
   HABS n'a pas la manipulation « regarder sans noter » des expériences sur la
   dépendance sérielle. Une persistance réelle du ressenti ne pourra donc pas être
   séparée complètement d'une persistance de la manière de répondre.
3. **Cinétique et qualité.** Contrôler disponibilité, amplitude et qualité des
   capteurs. L'ECG et surtout l'EDA peuvent conserver une autocorrélation physique.
   Ne pas l'interpréter comme mémoire émotionnelle sans examen supplémentaire.
4. **Tendance lente.** Comparer à une référence incluant temps, ordre et délai.
   Utiliser des décalages/permutations par blocs seulement s'ils conservent ces
   dépendances ; un mélange arbitraire des lignes ne fournit pas un bon nul.
5. **Frontières et répétitions.** Ne jamais relier deux jours, deux segments ou
   des clips séparés par un événement non observé. Pour la vérification sur des
   contenus nouveaux à la personne, rechercher les expositions antérieures dans
   l'inventaire complet, même sans note utilisable.
6. **Erreur de mesure.** Évaluer la précision des caractéristiques et la stabilité
   des contrastes. Des différences de bruit entre notes et EEG peuvent créer une
   différence apparente de dépendance temporelle.

Les analyses ECG, EDA, EEG apériodique et fenêtres alternatives sont secondaires,
avec leurs propres critères de validité et sans sélection du résultat le plus
favorable. Aucun HRV de dix secondes n'est proposé comme mesure fiable de la
balance sympathique–vagale.

## Première étape et conditions d'arrêt

La première étape est un comptage sans ajustement d'association : vraies paires
consécutives, délais, données manquantes, vidéos communes et support par
personne/séance. Un effectif brut élevé ne remplace pas cette vérification.

Avant les associations, une simulation fondée sur le plan observé, sans effets
estimés sur les réponses réservées, doit déterminer quels contrastes pourraient
être estimés avec une précision utile. Si les pentes personnelles ne sont pas
identifiables ou suffisamment précises, abandonner la revendication de signature
individuelle. Une analyse moyenne du contexte peut rester descriptive, mais ne
répondrait pas à la question nouvelle annoncée.

L'expérience 01 est terminée et son test a été ouvert. Le nouveau projet est
donc exploratoire sur ce corpus, même si son analyse est fixée avant calcul.
Réserver à nouveau les mêmes personnes ne crée pas une confirmation indépendante.

## Portée d'un éventuel résultat

Un résultat positif soutiendrait une différence personnelle reproductible dans
la façon dont l'histoire récente se retrouve dans **les notes et l'EEG mesurés**.
Il ne démontrerait ni une classe biologique de personnes, ni un déficit
d'interoception, ni une mesure clinique de stress, ni une explication des qualia.
Un résultat négatif préciserait les limites de cette hypothèse et de sa mesure
dans HABS, sans conclure à l'absence d'expérience subjective ou de mécanisme neural.
