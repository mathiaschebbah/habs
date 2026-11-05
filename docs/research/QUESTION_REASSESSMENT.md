# Question scientifique après vérification des répétitions

État au 5 novembre 2026, avant toute performance sur les réponses réelles réservées.

## Ce que les données montrent

L'[audit corrigé](../../results/repetitions.json) trouve neuf paires de notes
pour un même identifiant et une même vidéo à des sessions différentes, réparties
sur six identifiants. Huit paires concernent la cohorte 2 et une la cohorte 4.
Le comptage des événements enregistrés, sans exiger deux notes appariables,
ne trouve que seize paires sur dix identifiants. L'identité vidéo est celle
du nom de fichier fourni ; les contenus ne sont pas distribués.

Ce résultat corrige le constat initial de zéro répétition dans la cohorte 2.
Il ne fournit cependant pas un échantillon adapté à l'apprentissage et au test
indépendants d'un modèle de variations à contenu constant. Les notes stables
n'ont pas été retirées du comptage. Le nombre de sessions répétées chez une
personne ne doit pas être confondu avec le nombre de répétitions du même contenu.

## Questions possibles

### Changements face au même contenu

**Affirmation.** Une variation physiologique prédit la variation de la note
d'une même personne face au même contenu à une autre séance.

Le contraste demanderait des différences appariées de notes et de signaux,
avec des personnes indépendantes pour le développement et le test. Les neuf
paires disponibles sont insuffisantes pour ce plan. Un gain utile pourrait
établir une prédiction des changements déclarés, mais ne séparerait pas
familiarisation, changement d'usage de l'échelle et erreur de mesure. Une
absence de significativité sur six identifiants ne permettrait pas de conclure
à l'absence de cette association. Cette question n'est pas retenue avec la
livraison actuelle.

### Utilité transférable d'un historique personnel

**Question recommandée.** Les réponses d'une première séance réduisent-elles
utilement l'erreur de prédiction des autoévaluations d'une séance ultérieure
sur d'autres vidéos, par rapport à une référence vidéo et contexte apprise
chez d'autres personnes ?

Le contraste principal compare une calibration personnelle simple et
régularisée à la même référence sans historique personnel. La MAE brute
moyenne par personne doit être le critère principal : l'erreur centrée annule
par construction le bénéfice d'une correction constante de niveau.

La sélection des paramètres devra utiliser la MAE dans le développement,
avant le test réservé. Les rares vidéos déjà vues seront exclues de la cible
future. Une question portant sur les notes n'exige pas que la physiologie soit
exploitable ; le manifeste doit donc être reconstruit sans cette restriction.
Les cohortes resteront séparées faute de correspondance des identités biologiques.

Le seuil proposé est un gain de 0,10 point de MAE sur l'échelle de 1 à 9.
C'est un choix opérationnel à fixer avant le test, sans validation clinique.
Un intervalle entièrement au-dessus du seuil soutiendrait un gain utile.
Une borne supérieure sous ce seuil exclurait un gain de cette ampleur pour
la famille de modèles étudiée. Un intervalle qui traverse le seuil laisserait
la question indéterminée. L'équivalence pratique entre les méthodes demanderait
en plus de borner les différences dans les deux sens.

Le résultat concernerait l'utilité prédictive d'un historique de réponses.
Il ne distinguerait pas style de notation, dispositions affectives et
préférences stables, et ne validerait pas les capteurs. L'intérêt pour HABS
serait d'estimer le bénéfice d'une séance de calibration. L'originalité de
la méthode serait modeste ; l'apport serait une estimation reproductible
sur cette livraison et ce catalogue.

### Valeur des capteurs sur des personnes et vidéos inconnues

**Affirmation.** Les mesures courantes apportent un gain utile lorsque ni la
personne ni la vidéo cible ne figurent dans l'apprentissage de population.

Ce plan demande une double séparation des personnes et des vidéos, une
référence applicable aux contenus inconnus, et un décompte des observations
restantes. Une moyenne par identité vidéo ne convient pas. L'incertitude doit
tenir compte des personnes et des vidéos ; un simple bootstrap des personnes
ne suffit pas à généraliser à de nouveaux stimuli.

Un gain utile appuierait la transférabilité opérationnelle des modèles testés.
Une borne excluant un gain utile en limiterait l'intérêt dans ce scénario,
sans démontrer l'absence d'information dans tous les signaux ou tous les modèles.
Les artefacts, les différences de sessions et la sélection personnalisée des
stimuli resteraient des explications concurrentes. Ce scénario peut être utile,
mais il n'est pas une exigence produit établie par les documents disponibles.
Sa faisabilité et sa précision demandent une nouvelle étude de support.

## Décision proposée et portée des résultats négatifs

La calibration transférable est la recommandation principale, car elle porte
sur une information disponible, un usage précis et un contraste interprétable.
Elle constitue une nouvelle question, pas une validation rétrospective de
l'hypothèse d'un état émotionnel individuel. Aucun jeu de données fixe ne
garantit une conclusion positive ou négative ; une estimation trop imprécise
doit rester indéterminée.

La distinction entre absence de significativité et exclusion d'un effet utile
suit les principes des tests d'équivalence. Les bornes doivent précéder la
consultation du test. Une procédure TOST à 5 % emploie un intervalle à 90 % ;
une règle fondée sur un intervalle à 95 % entièrement contenu dans les bornes
est plus conservatrice et doit être annoncée comme telle.
[Lakens, 2017](https://doi.org/10.1177/1948550617697177).

Les 879 répétitions synthétiques déjà terminées restent archivées dans
[`results/validation_pilot/`](../../results/validation_pilot/). Elles ne
résolvent pas l'identification d'un état latent et ne justifient pas à elles
seules de conserver l'ancienne question. Les performances réelles réservées
n'ont pas été calculées.
