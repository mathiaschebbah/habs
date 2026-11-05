# Reproduire l'étude

L'analyse porte sur la livraison HABS montée par la plateforme le 5 novembre
2026. Le dossier `sample/` permet de vérifier un lecteur BDF, mais ne contient
pas les annotations et les enregistrements nécessaires à l'étude complète.
Les données individuelles requièrent l'accès autorisé des organisateurs.

## Environnement

L'exécution de référence utilise Python 3.12.3. Les versions exactes figurent
dans les résultats JSON et les dépendances sont épinglées.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Le test portant sur une table réelle est ignoré en l'absence de `HABS_FEATURES`.
L'expérience exécute également son contrôle de prédiction sur les données
complètes avant de calculer les résultats. Les autres tests utilisent des cas
synthétiques et le BDF d'exemple ; leurs scores ne sont pas des résultats HABS.

## Préparation et évaluation

Le point d'entrée HABS exécute actuellement l'audit des répétitions, pendant le
réexamen de la question centrale. Cet inventaire ne lance aucun modèle :

```bash
.venv/bin/python process.py /data out/repetitions --stage repetitions
```

Depuis la racine du dépôt, remplacer `/data` par la racine de la livraison :

```bash
.venv/bin/python process.py /data out/reproduction --stage extract
.venv/bin/python process.py /data out/reproduction --stage development
.venv/bin/python process.py /data out/reproduction --stage final
```

L'étape d'extraction écrit `trials.csv.gz`, `alignment.json` et `quality.json`.
Les étapes suivantes réutilisent cette table. Pour vérifier de nouveau les
signaux, choisir un nouveau dossier de sortie. Le développement peut être omis
pour une simple reproduction des résultats finaux : ceux-ci reprennent les
paramètres versionnés dans `results/development.json`, sans les sélectionner
de nouveau. Une empreinte différente de la table ou de la partition arrête
l'évaluation. Une livraison modifiée nécessite une nouvelle étude explicitement
distinguée de cette reproduction.

La sortie finale se trouve dans `out/reproduction/study/`. Elle contient les
prédictions, les scores par personne, les partitions, les paramètres et un
rapport skore local. Son audit relit le rapport enregistré. Les JSON de synthèse
contiennent les effectifs et statistiques agrégées. Les tables individuelles
ne doivent pas être ajoutées au dépôt public.

## Simulation

```bash
.venv/bin/python process.py /data out/reproduction --stage validation
```

Cette commande lance un lot d'au plus 40 minutes de nouvelles tâches, puis
attend les tâches déjà lancées. Elle reprend d'abord le fichier de répétitions
du dossier de sortie, sinon le checkpoint publié dans `results/`. Relancer la
même commande poursuit les répétitions restantes jusqu'à 1 000 par mécanisme.
La reprise compare les empreintes des données et du code ainsi que les versions
numériques. Les simulations utilisent des réponses synthétiques, jamais les
notes réelles réservées pour choisir les modèles.

Pour recommencer la validation indépendamment d'un checkpoint publié :

```bash
.venv/bin/python -m experiments.02_method_validation \
  out/reproduction/trials.csv.gz out/new_validation \
  --replicates 1000 --workers 24 --max-seconds 2400 \
  --frozen results/development.json
```

Le dossier `out/new_validation` doit être neuf. Les lots suivants ajoutent
`--resume out/new_validation/simulation_replicates.csv` à la même commande.
Une durée de lot écoulée ne signifie pas que les 4 000 répétitions sont terminées ;
le champ `complete` et les effectifs par mécanisme permettent de le vérifier.

## Figures et article

Les figures ne nécessitent que les agrégats publiés, Matplotlib et NumPy.
La compilation du PDF demande une installation LaTeX avec Babel français,
Latin Modern, natbib, microtype et les extensions usuelles du document.

```bash
make figures
make paper
```

Le PDF est écrit dans `output/pdf/habs_arousal.pdf`. Les fichiers temporaires
de compilation restent dans `out/paper-build/`. Les tableaux et figures sont
reconstruits à partir des JSON ; les mécanismes synthétiques sont présentés
séparément des performances sur les notes réelles.

## Organisation du code

| Fichier | Fonction |
|---|---|
| `alignment.py` | Relier chaque annotation à un événement unique |
| `features.py` | Vérifier les signaux et calculer les caractéristiques |
| `model.py` | Référence par vidéo, calibration personnelle et correction physiologique |
| `study.py` | Partitions, sélection des pénalités et statistiques par personne |
| `simulation.py` | Générer les réponses synthétiques et rejouer la procédure |
| `experiments/` | Exécuter l'étude ou sa validation |
| `audit/` | Relire les résultats enregistrés |
| `figures.py` | Produire les figures et tableaux de l'article |

La [bibliographie](docs/research/README.md), les
[amendements du protocole](journal/01_individual_arousal.md) et les empreintes
des exécutions permettent de distinguer les choix antérieurs au test des résultats.
