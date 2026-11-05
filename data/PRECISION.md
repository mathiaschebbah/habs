# Précision attendue et seuil d'intérêt

Le nombre pertinent est celui des personnes de test, pas le nombre de fenêtres. Le plan proposé sur les 75 identifiants admissibles avant contrôle qualité réserve 25 personnes au test. Le critère principal est une différence appariée d'erreur quadratique centrée, en points², par personne.

Le [calcul reproductible](planning.py) produit [80 scénarios](precision_scenarios.csv) pour 20, 25, 30, 50 ou 75 personnes de test, quatre dispersions interpersonnelles supposées et quatre effets moyens. Seul le scénario à 25 personnes correspond à la partition envisagée actuellement ; les autres servent à comprendre la sensibilité à la taille. Aucune dispersion des gains HABS n'a encore été estimée.

## Illustration pour 25 personnes de test

| Écart-type supposé des gains individuels | Demi-largeur normale de l'IC à 95 % | Gain détectable approximatif à 80 % | Puissance normale pour un gain de 0,10 |
|---:|---:|---:|---:|
| 0,10 | 0,039 | 0,056 | 99,9 % |
| 0,25 | 0,098 | 0,140 | 51,6 % |
| 0,50 | 0,196 | 0,280 | 17,0 % |
| 1,00 | 0,392 | 0,560 | 8,0 % |

Le tableau utilise un test normal bilatéral à 5 %, une variance supposée connue et des personnes indépendantes, conditionnellement au modèle et au catalogue. L'effet détectable est l'approximation `(z_0.975 + z_0.80) × sigma / sqrt(n)`. Les limites liées à l'estimation de sigma et au bootstrap sur un petit nombre de personnes ne sont pas simulées ici. Le calcul ne garantit donc pas la puissance du futur test.

Le seuil proposé de 0,10 point² est un choix opérationnel, pas un effet minimum validé par la littérature. Il doit être confirmé avant l'évaluation finale. Si la dispersion est de 0,25 point², 25 personnes ne suffisent pas à assurer 80 % de puissance pour ce seuil. Accumuler plus de fenêtres chez les mêmes personnes ne supprime pas ce problème.

## Simulation à réaliser avant le test définitif

Le protocole prévoit une simulation du plan complet : matrice personne–vidéo observée, différences de niveau et de pente, dérive de session, sélection des stimuli, pertes de signal et même sélection de modèles. Les scénarios nuls « identité seule » et « stimulus seul » doivent être inclus. Il faut estimer le taux de faux positifs, la couverture des intervalles et la puissance avec leur erreur Monte-Carlo.

Cette simulation nécessite la chaîne d'extraction et les modèles définitifs. Elle n'a pas été exécutée dans la préparation ; aucun résultat de simulation complète ni aucune puissance « atteinte » ne sont revendiqués.

## Reproduction

Depuis la racine du dépôt, avec Python 3.9 ou supérieur :

```bash
python3 data/planning.py data/precision_scenarios.csv
python3 -m unittest discover -s tests -v
```

Le test d'invariance vérifie qu'une correction constante propre à la personne ne peut pas améliorer le critère centré. Un autre test vérifie les exclusions de jointure sur une petite structure synthétique.
