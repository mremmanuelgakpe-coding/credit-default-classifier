# Fiche modèle — Credit Default Classifier v1.0.0

| Élément | Détail |
|---|---|
| Type | Classification binaire, probabilité de défaut (PD) |
| Algorithme | LightGBM (600 arbres, learning rate 0,05, 31 feuilles) dans un pipeline scikit-learn |
| Variables | 11 (variante « complète ») ou 9 (variante « octroi », sans `loan_grade` ni `loan_int_rate`) |
| Données | Kaggle *Credit Risk Dataset*, 32 411 prêts après nettoyage, 21,9 % de défauts |
| Découpage | 80 % entraînement (CV stratifiée 5 plis) / 20 % test, graine 42 |
| Seuil de décision | 0,14 (coût FN = 5, coût FP = 1, choisi hors test) |
| Traçabilité | Hash SHA-256 des données et du modèle dans `models/model_full.meta.json` |

## Performances (test, n = 6 483)

| Variante | AUC | Gini | KS | PR-AUC | Brier |
|---|---|---|---|---|---|
| Complète | 0,950 | 0,899 | 0,766 | 0,909 | 0,050 |
| Octroi | 0,908 | 0,815 | 0,655 | 0,827 | 0,080 |

## Usage prévu
Démonstration pédagogique et de portfolio d'un score d'octroi de crédit à la consommation. **Pas destiné à une décision réelle** sans validation sur des données internes, validation *out-of-time* et revue par une fonction de validation des modèles indépendante.

## Limites et risques
- Données sans horodatage : stabilité dans le temps non évaluée.
- Pas de variables protégées dans le jeu, mais `person_age` est utilisée : l'usage de l'âge dans une décision de crédit doit être justifié et encadré (non-discrimination).
- Un modèle de boosting est moins directement explicable qu'une grille de score logistique : chaque prédiction de l'API est donc accompagnée de ses motifs (valeurs SHAP exactes, TreeSHAP). Ces motifs expliquent le modèle, pas une causalité.
- Les classes de risque A → E de l'API sont des seuils illustratifs, non calibrés sur un appétit au risque.

## Réentraînement
`make train-all` — régénère modèles, métadonnées, métriques et figures de façon reproductible.
