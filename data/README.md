# Données

Dataset public **Credit Risk Dataset** (Kaggle, `laotse/credit-risk-dataset`) — 32 581 prêts, 12 variables.

```bash
# Option 1 : CLI Kaggle (nécessite ~/.kaggle/kaggle.json, à ne jamais committer)
kaggle datasets download -d laotse/credit-risk-dataset -p data/raw --unzip

# Option 2 : téléchargement manuel depuis
# https://www.kaggle.com/datasets/laotse/credit-risk-dataset  -> data/raw/credit_risk_dataset.csv
```

Empreinte de contrôle (vérifiée automatiquement au chargement) :
`sha256 = ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8`

Licence : **CC0 : domaine public** (indiquée sur la page Kaggle du dataset, auteur « Lao Tse »).
Le fichier n'est pas versionné (`.gitignore`) pour garder le dépôt léger : on documente comment l'obtenir.

| Variable | Description |
|---|---|
| person_age | Âge |
| person_income | Revenu annuel |
| person_home_ownership | Statut d'occupation (RENT, MORTGAGE, OWN, OTHER) |
| person_emp_length | Ancienneté professionnelle (années) |
| loan_intent | Objet du prêt |
| loan_grade | Note interne du prêteur (A → G) |
| loan_amnt | Montant du prêt |
| loan_int_rate | Taux d'intérêt |
| loan_status | **Cible** : 1 = défaut, 0 = remboursé |
| loan_percent_income | Montant / revenu |
| cb_person_default_on_file | Défaut antérieur au bureau de crédit (Y/N) |
| cb_person_cred_hist_length | Ancienneté de l'historique de crédit (années) |
