# Decision Tree model artifacts
This folder contains the trained Decision Tree model and related metadata.

Files created:
- model.joblib: trained Decision Tree model (joblib)
- scaler.joblib: StandardScaler used for preprocessing
- features.txt: full original feature list
- used_features.txt: features selected by mutual information and model metadata
- readme.md: this file

Model summary:
- Model type: Decision Tree
- Best hyperparameters: {'max_depth': 14, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy'}
- Final accuracy: 0.9998959498478267
