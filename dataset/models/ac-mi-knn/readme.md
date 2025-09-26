# AC-MI-KNN Model

## K-Nearest Neighbors with Mutual Information Feature Selection

### Model Details:
- **Algorithm**: K-Nearest Neighbors Classifier
- **Features**: 62 selected using Mutual Information (MI ≥ 0.01)
- **Data Balancing**: SMOTE (Synthetic Minority Oversampling Technique)
- **Scaling**: StandardScaler
- **Hyperparameter Optimization**: Optuna (20 trials)

### Best Hyperparameters:
- **n_neighbors**: 2
- **weights**: distance
- **algorithm**: ball_tree
- **p**: 1

### Performance Metrics:
- **Accuracy**: 99.15%
- **Precision**: 99.15%
- **Recall**: 99.15%
- **F1-score**: 99.15%

### Files:
- `model.joblib`: Trained KNN model
- `scaler.joblib`: StandardScaler for feature preprocessing
- `features.txt`: List of selected features
- `used_features.txt`: Detailed feature usage information
- `acmiknn.ipynb`: Training notebook
