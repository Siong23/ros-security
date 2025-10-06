# AC-MI-KNN Model (Experimental)

## K-Nearest Neighbors with Mutual Information Feature Selection

> **Status**: ✅ **Experimental/Research Model** - This model serves as a research baseline. For production IDS deployment, use the deep learning models (AC-CNN-LSTM or AC-CNN-MHA).

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
- `scaler.joblib`: StandardScaler for feature preprocessing
- `features.txt`: List of selected features
- `used_features.txt`: Detailed feature usage information
- `acmiknn.ipynb`: Training notebook

### Recommended Usage:
- **Research and Analysis**: Baseline comparison for deep learning models
- **Feature Similarity Studies**: Understanding distance-based classification
- **Educational Purposes**: Demonstrating traditional ML approaches to cybersecurity
- **Quick Prototyping**: Fast implementation for initial testing

### Comparison with Primary IDS Models:
- **vs AC-CNN-LSTM**: Simpler architecture but lacks temporal sequence modeling
- **vs AC-CNN-MHA**: No attention mechanism for feature importance weighting
- **Performance**: Good for research but deep learning models recommended for production

> **For Production Deployment**: Use AC-CNN-LSTM or AC-CNN-MHA models for superior accuracy and real-world performance.
