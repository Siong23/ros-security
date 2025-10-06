# AC-MI-DT Model (Experimental)

## Decision Tree Classifier with Mutual Information Feature Selection

> **Status**: ✅ **Experimental/Research Model** - This model serves as a research baseline. For production IDS deployment, use the deep learning models (AC-CNN-LSTM or AC-CNN-MHA).

This folder contains the trained Decision Tree model and related metadata.

Files created:
- model.joblib: trained Decision Tree model (joblib)
- scaler.joblib: StandardScaler used for preprocessing
- features.txt: full original feature list
- used_features.txt: features selected by mutual information and model metadata
- readme.md: this file

### Model Details:
- **Algorithm**: Decision Tree Classifier
- **Feature Selection**: Mutual Information based selection
- **Hyperparameter Optimization**: Optuna-based tuning
- **Best Parameters**: 
  - max_depth: 14
  - min_samples_split: 2  
  - min_samples_leaf: 2
  - criterion: entropy

### Performance:
- **Research Accuracy**: High performance on training data
- **Use Case**: Feature importance analysis and interpretability studies

### Recommended Usage:
- **Research and Analysis**: Understanding decision boundaries and feature splits
- **Interpretability Studies**: Clear visualization of decision paths
- **Educational Purposes**: Demonstrating tree-based classification
- **Feature Analysis**: Understanding which features are most discriminative

### Comparison with Primary IDS Models:
- **vs AC-CNN-LSTM**: Simpler interpretable model but lacks sequence modeling capability
- **vs AC-CNN-MHA**: No attention mechanism or deep feature extraction
- **Interpretability**: Higher than deep learning models but lower real-world robustness

> **For Production Deployment**: Use AC-CNN-LSTM or AC-CNN-MHA models for better generalization and real-world performance.
