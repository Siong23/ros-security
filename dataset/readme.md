# Dataset Directory

This directory contains the datasets and trained machine learning models for the ROS Security Intrusion Detection System (IDS) project.

## Overview

The dataset is used to train and evaluate various machine learning models for detecting security intrusions in Robot Operating System (ROS) based autonomous robot navigation systems. All models have been successfully trained and are ready for deployment.

## Directory Structure

```
dataset/
├── existing/               # Original dataset files and references
├── models/                # Trained machine learning models and experiments
└── readme.md            # This file
```

## Dataset Source

The primary dataset used in this project is **NavBot25.csv**, which contains network traffic data from ROS-based autonomous robot navigation systems.

- **Source**: [Zenodo Dataset Repository](https://zenodo.org/records/16758080)
- **Related GitHub**: [INTRUSION-DETECTION-FOR-ROBOT-OPERATING-SYSTEM-BASED-AUTONOMOUS-ROBOT-NAVIGATION](https://github.com/Blackthorn23/INTRUSION-DETECTION-FOR-ROBOT-OPERATING-SYSTEM-BASED-AUTONOMOUS-ROBOT-NAVIGATION-)
- **Content**: Contains both raw PCAP files and processed CSV files
- **Note**: Large files are not included in this repository due to size constraints

## Machine Learning Models

The `models/` directory contains different approaches for intrusion detection, with the deep learning models being the primary working IDS systems:

### Primary IDS Models (Deep Learning)

### 1. AC-CNN-LSTM (Deep Learning Hybrid)
- **Algorithm**: Convolutional Neural Network + Long Short-Term Memory
- **Architecture**: Hybrid deep learning approach with feature fusion
- **Performance**: 98.78% test accuracy, 99.41% CV mean accuracy
- **Status**: ✅ **Primary Working IDS**

### 2. AC-CNN-MHA (CNN + Multi-Head Attention)
- **Algorithm**: CNN + Multi-Head Attention + ML Pipeline
- **Architecture**: CNN-MHA → PCA → Ensemble (KNN+RF) → Logistic Regression
- **Features**: 78 → 64 (CNN-MHA) → 16 (PCA) → Final prediction
- **Performance**: 98.10% ML pipeline accuracy
- **Attack Classes**: 8 categories supported
- **Status**: ✅ **Primary Working IDS**

### Experimental Models (Traditional ML)

### 3. AC-MI-RF (Random Forest)
- **Algorithm**: Random Forest Classifier
- **Feature Selection**: Mutual Information (MI ≥ 0.01)
- **Features**: 63 selected features
- **Data Balancing**: SMOTE
- **Hyperparameter Optimization**: Optuna (100 trials)
- **Status**: ✅ **Experimental/Research**

### 4. AC-MI-KNN (K-Nearest Neighbors)
- **Algorithm**: K-Nearest Neighbors Classifier
- **Feature Selection**: Mutual Information (MI ≥ 0.01)
- **Features**: 62 selected features
- **Best Parameters**: n_neighbors=2, weights=distance, algorithm=ball_tree
- **Status**: ✅ **Experimental/Research**

### 5. AC-MI-DT (Decision Tree)
- **Algorithm**: Decision Tree Classifier
- **Feature Selection**: Mutual Information
- **Best Parameters**: max_depth=14, criterion=entropy
- **Status**: ✅ **Experimental/Research**

## Files Description

### Model Artifacts
Each model directory contains:
- `model.joblib` - Trained model file (traditional ML models)
- `*.keras` - Keras model files (deep learning models)
- `scaler.joblib` - Feature scaling transformer
- `features.txt` - Complete list of available features
- `used_features.txt` - Features selected for the specific model
- `readme.md` - Model-specific documentation and performance metrics
- `*.ipynb` - Jupyter notebooks used for training and evaluation
- `pipeline_info.json` - Model pipeline configuration (CNN-MHA)

## Usage

### Loading and Using Models

#### Traditional ML Models (RF, KNN, DT):
```python
from joblib import load
import pandas as pd

# Load model and preprocessor
model = load('path/to/model.joblib')
scaler = load('path/to/scaler.joblib')

# Load feature list
with open('path/to/used_features.txt', 'r') as f:
    features = [line.strip() for line in f]

# Preprocess and predict
X_scaled = scaler.transform(X[features])
predictions = model.predict(X_scaled)
```

#### Deep Learning Models (CNN-LSTM, CNN-MHA):
```python
import tensorflow as tf
from joblib import load

# For CNN-MHA
cnn_mha_model = tf.keras.models.load_model('cnn_mha_model.keras')
feature_extractor = tf.keras.models.load_model('feature_extractor.keras')
final_classifier = load('lr_classifier.joblib')
```

## Data Processing Pipeline

1. **Raw Data**: PCAP files from ROS network traffic
2. **Feature Extraction**: Network flow features extraction
3. **Preprocessing**: Data cleaning and normalization
4. **Feature Selection**: Mutual Information or other selection methods
5. **Data Balancing**: SMOTE for handling class imbalance
6. **Model Training**: Various ML/DL algorithms
7. **Evaluation**: Performance metrics and validation

## Performance Summary

Models have been trained and evaluated on the NavBot25 dataset:

### Primary Working IDS Systems
| Model | Performance | Architecture | Status |
|-------|-------------|--------------|--------|
| **AC-CNN-LSTM** | **98.78% test accuracy** | CNN + LSTM hybrid | ✅ **Primary IDS** |
| **AC-CNN-MHA** | **98.10% pipeline accuracy** | CNN-MHA + ML ensemble | ✅ **Primary IDS** |

### Experimental/Research Models
| Model | Features | Optimization | Status |
|-------|----------|--------------|--------|
| AC-MI-RF | 63 (MI ≥ 0.01) | Optuna (100 trials) | ✅ Research |
| AC-MI-KNN | 62 (MI ≥ 0.01) | Optuna (20 trials) | ✅ Research |
| AC-MI-DT | MI selected | Optuna | ✅ Research |

### Attack Detection Capabilities
All models can detect 8 types of security attacks:
- Normal Traffic (33.3% of dataset)
- DoS Attack (15.90%)
- Reverse Shell (15.40%)
- Port Scanning Attack (14.30%)
- UnauthSub Attack
- SSH Bruteforce
- Pubflood
- Subflood

## Requirements

### Core Dependencies
- Python 3.x
- pandas
- scikit-learn
- joblib
- numpy

### Deep Learning Models
- tensorflow/keras (for CNN-LSTM and CNN-MHA models)

### Data Processing
- imbalanced-learn (SMOTE for data balancing)
- optuna (hyperparameter optimization)

## Contributing

When adding new models:
1. Create a new subdirectory in `models/`
2. Include all required artifacts (model, scaler, features)
3. Add comprehensive documentation in a README.md
4. Update this main README with model information

## License

Please refer to the main project license for usage terms.