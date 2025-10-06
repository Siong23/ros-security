# Machine Learning Models for ROS Security IDS

This directory contains five different intrusion detection models trained on the NavBot25 dataset for ROS-based autonomous robot security.

## Primary Working IDS Models (Deep Learning)

### 🧠 AC-CNN-LSTM (Hybrid Deep Learning)
- **Status**: ✅ **Primary Working IDS**
- **Architecture**: Convolutional Neural Network + Long Short-Term Memory
- **Performance**: 98.78% test accuracy, 99.41% CV mean accuracy
- **Use Case**: Real-time sequence-based anomaly detection
- **Files**: `model.joblib`, `scaler.joblib`, `features.txt`, `used_features.txt`

### 🎯 AC-CNN-MHA (CNN + Multi-Head Attention)
- **Status**: ✅ **Primary Working IDS**  
- **Architecture**: CNN-MHA → PCA → Ensemble (KNN+RF) → Logistic Regression
- **Feature Pipeline**: 78 → 64 (CNN-MHA) → 16 (PCA) → Final prediction
- **Performance**: 98.10% ML pipeline accuracy
- **Attack Classes**: 8 categories (Normal, DoS, SSH Bruteforce, etc.)
- **Files**: `cnn_mha_model.keras`, `feature_extractor.keras`, `pipeline_info.json`

## Experimental Models (Traditional ML)

### 🌳 AC-MI-RF (Random Forest)
- **Status**: ✅ **Experimental/Research**
- **Algorithm**: Random Forest with Mutual Information feature selection
- **Features**: 63 selected features (MI ≥ 0.01)
- **Optimization**: Optuna hyperparameter tuning (100 trials)
- **Files**: `model.joblib`, `scaler.joblib`, `acmirf.ipynb`

### 🔍 AC-MI-KNN (K-Nearest Neighbors)
- **Status**: ✅ **Experimental/Research**
- **Algorithm**: KNN with distance-based weighting
- **Features**: 62 selected features (MI ≥ 0.01)
- **Best Parameters**: n_neighbors=2, weights=distance, algorithm=ball_tree
- **Files**: `scaler.joblib`, `features.txt`, `acmiknn.ipynb`

### 📊 AC-MI-DT (Decision Tree)  
- **Status**: ✅ **Experimental/Research**
- **Algorithm**: Decision Tree with entropy criterion
- **Best Parameters**: max_depth=14, criterion=entropy
- **Files**: `model.joblib`, `scaler.joblib`, `acmidt.ipynb`

## Model Usage

### For Primary IDS Models (Deep Learning):
- Use **AC-CNN-LSTM** for real-time sequential anomaly detection
- Use **AC-CNN-MHA** for complex multi-class attack classification

### For Experimental Models:
- Available for research comparison and feature analysis
- Useful for understanding traditional ML performance baselines

## Attack Detection Capabilities

All models can detect the following attack types:
- **Normal Traffic** (33.3% of dataset)
- **DoS Attack** (15.90%)
- **Reverse Shell** (15.40%) 
- **Port Scanning Attack** (14.30%)
- **UnauthSub Attack**
- **SSH Bruteforce**
- **Pubflood**
- **Subflood**

## Next Steps

The deep learning models (AC-CNN-LSTM and AC-CNN-MHA) are the recommended production-ready IDS solutions for ROS-based autonomous systems, while the traditional ML models serve as valuable research baselines and comparison points.
