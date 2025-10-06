# AC-MI-RF Model (Experimental)

## Random Forest Classifier with Mutual Information Feature Selection

> **Status**: ✅ **Experimental/Research Model** - This model serves as a research baseline. For production IDS deployment, use the deep learning models (AC-CNN-LSTM or AC-CNN-MHA).

### Model Details:
- **Algorithm**: Random Forest Classifier  
- **Features**: 63 selected using Mutual Information (MI ≥ 0.01)
- **Data Balancing**: SMOTE (Synthetic Minority Oversampling Technique)
- **Scaling**: StandardScaler
- **Hyperparameter Optimization**: Optuna (100 trials)

### Best Hyperparameters:
- **n_estimators**: 300
- **max_depth**: None
- **min_samples_split**: 2
- **min_samples_leaf**: 1
- **criterion**: gini
- **max_features**: sqrt

### Performance Metrics:
- **Accuracy**: 99.99%
- **Precision**: 99.99%
- **Recall**: 99.99%
- **F1-score**: 99.99%

### Attack Detection Results:
- **Normal Traffic**: 33.3%
- **Attack Traffic**: 66.7%
- **Top Attacks**: DoS Attack (15.90%), Reverse Shell (15.40%), Port Scanning (14.30%)

### Files:
- `model.joblib`: Trained Random Forest model
- `scaler.joblib`: StandardScaler for feature preprocessing
- `features.txt`: List of selected features (63 features)
- `used_features.txt`: Detailed feature usage information
- `acmirf.ipynb`: Training notebook
- `datasetvalidation.ipynb`: Model validation and testing

### Dataset Information:
- **Source**: NavBot25.csv (ROS Security Dataset)
- **Total Samples**: 192,213
- **Attack Types**: 8 categories (Normal, DoS Attack, UnauthSub Attack, SSH Bruteforce, Pubflood, Subflood, Reverse Shell, Port Scanning Attack)
- **Feature Selection**: Mutual Information threshold ≥ 0.01
- **Validation Sample**: 1,000 samples for performance testing

### Usage:
1. Load the model: `model = joblib.load('model.joblib')`
2. Load the scaler: `scaler = joblib.load('scaler.joblib')`
3. Load features: Read `features.txt` for feature list
4. Preprocess data using the scaler
5. Predict using the loaded model

### Model Architecture:
- **Type**: Ensemble of 300 Decision Trees
- **Max Depth**: Unlimited (None)
- **Splitting Criterion**: Gini impurity
- **Feature Selection**: Top 63 features based on mutual information
- **Training Data**: SMOTE-balanced dataset with equal class distribution
- **Bootstrap**: True (random sampling with replacement)

### Comparison with Other Models:
- **vs Deep Learning Models (Primary IDS)**: Lower complexity but less sophisticated feature extraction
- **vs Decision Tree**: Similar feature selection but more robust due to ensemble approach  
- **vs k-NN**: Faster prediction time, better interpretability
- **Attack Detection**: Consistent ~67% attack rate across all traditional ML models

### Recommended Usage:
- **Research and Analysis**: Baseline comparison for deep learning models
- **Feature Importance Studies**: Understanding mutual information-based feature selection
- **Educational Purposes**: Demonstrating traditional ML approaches to cybersecurity
- **Resource-Constrained Environments**: When deep learning models are not feasible

> **For Production Deployment**: Use AC-CNN-LSTM or AC-CNN-MHA models for better accuracy and real-world performance.
