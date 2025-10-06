# 🛡️ ROS-Guard IDS Nodes

This directory contains the primary intrusion detection system nodes for real-time security monitoring of ROS-based autonomous systems.

## 🎯 Primary IDS Nodes

### 🧠 CNN-LSTM IDS Node (`cnn_lstm_ids_node_linux.py`)

**Deep Learning Hybrid Architecture for Sequential Anomaly Detection**

#### Features:
- **Architecture**: Convolutional Neural Network + Long Short-Term Memory
- **Performance**: 98.78% test accuracy, 99.41% CV mean accuracy
- **Pipeline**: CNN-LSTM → Feature Extraction → KNN+RF Fusion → Logistic Regression
- **Specialization**: Temporal sequence analysis for time-series attack patterns

#### Usage:
```bash
rosrun ros_ids cnn_lstm_ids_node_linux.py
```

#### Model Components Required:
- `feature_extractor.keras` - CNN+LSTM feature extraction model
- `scaler.joblib` - Feature preprocessing scaler
- `pca.joblib` - Principal Component Analysis transformer
- `knn_classifier.joblib` - K-Nearest Neighbors classifier
- `rf_classifier.joblib` - Random Forest classifier  
- `lr_classifier.joblib` - Logistic Regression final classifier
- `features.txt` - Feature names and definitions
- `class_mapping.json` - Attack class mappings
- `pipeline_info.json` - Model pipeline configuration

#### Model Paths:
- Primary: `/home/jakelcj/ids_ws/src/ros_ids/models/cnnlstm2`
- Alternative: `~/ids_ws/src/ros_ids/models/cnnlstm2`
- Local: `./cnnlstm2`

---

### 🎯 CNN-MHA IDS Node (`cnn_mha_ids_node_linux.py`)

**Multi-Head Attention Architecture for Complex Attack Classification**

#### Features:
- **Architecture**: CNN + Multi-Head Attention + ML Pipeline
- **Performance**: 98.10% ML pipeline accuracy
- **Pipeline**: CNN-MHA (78→64) → PCA (64→16) → Ensemble → Final Classification
- **Specialization**: Attention-based feature importance weighting for multi-class detection

#### Usage:
```bash
rosrun ros_ids cnn_mha_ids_node_linux.py
```

#### Model Components Required:
- `cnn_mha_model.keras` - CNN+Multi-Head Attention model
- `feature_extractor.keras` - Feature extraction component
- `scaler.joblib` - Feature preprocessing scaler
- `pca.joblib` - PCA dimensionality reduction (64→16)
- `knn_classifier.joblib` - K-Nearest Neighbors classifier
- `rf_classifier.joblib` - Random Forest classifier
- `lr_classifier.joblib` - Logistic Regression ensemble classifier
- `features.txt` - Feature definitions (78 features)
- `pipeline_info.json` - Complete pipeline configuration

#### Model Paths:
- Primary: `/home/jakelcj/ids_ws/src/ros_ids/models/cnnmha`
- Alternative: `~/ids_ws/src/ros_ids/models/cnnmha`
- Local: `./models/cnnmha`

---

## 🔧 System Requirements

### Dependencies:
```bash
sudo apt update
sudo apt install python3-pip python3-dev
pip3 install rospy std_msgs pandas numpy scikit-learn joblib tensorflow glob2 watchdog
```

### Prerequisites:
1. **ROS Noetic** installed and configured
2. **CICFlowMeter** running and generating CSV output at `/home/jakelcj/output.csv`
3. **Model files** properly placed in designated model directories
4. **Network interface** monitoring active (usually `wlp0s20f3`)

---

## 🚀 Operation Workflow

### 1. Start CICFlowMeter (Feature Extraction)
```bash
cd CICFlowMeter
sudo cicflowmeter -i wlp0s20f3 -c /home/jakelcj/output.csv
```

### 2. Launch IDS Node (Choose One)
```bash
# For sequential pattern analysis
rosrun ros_ids cnn_lstm_ids_node_linux.py

# OR for attention-based multi-class detection  
rosrun ros_ids cnn_mha_ids_node_linux.py
```

### 3. Monitor Detections
```bash
# View real-time alerts
rostopic echo /ids/alerts

# Check detection logs
tail -f ~/ids_logs/cnn_lstm_ids_*.log
# OR
tail -f ~/ids_logs/cnn_mha_ids_*.log
```

---

## 🎭 Attack Detection Capabilities

Both nodes can detect **8 attack categories**:

| Attack Type | Description | Detection Strength |
|-------------|-------------|-------------------|
| **Normal** | Legitimate robot traffic | ✅ Baseline |
| **DoS Attack** | Command velocity flooding | 🔴 High Priority |
| **Pubflood** | Publication flooding attacks | 🟠 Medium Priority |
| **Subflood** | Subscription flooding attacks | 🟠 Medium Priority |
| **UnauthSub Attack** | Unauthorized subscribers | 🟡 Low Priority |
| **SSH Bruteforce** | SSH login attempts | 🔴 High Priority |
| **Reverse Shell** | Remote shell access | 🔴 Critical |
| **Port Scanning** | Network reconnaissance | 🟠 Medium Priority |

---

## 📊 Performance Characteristics

### CNN-LSTM Node:
- **Strength**: Sequential pattern recognition, temporal dependencies
- **Best For**: Time-series attacks, behavioral anomalies
- **Processing**: ~2-3 seconds per detection cycle
- **Memory**: ~500MB RAM usage

### CNN-MHA Node:
- **Strength**: Multi-class classification, feature attention
- **Best For**: Complex multi-vector attacks, simultaneous threats
- **Processing**: ~3-4 seconds per detection cycle  
- **Memory**: ~600MB RAM usage

---

## 🔍 Real-time Monitoring Features

### Both Nodes Provide:
- **🕐 Timestamp Logging**: Precise attack detection times
- **📊 Confidence Scores**: Prediction probability for each attack type
- **📈 Feature Analysis**: Top contributing network features
- **🚨 ROS Topic Alerts**: Real-time notifications via `/ids/alerts`
- **📝 Detailed Logging**: Comprehensive log files with detection history
- **⚡ Live Monitoring**: Continuous CSV file watching with 2Hz frequency

### Log File Locations:
- CNN-LSTM: `~/ids_logs/cnn_lstm_ids_YYYYMMDD_HHMMSS.log`
- CNN-MHA: `~/ids_logs/cnn_mha_ids_YYYYMMDD_HHMMSS.log`

---

## 🛠️ Troubleshooting

### Common Issues:

**Model Files Not Found:**
```bash
# Verify model directory exists
ls -la /home/jakelcj/ids_ws/src/ros_ids/models/
# Check specific model files
ls -la /home/jakelcj/ids_ws/src/ros_ids/models/cnnlstm2/
ls -la /home/jakelcj/ids_ws/src/ros_ids/models/cnnmha/
```

**CSV File Not Found:**
```bash
# Check CICFlowMeter output
head -n 5 /home/jakelcj/output.csv
# Verify CICFlowMeter is running
ps aux | grep cicflowmeter
```

**ROS Connection Issues:**
```bash
# Verify ROS master is running
rostopic list
# Check node status
rosnode list | grep ids
```

**Memory Issues:**
```bash
# Monitor system resources
htop
# Check available memory
free -h
```

---

## 🎯 Model Selection Guide

### Use CNN-LSTM When:
- ✅ Analyzing **temporal attack patterns**
- ✅ Detecting **sequence-based anomalies** (e.g., sustained DoS)
- ✅ Monitoring **behavioral changes** over time
- ✅ Need **highest accuracy** (98.78%)

### Use CNN-MHA When:
- ✅ Handling **multiple simultaneous attacks**
- ✅ Requiring **detailed feature attention** analysis
- ✅ Need **multi-class classification** with confidence scores
- ✅ Analyzing **complex attack combinations**

---

## 📈 Integration with Demo Setup

These nodes integrate seamlessly with the demo environment:
- **Master Laptop**: `192.168.0.142` (runs IDS nodes)
- **TurtleBot3**: `192.168.0.141` (generates traffic)
- **Attacker VM**: `192.168.0.181` (simulates attacks)

The IDS nodes monitor network traffic from all sources and provide real-time detection alerts for demonstration purposes.

---

*For complete demo instructions, see `/demo/README.md`* 🎯