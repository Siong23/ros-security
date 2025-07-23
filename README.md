# 🤖 ROS-Guard: Machine Learning-based Intrusion Detection for Autonomous Robots

## 📌 Overview

As autonomous systems become increasingly prevalent across sectors such as healthcare, finance, manufacturing, and smart infrastructure, cybersecurity is emerging as a critical concern. Robotic platforms, particularly those built on the **Robot Operating System (ROS)**, remain under-protected, exposing them to threats like unauthorized access, data tampering, and network-based attacks.

**ROS-Guard** is a research-driven project that aims to develop a real-time, ML-powered **Intrusion Detection System (IDS)** tailored to ROS-based autonomous navigation systems. It not only detects cyber-attacks through network traffic analysis but also contributes to the research community by introducing novel ROS-specific attack scenarios and providing an open-source dataset collected from real robots.

---

## 🎯 Objectives

- 🔍 Identify security gaps in ROS (authentication, data integrity, communication protocols).
- 📡 Capture and label network traffic in ROS environments under benign and attack conditions.
- 🧠 Build a machine learning-based IDS to detect suspicious behaviors in real-time.
- 📊 Share an open dataset for ROS security research.
- 🔐 Enable robust and adaptive protection for networked robotic systems.

---

## 🛠️ Features

- ✅ Real-time anomaly detection for ROS-based systems
- ✅ Dataset generation from real-world robot traffic
- ✅ Support for known attacks (e.g., spoofing, replay, DoS)
- ✅ ML model training and evaluation pipeline
- ✅ ROS 1 support (ROS 2 coming soon)

---

## 🏗️ System Architecture

```
+---------------------+
|   ROS Navigation    |
|   Stack (ROS 1)     |
+---------+-----------+
          |
          v
+---------------------+        +----------------------+
|   Network Traffic   | -----> |   IDS Packet Sniffer |
|   (Benign/Attacks)  |        +----------------------+
+---------------------+                   |
                                          v
                     +----------------------+
                     |  Feature Extractor   |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |   ML Classification  |
                     |   (Anomaly Detector) |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |  Alert & Response    |
                     +----------------------+
```

## 📂 Dataset
- A curated dataset will be provided in this repository, including:
- Network capture files (.pcap)
- Labeled CSVs for model training/testing
- Attack scenario documentation
  💡 The dataset includes traffic from both normal operations and crafted attack scenarios in ROS environments.

## 📦 Installation
Requirements
- Python 3.8+
- ROS Noetic (for data collection & replay)
- scikit-learn, pandas, numpy, matplotlib
- Wireshark / tcpdump (for packet capture)

Setup
```
git clone https://github.com/your-org/ros-guard.git
cd ros-guard
pip install -r requirements.txt
```
🚀 Usage
1. Simulate or run real robot navigation
```
roslaunch turtlebot3_navigation turtlebot3_navigation.launch
```
2. Start packet capture
```
sudo tcpdump -i <robot_interface> -w output.pcap
```
3. Run feature extraction and ML model
```
python scripts/extract_features.py --input output.pcap
python scripts/train_model.py
python scripts/detect_intrusions.py
```

## 📈 ML Models
We evaluate multiple algorithms including:
- Random Forest
- SVM
- LSTM / GRU (for time-series)
- Isolation Forest (for anomaly detection)
- Performance metrics like precision, recall, and F1-score will be provided.
