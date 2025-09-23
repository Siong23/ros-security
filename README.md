# 🤖 ROS-Guard: Machine Learning-based Intrusion Detection for Autonomous Robots

� **ROS-Guard** is a research-driven project that develops a real-time, ML-powered **Intrusion Detection System (IDS)** specifically designed for ROS-based autonomous navigation systems. It enables comprehensive cybersecurity protection through network traffic analysis, attack detection, and provides an open-source dataset collected from real robotic systems for the research community.

As autonomous systems become increasingly prevalent across sectors such as healthcare, finance, manufacturing, and smart infrastructure, cybersecurity is emerging as a critical concern. Robotic platforms, particularly those built on the **Robot Operating System (ROS)**, remain under-protected, exposing them to threats like unauthorized access, data tampering, and network-based attacks.

---

## 🧠 Key Features

- **Real-time Anomaly Detection** for ROS-based autonomous systems
- **ML-powered Classification** using Random Forest, Decision Trees, and KNN algorithms
- **Comprehensive Attack Simulation** including DoS, MitM, unauthorized access, and topic flooding
- **Open Dataset Generation** from real-world robot traffic with labeled attack scenarios
- **Multi-algorithm Support** with performance comparison and optimization
- **Feature Engineering Pipeline** using 76+ network traffic characteristics
- Support for **ROS 1** environments (ROS 2 compatibility planned)

---

## 🗝️ Architecture

```
                    +----------------------------------------+
                    |           ROS-Guard IDS System          |
                    |     (Real-time Security Monitoring)     |
                    +------------------+---------------------+
                                       |
    +----------------------------------+----------------------------------+
    |                                                                     |
    |                                                                     |
+----+------------+                                      +---------------+-+
|  Physical Robot |                                      |   IDS Engine    |
| (TurtleBot3 +   |                                      | (ML-Powered)    |
|  ROS Noetic)    |                                      |                 |
+-----------------+                                      +-----------------+
|                 |              +-------------+         |                 |
| +-------------+ |------------->| PCAP Capture|-------->| +-------------+ |
| | ROS Topics  | |              | (tcpdump/   |         | | Feature     | |
| | /cmd_vel    | |              | Wireshark)  |         | | Extractor   | |
| | /scan       | |              +-------------+         | | (76 features)| |
| | /odom       | |                                      | +-------------+ |
| +-------------+ |                                      |                 |
|                 |                                      | +-------------+ |
| +-------------+ |                                      | | ML Models   | |
| | Navigation  | |                                      | | • Random    | |
| | Stack       | |                                      | |   Forest    | |
| | (SLAM)      | |                                      | | • Decision  | |
| +-------------+ |                                      | |   Tree      | |
+-----------------+                                      | | • KNN       | |
                                                         | +-------------+ |
                                                         |                 |
                                                         | +-------------+ |
                                                         | | Alert &     | |
                                                         | | Response    | |
                                                         | | System      | |
                                                         | +-------------+ |
                                                         +-----------------+
```

**Attack Simulation Layer:**
- **DoS Attacks**: Command velocity flooding
- **MitM Attacks**: Message manipulation and interception  
- **Unauthorized Access**: Fake publishers and subscribers
- **Topic Flooding**: High-frequency malicious data injection

---

## � Components

| Component              | Description                                                    |
|------------------------|----------------------------------------------------------------|
| **TurtleBot3**         | Physical robot platform for real-world data collection        |
| **ROS Noetic**         | Robot Operating System for navigation and communication       |
| **CICFlowMeter**       | Network traffic feature extraction (76 features)             |
| **scikit-learn**       | Machine learning framework for model training                 |
| **PCAP Tools**         | Wireshark/tcpdump for network packet capture                  |
| **Attack Scripts**     | Custom Python scripts for simulating cyber attacks           |
| **SMOTE**              | Synthetic minority oversampling for balanced datasets         |

---

## 📁 Repository Structure

```
ros-security/
├── README.md                    # Project overview and documentation
├── dataset/                     # Dataset and validation tools
│   ├── datasetvalidation.ipynb # Data preprocessing and validation
│   ├── test.py                  # Model testing and prediction script
│   ├── existing/                # External dataset references
│   └── models/                  # Trained ML models and notebooks
│       ├── ac-mi-rf/           # Random Forest implementation
│       │   ├── acmirf.ipynb    # Complete RF training pipeline
│       │   ├── model.joblib    # Trained Random Forest model
│       │   ├── scaler.joblib   # Data preprocessing scaler
│       │   └── features.txt    # Feature definitions (76 features)
│       ├── navbot25_ac_mi_knn.ipynb  # K-Nearest Neighbors model
│       ├── navbot25-ac-mi-dt.ipynb   # Decision Tree model
│       └── navbot25-ac-mi-rf.ipynb   # Random Forest model
├── doc/                         # Documentation and setup guides
│   ├── README.md               # CICFlowMeter installation guide
│   └── ROS TB3 and RGBD.pdf    # Hardware setup documentation
└── milestones/                  # Project progress and milestones
    └── readme.md               # Month 1 achievements and roadmap
```

---

## 📦 Getting Started

### Prerequisites
- Python 3.8+
- ROS Noetic (for data collection & replay)
- Ubuntu 20.04 LTS (recommended)
- scikit-learn, pandas, numpy, matplotlib
- Wireshark / tcpdump (for packet capture)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Siong23/ros-security.git
   cd ros-security
   ```

2. **Install Python Dependencies**
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn joblib imbalanced-learn
   ```

3. **Set Up ROS Environment**
   ```bash
   # Install ROS Noetic (if not already installed)
   sudo apt update
   sudo apt install ros-noetic-desktop-full
   
   # Install TurtleBot3 packages
   sudo apt install ros-noetic-turtlebot3-*
   ```

4. **Configure CICFlowMeter** (for feature extraction)
   ```bash
   # Follow the detailed installation guide: doc/README.md
   sudo apt install -y openjdk-11-jdk maven unzip libpcap0.8 libpcap-dev
   git clone https://github.com/ahlashkari/CICFlowMeter.git
   ```
   
   > 📖 **For complete CICFlowMeter installation guide, see:** [doc/README.md](./doc/README.md)

### Quick Start

1. **Set Up Robot Navigation**
   ```bash
   # Export TurtleBot3 model
   export TURTLEBOT3_MODEL=burger
   
   # Launch navigation stack
   roslaunch turtlebot3_navigation turtlebot3_navigation.launch
   ```

2. **Capture Network Traffic**
   ```bash
   # Start packet capture (replace with your network interface)
   sudo tcpdump -i wlp0s20f3 -w robot_traffic.pcap
   ```

3. **Run Attack Simulations**
   ```bash
   # Example: DoS attack simulation
   python attack_scripts/dos_attack.py
   ```

4. **Extract Features and Run Model**
   ```bash
   # Feature extraction using CICFlowMeter (CLI method)
   sudo cicflowmeter -i wlp0s20f3 -c /home/jakelcj/output.csv
   
   # Verify the output CSV file
   head -n 5 /home/jakelcj/output.csv
   
   # Run prediction with trained model
   cd dataset
   python test.py
   ```
   
   > **IDS Operation**: The system runs CICFlowMeter through CLI to extract network features in real-time, then processes them through the trained ML models for intrusion detection.

---

## 🧪 Use Cases

- **Real-time Intrusion Detection** for autonomous robot deployments
- **Cybersecurity Research** with comprehensive attack scenario datasets  
- **Network Traffic Analysis** for ROS-based communication patterns
- **AI/ML Security Testing** for robotic system protection
- **Academic Research** with open-source datasets and reproducible results
- **Industrial Robot Protection** in manufacturing and healthcare environments

---

## ⚙️ How the IDS Operates

The ROS-Guard IDS operates through a streamlined CLI-based workflow using CICFlowMeter for real-time network traffic analysis:

### Real-time Detection Process

1. **Network Traffic Capture**
   ```bash
   # Run CICFlowMeter through CLI to monitor network interface
   sudo cicflowmeter -i wlp0s20f3 -c /home/jakelcj/output.csv
   ```

2. **Feature Extraction**
   - CICFlowMeter extracts **76 network traffic features** in real-time
   - Features include flow duration, packet sizes, inter-arrival times, protocol flags
   
3. **Verification and Monitoring**
   ```bash
   # Check the extracted features
   head -n 5 /home/jakelcj/output.csv
   ```

4. **ML-based Classification**
   - Processed features are fed into trained Random Forest model
   - Real-time anomaly detection with ~95% accuracy
   - Alerts generated for suspicious network behavior

### Command Reference
```bash
# General syntax
sudo cicflowmeter -i [interface] -c [directory/output.csv]

# Example with specific network interface
sudo cicflowmeter -i wlp0s20f3 -c /home/jakelcj/output.csv
```

> 🔗 **Installation Guide**: For complete CICFlowMeter setup instructions, see [doc/README.md](./doc/README.md)

---

## 📊 Dataset & Models

### Machine Learning Models

| Model Type           | Status      | Accuracy | Features Used | Implementation |
|---------------------|-------------|----------|---------------|----------------|
| **Random Forest**   | ✅ Complete | ~95%     | 63/76         | `ac-mi-rf/`    |
| **Decision Tree**   | 🔄 Training | TBD      | 76            | `navbot25-ac-mi-dt.ipynb` |
| **K-Nearest Neighbors** | 🔄 Training | TBD   | 76            | `navbot25_ac_mi_knn.ipynb` |

### Feature Engineering
- **76 Network Traffic Features** extracted using CICFlowMeter
- **Statistical Analysis**: Flow duration, packet sizes, inter-arrival times
- **Protocol Features**: TCP flags, header lengths, window sizes  
- **Behavioral Patterns**: Active/idle times, up/down ratios

---

## 📚 Documentation

See the project folders for detailed information:

- **[doc/README.md](./doc/README.md)** - CICFlowMeter installation and setup guide
- **[milestones/readme.md](./milestones/readme.md)** - Project progress and achievements  
- **[dataset/models/](./dataset/models/)** - ML model implementations and notebooks
- **[dataset/existing/](./dataset/existing/)** - External dataset references and links

### Research Papers & References
- **External Dataset**: [Zenodo Repository](https://zenodo.org/records/16758080)
- **Related Work**: [INTRUSION-DETECTION-FOR-ROBOT-OPERATING-SYSTEM](https://github.com/Blackthorn23/INTRUSION-DETECTION-FOR-ROBOT-OPERATING-SYSTEM-BASED-AUTONOMOUS-ROBOT-NAVIGATION-)

---

## 🎯 Project Objectives & Impact

- 🔍 **Identify security gaps** in ROS authentication, data integrity, and communication protocols
- 📡 **Generate comprehensive datasets** from real robot traffic under attack conditions  
- 🧠 **Develop ML-based IDS** for real-time suspicious behavior detection
- 📊 **Contribute to research community** with open-source datasets and reproducible methods
- 🔐 **Enable robust protection** for networked robotic systems in critical applications

---

## 🚀 Future Development

- 🔄 **ROS 2 Compatibility** - Extend support to next-generation ROS
- 🤖 **Multi-Robot Systems** - Scale detection for robot swarms and fleets
- 🧠 **Advanced ML Models** - LSTM/GRU for time-series anomaly detection  
- 🔒 **Real-time Response** - Automated countermeasures and threat mitigation
- 🌐 **Edge Deployment** - Lightweight models for embedded robot systems

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ZTE-MMU Training Center** for this project opportunity
- **TurtleBot3** team for excellent robotic platform
- **ROS Community** for comprehensive robotics framework
- **CICFlowMeter** developers for network traffic analysis tools
- **Open5GS & UERANSIM** projects for inspiration in network simulation approaches
