# 🤖 ROS-Guard: Machine Learning-based Intrusion Detection for Autonomous Robots

🤖 **ROS-Guard** is a research-driven project that develops a real-time, ML-powered **Intrusion Detection System (IDS)** specifically designed for ROS-based autonomous navigation systems. It enables comprehensive cybersecurity protection through network traffic analysis, attack detection, and provides an open-source dataset collected from real robotic systems for the research community.

As autonomous systems become increasingly prevalent across sectors such as healthcare, finance, manufacturing, and smart infrastructure, cybersecurity is emerging as a critical concern. Robotic platforms, particularly those built on the **Robot Operating System (ROS)**, remain under-protected, exposing them to threats like unauthorized access, data tampering, and network-based attacks.

---

## 🧠 Key Features

- **Real-time Anomaly Detection** for ROS-based autonomous systems
- **ML-powered Classification** using Deep Learning (CNN-LSTM, CNN-MHA) and traditional algorithms
- **Comprehensive Attack Simulation** including DoS, MitM, unauthorized access, and topic flooding
- **Open Dataset Generation** from real-world robot traffic with labeled attack scenarios
- **Multi-algorithm Support** with deep learning and traditional ML approaches  
- **Feature Engineering Pipeline** using 78+ network traffic characteristics
- **Real-time ROS Integration** with live threat detection and alerting
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
| | Navigation  | |                                      | | • MHA       | |
| | Stack       | |                                      | | • LSTM      | |
| | (SLAM)      | |                                      | +-------------+ |                                     
+-----------------+                                      | |             | |
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

## 🔧 Components

| Component              | Description                                                    |
|------------------------|----------------------------------------------------------------|
| **TurtleBot3**         | Physical robot platform for real-world data collection        |
| **ROS Noetic**         | Robot Operating System for navigation and communication       |
| **IDS ROS Node**       | Real-time intrusion detection node (`codes/ids_node.py`)      |
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
├── codes/                       # ROS implementation and runtime scripts
│   └── ids_node.py             # Real-time IDS ROS node for live detection
├── dataset/                     # Dataset and trained models
│   ├── existing/                # External dataset references
│   └── models/                  # Trained ML/DL models and notebooks
│       ├── ac-cnn-lstm/        # CNN-LSTM Deep Learning IDS (Primary)
│       │   ├── model.joblib    # Trained hybrid model
│       │   ├── scaler.joblib   # Feature preprocessing
│       │   └── features.txt    # Feature definitions
│       ├── ac-cnn-mha/         # CNN Multi-Head Attention IDS (Primary)  
│       │   ├── cnn_mha_model.keras # Trained CNN-MHA model
│       │   ├── feature_extractor.keras # Feature extraction model
│       │   └── pipeline_info.json # Model pipeline configuration
│       ├── ac-mi-rf/           # Random Forest (Experimental)
│       ├── ac-mi-knn/          # K-Nearest Neighbors (Experimental)
│       └── ac-mi-dt/           # Decision Tree (Experimental)
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

   **Method A: CLI (Real-time)**
   ```bash
   # Feature extraction using CICFlowMeter CLI for live traffic
   sudo cicflowmeter -i wlp0s20f3 -c /home/jakelcj/output.csv
   
   # Verify the output CSV file
   head -n 5 /home/jakelcj/output.csv
   ```

   **Method B: GUI (Offline PCAP Analysis)**
   ```bash
   # Launch CICFlowMeter GUI
   sudo ./gradlew execute
   ```
   
   **GUI Steps:**
   1. Click **"Network"** button in the interface
   2. Press **"Offline"** option 
   3. **Browse PCAP file**: Click browse button next to "PCAP Dir" and select your `.pcap` file
   4. **Set Output directory**: Click browse button next to "Output Dir" to choose where the `.csv` file will be saved
   5. **Configure timeouts**:
      - Flow Timeout: `120000000`
      - Activity Timeout: `5000000`
   6. Press **"OK"** to start feature extraction
   7. Check the Output directory for the generated CSV file with extracted flow features

   **Run Model Prediction:**
   ```bash
   # Run prediction with trained model (works with both methods)
   cd dataset
   python test.py
   ```

5. **Run Real-time IDS with ROS Integration**
   ```bash
   # Start the IDS ROS node for live detection
   rosrun ros_ids ids_node.py
   
   # Monitor detection alerts (in another terminal)
   rostopic echo /ids/alerts
   ```
   
   > **IDS Operation**: The system runs CICFlowMeter through CLI to extract network features in real-time, then processes them through the trained ML models for intrusion detection. The ROS node provides seamless integration with existing robot systems.

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

The ROS-Guard IDS supports both real-time and offline analysis workflows using CICFlowMeter for network traffic feature extraction:

### Real-time Detection Process (CLI Method)

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

### Offline Analysis Process (GUI Method)

1. **PCAP File Preparation**
   - Capture network traffic using `tcpdump` or Wireshark
   - Save as `.pcap` file for offline analysis

2. **GUI-based Feature Extraction**
   ```bash
   # Launch CICFlowMeter GUI
   sudo ./gradlew execute
   ```
   
   **GUI Workflow:**
   - Click **Network** → **Offline**
   - **PCAP Dir**: Browse and select your `.pcap` file
   - **Output Dir**: Choose destination for extracted `.csv` file  
   - **Configure timeouts**: Flow Timeout: `120000000`, Activity Timeout: `5000000`
   - Press **OK** to extract flow features

3. **ML-based Classification & ROS Integration**
   - Processed features are fed into trained Random Forest model
   - Real-time anomaly detection with ~95% accuracy
   - **ROS Node Integration**: `ids_node.py` continuously monitors CSV output
   - Publishes alerts to `/ids/alerts` topic for ROS ecosystem integration
   - Logs detection results for analysis and monitoring

### ROS Node Operation
```bash
# Run the IDS ROS node for real-time detection
rosrun ros_ids ids_node.py

# Monitor alerts in another terminal
rostopic echo /ids/alerts
```

The IDS node (`codes/ids_node.py`) features:
- **Continuous monitoring** of CICFlowMeter CSV output at 2Hz
- **Feature alignment** ensuring compatibility with trained models  
- **Real-time publishing** of detection results to ROS topics
- **Comprehensive logging** for forensic analysis

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
| **CNN-LSTM** (Primary)   | ✅ Complete | 98.78%   | Deep features | `ac-cnn-lstm/` |
| **CNN-MHA** (Primary)    | ✅ Complete | 98.10%   | 78→64→16 pipeline | `ac-cnn-mha/` |
| **Random Forest**        | ✅ Experimental | Research | 63/78 | `ac-mi-rf/`    |
| **Decision Tree**        | ✅ Experimental | Research | MI selected | `ac-mi-dt/` |
| **K-Nearest Neighbors**  | ✅ Experimental | Research | 62/78 | `ac-mi-knn/` |

### Feature Engineering
- **78 Network Traffic Features** extracted using CICFlowMeter
- **Deep Learning Pipeline**: CNN-MHA feature extraction (78→64→16)
- **Statistical Analysis**: Flow duration, packet sizes, inter-arrival times
- **Protocol Features**: TCP flags, header lengths, window sizes  
- **Behavioral Patterns**: Active/idle times, up/down ratios
- **Attack Classification**: 8 attack types including DoS, MitM, Port Scanning

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
