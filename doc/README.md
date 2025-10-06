# 📚 ROS-Guard Documentation

This directory contains technical documentation and setup guides for the ROS-Guard intrusion detection system.

## 📋 Contents

- **CICFlowMeter Installation Guide** - Network traffic feature extraction setup
- **Hardware Setup Documentation** - TurtleBot3 and RGBD sensor configuration
- **System Requirements** - Complete dependency and environment setup

---

# 🔧 CICFlowMeter Installation on Ubuntu

This guide explains how to install and run **CICFlowMeter** on Ubuntu for network traffic feature extraction.  

## Steps  

1. **Update packages**  
   ```bash
   sudo apt update
2. **Remove old CICFlowMeter (if exists)**
   ````bash
   sudo rm -rf CICFlowMeter
3. Install dependencies
   ````bash
   sudo apt install -y openjdk-11-jdk maven unzip libpcap0.8 libpcap-dev
4. Clone the CICFlowMeter repository
   ````bash
   git clone https://github.com/ahlashkari/CICFlowMeter.git
5. Navigate to the directory
   ````bash
   cd CICFlowMeter
6. Install JNetPcap library
   ````bash
   sudo mvn install:install-file \
   Dfile=jnetpcap/linux/jnetpcap-1.4.r1425/jnetpcap.jar \
   DgroupId=org.jnetpcap \
   DartifactId=jnetpcap \
   Dversion=1.4.1 \
   Dpackaging=jar
7. Give execution permissions
   ````bash
   chmod +x gradlew
8. Install Java 8
   ````bash
   sudo apt install -y openjdk-8-jdk
9. Configure default Java version
    ````bash
    sudo update-alternatives --config java
> Select Java 8 from the list.
10. Verify Java version
    ````bash
    java -version

11. Run CICFlowMeter
````bash
sudo ./gradlew execute
````
---

## 🚀 How to Use CICFlowMeter

After successful installation, CICFlowMeter can be used in two modes:

### 🖥️ GUI Mode (Offline Analysis)
```bash
sudo ./gradlew execute
```
- Use for analyzing existing PCAP files
- Select input PCAP directory and output CSV location
- Configure timeouts: Flow Timeout: `120000000`, Activity Timeout: `5000000`

### ⚡ CLI Mode (Real-time Analysis)
```bash
sudo cicflowmeter -i wlp0s20f3 -c /home/jakelcj/output.csv
```

**Command Format:**
```bash
sudo cicflowmeter -i [network_interface] -c [output_csv_path]
```

**Verify Operation:**
```bash
head -n 5 /home/jakelcj/output.csv
```

### 🔍 Integration with ROS-Guard IDS
The extracted CSV features are automatically processed by the ROS IDS node (`ids_node.py`) for real-time intrusion detection.

---

## 📊 Feature Extraction Details

CICFlowMeter extracts **78 network traffic features** including:
- Flow duration and packet statistics
- Inter-arrival times and flow rates  
- TCP flag distributions
- Packet size distributions
- Protocol-specific features

These features are then used by the machine learning models for attack classification.

---

## 🆘 Troubleshooting

**Common Issues:**
- **Java Version**: Ensure Java 8 is selected as default
- **Permissions**: Run with `sudo` for network interface access
- **Network Interface**: Use `ifconfig` to find correct interface name
- **Dependencies**: Verify all packages are properly installed

**For Additional Support:** See the main project README and model documentation in `/dataset/models/`

