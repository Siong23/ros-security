# 🎯 ROS-Guard IDS Live Demo Guide

This guide provides step-by-step instructions for demonstrating the ROS-Guard Intrusion Detection System during your internship presentation.

## 🖥️ Hardware Setup Overview

| Device | Role | IP Address | OS |
|--------|------|------------|-----|
| **Master Laptop** | Control Station | `192.168.0.142` | Ubuntu 20.04 |
| **TurtleBot3 Burger** | Robot Platform | `192.168.0.141` | Ubuntu (TurtleBot) |
| **Attacker Machine** | Attack Simulation | `192.168.0.181` | Kali Linux (VM) |

---

## 🚀 Part 1: System Initialization

### Step 1: Start Master Laptop (Ubuntu - 192.168.0.142)

**Terminal 1 - SSH to TurtleBot:**
```bash
ssh ubuntu@192.168.0.141
# Password: turtlebot
```

**Terminal 2 - Start ROS Master:**
```bash
roscore
```

**Terminal 1 (SSH session) - Launch TurtleBot:**
```bash
roslaunch turtlebot3_bringup turtlebot3_robot.launch
```

✅ **Checkpoint**: Master laptop and TurtleBot can now communicate

---

## 📊 Part 2: Start Intrusion Detection System

### Step 2: Initialize CICFlowMeter (Network Traffic Monitoring)

**Terminal 3 - Start Feature Extraction:**
```bash
cd CICFlowMeter
sudo cicflowmeter -i wlp0s20f3 -c /home/jakelcj/output.csv
```

**Terminal 4 - Verify Data Collection:**
```bash
head -n 5 /home/jakelcj/output.csv
```

✅ **Checkpoint**: Network traffic features are being extracted in real-time

### Step 3: Launch IDS Models

Choose one of the primary deep learning models:

**Option A - CNN Multi-Head Attention IDS:**
```bash
rosrun ros_ids cnn_mha_ids_node_linux.py
```

**Option B - CNN-LSTM Hybrid IDS:**
```bash
rosrun ros_ids cnn_lstm_ids_node_linux.py
```

✅ **Checkpoint**: IDS is now monitoring and ready to detect attacks

---

## 🥷 Part 3: Attack Simulation (Kali VM - 192.168.0.181)

### Step 4: Setup Attack Environment

**Initialize Docker and ROS:**
```bash
sudo systemctl start docker
docker run -it --rm --net=host -v /home/kali/ros_attacks:/attacks ros:noetic-ros-core
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://192.168.0.142:11311
export ROS_IP=192.168.0.181
```

**Verify Connection:**
```bash
rostopic list
cd attacks
```

✅ **Checkpoint**: Attacker machine connected to ROS network

---

## ⚡ Part 4: Live Attack Demonstrations

### Available Attack Scripts & Expected Detections:

| Attack Script | Target | Primary Detection | Secondary Detection |
|---------------|--------|------------------|-------------------|
| **`dos.py`** | `/cmd_vel` topic | **DoS Attack** | - |
| **`pubflood_attack.py 192.168.0.141`** | TurtleBot | **Pubflood** | - |
| **`pubflood_attack.py 192.168.0.142`** | Master | **DoS Pubflood** | - |
| **`unauth_publisher.py`** | ROS Topics | **Pubflood** | **DoS Attack** |
| **`unauthsub_attack.py 192.168.0.142`** | Master | **Pubflood, DoS** | **UnauthSub Attack** |
| **`port_scan.py 192.168.0.142`** | Master | **DoS, Pubflood** | **UnauthSub Attack** |
| **`subandpubflood.py`** | Subscriptions/Publications | **Pubflood/Subflood** | - |
| **`topicflooding.py`** | Topics | **Pubflood** | **(UnauthSub?)** |
| **`v2pub_flood_sim.py`** | Publications | **Pubflood** | - |

### Demo Attack Sequence (Suggested):

1. **🎯 Basic DoS Attack:**
   ```bash
   python3 Dos.py
   ```
   *Show IDS detecting DoS attack in real-time*

2. **📡 Publication Flooding:**
   ```bash
   python3 Pubflood_attack.py 192.168.0.141
   ```
   *Demonstrate Pubflood detection*

3. **🔓 Unauthorized Publisher:**
   ```bash
   python3 Unauth_publisher.py
   ```
   *Show multiple attack types detected simultaneously*

4. **🕵️ Port Scanning:**
   ```bash
   python3 Port_scan.py 192.168.0.142
   ```
   *Demonstrate network-level attack detection*

---
