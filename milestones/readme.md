# 📍 Month 1 Milestone: System Setup & Attack Simulation (Week 1 – 4)

This milestone focused on setting up a working ROS environment, familiarizing with ROS-based robot operation, and simulating common cyber attacks to generate labeled datasets for IDS training.

---

## ✅ Key Achievements

### 🤖 ROS Environment Setup
- Configured **TurtleBot3 (Burger)** using ROS Noetic on Ubuntu.
- Established stable communication between **Host PC ↔ Robot** over Wi-Fi using:
  - Correct `ROS_MASTER_URI` and `ROS_HOSTNAME` settings in `.bashrc`.
- Verified SLAM operation in RViz with live map updates.

### 🧪 Attack Scenario Implementation

| Attack Type              | Script                     | Status     | Description                                     |
|--------------------------|----------------------------|------------|-------------------------------------------------|
| DoS                      | `dos.py`              | ✅ Working | Floods `/cmd_vel` with movement commands        |
| MitM                      | `mitm.py`              | ✅ Working | xxxxx        |
| Unauthorized Publisher   | `unauthorized_publisher.py`| ✅ Working | Injects fake control messages                   |
| Unauthorized Subscriber  | `unauthorized_subscriber.py`| ✅ Working | Eavesdrops data from `/scan` topic              |
| Topic Flooding           | `topic_flooding.py`        | ✅ Working | Publishes to `/scan` with high frequency        |
| All-Angle Movement Spam  | `rosallangle.py`           | ✅ Working | Sends data to all linear/angular channels        |
| Service Exploitation     | `service_exploitation.py`  | ❌ Failed  | Attempted `/clear_costmap` service flooding [only worked in mapped environment]     |

> All scripts executed during SLAM operation and teleoperation to assess real-time impact.

# To be done in the future
-	Attacks not done before, use new attack to get new data
-	Consider working on ML model for those new attacks

# What have been done in week 3
- Focused on attak called MITM where it is a message manipulation attack
- Cleaned the MITM dataset
- Try to follow Nawfal's code
- ROSIDS23 and NavBot25, what do I do with these existing dataset?

# What have been done in week 4
- Recorded 3 attacks

# Milestone #1
- Try to complete the dataset as much as possible.


# Milestone #2
- CICFlowMeter working in Ubuntu to see real life incoming traffic
- Export models created in NavBot25
- Validating the data of models

# Milestone #3
- 
