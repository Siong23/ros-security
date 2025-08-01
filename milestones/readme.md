# 📍 Month 1 Milestone: System Setup & Attack Simulation (Weeks 1–4)

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
| DoS                      | `ros_dos2.py`              | ✅ Working | Floods `/cmd_vel` with movement commands        |
| Unauthorized Publisher   | `unauthorized_publisher.py`| ✅ Working | Injects fake control messages                   |
| Unauthorized Subscriber  | `unauthorized_subscriber.py`| ✅ Working | Eavesdrops data from `/scan` topic              |
| Topic Flooding           | `topic_flooding.py`        | ⚠️ Partial | Publishes to `/scan` with high frequency        |
| All-Angle Movement Spam  | `rosallangle.py`           | ✅ Working | Sends data to all linear/angular channels       |
| Service Exploitation     | `service_exploitation.py`  | ❌ Failed  | Attempted `/clear_costmap` service flooding     |

> All scripts executed during SLAM operation and teleoperation to assess real-time impact.

# To be done in the future
-	Attacks not done before, use new attack to get new data
-	Consider working on ML model for those new attacks
