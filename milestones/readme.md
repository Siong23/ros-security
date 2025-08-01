🛡️ ROS-Guard Internship - Month 1 Milestone
📅 Duration:
Week 1 – Week 4

📍 Objective:
Build foundational understanding of ROS-based robotic systems and demonstrate basic exploitation techniques to identify vulnerabilities in ROS communication.

✅ Completed Tasks
🧠 System Familiarization
Setup TurtleBot3 (Burger) in SLAM environment.

Successfully connected Host PC and TurtleBot via ROS over SSH and Wi-Fi.

Modified .bashrc on both devices for ROS_MASTER_URI and ROS_HOSTNAME.

🤖 ROS Operations
Launched key nodes:

roscore

turtlebot3_bringup

turtlebot3_slam

turtlebot3_teleop

Used RViz for live robot visualization and navigation.

🧪 Attack Implementation
Attack Type	Script	Status	Notes
DoS	ros_dos2.py	✅ Working	Floods /cmd_vel
Unauthorized Publisher	unauthorized_publisher.py	✅ Working	Controls robot without permission
Unauthorized Subscriber	unauthorized_subscriber.py	✅ Working	Eavesdrops movement commands
Topic Flooding	topic_flooding.py	⚠️ Partial	Logs flood on /scan, no physical effect
All-Angle Flood	rosallangle.py	✅ Working	Modified version of DoS
Service Exploitation	service_exploitation.py	❌ Not Working	Investigating /clear_costmap behavior

🧾 Data Collection for IDS
Used rosbag to record:

✅ Normal operations (normal_operation.bag)

✅ Attack scenarios (attack_scenario.bag)

Extracted specific topics like /cmd_vel to .csv for ML training:

bash
Copy
Edit
rostopic echo -b attack_scenario.bag -p /cmd_vel > attack.csv
📂 Next Steps (Month 2 Preview)
Label and preprocess .csv data.

Begin feature engineering from flow-based traffic using CICFlowMeter.

Start training ML models for intrusion detection (e.g., Random Forest, XGBoost).

Evaluate detection accuracy against various attacks.

