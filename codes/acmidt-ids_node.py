#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import pandas as pd
from joblib import load

CSV_PATH = "/home/jakelcj/output.csv"
MODEL_PATH = "/home/jakelcj/ids_ws/src/ros_ids/models/dt/model.joblib"
SCALER_PATH = "/home/jakelcj/ids_ws/src/ros_ids/models/dt/scaler.joblib"
FEATURES_PATH = "/home/jakelcj/ids_ws/src/ros_ids/models/dt/used_features.txt"
OUTPUT_LOG_PATH = "/home/jakelcj/acmidt-ids_output.log"  # Path to the output log file

def load_pipeline():
    """Load trained model, scaler, and expected features."""
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)

    with open(FEATURES_PATH) as f:
        used_features = [line.strip() for line in f.readlines()]

    rospy.loginfo(f"✅ Loaded model, scaler, and {len(used_features)} features")
    return model, scaler, used_features

def ids_node():
    rospy.init_node("ids_node", anonymous=True)
    pub = rospy.Publisher("/ids/alerts", String, queue_size=10)
    rate = rospy.Rate(2)  # 2 Hz

    model, scaler, used_features = load_pipeline()
    seen_rows = 0

    while not rospy.is_shutdown():
        try:
            # Load CSV
            df = pd.read_csv(CSV_PATH)

            if df.shape[0] > seen_rows:
                # Get only new rows
                new_df = df.iloc[seen_rows:].copy()  # <-- copy() to avoid SettingWithCopyWarning

                # Ensure feature alignment
                for feat in used_features:
                    if feat not in new_df.columns:
                        new_df[feat] = 0  # fill missing

                new_df = new_df[used_features]  # reorder
                new_df = new_df.apply(pd.to_numeric, errors="coerce").fillna(0)

                # Scale + predict
                X_scaled = scaler.transform(new_df)
                preds = model.predict(X_scaled)

                # Publish and log each prediction
                with open(OUTPUT_LOG_PATH, "a") as log_file:
                    for p in preds:
                        msg = "ALERT: Attack detected!" if p == 1 else "Normal traffic"
                        rospy.loginfo(msg)
                        pub.publish(msg)
                        log_file.write(msg + "\n")  # Write to log file

                # Update counter
                seen_rows = df.shape[0]

        except Exception as e:
            rospy.logerr(f"IDS error: {e}")

        rate.sleep()

if __name__ == "__main__":
    try:
        ids_node()
    except rospy.ROSInterruptException:
        pass
