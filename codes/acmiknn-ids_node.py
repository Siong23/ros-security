#!/usr/bin/env python3
import json
import rospy
from std_msgs.msg import String
import pandas as pd
from joblib import load

# Default (can be overridden via ROS params: ~model_path, ~scaler_path, ~features_path, ~csv_path, ~output_log)
CSV_PATH = "/home/jakelcj/output.csv"
MODEL_PATH = "/home/jakelcj/ids_ws/src/ros_ids/models/knn/acmiknn-model.joblib"
SCALER_PATH = "/home/jakelcj/ids_ws/src/ros_ids/models/knn/acmiknn-scaler.joblib"
FEATURES_PATH = "/home/jakelcj/ids_ws/src/ros_ids/models/knn/acmiknn-used_features.txt"
OUTPUT_LOG_PATH = "/home/jakelcj/acmiknn-ids_output.log"  # Path to the output log file

def load_pipeline(model_path, scaler_path, features_path):
    """Load trained model, scaler, and expected features from given paths."""
    model = load(model_path)
    scaler = load(scaler_path)

    with open(features_path) as f:
        used_features = [line.strip() for line in f.readlines()]

    rospy.loginfo(f"✅ Loaded model, scaler, and {len(used_features)} features from {features_path}")
    return model, scaler, used_features

def ids_node():
    rospy.init_node("ids_node", anonymous=True)
    pub = rospy.Publisher("/ids/alerts", String, queue_size=10)
    rate = rospy.Rate(2)  # 2 Hz

    # Allow overriding paths via ROS params (private namespace)
    model_path = rospy.get_param('~model_path', MODEL_PATH)
    scaler_path = rospy.get_param('~scaler_path', SCALER_PATH)
    features_path = rospy.get_param('~features_path', FEATURES_PATH)
    csv_path = rospy.get_param('~csv_path', CSV_PATH)
    output_log = rospy.get_param('~output_log', OUTPUT_LOG_PATH)

    # Attack classes: either a list like [1,3] or comma-separated string '1,3'
    attack_classes_param = rospy.get_param('~attack_classes', None)
    attack_classes = None
    if attack_classes_param is not None:
        if isinstance(attack_classes_param, (list, tuple)):
            attack_classes = set(int(x) for x in attack_classes_param)
        elif isinstance(attack_classes_param, str):
            attack_classes = set(int(x.strip()) for x in attack_classes_param.split(',') if x.strip())
    # If not provided, we'll default to interpreting any class != 0 as attack (but log a warning)

    model, scaler, used_features = load_pipeline(model_path, scaler_path, features_path)
    seen_rows = 0

    # Validate feature count against model if possible
    if hasattr(model, 'n_features_in_'):
        expected = int(model.n_features_in_)
        if len(used_features) != expected:
            rospy.logwarn(f"Feature count mismatch: used_features ({len(used_features)}) != model.n_features_in_ ({expected})." )

    while not rospy.is_shutdown():
        try:
            # Load CSV (path may be overridden)
            df = pd.read_csv(csv_path)

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
                with open(output_log, "a") as log_file:
                    for p in preds:
                        # Determine if this prediction should be considered an attack
                        try:
                            p_int = int(p)
                        except Exception:
                            # some models may return numpy types or strings
                            p_int = int(p)

                        if attack_classes is not None:
                            is_attack = (p_int in attack_classes)
                        else:
                            is_attack = (p_int != 0)

                        human = "ALERT: Attack detected" if is_attack else "Normal traffic"
                        payload = {
                            'class': int(p_int),
                            'is_attack': bool(is_attack),
                            'message': human
                        }
                        msg = json.dumps(payload)
                        rospy.loginfo(human + f" (class={p_int})")
                        pub.publish(msg)
                        log_file.write(msg + "\n")  # Write JSON line to log file

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
