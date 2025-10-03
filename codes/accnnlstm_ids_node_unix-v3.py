#!/usr/bin/env python3
"""
Fixed CNN+LSTM IDS Node - Complete Pipeline Implementation
=========================================================

This version properly implements the full transformation pipeline:
Raw(78) -> Scale(78) -> PCA(16) -> KNN_proba(8)+RF_proba(8) -> Fuse(32) -> LR_predict

Fixes the "78 features vs 32 features" error by applying the complete
transformation sequence that was used during training.

NEW FEATURES:
- Continuous file monitoring: Watches for new data appended to CSV files
- New file detection: Automatically processes new CSV files in watch directory  
- Real-time alerts: Publishes attack detections as they are found
- Incremental processing: Only processes new data, not entire file each time
"""

import rospy
import pandas as pd
import numpy as np
from std_msgs.msg import String
from joblib import load
import json
import os
import sys
import time
import glob
from pathlib import Path

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR DEPLOYMENT
# =============================================================================

# Model file paths - Points to the cnnlstm subfolder where your joblib files are
MODEL_BASE_PATH = "/home/jakelcj/ids_ws/src/ros_ids/models/cnnlstm"

SCALER_PATH = os.path.join(MODEL_BASE_PATH, "scaler.joblib")
PCA_PATH = os.path.join(MODEL_BASE_PATH, "pca.joblib")
KNN_PATH = os.path.join(MODEL_BASE_PATH, "knn_classifier.joblib")  
RF_PATH = os.path.join(MODEL_BASE_PATH, "rf_classifier.joblib")
LR_PATH = os.path.join(MODEL_BASE_PATH, "lr_classifier.joblib")
FEATURES_PATH = os.path.join(MODEL_BASE_PATH, "features.txt")
PIPELINE_INFO_PATH = os.path.join(MODEL_BASE_PATH, "pipeline_info.json")

# Data file paths - UPDATE THESE for monitoring
CSV_FILE_PATH = "/home/jakelcj/output.csv"  # Primary CSV file to monitor
CSV_WATCH_DIR = "/home/jakelcj/"  # Directory to watch for new CSV files
CSV_PATTERN = "*.csv"  # Pattern for CSV files to monitor

# Output paths
RESULTS_PATH = "/home/jakelcj/ids_ws/src/ros_ids/models/cnnlstm/result/accnnlstm-ids_results.csv"  # UPDATE THIS PATH
OUTPUT_LOG_PATH = "/home/jakelcj/ids_ws/src/ros_ids/models/cnnlstm/result/accnnlstm-ids_output.log"  # UPDATE THIS PATH

def load_complete_pipeline():
    """
    Load the complete CNN+LSTM fusion pipeline.
    
    Returns:
        tuple: (scaler, pca, knn, rf, lr, features, pipeline_info)
    """
    try:
        rospy.loginfo("🔄 Loading pipeline components...")
        
        # Check if all files exist
        required_files = [SCALER_PATH, PCA_PATH, KNN_PATH, RF_PATH, LR_PATH, FEATURES_PATH, PIPELINE_INFO_PATH]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        # Load all components
        scaler = load(SCALER_PATH)
        pca = load(PCA_PATH) 
        knn_classifier = load(KNN_PATH)
        rf_classifier = load(RF_PATH)
        lr_classifier = load(LR_PATH)
        
        # Load feature names
        with open(FEATURES_PATH, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        
        # Load pipeline info
        with open(PIPELINE_INFO_PATH, 'r') as f:
            pipeline_info = json.load(f)
            
        rospy.loginfo("✅ All pipeline components loaded successfully")
        rospy.loginfo(f"   - Expected features: {len(features)}")
        rospy.loginfo(f"   - PCA components: {pca.n_components_}")
        rospy.loginfo(f"   - Final classes: {len(pipeline_info.get('attack_classes', {}))}")
        
        return scaler, pca, knn_classifier, rf_classifier, lr_classifier, features, pipeline_info
        
    except Exception as e:
        rospy.logerr(f"❌ Failed to load pipeline: {e}")
        raise

def apply_complete_pipeline(raw_data, scaler, pca, knn, rf, lr):
    """
    Apply the complete transformation pipeline to raw features.
    
    Args:
        raw_data (pd.DataFrame or np.ndarray): Raw feature matrix (n_samples, 78)
        scaler: Fitted StandardScaler
        pca: Fitted PCA transformer  
        knn: Fitted KNN classifier
        rf: Fitted RandomForest classifier
        lr: Fitted LogisticRegression classifier
        
    Returns:
        np.ndarray: Final predictions
    """
    try:
        # Convert to numpy if DataFrame (but preserve feature names for scaler)
        if hasattr(raw_data, 'values'):
            # It's a DataFrame - scaler can use feature names
            X_scaled = scaler.transform(raw_data)
        else:
            # It's already numpy array
            X_scaled = scaler.transform(raw_data)
        rospy.logdebug(f"After scaling: {X_scaled.shape}")
        
        # Step 2: Apply PCA (78 -> 16)  
        X_pca = pca.transform(X_scaled)
        rospy.logdebug(f"After PCA: {X_pca.shape}")
        
        # Step 3: Get classifier probabilities on PCA features
        knn_proba = knn.predict_proba(X_pca)  # (n_samples, 8)
        rf_proba = rf.predict_proba(X_pca)    # (n_samples, 8)
        rospy.logdebug(f"KNN probabilities: {knn_proba.shape}")
        rospy.logdebug(f"RF probabilities: {rf_proba.shape}")
        
        # Step 4: Fuse features (16 + 8 + 8 = 32)
        X_fused = np.concatenate([X_pca, knn_proba, rf_proba], axis=1)
        rospy.logdebug(f"After fusion: {X_fused.shape}")
        
        # Step 5: Final prediction (32 -> classes)
        final_predictions = lr.predict(X_fused)
        rospy.logdebug(f"Final predictions: {final_predictions.shape}")
        
        return final_predictions
        
    except Exception as e:
        rospy.logerr(f"❌ Pipeline transformation failed: {e}")
        raise

def prepare_features(df, expected_features):
    """
    Prepare and align features from CSV data with column name mapping.
    Handles conversion from underscore format to space format.
    
    Args:
        df (pd.DataFrame): Raw CSV data  
        expected_features (list): List of expected feature names
        
    Returns:
        pd.DataFrame: Aligned feature matrix
    """
    try:
        # Column name mapping from CSV format to model expected format
        column_mapping = {
            # Port and flow features
            'src_port': 'Src Port', 'dst_port': 'Dst Port', 
            'flow_duration': 'Flow Duration',
            'tot_fwd_pkts': 'Tot Fwd Pkts', 'tot_bwd_pkts': 'Tot Bwd Pkts',
            'totlen_fwd_pkts': 'TotLen Fwd Pkts', 'totlen_bwd_pkts': 'TotLen Bwd Pkts',
            
            # Forward packet features
            'fwd_pkt_len_max': 'Fwd Pkt Len Max', 'fwd_pkt_len_min': 'Fwd Pkt Len Min',
            'fwd_pkt_len_mean': 'Fwd Pkt Len Mean', 'fwd_pkt_len_std': 'Fwd Pkt Len Std',
            
            # Backward packet features  
            'bwd_pkt_len_max': 'Bwd Pkt Len Max', 'bwd_pkt_len_min': 'Bwd Pkt Len Min',
            'bwd_pkt_len_mean': 'Bwd Pkt Len Mean', 'bwd_pkt_len_std': 'Bwd Pkt Len Std',
            
            # Flow rate features
            'flow_byts_s': 'Flow Byts/s', 'flow_pkts_s': 'Flow Pkts/s',
            
            # Flow IAT features
            'flow_iat_mean': 'Flow IAT Mean', 'flow_iat_std': 'Flow IAT Std',
            'flow_iat_max': 'Flow IAT Max', 'flow_iat_min': 'Flow IAT Min',
            
            # Forward IAT features
            'fwd_iat_tot': 'Fwd IAT Tot', 'fwd_iat_mean': 'Fwd IAT Mean',
            'fwd_iat_std': 'Fwd IAT Std', 'fwd_iat_max': 'Fwd IAT Max', 'fwd_iat_min': 'Fwd IAT Min',
            
            # Backward IAT features
            'bwd_iat_tot': 'Bwd IAT Tot', 'bwd_iat_mean': 'Bwd IAT Mean', 
            'bwd_iat_std': 'Bwd IAT Std', 'bwd_iat_max': 'Bwd IAT Max', 'bwd_iat_min': 'Bwd IAT Min',
            
            # Flag features
            'fwd_psh_flags': 'Fwd PSH Flags', 'bwd_psh_flags': 'Bwd PSH Flags',
            'fwd_urg_flags': 'Fwd URG Flags', 'bwd_urg_flags': 'Bwd URG Flags',
            
            # Header features
            'fwd_header_len': 'Fwd Header Len', 'bwd_header_len': 'Bwd Header Len',
            
            # Packets per second
            'fwd_pkts_s': 'Fwd Pkts/s', 'bwd_pkts_s': 'Bwd Pkts/s',
            
            # Packet length features
            'pkt_len_min': 'Pkt Len Min', 'pkt_len_max': 'Pkt Len Max',
            'pkt_len_mean': 'Pkt Len Mean', 'pkt_len_std': 'Pkt Len Std', 'pkt_len_var': 'Pkt Len Var',
            
            # Flag counts
            'fin_flag_cnt': 'FIN Flag Cnt', 'syn_flag_cnt': 'SYN Flag Cnt', 'rst_flag_cnt': 'RST Flag Cnt',
            'psh_flag_cnt': 'PSH Flag Cnt', 'ack_flag_cnt': 'ACK Flag Cnt', 'urg_flag_cnt': 'URG Flag Cnt',
            'cwe_flag_count': 'CWE Flag Count', 'ece_flag_cnt': 'ECE Flag Cnt',
            
            # Ratio and average features
            'down_up_ratio': 'Down/Up Ratio', 'pkt_size_avg': 'Pkt Size Avg',
            'fwd_seg_size_avg': 'Fwd Seg Size Avg', 'bwd_seg_size_avg': 'Bwd Seg Size Avg',
            
            # Bulk features
            'fwd_byts_b_avg': 'Fwd Byts/b Avg', 'fwd_pkts_b_avg': 'Fwd Pkts/b Avg',
            'fwd_blk_rate_avg': 'Fwd Blk Rate Avg', 'bwd_byts_b_avg': 'Bwd Byts/b Avg',
            'bwd_pkts_b_avg': 'Bwd Pkts/b Avg', 'bwd_blk_rate_avg': 'Bwd Blk Rate Avg',
            
            # Subflow features
            'subflow_fwd_pkts': 'Subflow Fwd Pkts', 'subflow_fwd_byts': 'Subflow Fwd Byts',
            'subflow_bwd_pkts': 'Subflow Bwd Pkts', 'subflow_bwd_byts': 'Subflow Bwd Byts',
            
            # Window features
            'init_fwd_win_byts': 'Init Fwd Win Byts', 'init_bwd_win_byts': 'Init Bwd Win Byts',
            
            # Active data and segment features
            'fwd_act_data_pkts': 'Fwd Act Data Pkts', 'fwd_seg_size_min': 'Fwd Seg Size Min',
            
            # Active/Idle features
            'active_mean': 'Active Mean', 'active_std': 'Active Std',
            'active_max': 'Active Max', 'active_min': 'Active Min',
            'idle_mean': 'Idle Mean', 'idle_std': 'Idle Std', 
            'idle_max': 'Idle Max', 'idle_min': 'Idle Min'
        }
        
        rospy.loginfo(f"📊 Input CSV has {len(df.columns)} columns")
        
        # Create aligned DataFrame with proper column mapping
        df_aligned = pd.DataFrame()
        mapped_count = 0
        missing_count = 0
        
        for expected_feat in expected_features:
            found = False
            
            # Check direct name match first
            if expected_feat in df.columns:
                df_aligned[expected_feat] = df[expected_feat]
                found = True
                mapped_count += 1
            else:
                # Check reverse mapping (find CSV column that maps to expected feature)
                for csv_col, model_feat in column_mapping.items():
                    if model_feat == expected_feat and csv_col in df.columns:
                        df_aligned[expected_feat] = df[csv_col]
                        found = True
                        mapped_count += 1
                        break
            
            if not found:
                # Fill missing features with 0
                df_aligned[expected_feat] = 0.0
                missing_count += 1
                rospy.logwarn(f"⚠️  Missing feature '{expected_feat}' filled with 0")
        
        rospy.loginfo(f"✅ Mapped {mapped_count} features, {missing_count} missing")
        
        # Convert to numeric and handle NaN/inf
        df_aligned = df_aligned.apply(pd.to_numeric, errors='coerce')
        df_aligned = df_aligned.fillna(0)
        df_aligned = df_aligned.replace([np.inf, -np.inf], 0)
        
        rospy.logdebug(f"Feature alignment complete: {df_aligned.shape}")
        return df_aligned
        
    except Exception as e:
        rospy.logerr(f"❌ Feature preparation failed: {e}")
        raise

def format_prediction_message(predictions, pipeline_info):
    """
    Format prediction results into human-readable messages.
    
    Args:
        predictions (np.ndarray): Numeric predictions
        pipeline_info (dict): Pipeline information with class mappings
        
    Returns:
        list: List of formatted messages
    """
    try:
        # Get class mappings from pipeline_info (keys are strings in JSON)
        attack_classes_raw = pipeline_info.get('attack_classes', {
            "0": "Normal", "1": "DoS Attack", "2": "UnauthSub Attack", "3": "SSH Bruteforce",
            "4": "Pubflood", "5": "Subflood", "6": "Reverse Shell", "7": "Port Scanning Attack"
        })
        
        # Convert string keys to integer keys for proper lookup
        class_names = {}
        for key, value in attack_classes_raw.items():
            try:
                class_names[int(key)] = value
            except (ValueError, TypeError):
                rospy.logwarn(f"Invalid class key in pipeline_info: {key}")
        
        # Debug: Log the class mappings being used
        rospy.logdebug(f"Converted class mappings: {class_names}")
        
        messages = []
        for pred in predictions:
            # Ensure proper integer conversion
            pred_int = int(pred) if hasattr(pred, '__int__') else pred
            
            # Debug logging for troubleshooting
            rospy.logdebug(f"Raw prediction: {pred} (type: {type(pred)})")
            rospy.logdebug(f"Converted to int: {pred_int} (type: {type(pred_int)})")
            rospy.logdebug(f"Available keys in class_names: {list(class_names.keys())}")
            
            class_name = class_names.get(pred_int, f"Unknown({pred_int})")
            timestamp = rospy.Time.now().to_sec()
            
            if pred_int == 0:  # Normal traffic
                msg = f"[{timestamp:.2f}] Normal traffic detected"
            else:  # Attack detected
                msg = f"[{timestamp:.2f}] 🚨 ATTACK DETECTED: {class_name} (Class {pred_int})"
                
            messages.append(msg)
            
        return messages
        
    except Exception as e:
        rospy.logerr(f"❌ Message formatting failed: {e}")
        return [f"Prediction: {pred}" for pred in predictions]

class CNNLSTMIDSNode:
    """
    CNN+LSTM based Intrusion Detection System ROS Node
    """
    
    def __init__(self):
        """Initialize the IDS node"""
        rospy.init_node('cnn_lstm_ids_node', anonymous=True)
        rospy.loginfo("🚀 Initializing CNN+LSTM IDS Node...")
        
        # Load the complete pipeline
        try:
            self.scaler, self.pca, self.knn, self.rf, self.lr, self.features, self.pipeline_info = load_complete_pipeline()
            rospy.loginfo("✅ Pipeline loaded successfully")
        except Exception as e:
            rospy.logerr(f"❌ Failed to initialize pipeline: {e}")
            rospy.signal_shutdown("Pipeline initialization failed")
            return
            
        # ROS Publisher for alerts
        self.alert_pub = rospy.Publisher('/ids_alerts', String, queue_size=10)
        
        # Processing parameters
        self.batch_size = 100
        self.log_file = OUTPUT_LOG_PATH
        
        # Monitoring parameters
        self.last_file_size = 0
        self.last_processed_line = 0
        self.processed_files = set()
        self.monitor_interval = 5.0  # Check every 5 seconds
        
        rospy.loginfo("🎯 CNN+LSTM IDS Node ready for processing")
        
    def process_csv_data(self):
        """
        Process CSV data and publish predictions
        """
        try:
            if not os.path.exists(CSV_FILE_PATH):
                rospy.logerr(f"❌ CSV file not found: {CSV_FILE_PATH}")
                return
                
            rospy.loginfo(f"📂 Loading CSV data from: {CSV_FILE_PATH}")
            
            # Read CSV data in chunks for memory efficiency
            chunk_num = 0
            total_processed = 0
            
            for chunk in pd.read_csv(CSV_FILE_PATH, chunksize=self.batch_size):
                chunk_num += 1
                rospy.loginfo(f"🔄 Processing chunk {chunk_num} ({len(chunk)} rows)")
                
                # Prepare features with column mapping
                X_prepared = prepare_features(chunk, self.features)
                
                # Apply complete pipeline transformation (pass DataFrame to preserve feature names)
                predictions = apply_complete_pipeline(
                    X_prepared, self.scaler, self.pca, 
                    self.knn, self.rf, self.lr
                )
                
                # Format and publish alerts
                messages = format_prediction_message(predictions, self.pipeline_info)
                
                for msg in messages:
                    # Publish to ROS topic
                    self.alert_pub.publish(String(data=msg))
                    
                    # Log to file
                    with open(self.log_file, 'a') as f:
                        f.write(msg + "\n")
                    
                    # Print to console for debugging
                    rospy.loginfo(msg)
                
                total_processed += len(chunk)
                rospy.logdebug(f"Processed {total_processed} total samples")
                
                # Small delay to prevent overwhelming the system
                rospy.sleep(0.1)
                
            rospy.loginfo(f"✅ Processing complete: {total_processed} samples processed")
            
        except Exception as e:
            rospy.logerr(f"❌ CSV processing failed: {e}")
            raise

    def check_file_changes(self):
        """
        Check if the primary CSV file has new data
        
        Returns:
            bool: True if new data is available
        """
        try:
            if not os.path.exists(CSV_FILE_PATH):
                return False
                
            current_size = os.path.getsize(CSV_FILE_PATH)
            
            if current_size > self.last_file_size:
                rospy.loginfo(f"📈 File size increased: {self.last_file_size} -> {current_size} bytes")
                self.last_file_size = current_size
                return True
                
            return False
            
        except Exception as e:
            rospy.logwarn(f"Error checking file changes: {e}")
            return False

    def check_new_files(self):
        """
        Check for new CSV files in the watch directory
        
        Returns:
            list: List of new CSV files to process
        """
        try:
            # Get all CSV files in the watch directory
            csv_pattern = os.path.join(CSV_WATCH_DIR, CSV_PATTERN)
            all_csv_files = glob.glob(csv_pattern)
            
            # Find new files not yet processed
            new_files = []
            for csv_file in all_csv_files:
                if csv_file not in self.processed_files:
                    new_files.append(csv_file)
                    self.processed_files.add(csv_file)
                    
            if new_files:
                rospy.loginfo(f"📁 Found {len(new_files)} new CSV files: {new_files}")
                
            return new_files
            
        except Exception as e:
            rospy.logwarn(f"Error checking for new files: {e}")
            return []

    def process_new_data_only(self):
        """
        Process only the new lines added to the primary CSV file
        """
        try:
            if not os.path.exists(CSV_FILE_PATH):
                rospy.logwarn(f"CSV file not found: {CSV_FILE_PATH}")
                return
                
            # Read all lines and skip already processed ones
            df = pd.read_csv(CSV_FILE_PATH)
            total_lines = len(df)
            
            if total_lines <= self.last_processed_line:
                return  # No new data
                
            # Process only new lines
            new_data = df.iloc[self.last_processed_line:]
            rospy.loginfo(f"🔄 Processing {len(new_data)} new rows (lines {self.last_processed_line + 1}-{total_lines})")
            
            # Process in chunks
            chunk_size = self.batch_size
            for i in range(0, len(new_data), chunk_size):
                chunk = new_data.iloc[i:i + chunk_size]
                
                # Prepare features with column mapping
                X_prepared = prepare_features(chunk, self.features)
                
                # Apply complete pipeline transformation
                predictions = apply_complete_pipeline(
                    X_prepared, self.scaler, self.pca, 
                    self.knn, self.rf, self.lr
                )
                
                # Format and publish alerts
                messages = format_prediction_message(predictions, self.pipeline_info)
                
                for msg in messages:
                    # Publish to ROS topic
                    self.alert_pub.publish(String(data=msg))
                    
                    # Log to file
                    with open(self.log_file, 'a') as f:
                        f.write(msg + "\n")
                    
                    # Print to console
                    rospy.loginfo(msg)
                
                # Small delay between chunks
                rospy.sleep(0.1)
            
            # Update processed line count
            self.last_processed_line = total_lines
            rospy.loginfo(f"✅ Processed up to line {self.last_processed_line}")
            
        except Exception as e:
            rospy.logerr(f"❌ New data processing failed: {e}")

    def process_new_file(self, file_path):
        """
        Process a completely new CSV file
        
        Args:
            file_path (str): Path to the new CSV file
        """
        try:
            rospy.loginfo(f"📄 Processing new file: {file_path}")
            
            # Read and process the entire new file
            for chunk in pd.read_csv(file_path, chunksize=self.batch_size):
                rospy.loginfo(f"🔄 Processing chunk from {file_path} ({len(chunk)} rows)")
                
                # Prepare features with column mapping
                X_prepared = prepare_features(chunk, self.features)
                
                # Apply complete pipeline transformation
                predictions = apply_complete_pipeline(
                    X_prepared, self.scaler, self.pca, 
                    self.knn, self.rf, self.lr
                )
                
                # Format and publish alerts
                messages = format_prediction_message(predictions, self.pipeline_info)
                
                for msg in messages:
                    # Add file info to message
                    file_name = os.path.basename(file_path)
                    enhanced_msg = f"[{file_name}] {msg}"
                    
                    # Publish to ROS topic
                    self.alert_pub.publish(String(data=enhanced_msg))
                    
                    # Log to file
                    with open(self.log_file, 'a') as f:
                        f.write(enhanced_msg + "\n")
                    
                    # Print to console
                    rospy.loginfo(enhanced_msg)
                
                # Small delay between chunks
                rospy.sleep(0.1)
                
            rospy.loginfo(f"✅ Completed processing new file: {file_path}")
            
        except Exception as e:
            rospy.logerr(f"❌ New file processing failed for {file_path}: {e}")
            
    def run(self):
        """
        Main processing loop with continuous monitoring
        """
        try:
            # Initial processing of existing data
            rospy.loginfo("🚀 Starting initial data processing...")
            self.process_csv_data()
            
            # Initialize file monitoring
            if os.path.exists(CSV_FILE_PATH):
                self.last_file_size = os.path.getsize(CSV_FILE_PATH)
                # Count existing lines for incremental processing
                try:
                    df = pd.read_csv(CSV_FILE_PATH)
                    self.last_processed_line = len(df)
                    rospy.loginfo(f"📊 Initial file has {self.last_processed_line} lines")
                except Exception:
                    self.last_processed_line = 0
            
            # Mark initial files as processed
            initial_files = glob.glob(os.path.join(CSV_WATCH_DIR, CSV_PATTERN))
            self.processed_files.update(initial_files)
            rospy.loginfo(f"📁 Marked {len(initial_files)} existing files as processed")
            
            # Enter continuous monitoring mode
            rospy.loginfo("🔄 Entering continuous monitoring mode...")
            rospy.loginfo(f"   📂 Watching directory: {CSV_WATCH_DIR}")
            rospy.loginfo(f"   📄 Primary file: {CSV_FILE_PATH}")
            rospy.loginfo(f"   ⏱️  Check interval: {self.monitor_interval} seconds")
            
            while not rospy.is_shutdown():
                try:
                    # Check for new data in existing file
                    if self.check_file_changes():
                        rospy.loginfo("📈 New data detected in primary file")
                        self.process_new_data_only()
                    
                    # Check for completely new files
                    new_files = self.check_new_files()
                    for new_file in new_files:
                        self.process_new_file(new_file)
                    
                    # Wait before next check
                    rospy.sleep(self.monitor_interval)
                    
                except Exception as e:
                    rospy.logerr(f"❌ Error in monitoring loop: {e}")
                    rospy.sleep(self.monitor_interval)  # Continue monitoring despite errors
                
        except KeyboardInterrupt:
            rospy.loginfo("🛑 Shutdown requested by user")
        except Exception as e:
            rospy.logerr(f"❌ Node execution failed: {e}")
        finally:
            rospy.loginfo("🏁 CNN+LSTM IDS Node shutting down")

def main():
    """
    Main entry point
    """
    try:
        # Create and run the IDS node
        ids_node = CNNLSTMIDSNode()
        ids_node.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 ROS interrupt received")
    except Exception as e:
        rospy.logerr(f"❌ Main execution failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
