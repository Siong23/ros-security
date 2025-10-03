#!/usr/bin/env python3
"""
CNN+MHA IDS Node for Ubuntu Linux
Real-time Intrusion Detection System using CNN+Multi-Head Attention

Author: eusoff "ae1207" aminurrashid
Date: 2 oct 2025
Version: 2.0

Dependencies:
    sudo apt update
    sudo apt install python3-pip python3-dev
    pip3 install rospy std_msgs pandas numpy scikit-learn joblib tensorflow glob2 watchdog

Usage:
    rosrun ros_ids cnn_mha_ids_node_linux.py
    
Model Files Required:
    - scaler.joblib
    - feature_extractor.keras  
    - pca.joblib
    - knn_classifier.joblib
    - rf_classifier.joblib
    - lr_classifier.joblib
    - features.txt
    - pipeline_info.json
"""

import rospy
import sys
import os
import json
import time
import threading
import glob
from datetime import datetime
from pathlib import Path

# Suppress scikit-learn version warnings and other warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*version.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# Scientific computing
import numpy as np
import pandas as pd

# Machine Learning
try:
    import joblib
    print("✅ joblib imported successfully")
except ImportError as e:
    print(f"❌ joblib import error: {e}")
    print("Install with: pip3 install joblib")
    sys.exit(1)

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    print("✅ scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ scikit-learn import error: {e}")
    print("Install with: pip3 install scikit-learn==1.3.2")
    sys.exit(1)

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print("✅ TensorFlow imported successfully")
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow import error: {e}")
    print("Install with: pip3 install tensorflow")
    sys.exit(1)

# ROS
try:
    from std_msgs.msg import String
    print("✅ ROS std_msgs imported successfully")
except ImportError as e:
    print(f"❌ ROS import error: {e}")
    print("Make sure ROS is properly installed and sourced")
    sys.exit(1)

# File monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    print("✅ watchdog imported successfully")
except ImportError as e:
    print(f"❌ watchdog import error: {e}")
    print("Install with: pip3 install watchdog")
    sys.exit(1)


class CNNMHAIDSNode:
    """CNN+MHA based Intrusion Detection System Node for ROS"""
    
    def __init__(self):
        """Initialize the CNN+MHA IDS Node"""
        rospy.init_node('cnn_mha_ids_node', anonymous=True)
        rospy.loginfo("🚀 Starting CNN+MHA IDS Node...")
        
        # Configuration
        self.node_name = "CNN+MHA IDS"
        self.model_dir = self._get_model_directory()
        self.csv_dir = self._get_csv_directory()
        
        # Model components
        self.scaler = None
        self.feature_extractor = None
        self.pca = None
        self.knn_classifier = None
        self.rf_classifier = None
        self.lr_classifier = None
        self.feature_names = []
        self.pipeline_info = {}
        self.class_names = []
        
        # Data processing
        self.processed_files = set()
        self.last_processed_rows = {}
        self.monitoring = True
        
        # ROS Publisher
        self.alert_publisher = rospy.Publisher('/ids_alerts', String, queue_size=10)
        
        # Statistics
        self.total_predictions = 0
        self.attack_detections = 0
        self.start_time = time.time()
        
        # Column mapping for CSV compatibility
        self.column_mapping = self._create_column_mapping()
        
        # Load the complete pipeline
        self._load_pipeline()
        
        # Start file monitoring
        self._start_file_monitoring()
        
        rospy.loginfo("✅ CNN+MHA IDS Node initialized successfully")
        rospy.loginfo(f"📁 Model directory: {self.model_dir}")
        rospy.loginfo(f"📂 CSV directory: {self.csv_dir}")
        rospy.loginfo(f"🔍 Monitoring for new CSV files...")
        
    def _get_model_directory(self):
        """Get the model directory path"""
        # Try different possible locations
        possible_paths = [
            "/home/jakelcj/ids_ws/src/ros_ids/models/navbot25_ac_cnn_mha_knn",
            "~/ids_ws/src/ros_ids/models/navbot25_ac_cnn_mha_knn", 
            "./models/navbot25_ac_cnn_mha_knn",
            rospy.get_param('~model_path', '/home/jakelcj/ids_ws/src/ros_ids/models/navbot25_ac_cnn_mha_knn')
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                rospy.loginfo(f"📁 Found model directory: {expanded_path}")
                return expanded_path
                
        rospy.logerr(f"❌ Model directory not found in any of these locations:")
        for path in possible_paths:
            rospy.logerr(f"   - {os.path.expanduser(path)}")
        rospy.logerr("Please ensure the model files are properly placed")
        sys.exit(1)
        
    def _get_csv_directory(self):
        """Get the CSV monitoring directory"""
        default_csv_dir = "/home/jakelcj"  # Directory where output.csv is located
        csv_dir = rospy.get_param('~csv_path', default_csv_dir)
        csv_dir = os.path.expanduser(csv_dir)
        
        # Create directory if it doesn't exist
        os.makedirs(csv_dir, exist_ok=True)
        return csv_dir
        
    def _create_column_mapping(self):
        """Create mapping from CSV column names to model feature names"""
        return {
            # Network flow features
            'src_port': 'Src Port', 'dst_port': 'Dst Port',
            'flow_duration': 'Flow Duration', 'tot_fwd_pkts': 'Tot Fwd Pkts',
            'tot_bwd_pkts': 'Tot Bwd Pkts', 'totlen_fwd_pkts': 'TotLen Fwd Pkts',
            'totlen_bwd_pkts': 'TotLen Bwd Pkts', 'fwd_pkt_len_max': 'Fwd Pkt Len Max',
            'fwd_pkt_len_min': 'Fwd Pkt Len Min', 'fwd_pkt_len_mean': 'Fwd Pkt Len Mean',
            'fwd_pkt_len_std': 'Fwd Pkt Len Std', 'bwd_pkt_len_max': 'Bwd Pkt Len Max',
            'bwd_pkt_len_min': 'Bwd Pkt Len Min', 'bwd_pkt_len_mean': 'Bwd Pkt Len Mean',
            'bwd_pkt_len_std': 'Bwd Pkt Len Std', 'flow_byts_s': 'Flow Byts/s',
            'flow_pkts_s': 'Flow Pkts/s', 'flow_iat_mean': 'Flow IAT Mean',
            'flow_iat_std': 'Flow IAT Std', 'flow_iat_max': 'Flow IAT Max',
            'flow_iat_min': 'Flow IAT Min', 'fwd_iat_tot': 'Fwd IAT Tot',
            'fwd_iat_mean': 'Fwd IAT Mean', 'fwd_iat_std': 'Fwd IAT Std',
            'fwd_iat_max': 'Fwd IAT Max', 'fwd_iat_min': 'Fwd IAT Min',
            'bwd_iat_tot': 'Bwd IAT Tot', 'bwd_iat_mean': 'Bwd IAT Mean',
            'bwd_iat_std': 'Bwd IAT Std', 'bwd_iat_max': 'Bwd IAT Max',
            'bwd_iat_min': 'Bwd IAT Min', 'fwd_psh_flags': 'Fwd PSH Flags',
            'bwd_psh_flags': 'Bwd PSH Flags', 'fwd_urg_flags': 'Fwd URG Flags',
            'bwd_urg_flags': 'Bwd URG Flags', 'fwd_header_len': 'Fwd Header Len',
            'bwd_header_len': 'Bwd Header Len', 'fwd_pkts_s': 'Fwd Pkts/s',
            'bwd_pkts_s': 'Bwd Pkts/s', 'pkt_len_min': 'Pkt Len Min',
            'pkt_len_max': 'Pkt Len Max', 'pkt_len_mean': 'Pkt Len Mean',
            'pkt_len_std': 'Pkt Len Std', 'pkt_len_var': 'Pkt Len Var',
            'fin_flag_cnt': 'FIN Flag Cnt', 'syn_flag_cnt': 'SYN Flag Cnt',
            'rst_flag_cnt': 'RST Flag Cnt', 'psh_flag_cnt': 'PSH Flag Cnt',
            'ack_flag_cnt': 'ACK Flag Cnt', 'urg_flag_cnt': 'URG Flag Cnt',
            'cwe_flag_count': 'CWE Flag Count', 'ece_flag_cnt': 'ECE Flag Cnt',
            'down_up_ratio': 'Down/Up Ratio', 'pkt_size_avg': 'Pkt Size Avg',
            'fwd_seg_size_avg': 'Fwd Seg Size Avg', 'bwd_seg_size_avg': 'Bwd Seg Size Avg',
            'fwd_byts_b_avg': 'Fwd Byts/b Avg', 'fwd_pkts_b_avg': 'Fwd Pkts/b Avg',
            'fwd_blk_rate_avg': 'Fwd Blk Rate Avg', 'bwd_byts_b_avg': 'Bwd Byts/b Avg',
            'bwd_pkts_b_avg': 'Bwd Pkts/b Avg', 'bwd_blk_rate_avg': 'Bwd Blk Rate Avg',
            'subflow_fwd_pkts': 'Subflow Fwd Pkts', 'subflow_fwd_byts': 'Subflow Fwd Byts',
            'subflow_bwd_pkts': 'Subflow Bwd Pkts', 'subflow_bwd_byts': 'Subflow Bwd Byts',
            'init_fwd_win_byts': 'Init Fwd Win Byts', 'init_bwd_win_byts': 'Init Bwd Win Byts',
            'fwd_act_data_pkts': 'Fwd Act Data Pkts', 'fwd_seg_size_min': 'Fwd Seg Size Min',
            'active_mean': 'Active Mean', 'active_std': 'Active Std',
            'active_max': 'Active Max', 'active_min': 'Active Min',
            'idle_mean': 'Idle Mean', 'idle_std': 'Idle Std',
            'idle_max': 'Idle Max', 'idle_min': 'Idle Min'
        }
    
    def _load_pipeline(self):
        """Load the complete CNN+MHA pipeline"""
        try:
            rospy.loginfo("📥 Loading CNN+MHA pipeline components...")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            self.scaler = joblib.load(scaler_path)
            rospy.loginfo("✅ Loaded scaler.joblib")
            
            # Load feature extractor (CNN+MHA)
            feature_extractor_path = os.path.join(self.model_dir, 'feature_extractor.keras')
            self.feature_extractor = load_model(feature_extractor_path, compile=False)
            
            # Compile the model to suppress warning
            self.feature_extractor.compile(optimizer='adam', loss='categorical_crossentropy')
            rospy.loginfo("✅ Loaded feature_extractor.keras")
            
            # Load PCA
            pca_path = os.path.join(self.model_dir, 'pca.joblib')
            self.pca = joblib.load(pca_path)
            rospy.loginfo("✅ Loaded pca.joblib")
            
            # Load KNN classifier
            knn_path = os.path.join(self.model_dir, 'knn_classifier.joblib')
            self.knn_classifier = joblib.load(knn_path)
            rospy.loginfo("✅ Loaded knn_classifier.joblib")
            
            # Load Random Forest classifier
            rf_path = os.path.join(self.model_dir, 'rf_classifier.joblib')
            self.rf_classifier = joblib.load(rf_path)
            rospy.loginfo("✅ Loaded rf_classifier.joblib")
            
            # Load Logistic Regression classifier
            lr_path = os.path.join(self.model_dir, 'lr_classifier.joblib')
            self.lr_classifier = joblib.load(lr_path)
            rospy.loginfo("✅ Loaded lr_classifier.joblib")
            
            # Load feature names
            features_path = os.path.join(self.model_dir, 'features.txt')
            with open(features_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            rospy.loginfo(f"✅ Loaded {len(self.feature_names)} feature names")
            
            # Load pipeline info
            pipeline_info_path = os.path.join(self.model_dir, 'pipeline_info.json')
            with open(pipeline_info_path, 'r') as f:
                self.pipeline_info = json.load(f)
            
            # Extract class names
            if 'class_names' in self.pipeline_info:
                self.class_names = self.pipeline_info['class_names']
            else:
                # Fallback class names
                self.class_names = ["Normal", "DoS Attack", "UnauthSub Attack", 
                                  "SSH Bruteforce", "Pubflood", "Subflood", 
                                  "Reverse Shell", "Port Scanning Attack"]
            
            rospy.loginfo("✅ Loaded pipeline_info.json")
            rospy.loginfo(f"🎯 Model Architecture: {self.pipeline_info.get('model_architecture', 'CNN+MHA Pipeline')}")
            rospy.loginfo(f"🔢 Number of classes: {self.pipeline_info.get('num_classes', 8)}")
            rospy.loginfo(f"📊 Feature flow: {self.pipeline_info.get('feature_flow', 'N/A')}")
            
        except Exception as e:
            rospy.logerr(f"❌ Failed to load pipeline: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            sys.exit(1)
    
    def _apply_complete_pipeline(self, X):
        """Apply the complete CNN+MHA pipeline for prediction"""
        try:
            # Step 1: Scale the input features (convert to numpy to avoid feature name warnings)
            X_array = X.values if hasattr(X, 'values') else X
            X_scaled = self.scaler.transform(X_array)
            
            # Step 2: Reshape for CNN+MHA (add time dimension)
            n_samples, n_features = X_scaled.shape
            X_reshaped = X_scaled.reshape(n_samples, n_features, 1)
            
            # Step 3: Extract features using CNN+MHA
            cnn_mha_features = self.feature_extractor.predict(X_reshaped, verbose=0)
            
            # Step 4: Apply PCA
            pca_features = self.pca.transform(cnn_mha_features)
            
            # Step 5: Get probabilities from KNN and RF
            knn_proba = self.knn_classifier.predict_proba(pca_features)
            rf_proba = self.rf_classifier.predict_proba(pca_features)
            
            # Step 6: Concatenate probabilities
            combined_features = np.concatenate([knn_proba, rf_proba], axis=1)
            
            # Step 7: Final prediction with Logistic Regression
            final_predictions = self.lr_classifier.predict(combined_features)
            final_probabilities = self.lr_classifier.predict_proba(combined_features)
            
            return final_predictions, final_probabilities
            
        except Exception as e:
            rospy.logerr(f"❌ Pipeline prediction error: {str(e)}")
            return None, None
    
    def _prepare_features(self, df):
        """Prepare and validate features for the model"""
        try:
            # Apply column mapping
            df_mapped = df.copy()
            
            # Rename columns using mapping
            for csv_col, model_col in self.column_mapping.items():
                if csv_col in df_mapped.columns:
                    df_mapped = df_mapped.rename(columns={csv_col: model_col})
            
            # Select only the features the model expects
            available_features = []
            missing_features = []
            
            for feature in self.feature_names:
                if feature in df_mapped.columns:
                    available_features.append(feature)
                else:
                    missing_features.append(feature)
            
            if missing_features:
                rospy.logwarn(f"⚠️ Missing {len(missing_features)} features: {missing_features[:5]}...")
            
            # Create feature matrix with all expected features
            X = pd.DataFrame(0.0, index=df_mapped.index, columns=self.feature_names)
            
            # Fill in available features
            for feature in available_features:
                X[feature] = pd.to_numeric(df_mapped[feature], errors='coerce')
            
            # Handle infinite and missing values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(0.0, inplace=True)
            
            rospy.loginfo(f"📊 Prepared {len(available_features)}/{len(self.feature_names)} features")
            return X.values
            
        except Exception as e:
            rospy.logerr(f"❌ Feature preparation error: {str(e)}")
            return None
    
    def _process_csv_file(self, file_path):
        """Process a single CSV file"""
        try:
            rospy.loginfo(f"📄 Processing file: {os.path.basename(file_path)}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            rospy.logdebug(f"📊 Loaded {len(df)} rows from {file_path}")
            
            # Handle incremental processing
            start_idx = 0
            if file_path in self.last_processed_rows:
                start_idx = self.last_processed_rows[file_path]
            
            # Process only new rows
            new_rows = df.iloc[start_idx:]
            if len(new_rows) == 0:
                return
                
            rospy.loginfo(f"🔄 Processing {len(new_rows)} new rows (starting from row {start_idx})")
            
            # Remove label columns if present
            feature_df = new_rows.drop(columns=['Label'], errors='ignore')
            feature_df = feature_df.drop(columns=['label'], errors='ignore')
            
            # Prepare features
            X = self._prepare_features(feature_df)
            if X is None:
                return
                
            # Make predictions
            predictions, probabilities = self._apply_complete_pipeline(X)
            if predictions is None:
                return
            
            # Process predictions
            for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
                self.total_predictions += 1
                
                # Get attack class name
                attack_class = self.class_names[pred] if pred < len(self.class_names) else f"Unknown({pred})"
                confidence = np.max(proba)
                
                # Create alert message
                alert_data = {
                    'timestamp': datetime.now().isoformat(),
                    'file': os.path.basename(file_path),
                    'row': int(start_idx + i),
                    'prediction': int(pred),
                    'attack_class': str(attack_class),
                    'confidence': float(confidence),
                    'is_attack': bool(pred != 0),
                    'probabilities': [float(p) for p in proba]
                }
                
                # Log and publish alerts for attacks
                if pred != 0:  # Not normal traffic
                    self.attack_detections += 1
                    rospy.logwarn(f"🚨 ATTACK DETECTED: {attack_class} (confidence: {confidence:.3f})")
                    
                    # Publish ROS message
                    alert_msg = String()
                    alert_msg.data = json.dumps(alert_data)
                    self.alert_publisher.publish(alert_msg)
                else:
                    rospy.logdebug(f"✅ Normal traffic (confidence: {confidence:.3f})")
            
            # Update processed rows counter
            self.last_processed_rows[file_path] = len(df)
            
            rospy.loginfo(f"✅ Processed {len(new_rows)} rows from {os.path.basename(file_path)}")
            
        except Exception as e:
            rospy.logerr(f"❌ Error processing {file_path}: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def _scan_existing_files(self):
        """Scan for existing CSV files in the directory"""
        try:
            # Focus specifically on output.csv
            target_file = os.path.join(self.csv_dir, "output.csv")
            
            if os.path.exists(target_file):
                rospy.loginfo(f"🔍 Found target file: output.csv")
                if target_file not in self.processed_files:
                    self._process_csv_file(target_file)
                    self.processed_files.add(target_file)
            else:
                rospy.loginfo(f"📂 Target file output.csv not found in {self.csv_dir}")
                rospy.loginfo("   Will monitor directory for when output.csv appears...")
                
        except Exception as e:
            rospy.logerr(f"❌ Error scanning existing files: {str(e)}")
    
    def _start_file_monitoring(self):
        """Start monitoring for new CSV files"""
        try:
            # Process existing files first
            self._scan_existing_files()
            
            # Set up file system watcher
            event_handler = CSVFileHandler(self)
            observer = Observer()
            observer.schedule(event_handler, self.csv_dir, recursive=False)
            observer.start()
            
            rospy.loginfo(f"👁️ Started file monitoring on {self.csv_dir}")
            
            # Keep observer running
            self.observer = observer
            
        except Exception as e:
            rospy.logerr(f"❌ Failed to start file monitoring: {str(e)}")
    
    def _print_statistics(self):
        """Print detection statistics"""
        uptime = time.time() - self.start_time
        attack_rate = (self.attack_detections / max(self.total_predictions, 1)) * 100
        
        rospy.loginfo("📈 CNN+MHA IDS Statistics:")
        rospy.loginfo(f"   ⏱️ Uptime: {uptime:.1f}s")
        rospy.loginfo(f"   🔢 Total predictions: {self.total_predictions}")
        rospy.loginfo(f"   🚨 Attack detections: {self.attack_detections}")
        rospy.loginfo(f"   📊 Attack rate: {attack_rate:.2f}%")
        rospy.loginfo(f"   📁 Files processed: {len(self.processed_files)}")
    
    def run(self):
        """Main execution loop"""
        rospy.loginfo("🎯 CNN+MHA IDS Node is running...")
        
        # Print statistics periodically
        rate = rospy.Rate(0.1)  # 10-second intervals
        last_stats_time = 0
        
        try:
            while not rospy.is_shutdown() and self.monitoring:
                current_time = time.time()
                
                # Print statistics every 60 seconds
                if current_time - last_stats_time > 60:
                    self._print_statistics()
                    last_stats_time = current_time
                
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("👋 Shutting down CNN+MHA IDS Node...")
        finally:
            self.monitoring = False
            if hasattr(self, 'observer'):
                self.observer.stop()
                self.observer.join()
            
            # Final statistics
            self._print_statistics()
            rospy.loginfo("✅ CNN+MHA IDS Node shutdown complete")


class CSVFileHandler(FileSystemEventHandler):
    """Handler for CSV file system events"""
    
    def __init__(self, ids_node):
        self.ids_node = ids_node
        super().__init__()
    
    def on_created(self, event):
        """Handle new file creation"""
        if not event.is_directory and event.src_path.endswith('output.csv'):
            rospy.loginfo(f"📝 Target file detected: {os.path.basename(event.src_path)}")
            # Wait a moment for file to be fully written
            time.sleep(1)
            self.ids_node._process_csv_file(event.src_path)
            self.ids_node.processed_files.add(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification (new data appended)"""
        if not event.is_directory and event.src_path.endswith('output.csv'):
            if event.src_path in self.ids_node.processed_files:
                rospy.logdebug(f"📝 Target file updated: {os.path.basename(event.src_path)}")
                # Wait a moment for file to be fully written
                time.sleep(0.5)
                self.ids_node._process_csv_file(event.src_path)


def main():
    """Main function"""
    try:
        # Print system information
        print("\n" + "="*60)
        print("🛡️  CNN+MHA INTRUSION DETECTION SYSTEM")
        print("="*60)
        print(f"🐧 Platform: Ubuntu Linux")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print(f"🧠 TensorFlow: {tf.__version__}")
        print(f"⚙️ Node: cnn_mha_ids_nodev3.py")
        print("="*60 + "\n")
        
        # Create and run IDS node
        ids_node = CNNMHAIDSNode()
        ids_node.run()
        
    except Exception as e:
        rospy.logerr(f"❌ Fatal error: {str(e)}")
        import traceback
        rospy.logerr(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()