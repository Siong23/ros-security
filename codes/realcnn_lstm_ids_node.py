#!/usr/bin/env python3
"""
CNN+LSTM IDS Node for Ubuntu Linux
Real-time Intrusion Detection System using CNN+LSTM with KNN+RF fusion

Author: eusoff "ae1207" aminurrashid  
Date: 6 oct 2025
Version: 2.0

Dependencies:
    sudo apt update
    sudo apt install python3-pip python3-dev
    pip3 install rospy std_msgs pandas numpy scikit-learn joblib tensorflow glob2 watchdog

Usage:
    rosrun ros_ids cnn_lstm_ids_node_linux.py
    
Model Files Required:
    - scaler.joblib
    - feature_extractor.keras  
    - pca.joblib
    - knn_classifier.joblib
    - rf_classifier.joblib
    - lr_classifier.joblib
    - features.txt
    - class_mapping.json
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
    print(f"❌ joblib import failed: {e}")
    print("💡 Install with: pip3 install joblib")
    sys.exit(1)

try:
    import sklearn
    print(f"✅ scikit-learn imported successfully (version: {sklearn.__version__})")
except ImportError as e:
    print(f"❌ scikit-learn import failed: {e}")
    print("💡 Install with: pip3 install scikit-learn")
    sys.exit(1)

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print(f"✅ TensorFlow imported successfully (version: {tf.__version__})")
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    print("💡 Install with: pip3 install tensorflow")
    sys.exit(1)

# ROS
try:
    from std_msgs.msg import String
    print("✅ ROS std_msgs imported successfully")
except ImportError as e:
    print(f"❌ ROS import failed: {e}")
    print("💡 Make sure ROS is properly installed and sourced")
    sys.exit(1)

# File monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    print("✅ Watchdog imported successfully")
except ImportError as e:
    print(f"❌ Watchdog import failed: {e}")
    print("💡 Install with: pip3 install watchdog")
    sys.exit(1)

class CNNLSTMIDSNode:
    """CNN+LSTM-based Intrusion Detection System Node"""
    
    def __init__(self):
        """Initialize the CNN+LSTM IDS Node"""
        
        print("\n" + "="*60)
        print("🛡️  CNN+LSTM INTRUSION DETECTION SYSTEM")
        print("="*60)
        print("🐧 Platform: Ubuntu Linux")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print(f"🧠 TensorFlow: {tf.__version__}")
        print("⚙️ Node: cnn_lstm_ids_node_linux.py")
        print("="*60)
        
        # Initialize ROS node
        rospy.init_node('cnn_lstm_ids_node', anonymous=True)
        rospy.loginfo("🚀 Starting CNN+LSTM IDS Node...")
        
        # Model directory
        self.model_dir = self._get_model_directory()
        rospy.loginfo(f"📁 Found model directory: {self.model_dir}")
        
        # CSV monitoring directory  
        self.csv_dir = self._get_csv_directory()
        
        # Initialize components
        self.scaler = None
        self.feature_extractor = None
        self.pca = None
        self.knn_classifier = None
        self.rf_classifier = None
        self.lr_classifier = None
        self.feature_names = []
        self.class_names = []
        self.pipeline_info = {}
        
        # Processing tracking
        self.processed_files = set()
        self.last_processed_rows = {}
        
        # Statistics
        self.total_predictions = 0
        self.attack_detections = 0
        self.start_time = time.time()
        
        # ROS publisher
        self.alert_publisher = rospy.Publisher('/ids_alerts', String, queue_size=10)
        
        # Load models
        self._load_models()
        
        # File monitoring
        self.file_handler = CSVFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(self.file_handler, self.csv_dir, recursive=False)
        
        # Scan existing files
        self._scan_existing_files()
        
    def _get_model_directory(self):
        """Get the model directory path"""
        possible_paths = [
            "/home/jakelcj/ids_ws/src/ros_ids/models/navbot25_ac_cnn_lstm_complete",
            "/home/jakelcj/ids_ws/src/ros_ids/models/cnnlstm",
            "~/ids_ws/src/ros_ids/models/navbot25_ac_cnn_lstm_complete",
            "./navbot25_ac_cnn_lstm_complete"
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
                
        # If not found, create default path
        default_path = "/home/jakelcj/ids_ws/src/ros_ids/models/navbot25_ac_cnn_lstm_complete"
        rospy.logwarn(f"⚠️ Model directory not found. Expected: {default_path}")
        return default_path
    
    def _get_csv_directory(self):
        """Get the CSV monitoring directory"""
        return "/home/jakelcj"
    
    def _load_models(self):
        """Load all CNN+LSTM pipeline components"""
        rospy.loginfo("📥 Loading CNN+LSTM pipeline components...")
        
        try:
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            self.scaler = joblib.load(scaler_path)
            rospy.loginfo("✅ Loaded scaler.joblib")
            
            # Load feature extractor (CNN+LSTM)
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
            
            # Load class mapping
            class_mapping_path = os.path.join(self.model_dir, 'class_mapping.json')
            with open(class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            # Convert to list indexed by class ID
            self.class_names = [''] * len(class_mapping)
            for name, idx in class_mapping.items():
                self.class_names[idx] = name
            
            # Load pipeline info
            pipeline_info_path = os.path.join(self.model_dir, 'pipeline_info.json')
            with open(pipeline_info_path, 'r') as f:
                self.pipeline_info = json.load(f)
            rospy.loginfo("✅ Loaded pipeline_info.json")
            
            # Display pipeline info
            rospy.loginfo("🎯 Model Architecture: CNN+LSTM -> PCA -> KNN+RF -> LogisticRegression")
            rospy.loginfo(f"🔢 Number of classes: {len(self.class_names)}")
            rospy.loginfo(f"📊 Feature flow: {self.pipeline_info.get('feature_flow', 'N/A')}")
            
        except Exception as e:
            rospy.logerr(f"❌ Error loading models: {str(e)}")
            raise e
    
    def _prepare_features(self, df):
        """Prepare features from dataframe"""
        try:
            # Ensure we have the expected number of features
            if len(df.columns) != len(self.feature_names):
                rospy.logwarn(f"⚠️ Feature mismatch: got {len(df.columns)}, expected {len(self.feature_names)}")
                return None
            
            # Align column order with trained model
            df_aligned = df[self.feature_names].copy()
            
            rospy.loginfo(f"📊 Prepared {len(self.feature_names)}/{len(self.feature_names)} features")
            return df_aligned
            
        except Exception as e:
            rospy.logerr(f"❌ Error preparing features: {str(e)}")
            return None
    
    def _apply_complete_pipeline(self, X):
        """Apply the complete CNN+LSTM pipeline for prediction"""
        try:
            # Step 1: Scale the input features (convert to numpy to avoid feature name warnings)
            X_array = X.values if hasattr(X, 'values') else X
            X_scaled = self.scaler.transform(X_array)
            
            # Step 2: Reshape for CNN+LSTM (add time dimension)
            n_samples, n_features = X_scaled.shape
            timesteps = 1  # Single timestep for real-time data
            X_reshaped = X_scaled.reshape(n_samples, timesteps, n_features)
            
            # Step 3: Extract features using CNN+LSTM
            extracted_features = self.feature_extractor.predict(X_reshaped, verbose=0)
            
            # Step 4: Apply PCA for dimensionality reduction  
            pca_features = self.pca.transform(extracted_features)
            
            # Step 5: Get probability predictions from KNN and RF
            knn_proba = self.knn_classifier.predict_proba(pca_features)
            rf_proba = self.rf_classifier.predict_proba(pca_features)
            
            # Step 6: Fuse KNN and RF probabilities
            fused_features = np.hstack([knn_proba, rf_proba])
            
            # Step 7: Final prediction using Logistic Regression
            final_predictions = self.lr_classifier.predict(fused_features)
            final_probabilities = self.lr_classifier.predict_proba(fused_features)
            
            return final_predictions, final_probabilities
            
        except Exception as e:
            rospy.logerr(f"❌ Error in pipeline: {str(e)}")
            return None, None
    
    def _process_csv_file(self, file_path):
        """Process a CSV file for intrusion detection"""
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if we've processed this file before
            start_idx = self.last_processed_rows.get(file_path, 0)
            
            if len(df) <= start_idx:
                return  # No new rows
            
            # Get new rows
            new_rows = df.iloc[start_idx:].copy()
            
            if len(new_rows) == 0:
                return
            
            rospy.loginfo(f"🔄 Processing {len(new_rows)} new rows (starting from row {start_idx})")
            
            # Prepare features
            feature_df = new_rows.iloc[:, :-1] if new_rows.shape[1] > len(self.feature_names) else new_rows
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
                    rospy.loginfo(f"📄 Processing file: output.csv")
            else:
                rospy.logwarn(f"⚠️ Target file not found: output.csv")
                
        except Exception as e:
            rospy.logerr(f"❌ Error scanning files: {str(e)}")
    
    def start_monitoring(self):
        """Start file monitoring"""
        try:
            self.observer.start()
            rospy.loginfo(f"👁️ Started file monitoring on {self.csv_dir}")
            rospy.loginfo("✅ CNN+LSTM IDS Node initialized successfully")
            rospy.loginfo(f"📁 Model directory: {self.model_dir}")
            rospy.loginfo(f"📂 CSV directory: {self.csv_dir}")
            rospy.loginfo("🔍 Monitoring for new CSV files...")
            
            # Main loop
            rate = rospy.Rate(0.1)  # 0.1 Hz = 10 second intervals
            while not rospy.is_shutdown():
                rospy.loginfo("🎯 CNN+LSTM IDS Node is running...")
                
                # Print statistics
                uptime = time.time() - self.start_time
                attack_rate = (self.attack_detections / self.total_predictions * 100) if self.total_predictions > 0 else 0
                
                rospy.loginfo("📈 CNN+LSTM IDS Statistics:")
                rospy.loginfo(f"   ⏱️ Uptime: {uptime:.1f}s")
                rospy.loginfo(f"   🔢 Total predictions: {self.total_predictions}")
                rospy.loginfo(f"   🚨 Attack detections: {self.attack_detections}")
                rospy.loginfo(f"   📊 Attack rate: {attack_rate:.2f}%")
                rospy.loginfo(f"   📁 Files processed: {len(self.processed_files)}")
                
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("👋 Shutting down CNN+LSTM IDS Node...")
        except Exception as e:
            rospy.logerr(f"❌ Error in monitoring loop: {str(e)}")
        finally:
            self.observer.stop()
            self.observer.join()

class CSVFileHandler(FileSystemEventHandler):
    """Handle CSV file system events"""
    
    def __init__(self, ids_node):
        self.ids_node = ids_node
        
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory and event.src_path.endswith('output.csv'):
            rospy.loginfo(f"📄 New CSV file detected: {event.src_path}")
            self.ids_node._process_csv_file(event.src_path)
            self.ids_node.processed_files.add(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path.endswith('output.csv'):
            rospy.logdebug(f"📝 CSV file modified: {event.src_path}")
            self.ids_node._process_csv_file(event.src_path)

def main():
    """Main function"""
    try:
        # Create and start the CNN+LSTM IDS node
        ids_node = CNNLSTMIDSNode()
        ids_node.start_monitoring()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("👋 CNN+LSTM IDS Node interrupted")
    except Exception as e:
        rospy.logerr(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()