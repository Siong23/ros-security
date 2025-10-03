#!/usr/bin/env python3
"""
Fixed CNN+LSTM IDS Node - Complete Pipeline Implementation
=========================================================

This version properly implements the full transformation pipeline:
Raw features (78) -> Scaler -> PCA (16) -> KNN+RF probabilities -> Fusion (32) -> LogisticRegression

Key changes from original:
1. Loads all pipeline components (scaler, pca, knn, rf, lr)
2. Applies complete transformation sequence  
3. Handles feature alignment correctly
4. Includes error handling and logging

Usage:
1. Copy all .joblib files from notebook to MODEL_DIR
2. Update file paths below
3. Run: python accnnlstm-ids_node_fixed.py
"""

import rospy
from std_msgs.msg import String
import pandas as pd
import numpy as np
from joblib import load
import json
import os

# =============================================================================
# CONFIGURATION - Update these paths for your deployment
# =============================================================================
CSV_PATH = "/home/jakelcj/output.csv"
MODEL_DIR = "/home/jakelcj/ids_ws/src/ros_ids/models/cnnlstm/"  # Directory containing all .joblib files

# Individual component paths
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
PCA_PATH = os.path.join(MODEL_DIR, "pca.joblib") 
KNN_PATH = os.path.join(MODEL_DIR, "knn_classifier.joblib")
RF_PATH = os.path.join(MODEL_DIR, "rf_classifier.joblib")
LR_PATH = os.path.join(MODEL_DIR, "lr_classifier.joblib")  # or model.joblib
FEATURES_PATH = os.path.join(MODEL_DIR, "features.txt")
PIPELINE_INFO_PATH = os.path.join(MODEL_DIR, "pipeline_info.json")

# Output log
OUTPUT_LOG_PATH = "/home/jakelcj/accnnlstm-ids_output.log"

def load_complete_pipeline():
    """
    Load the complete CNN+LSTM fusion pipeline.
    
    Returns:
        tuple: (scaler, pca, knn, rf, lr, features, pipeline_info)
    """
    try:
        # Load all components
        scaler = load(SCALER_PATH)
        pca = load(PCA_PATH) 
        knn_classifier = load(KNN_PATH)
        rf_classifier = load(RF_PATH)
        lr_classifier = load(LR_PATH)
        
        # Load feature names
        with open(FEATURES_PATH, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        
        # Load pipeline info (optional)
        pipeline_info = {}
        if os.path.exists(PIPELINE_INFO_PATH):
            with open(PIPELINE_INFO_PATH, 'r') as f:
                pipeline_info = json.load(f)
        
        # Validate pipeline components
        rospy.loginfo(f"✅ Loaded scaler: {type(scaler).__name__}")
        rospy.loginfo(f"✅ Loaded PCA: {pca.n_components_} components") 
        rospy.loginfo(f"✅ Loaded KNN: {knn_classifier.n_neighbors} neighbors")
        rospy.loginfo(f"✅ Loaded RF: {rf_classifier.n_estimators} trees")
        rospy.loginfo(f"✅ Loaded LR: {lr_classifier.n_features_in_} input features")
        rospy.loginfo(f"✅ Loaded {len(features)} original features")
        
        if pipeline_info:
            rospy.loginfo(f"✅ Pipeline info: {pipeline_info.get('pipeline_type', 'Unknown')}")
        
        return scaler, pca, knn_classifier, rf_classifier, lr_classifier, features, pipeline_info
        
    except Exception as e:
        rospy.logerr(f"❌ Failed to load pipeline: {e}")
        raise

def apply_complete_pipeline(raw_data, scaler, pca, knn, rf, lr):
    """
    Apply the complete transformation pipeline to raw network features.
    
    Args:
        raw_data (pd.DataFrame): Raw network features (78 columns)
        scaler: Fitted StandardScaler
        pca: Fitted PCA transformer  
        knn: Fitted KNN classifier
        rf: Fitted RandomForest classifier
        lr: Fitted LogisticRegression classifier
        
    Returns:
        np.ndarray: Final predictions
    """
    try:
        # Step 1: Scale raw features (78 -> 78)
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
    Prepare and align features from CSV data.
    
    Args:
        df (pd.DataFrame): Raw CSV data
        expected_features (list): List of expected feature names
        
    Returns:
        pd.DataFrame: Aligned feature matrix
    """
    try:
        # Ensure all expected features are present
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0  # Fill missing features with 0
                rospy.logwarn(f"⚠️  Missing feature '{feat}' filled with 0")
        
        # Select and reorder features
        df_aligned = df[expected_features].copy()
        
        # Convert to numeric and handle NaN/inf
        df_aligned = df_aligned.apply(pd.to_numeric, errors='coerce')
        df_aligned = df_aligned.fillna(0)
        df_aligned = df_aligned.replace([np.inf, -np.inf], 0)
        
        rospy.logdebug(f"Feature alignment: {df_aligned.shape}")
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
        # Get class mappings
        class_names = pipeline_info.get('attack_classes', {
            0: "Normal", 1: "DoS Attack", 2: "UnauthSub Attack", 3: "SSH Bruteforce",
            4: "Pubflood", 5: "Subflood", 6: "Reverse Shell", 7: "Port Scanning Attack"
        })
        
        messages = []
        for pred in predictions:
            class_name = class_names.get(int(pred), f"Unknown({pred})")
            if int(pred) == 0:
                msg = f"Normal traffic: {class_name}"
            else:
                msg = f"ALERT: Attack detected - {class_name}"
            messages.append(msg)
        
        return messages
        
    except Exception as e:
        rospy.logerr(f"❌ Message formatting failed: {e}")
        return [f"Prediction: {pred}" for pred in predictions]

def ids_node():
    """Main IDS node function."""
    rospy.init_node("cnnlstm_ids_node", anonymous=True)
    pub = rospy.Publisher("/ids/alerts", String, queue_size=10)
    rate = rospy.Rate(2)  # 2 Hz
    
    # Load complete pipeline
    try:
        scaler, pca, knn, rf, lr, features, pipeline_info = load_complete_pipeline()
        rospy.loginfo("🚀 CNN+LSTM IDS Node initialized successfully")
    except Exception as e:
        rospy.logerr(f"💥 Failed to initialize IDS node: {e}")
        return
    
    seen_rows = 0
    
    while not rospy.is_shutdown():
        try:
            # Load CSV data
            if not os.path.exists(CSV_PATH):
                rospy.logwarn_throttle(10, f"⚠️  CSV file not found: {CSV_PATH}")
                rate.sleep()
                continue
                
            df = pd.read_csv(CSV_PATH)
            
            if df.shape[0] > seen_rows:
                # Process only new rows
                new_df = df.iloc[seen_rows:].copy()
                rospy.loginfo(f"📊 Processing {len(new_df)} new samples")
                
                # Prepare features
                aligned_features = prepare_features(new_df, features)
                
                # Apply complete pipeline
                predictions = apply_complete_pipeline(
                    aligned_features, scaler, pca, knn, rf, lr
                )
                
                # Format and publish messages
                messages = format_prediction_message(predictions, pipeline_info)
                
                # Log and publish each prediction
                with open(OUTPUT_LOG_PATH, "a", encoding='utf-8') as log_file:
                    for msg in messages:
                        rospy.loginfo(msg)
                        pub.publish(msg)
                        log_file.write(f"{rospy.Time.now()}: {msg}\n")
                
                # Update counter
                seen_rows = df.shape[0]
                
                # Summary statistics
                attack_count = np.sum(predictions != 0)
                normal_count = len(predictions) - attack_count
                rospy.loginfo(f"📈 Batch summary: {normal_count} normal, {attack_count} attacks")
                
            else:
                rospy.logdebug("No new data to process")
                
        except Exception as e:
            rospy.logerr(f"❌ IDS processing error: {e}")
            
        rate.sleep()

if __name__ == "__main__":
    try:
        ids_node()
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 CNN+LSTM IDS Node stopped")
    except Exception as e:
        rospy.logerr(f"💥 Fatal error: {e}")