# 📊 External Dataset References

This directory contains references to external datasets used in the ROS-Guard project.

## 🎯 Primary Dataset: NavBot25

### Dataset Information:
- **Name**: NavBot25 ROS Security Dataset  
- **Total Samples**: 192,213 network traffic records
- **Attack Types**: 8 categories including Normal, DoS, SSH Bruteforce, Port Scanning
- **Features**: 78 network traffic characteristics extracted via CICFlowMeter
- **Source**: Real TurtleBot3 navigation system traffic

### 🔗 Dataset Links:

**Primary Repository:**
- **Zenodo**: https://zenodo.org/records/16758080
- **GitHub**: https://github.com/Blackthorn23/INTRUSION-DETECTION-FOR-ROBOT-OPERATING-SYSTEM-BASED-AUTONOMOUS-ROBOT-NAVIGATION-

### 📁 Dataset Contents:
- **Raw PCAP Files**: Original network packet captures from robot operations
- **Processed CSV Files**: CICFlowMeter-extracted features for ML training
- **Attack Scenarios**: Labeled data for 8 different attack types
- **Normal Operations**: Baseline traffic from legitimate robot navigation

### 📈 Attack Distribution:
- **Normal Traffic**: 33.3% of dataset
- **DoS Attack**: 15.90%  
- **Reverse Shell**: 15.40%
- **Port Scanning Attack**: 14.30%
- **UnauthSub Attack**: Various percentages
- **SSH Bruteforce**: Various percentages  
- **Pubflood**: Various percentages
- **Subflood**: Various percentages

### 💾 File Size Note:
Due to GitHub file size limitations, the complete dataset files are hosted externally. Please download from the provided Zenodo link for full access to:
- Complete PCAP captures
- Full CSV feature files  
- All attack scenario data

### 🤝 Usage in ROS-Guard:
This dataset serves as the training foundation for all five machine learning models in the ROS-Guard system:
- **Primary Models**: AC-CNN-LSTM, AC-CNN-MHA  
- **Experimental Models**: AC-MI-RF, AC-MI-KNN, AC-MI-DT

### 📚 Research Citation:
When using this dataset, please cite the original research work and the Zenodo repository for proper attribution to the research community.
