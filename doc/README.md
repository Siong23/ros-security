# Documentation for this repo

I will be uploading what I have used in this project for reusability.

# CICFlowMeter Installation on Ubuntu

This guide explains how to install and run **CICFlowMeter** on Ubuntu for network traffic feature extraction.  

## Steps  

1. **Update packages**  
   ```bash
   sudo apt update
2. **Remove old CICFlowMeter (if exists)**
   ````bash
   sudo rm -rf CICFlowMeter
3. Install dependencies
   ````bash
   sudo apt install -y openjdk-11-jdk maven unzip libpcap0.8 libpcap-dev
4. Clone the CICFlowMeter repository
   ````bash
   git clone https://github.com/ahlashkari/CICFlowMeter.git
5. Navigate to the directory
   ````bash
   cd CICFlowMeter
6. Install JNetPcap library
   ````bash
   sudo mvn install:install-file \
   Dfile=jnetpcap/linux/jnetpcap-1.4.r1425/jnetpcap.jar \
   DgroupId=org.jnetpcap \
   DartifactId=jnetpcap \
   Dversion=1.4.1 \
   Dpackaging=jar
7. Give execution permissions
   ````bash
   chmod +x gradlew
8. Install Java 8
   ````bash
   sudo apt install -y openjdk-8-jdk
9. Configure default Java version
    ````bash
    sudo update-alternatives --config java
> Select Java 8 from the list.
10. Verify Java version
    ````bash
    java -version

11. Run CICFlowMeter
````bash
sudo ./gradlew execute
````
# How to use CICFlowMeter
