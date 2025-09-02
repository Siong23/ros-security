# Tests Environment
Packets are 


- Normal is ran for 30 minutes - consists of small movements and idle

# CICFlowMeter Usage Guide

This repository demonstrates how to use **CICFlowMeter** to extract bidirectional network flows and generate statistical features (~80+) from PCAP files.  

CICFlowMeter is widely used in network intrusion detection and traffic analysis projects.

---

## 📌 What is CICFlowMeter?
- A Java-based tool that converts **PCAP network traffic** into **bidirectional flows**.  
- Extracts statistical features such as packet counts, durations, inter-arrival times, and byte distributions.  
- Generates a CSV file that can be used for **machine learning** and **network security research**.

---

## ⚡ Running CICFlowMeter

### Linux
```bash
java -jar CICFlowMeter.jar -i input.pcap -o flows.csv
