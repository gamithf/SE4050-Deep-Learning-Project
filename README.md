# Depression Detection using Deep Learning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning project for detecting Major Depressive Disorder (MDD) using actigraphy data and demographic features. This project implements and compares four different deep learning architectures.

## 🎯 Project Overview

This project aims to classify depression states using motor activity data from wearable devices. We use a real-world dataset containing actigraphy recordings from both depressed patients and healthy controls.

### Models Implemented
1. **1D CNN** - Spatial pattern detection in activity data
2. **LSTM** - Temporal sequence modeling
3. **Bidirectional LSTM** - Enhanced temporal understanding
4. **Hybrid CNN-LSTM** - Combined spatial and temporal feature extraction

## 📊 Dataset

**Source**: [Depresjon Motor Activity Database](https://datasets.simula.no/depresjon/)
- **55 participants** (23 condition, 32 control)
- **1,571,706 activity records** with 1-minute intervals
- **16 features** including temporal, demographic, and clinical data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/depression-detection-dl.git
   cd depression-detection-dl

2. **Setup**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate # On Mac: source venv/bin/activate
   pip install -r requirements.txt

3. **For team**
   ```bash
   Choose your model folder
   Modify ONLY the create_model() function in model.py
   Test your model using train.py
   example. 
   cd models/01_cnn
   python train.py