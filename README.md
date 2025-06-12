# 🧮 Matrix Operation Thread Optimizer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Site-blue)](https://thread-pt4k.onrender.com)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/ananthakr1shnan/Thread)

A machine-learning-powered tool that predicts the **optimal number of OpenMP threads** required for efficient execution of matrix operations. Say goodbye to manual tuning — this optimizer intelligently learns the best thread configurations for you.

---

## 📌 Overview

The **Matrix Operation Thread Optimizer** uses ML models trained on real execution data to recommend thread counts based on matrix dimensions and operation types. It's designed for high-performance computing scenarios where thread tuning can make a huge difference.

---

## ✨ Features

- 🧠 **Smart Thread Prediction** — ML model trained on benchmark data  
- 🔢 **Multiple Matrix Operations** — Supports a wide range from basic to advanced  
- 🌐 **Web Interface** — Lightweight UI for real-time predictions  
- 🚀 **Performance-Driven** — Up to 40% faster execution with 90%+ prediction accuracy

---

## ⚙️ How It Works

1. **Data Collection**  
   Benchmarked various matrix operations using different thread counts and captured performance metrics.

2. **Model Training**  
   A variety of supervised learning models were tested to predict the optimal thread count, including:
   Basic classifiers (Logistic Regression, Decision Trees, Random Forests),Gradient Boosting models,XGBoost,Neural Networks

   After extensive experimentation, XGBoost consistently delivered the highest accuracy and fastest inference.
   As a result, XGBoost was selected as the final model for production use

3. **Real-Time Prediction**  
   Users input matrix details through the web UI, and the model predicts the optimal thread count instantly.

---

## 🧪 Supported Matrix Operations

- Matrix Multiplication  
- Matrix Addition  
- Matrix Transposition  
- Matrix Determinant  
- Matrix Eigenvalue  
- Matrix LU Decomposition  
- Matrix Exponential  
- Matrix Logarithm  
- Matrix Scaling  
- Matrix Square Root

---

## 🖥️ Try It Out

### ▶️ Live Demo  
[🔗 https://thread-pt4k.onrender.com](https://thread-pt4k.onrender.com)

### 🛠️ Run Locally

#### 📋 Prerequisites
- Python 3.8+
- GCC or any OpenMP-compatible compiler

#### 🧱 Setup

```bash
# Clone the repository
git clone https://github.com/ananthakr1shnan/Thread.git
cd Thread

# Install required Python packages
pip install -r requirements.txt

# Start the Flask server
python app.py
