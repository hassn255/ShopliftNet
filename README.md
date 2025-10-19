# 🛒 Shoplifting Video Detection

---

## 🧭 Overview

This project focuses on **shoplifting behavior detection** from video footage using deep learning.  
The workflow is divided into two main parts:
1. **Data Cleaning** – Detect and remove duplicate videos to prevent data leakage.  
2. **Model Training** – Fine-tune a pretrained video classification model for final evaluation.

---

## 📊 Project Workflow

### 1️ - Data Cleaning & Preparation

- Extracted video features using a **pretrained R3D-18** model (`torchvision.models.video.r3d_18`).
- Applied **cosine similarity** between feature vectors within each class.
- Removed videos with a **similarity score above a defined threshold** (potential duplicates).
- Split the dataset into **Train** and **Validation** sets *after* cleaning to prevent data leakage.

> ⚙️ This step ensures a fair and generalizable evaluation of the models.

---

### 2️ - Video Classification Model

- Used a **pretrained R(2+1)D** model (`torchvision.models.video.r2plus1d`).
- Fine-tuned on the cleaned dataset with:
  - Input size: `16 × 256 × 256`
  - Optimizer: `Adam`
  - Learning Rate: `1e-3 -> 1e-4`
  - Device: **NVIDIA T4 GPU (Google Colab)**

#### 🧾 Evaluation Results

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.9434 |
| **Precision** | 0.9310 |
| **Recall** | 0.9643 |
| **F1-Score** | 0.9474 |

**Confusion Matrix**
```
[[46 4]
[ 2 54]]
```

**Classification Report**

|       | precision |   recall | f1-score |
|-------|-----------|----------|----------|
Non-Shoplifting |0.96| 0.92 |0.94
Shoplifting |0.93 |0.96 |0.95|

---

### ⚠️ ResNet50 + GRU (Data Leakage Case Study)

- Combined **pretrained ResNet50** for spatial features with a **GRU** for temporal modeling.
- Achieved perfect metrics (Accuracy = 1.0, F1 = 1.0) even after initial cleaning, indicating that residual data leakage was still present.
- The model’s high capacity enabled it to memorize subtle similarities between samples instead of learning generalizable features.

---

## Key Insights

- Data leakage can inflate performance metrics.
- Cleaning the dataset via **feature-based similarity** is a crucial step in video ML pipelines.  
- The **R(2+1)D** architecture provides a strong balance between temporal modeling and computational efficiency.

## Conclusion

Complex models are more prone to the effects of **data leakage**, since their higher capacity allows them to memorize patterns rather than generalize.  
After cleaning the dataset, performance normalized, reinforcing the importance of **robust data handling** before model evaluation.