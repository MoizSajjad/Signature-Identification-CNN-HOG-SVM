# Signature Identification: Deep Learning vs. Traditional Machine Learning

This project presents a comprehensive comparison between two distinct methodologies for multiclass signature identification: a deep learning approach using **Convolutional Neural Networks (CNN)** and a traditional machine learning pipeline using **Histogram of Oriented Gradients (HOG)** combined with **Support Vector Machines (SVM)**.

## 📌 Project Overview
The objective is to accurately identify individuals based on their genuine handwritten signatures. The project classifies signatures into one of 64 unique person-ID categories, exploring the trade-offs between automatic feature extraction and manual feature engineering.

## 📊 Performance Summary
| Model | Test Accuracy | Macro F1-Score |
| :--- | :--- | :--- |
| **HOG + SVM** | 99.60% | 99.60% |
| **CNN** | 98.02% | 86.26% |

The results indicate that traditional feature engineering (HOG) can outperform deep learning on clean, high-contrast signature datasets, while CNNs offer greater flexibility and generalization for real-world noisy data.

## 📁 Repository Structure
- `01_dataset_exploration.ipynb`: Initial analysis, data scanning, and visualization of the signature dataset.
- `02_cnn_training_multiclass.ipynb`: Implementation and training of the Convolutional Neural Network.
- `03_hog_svm_multiclass.ipynb`: Implementation of the HOG descriptor extraction and Linear SVM classification.
- `Signaturre_Identification.pdf`: Detailed technical report documenting methodology, experimental setup, and results.

## 🛠️ Methodology

### 1. Data Preprocessing
Standardized steps were applied across both models to ensure consistency:
- **Grayscale Conversion**: All images converted to single-channel format.
- **Resizing**: Standardized to $128 \times 128$ pixels.
- **Validation**: Filtering out unreadable or corrupted files.
- **Augmentation (CNN only)**: Geometric transformations including random rotation ($\pm3^{\circ}$) and affine shifts (translation 0.03, scaling 0.98-1.02) to improve robustness.

### 2. CNN Architecture
A compact, specialized architecture designed for signature patterns:
- **Feature Extraction**: Three convolutional blocks (Conv2D + ReLU + MaxPool2D).
  - Block 1: 32 channels
  - Block 2: 64 channels
  - Block 3: 128 channels
- **Classification**: A dropout layer (0.4) followed by a 256-unit fully connected layer and a 64-unit softmax output layer.

### 3. HOG + SVM Pipeline
- **HOG Descriptors**: Extracts local gradient patterns and edge orientations (9 orientations, $8 \times 8$ pixels per cell).
- **Linear SVM**: A Linear Support Vector Machine trained on the standardized HOG feature vectors.

## 🚀 Getting Started
1. **Dataset**: Download the [Signature Verification Dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) from Kaggle.
2. **Setup**:
   ```bash
   pip install torch torchvision scikit-learn scikit-image opencv-python pillow matplotlib seaborn
