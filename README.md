# Hybrid Deep Learning for Breast Cancer Histopathology Classification

This project presents the development, evaluation, and deployment of a hybrid deep learning model for binary classification (benign vs. malignant) of breast cancer histopathology images. The model combines EfficientNetV2-S and a Vision Transformer (ViT-B/16), trained on the BreaKHis dataset using a phased fine-tuning strategy. The project also explores domain generalization by testing on the BACH dataset with adaptive stain normalization and Test-Time Augmentation (TTA), and includes a deployed web application for interactive inference.

**Authors:** Yousef Khaled, Abdelrahman Ahmed, Belal Gamal, Menna Mohamed

**Supervisors:** Dr. Omar Fahmy & Dr. Eman Badr

**Affiliation:** Department of Communication and Information Engineering, Zewail City of Science & Technology (CIE 555 – CIE 552)

**Live Demo (Gradio Web App on Hugging Face Spaces):**
[https://yousefkhaled-breast-cancer-classification.hf.space/](https://yousefkhaled-breast-cancer-classification.hf.space/)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://yousefkhaled-breast-cancer-classification.hf.space/)


## Table of Contents
1.  [Introduction](#introduction)
2.  [Project Goals](#project-goals)
3.  [Features](#features)
4.  [Methodology](#methodology)
    *   [Datasets](#datasets)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Model Architecture](#model-architecture)
    *   [Training Strategy](#training-strategy)
    *   [Generalization Testing](#generalization-testing)
5.  [Results](#results)
6.  [Web Application Deployment](#web-application-deployment)
7.  [Setup and Usage (for running the code)](#setup-and-usage)
    *   [Prerequisites](#prerequisites)
    *   [Training the Model](#training-the-model)
    *   [Testing on External Dataset](#testing-on-external-dataset)
    *   [Running the API Locally](#running-the-api-locally)
8.  [File Structure](#file-structure)
9.  [Future Work](#future-work)
10. [References](#references)
11. [Acknowledgements](#acknowledgements)

## 1. Introduction
Breast cancer diagnosis relies heavily on the analysis of histopathological images, a task that is complex and can be subject to inter-observer variability. This project leverages deep learning to build a robust system for classifying breast cancer tissue as benign or malignant, aiming to assist pathologists and improve diagnostic consistency. We explore a hybrid architecture, advanced training techniques, and methods for domain adaptation.

## 2. Project Goals
*   Develop a high-performing hybrid deep learning model (EfficientNetV2-S + ViT-B/16) for binary breast cancer classification on the BreaKHis dataset.
*   Implement and evaluate a phased fine-tuning strategy.
*   Assess model generalization on an external dataset (BACH) using techniques like domain-adaptive stain normalization and Test-Time Augmentation (TTA).
*   Deploy the trained model as an interactive web application for demonstration and potential use.

## 3. Features
*   **Hybrid Model Architecture:** Combines CNN (EfficientNetV2-S) for local feature extraction and Transformer (ViT-B/16) for global context.
*   **Phased Fine-Tuning:** Progressive unfreezing of layers with differential learning rates for optimal adaptation of pre-trained weights.
*   **Class Imbalance Handling:** Uses weighted loss and a `WeightedRandomSampler`.
*   **Patient-Level Data Splitting:** Ensures robust evaluation by preventing data leakage from the same patient across train/validation/test sets using `GroupShuffleSplit`.
*   **Domain-Adaptive Stain Normalization:** Reinhard normalization applied to the external test dataset (BACH) using global statistics derived from the BreaKHis training set.
*   **Test-Time Augmentation (TTA):** Enhances prediction stability on test data.
*   **Interactive Web Deployment:** User-friendly Gradio interface hosted on Hugging Face Spaces.
*   **Comprehensive Evaluation:** Includes Accuracy, AUC, Precision, Recall, F1-Score, Confusion Matrices, and Classification Reports.

## 4. Methodology

### 4.1. Datasets
*   **Primary Training/Test Dataset: BreaKHis**
    *   7,909 images (700x460 pixels, PNG) of benign and malignant breast tumor tissue.
    *   Four magnification factors (40X, 100X, 200X, 400X).
    *   Reference: Spanhol et al., 2016a.
*   **External Generalization Test Dataset: BACH (dina0808/bach-icar-2018)**
    *   Source: Kaggle, derived from ICIAR 2018 Challenge.
    *   400 image patches (2048x1536 pixels, TIFF format).
    *   Original 4 classes (Normal, Benign, InSitu, Invasive) mapped to binary.

### 4.2. Data Preprocessing
*   **BreaKHis:** Images organized, metadata (label, subtype, magnification, slide ID) extracted.
*   **BACH:** Labels mapped to binary. Quality Control (QC) applied to filter blurry images (Laplacian variance < 75.0) and patches with low tissue content (< 40.0%), resulting in 275 images for evaluation.
*   **Label Encoding:** `sklearn.preprocessing.LabelEncoder` for 'benign'/'malignant' to 0/1.
*   **Stain Normalization (for BACH testing):** Reinhard normalization applied to QC-passed BACH images, targeting global LAB statistics (Means: `[180.8, 148.6, 116.5]`, Stds: `[35.4, 17.5, 11.1]`) derived from a 2000-image sample of the BreaKHis training set.
*   **Image Augmentation (Training - BreaKHis):** `torchvision.transforms` including Resize(256), RandomCrop(224), Flips, ColorJitter, Rotation, Affine, RandomErasing. Intensity varied per phase.
*   **Test Transforms (BreaKHis & BACH):** Resize (224x224), ToTensor, ImageNet Normalization.

### 4.3. Model Architecture (`HybridModel`)
A hybrid architecture fusing features from EfficientNetV2-S (for local features via `model.features` and Adaptive Average Pooling) and ViT-B/16 (for global context via CLS token, head removed).
*   **Fusion Block:** Concatenated features processed by `LayerNorm -> Linear(input_dim, 1024) -> GELU -> LayerNorm(1024) -> Dropout(0.6) -> Linear(1024, 1024)`. (input_dim = 2048 for this model version).
*   **Classifier Head:** `LayerNorm(1024) -> Linear(1024, 512) -> GELU -> Dropout(0.7) -> Linear(512, 2)`.
*   
![ModelArchitecture](https://github.com/user-attachments/assets/f8d35122-2161-418c-848d-f4e6e0184372)

### 4.4. Training Strategy (BreaKHis)
*   **Loss:** `nn.CrossEntropyLoss` with class weights and label smoothing (0.2).
*   **Optimizer:** `AdamW` (weight decay 1e-5).
*   **Phased Fine-Tuning:**
    1.  Phase 1 (7 epochs): Head/Fusion only (LR 1e-4). (Max Val AUC: 0.8874)
    2.  Phase 2 (10 epochs): Partial backbone unfreeze (Head LR 2e-5, Backbone LR 1e-6). (Max Val AUC: 0.9199 at Epoch 9)
    3.  Phase 3 (13 epochs): Full backbone unfreeze (Head LR 1e-5, Backbone LR 5e-7). (Did not improve Val AUC further).
*   **Scheduler:** `ReduceLROnPlateau` (Val AUC, patience 3, factor 0.2).
*   **Other:** Mixed Precision Training, `WeightedRandomSampler`.

### 4.5. Generalization Testing (BACH)
*   The BreaKHis-trained model (checkpoint from best Val AUC during Phase 2, Epoch 9) was evaluated on the QC-filtered, stain-normalized BACH dataset.
*   **Test-Time Augmentation (TTA):** Applied 7 augmentations (original, HFlip, VFlip, H+VFlip, Rot90, Rot180, Rot270) and averaged probabilities.

## 5. Results

### 5.1. Performance on BreaKHis Dataset
The model achieved its best validation performance (AUC 0.9199, Accuracy 84.64%) during Phase 2, Epoch 9.
The performance on the held-out **BreaKHis Test Set** using this checkpoint is as follows:

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 79.07%  |
| AUC       | 0.8048  |
| Precision (Mal) | 0.7847  |
| Recall (Mal)    | 0.8496  |
| F1-Score (Mal)  | 0.8159  |

*(Optional: Link to or embed Figure 5, 6, 7 - BreaKHis Training History & Confusion Matrix)*

### 5.2. Generalization to External BACH Dataset
**5.2.1. Performance on BACH Dataset (440 images, No QC, BreaKHis Norm + TTA)**

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 67.73%  |
| AUC       | 0.6811  |
| Precision (Mal) | 0.6726  |
| Recall (Mal)    | 0.6909  |
| F1-Score (Mal)  | 0.6816  |


**5.2.2. Performance on QC-Filtered BACH Dataset (275 images, BreaKHis Norm + TTA)**

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 63.27%  |
| AUC       | 0.6280  |
| Precision (Mal) | 0.6781  |
| Recall (Mal)    | 0.6471  |
| F1-Score (Mal)  | 0.6622  |


**Discussion Snippet from Paper (included here for context):**
The performance drop from BreaKHis to BACH illustrates domain shift. Interestingly, performance on the QC-filtered BACH set (AUC 0.6280) was slightly lower than on a larger, unfiltered BACH subset (AUC 0.6811). This might suggest QC removed some "easier" images, the remaining set was harder, or issues with smaller sample size variance.

## 6. Web Application Deployment
The BreaKHis-trained model (checkpoint achieving Val AUC 0.9199) was deployed as an interactive web application.
*   **Tools:** Gradio for UI, Hugging Face Spaces for hosting.
*   **Functionality:** Users can upload a PNG/JPG histopathology image and receive a benign/malignant classification with confidence scores. (Note: Stain normalization was not applied to user uploads in the demo for simplicity).
*   **Live Demo:** [https://yousefkhaled-breast-cancer-classification.hf.space/](https://yousefkhaled-breast-cancer-classification.hf.space/)

## 7. Setup and Usage

### 7.1. Prerequisites
*   Python 3.9+
*   PyTorch, TorchVision, OpenCV, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Tqdm, Torchinfo
*   (For API/Deployment): FastAPI, Uvicorn, python-multipart, Gradio, huggingface_hub
*   `kaggle.json` for dataset downloads.

### 7.2. Training the Model (BreaKHis)
Refer to the main Colab notebook (`deeplearningproject_colab_main.ipynb`) for the complete training pipeline. Key steps involve setting up the Kaggle API, downloading BreaKHis, preprocessing, and running the phased training loop.

### 7.3. Testing on External Dataset (BACH)
The Colab notebook (`Test_Data_QC.ipynb`) also contains cells to download the BACH dataset (dina0808/bach-icar-2018), calculate BreaKHis global LAB statistics for normalization, prepare BACH metadata (including QC), and run evaluation with TTA.

### 7.4. Running the Deployed Web App
Access the live demo via the Hugging Face Spaces link provided above.

## 8. Future Work
*   Multi-Class Classification (BreaKHis subtypes).
*   Advanced Domain Adaptation: Meta-Learning, sophisticated stain normalization (Macenko, GANs), Domain Adversarial Training.
*   Richer Feature Fusion: Cross-attention or compact bilinear pooling.
*   Magnification-Awareness Refinement.
*   Deployment Enhancements.

## 9. Acknowledgements
Special thanks to Dr. Omar Fahmy and Dr. Eman Badr for their supervision and guidance. We acknowledge the creators of the BreaKHis and BACH datasets.
