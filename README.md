# Advancing Brain Tumor Segmentation via Attention-Based 3D U-Net and Digital Image Processing  

---

## Project Overview

This repository implements an **Attention-Based 3D U-Net** architecture for accurate brain tumor segmentation using the BraTS 2020 dataset. The incorporation of attention mechanisms enhances the model’s focus on tumor regions, while digital image processing techniques are applied to mitigate class imbalance during training.

### Highlights
- Attention-augmented 3D U-Net for enhanced segmentation precision.  
- Application of digital image processing to address class imbalance challenges.  
- Evaluated on BraTS 2020 dataset achieving:  
  - **Dice Score:** 0.975  
  - **Sensitivity:** 0.995  
  - **Specificity:** 0.988  

---

## Dataset

- **BraTS 2020 Dataset**  
Download from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data).

---

## Repository Structure

- `utils.py` — Utility functions for data loading, preprocessing, and visualization.  
- `preprocess.py` — Data preprocessing including resizing, normalization, and augmentation.  
- `models.py` — Implementation of the Attention-Based 3D U-Net model.  
- `train.py` — Script for training the model.  
- `evaluate.py` — Evaluation script computing metrics such as Dice Score, IoU, Sensitivity, and Specificity.  
- `requirements.txt` — Python package dependencies.

---

## Performance Results

| Metric      | Score  |
|-------------|---------|
| Dice Score  | 0.975   |
| Sensitivity | 0.995   |
| Specificity | 0.988   |

---

## Prerequisites

- Python 3.8 or later  
- TensorFlow / Keras 2.x+  

Install dependencies using:  
```bash
pip install -r requirements.txt
```

---

## Usage
1. **Preprocess Data**:
   ```bash
   python preprocess.py
   ```
2. **Train the Model**:
   ```bash
   python train.py
   ```
3. **Evaluate the Model**:
   ```bash
   python evaluate.py
   ```
---

