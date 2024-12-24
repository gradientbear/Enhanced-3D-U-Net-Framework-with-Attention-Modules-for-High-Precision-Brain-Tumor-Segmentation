# [Advancing Brain Tumor Segmentation via Attention-Based 3D U-Net Architecture and Digital Image Processing](https://doi.org/10.1007/978-3-031-49333-1_18)

---

## Project Overview
This project implements an **Attention-Based 3D U-Net model** for brain tumor segmentation using the BraTS 2020 dataset. The integration of attention mechanisms enhances the model's ability to focus on tumor regions, while a digital image processing-based algorithm addresses class imbalance during training.

### Key Features:
- **Attention-Based 3D U-Net** for improved segmentation accuracy.
- **Digital Image Processing Techniques** to handle class imbalance.
- Evaluated on the **BraTS 2020 Dataset**, achieving:
  - **Dice Score**: 0.975
  - **Sensitivity**: 0.995
  - **Specificity**: 0.988

---
## Data:
- **BraTS2020 Dataset**: Download from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data).

## Repository Structure
- **`utils.py`**: Utility functions for loading data, preprocessing, and visualization.
- **`preprocess.py`**: Preprocessing steps such as resizing, normalization, and augmentation.
- **`models.py`**: Implementation of the Attention-Based 3D U-Net model.
- **`train.py`**: Script for training the model.
- **`evaluate.py`**: Evaluation script for metrics like Dice Score, IoU, Sensitivity, and Specificity.
- **`requirements.txt`**: List of dependencies.

---

## Results
The proposed model achieved the following results on the BraTS 2020 dataset:
- **Dice Score**: 0.975
- **Sensitivity**: 0.995
- **Specificity**: 0.988

---

## Prerequisites
- Python 3.8+
- TensorFlow/Keras (2.x+)
- Install all dependencies using:
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

## References
If you use this work, please cite:
```plaintext
@InProceedings{10.1007/978-3-031-49333-1_18,
author="Gad, Eyad
and Soliman, Seif
and Darweesh, M. Saeed",
editor="Mosbah, Mohamed
and Kechadi, Tahar
and Bellatreche, Ladjel
and Gargouri, Faiez",
title="Advancing Brain Tumor Segmentation via Attention-Based 3D U-Net Architecture and Digital Image Processing",
booktitle="Model and Data Engineering",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="245--258"
}
```
