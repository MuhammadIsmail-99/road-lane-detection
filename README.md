# Road Lane Detection using VGG16-UNet 🏎️🛣️

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://tensorflow.org)
[![Kaggle](https://img.shields.io/badge/Dataset-TUSimple-lightgrey.svg)](https://github.com/TuSimple/tusimple-benchmark)

Developed by **[Muhammad Ismail](https://github.com/MuhammadIsmail-99)**, this repository implements a high-performance semantic segmentation pipeline to detect highway lanes. Using the TUSimple dataset, the project transforms raw video frames into precise binary masks using a custom U-Net architecture.

## 📌 Project Overview
Lanes are the fundamental "rails" of autonomous driving. This project tackles the challenge of lane detection by:
1.  **Automated Preprocessing:** Parsing JSON annotations and generating ground-truth masks with OpenCV polylines.
2.  **Hybrid Architecture:** Implementing a **U-Net** with a pre-trained **VGG16 backbone** to leverage transfer learning.
3.  **Optimized Pipeline:** Utilizing `tf.data` with caching and prefetching for maximum GPU throughput on Tesla P100 hardware.

## 🏗️ Model Architecture
The model follows a classic Encoder-Decoder (U-Net) structure:
*   **Encoder:** Pre-trained **VGG16** (ImageNet weights) extracts complex hierarchical features.
*   **Decoder:** Symmetrical upsampling blocks with **Skip Connections** that concatenate high-resolution features from the encoder to recover spatial detail.
*   **Regularization:** Integrated **Batch Normalization** and **Dropout ($0.2$)** layers to ensure generalization and prevent overfitting.
*   **Activation:** ELU activation for smoother gradient flow and Sigmoid for the final pixel-wise classification.

## 📊 Metrics & Training
Since lane pixels are sparse compared to the road surface, standard accuracy is misleading. This project prioritizes:
*   **Dice Coefficient:** To measure the spatial overlap between predicted and actual lanes.
*   **Dice Loss:** Used as the primary objective function to combat class imbalance.
*   **Precision & Recall:** Smoothed metrics to ensure consistent detection even in varying light conditions.

### Training Results (Quick Look)
| Epoch | Training Loss | Val Loss | Val Dice Coeff | Val Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.4160 | 0.3342 | 0.6659 | 97.15% |
| 4 | 0.2589 | 0.2640 | 0.7360 | 97.70% |

## 🚀 Installation & Usage

1. **Clone the repo:**
   ```bash
   git clone https://github.com/MuhammadIsmail-99/road-lane-detection.git
   cd road-lane-detection
   ```

2. **Data Structure:**
   Ensure the TUSimple dataset is placed in the expected directory structure:
   `../input/tusimple/TUSimple/train_set/`

3. **Execution:**
   Run the notebook or script. The pipeline will automatically create the `tusimple_processed/` directory and begin image extraction and mask generation.

## 🛠️ Key Features
*   **Memory Management:** Custom callbacks use `gc.collect()` and `K.clear_session()` to handle large-scale image processing without memory leaks.
*   **Visual Monitoring:** Includes a `DisplayCallback` that renders predictions at the end of every epoch.
*   **Production Ready:** Exports the best weights to `best_model.keras` using `ModelCheckpoint`.

## 👤 Author
**Muhammad Ismail**
*   GitHub: [@MuhammadIsmail-99](https://github.com/MuhammadIsmail-99)

---
*License: MIT*
```
