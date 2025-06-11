# AQ-Net: Deep Spatio-Temporal Neural Network for Air Quality Reanalysis  
📄 *Accepted at SCIA 2025*  
🔗 [arXiv:2502.11941](https://arxiv.org/abs/2502.11941)
🧠 Poster AQ-Net (SCIA 2025)
[![Poster](img/poster-preview.png)](img/poster%20SCIA%202025V.pdf)

**Author:** Ammar Kheder  
**Affiliations:**  
- Lappeenranta–Lahti University of Technology (LUT)  
- Atmospheric Modelling Centre Lahti, Lahti University Campus  
- University of Helsinki  
- Chinese Academy of Sciences
## Spatial Reanalysis of PM2.5 Levels Across China

![Description de l'image](img/mapc.png)
 ![m](img/mapC.gif)  
This map shows the reanalysis of PM2.5 levels across all stations used in our study, whether visible or hidden. The color indicates the PM2.5 concentration (from dark purple for low values to yellow for high values), and the size of the circles reflects the relative measured intensity. The zoom on the Beijing area highlights a crucial aspect: some stations are extremely close to each other.
## Overview

**AQ-Net** is a deep learning-based model designed for air quality reanalysis in both spatial and temporal domains.  
Accepted at **SCIA 2025**, the paper is now publicly available on [arXiv](https://arxiv.org/abs/2502.11941).

The model leverages an **LSTM combined with multi-head attention** to capture temporal dependencies, while a **neural kNN module** provides spatial interpolation for unobserved monitoring stations.  
An innovative **Cyclic Encoding (CE)** technique is also introduced to ensure continuous time representation, overcoming issues with discrete time boundaries.

The project focuses on **PM2.5 analysis using data collected in northern China (2013–2017)**.  
Extensive experiments have demonstrated that AQ-Net is robust in capturing both **short-term (6–24 hours)** and **long-term (up to 7 days)** pollutant trends, making it a valuable tool for environmental reanalysis, public health alerts, and policy decisions.
![Description de l'image](img/mapB.png)

---

## Key Features

- **Hybrid Model Architecture**  
  - **LSTM + Multi-Head Attention:** Captures long-term temporal dependencies and selectively weights critical time steps.  
  - **Cyclic Encoding (CE):** Projects time-related features into a continuous 2D sinusoid space to avoid discontinuities.  
    ![CE](img/cyclic_encoding.gif)  
  - **Neural kNN Interpolation:** Fills spatial gaps by interpolating features from observed stations to unobserved ones.

- **Robust Reanalysis**  
  Designed to reconstruct historical air pollution levels even in regions with sparse sensor coverage, ensuring both spatial and temporal consistency.

- **Comprehensive Evaluation**  
  The model has been quantitatively and qualitatively evaluated against traditional methods (e.g., LSTM, linear regression, and PatchTST), showing improved performance in terms of MAE, RMSE, and R² scores.

---

## Overall Architecture

![Description de l'image](img/Artboard11.jpg)
---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AmmarKheder/AQ-Net.git
   cd AQ-Net
2. **Install dependencies:**
   pip install -r requirements.txt

3. **Usage**
Run the main script to execute the entire pipeline:
  python main.py
This will:
	1.	Load and preprocess the data (filtering, normalization, etc.).
	2.	Split the dataset into training, validation, and test sets.
	3.	Create PyTorch datasets and data loaders.
	4.	Define and train the AQ-Net model (LSTM + Attention + neural kNN).
	5.	Evaluate the model (MAE, RMSE, R²).
	6.	Visualize spatial predictions for unobserved stations.


