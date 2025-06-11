<h1 align="center">🌫 AQ-Net: Deep Spatio-Temporal Neural Network for Air Quality Reanalysis</h1>

<p align="center">
📄 <strong>Accepted at SCIA 2025</strong> • 🔗 <a href="https://arxiv.org/abs/2502.11941">arXiv:2502.11941</a> • 📌 <a href="img/poster%20SCIA%202025V.pdf">Poster (PDF)</a>  
</p>

<p align="center">
  <a href="img/poster%20SCIA%202025V.pdf">
    <img src="img/posterSCIA2025V.png" alt="AQ-Net Poster Preview" width="70%"/>
  </a>
</p>

---

## 👤 Author & Affiliations

**Ammar Kheder**  
- Lappeenranta–Lahti University of Technology (LUT)  
- Atmospheric Modelling Centre Lahti, Lahti University Campus  
- University of Helsinki  
- Chinese Academy of Sciences

---

## 🗺 Spatial Reanalysis of PM2.5 Across China

<p align="center">
  <img src="img/mapC.gif" alt="Beijing Zoomed View" width="49%"/>
</p>

This map shows the reanalysis of PM2.5 levels across all stations used in our study, whether visible or hidden.  
The color indicates the PM2.5 concentration (from dark purple to yellow), and the size of the circles reflects the relative intensity.  
The zoom on Beijing highlights how dense the sensor network is in certain urban regions.

---

## 📘 Overview

**AQ-Net** is a deep learning model for spatiotemporal air quality reanalysis, focused on **PM2.5 estimation across Northern China (2013–2017)**.

Key components:
- **LSTM + Multi-Head Attention** for temporal modeling.
- **Neural kNN module** for spatial interpolation.
- **Cyclic Encoding** for robust continuous time representation.

✅ Proven performance across:
- **Short-term (6–24h)** and  
- **Long-term (up to 7 days)** horizons.  

Ideal for environmental assessment, public health alerts, and policy development.

<p align="center">
  <img src="img/mapB.png" alt="PM2.5 Time Trends"/>
</p>

---

## 🚀 Key Features

### 🔁 Hybrid Model Architecture
- **LSTM + Multi-Head Attention**: Captures long-term dependencies & focuses on key time steps.  
- **Cyclic Encoding (CE)**: Projects time into 2D sinusoidal space.  
  <img src="img/cyclic_encoding.gif" width="400"/>  
- **Neural kNN**: Learns to interpolate spatial features for unobserved stations.

### 🧪 Robust Reanalysis
- Handles **sparse sensor networks**.
- Maintains **spatial + temporal consistency**.

### 📊 Comprehensive Evaluation
- Outperforms LSTM, Linear Regression, and PatchTST.  
- Metrics: **MAE**, **RMSE**, **R²**.

---

## 🧬 Model Architecture

<p align="center">
  <img src="img/Artboard11.jpg" alt="AQ-Net Architecture" width="80%"/>
</p>

---

## ⚙️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/AmmarKheder/AQ-Net.git
cd AQ-Net
