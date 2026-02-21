# ⚡ Electricity Load Early Warning System — Deep Learning

> An end-to-end deep learning pipeline for detecting high electricity load events from multivariate time-series data, benchmarking 8 models from classical ML to Transformer-based attention architectures.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Pipeline & Methodology](#pipeline--methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Winner: Transformer](#winner-transformer)
- [Key Inferences](#key-inferences)
- [Practical Recommendations](#practical-recommendations)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Future Work](#future-work)

---

## Overview

Unexpected electricity load spikes put enormous stress on power grids and can cascade into widespread outages with severe economic and societal consequences. This project builds an **early-warning classification system** that predicts whether a high-load event will occur within the next hour, giving grid operators actionable advance notice.

Eight models — spanning classical machine learning, recurrent neural networks, convolutional temporal networks, and attention-based Transformers — were designed, trained, and rigorously compared under identical conditions to identify the best architecture for real-world deployment.

---

## Problem Statement

Given a 6-hour window of multivariate electricity consumption readings (5 clients, 15-minute intervals = 24 timesteps × 5 features), predict:

> **"Will a high-load event occur in the next 15–60 minutes?"** (Binary classification: 0 = Normal, 1 = Early Warning)

A *high-load event* is defined as total system consumption exceeding the **95th percentile** of the historical distribution, sustained for at least 4 consecutive 15-minute timesteps (≈ 1 hour).

---

## Dataset

**Source:** [UCI Electricity Load Diagrams 2011–2014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)

| Property | Value |
|---|---|
| Raw shape | 140,256 timesteps × 370 clients |
| Sampling frequency | 15 minutes |
| Date range | 2011-01-01 → 2015-01-01 |
| Subset used | 5 clients, June–August 2013 (8,832 timesteps) |
| High-load threshold | 95th percentile of total system load (≈ 362,649 kW in this run) |
| Event prevalence | ~25% of windows are early-warning positives |

The summer 2013 subset was chosen because high-load precursor events are seasonally concentrated in summer months, providing a realistic class imbalance while still being learnable.

**Preprocessing steps:**
1. Aggregate consumption per client per timestep
2. Compute system-level total load and flag high-load events
3. Generate *early-warning labels*: label = 1 if an event occurs within the next 4 timesteps
4. Min-Max normalize each client's time series independently
5. Build sliding windows of 24 timesteps (6 hours) as model inputs

---

## Pipeline & Methodology

```
Raw CSV (140K rows × 370 clients)
       ↓
Aggregate & Clean
       ↓
Define High-Load Events (95th percentile, 4-step persistence)
       ↓
Early-Warning Labels (shift −4 timesteps)
       ↓
Subset (5 clients, summer 2013)
       ↓
Min-Max Normalization
       ↓
Sliding Window Construction (window=24, stride=1)
       → X: (8808, 24, 5)   y: (8808,)
       ↓
Time-Aware Train/Val Split (70% / 30%, no shuffling)
       → Train: 6,165 samples   Val: 2,643 samples
       ↓
Class-Weighted Training (ratio ~1:2.54 for events)
       ↓
Threshold Tuning (sweep 0.10–0.55) per model
       ↓
Evaluation: Accuracy / Precision / Recall / F1
```

**Key design decisions:**
- **Time-aware splitting** (no shuffling) prevents information leakage from future to past
- **Class-weighted training** compensates for the natural imbalance (~75% normal, ~25% event)
- **Threshold tuning** allows each model to be optimised for the desired precision-recall trade-off, reflecting real operational requirements

---

## Models Implemented

| # | Model | Type | Architecture Summary |
|---|-------|------|----------------------|
| 1 | **Logistic Regression** | Classical ML (Baseline) | Flattened 24×5 window → linear classifier |
| 2 | **LSTM (Base)** | Recurrent NN | LSTM(64) → Dropout(0.3) → Dense(1) |
| 3 | **GRU (Base)** | Recurrent NN | GRU(64) → Dropout(0.3) → Dense(1) |
| 4 | **LSTM (Tuned)** | Recurrent NN | LSTM(128) → Dropout(0.4) → Dense(32) → Dense(1), lr=0.0005 |
| 5 | **GRU (Tuned)** | Recurrent NN | GRU(64) → Dropout(0.3) → Dense(32) → Dense(1) + class weights |
| 6 | **TCN** | Convolutional | TCN(64 filters, kernel=3, dilations=[1,2,4,8,16]) → Dropout → Dense(1) |
| 7 | **Transformer** | Attention-Based | Positional Encoding + 2× [Multi-Head Attention + FFN + LayerNorm] + GlobalAvgPool → Dense(1) |
| 8 | **BiLSTM** | Recurrent NN | Bidirectional LSTM(64) → Dropout(0.3) → Dense(32) → Dense(1) |

All deep learning models used:
- **Binary cross-entropy** loss
- **Adam** optimiser
- **Early stopping** (patience 5–15, monitor `val_loss`, restore best weights)
- **Class-weighted training** to handle label imbalance

---

## Results

All metrics are evaluated on the held-out **validation set (2,643 samples)** using each model's individually tuned decision threshold.

### Full Comparison Table

| Model | Accuracy | Precision (Event) | Recall (Event) | F1 (Event) | Threshold Used |
|---|---|---|---|---|---|
| Logistic Regression | 0.710 | 0.791 | 0.318 | 0.453 | 0.50 |
| LSTM (Base) | 0.751 | 0.686 | 0.629 | 0.657 | 0.35 |
| GRU (Base) | 0.796 | 0.679 | 0.872 | 0.764 | 0.20 |
| LSTM (Tuned) | 0.746 | 0.642 | 0.741 | 0.688 | 0.35 |
| GRU (Tuned) | 0.811 | **0.796** | 0.672 | 0.729 | 0.20 |
| TCN | 0.709 | 0.572 | **0.928** | 0.708 | 0.15 |
| **Transformer** | **0.852** | 0.771 | 0.869 | **0.817** | 0.50 |
| BiLSTM | 0.842 | 0.746 | 0.883 | 0.809 | 0.25 |

### Awards

| Award | Model | Score |
|---|---|---|
| 🏆 Best Overall F1 | **Transformer** | F1 = **0.817** |
| 🎯 Highest Accuracy | **Transformer** | Acc = **85.2%** |
| 📡 Best Recall (Event Detection) | **TCN** | Recall = **0.928** |
| 🔬 Best Precision | **GRU (Tuned)** | Precision = **0.796** |

---

## Winner: Transformer

The **Transformer** is the overall winning model, achieving the best F1 score (0.817) and highest accuracy (85.2%).

### Why the Transformer wins

```
Input (24 timesteps × 5 features)
       ↓
Positional Encoding   ← injects sequence order information
       ↓
Multi-Head Self-Attention (×2 blocks)
  • Each timestep attends to ALL other timesteps simultaneously
  • Captures both short-range (adjacent hours) and long-range (daily) load patterns
       ↓
Feed-Forward Network + Layer Normalization + Residual Connections
       ↓
Global Average Pooling   ← aggregates the sequence into a fixed vector
       ↓
Dense(32, relu) → Dropout(0.1) → Dense(1, sigmoid)
       ↓
Binary prediction: Early Warning (1) or Normal (0)
```

**Transformer Classification Report (threshold = 0.50):**

```
              precision    recall  f1-score   support

           0       0.91      0.84      0.88      1642
           1       0.77      0.87      0.82      1001

    accuracy                           0.85      2643
   macro avg       0.84      0.86      0.85      2643
weighted avg       0.86      0.85      0.85      2643

Confusion Matrix:
[[1383  259]
 [ 131  870]]

✓ 870 of 1001 critical events correctly predicted (86.9% recall)
✓ Only 131 events missed out of 1001 total
✓ 259 false alarms out of 1642 normal windows (15.8% false-alarm rate)
```

**Model parameters:** 73,601 (287.5 KB) — lightweight enough for edge deployment.

---

## Key Inferences

### 1. Deep learning far outperforms classical ML
The Logistic Regression baseline achieves only **F1 = 0.453** for the event class — barely better than random — because it treats the time series as a flat feature vector and ignores temporal ordering. Every neural network architecture dramatically improves on this, confirming that **temporal pattern modelling is essential** for early-warning prediction.

### 2. Attention-based architectures are best suited for this task
The Transformer (F1 = 0.817) outperforms all recurrent models because:
- Self-attention assigns learned importance weights to every timestep, not just the most recent
- It can directly relate a spike beginning 5 hours ago to the current load trajectory, even if separated by many timesteps
- Unlike LSTM/GRU, it does not suffer from vanishing gradients over long sequences

### 3. BiLSTM is the strongest recurrent model
BiLSTM (F1 = 0.809, Acc = 84.2%) comes close to the Transformer by reading the sequence in both forward and backward directions. This bidirectionality helps because early-warning signatures often span multiple hours and are better characterized by looking at the context from both ends of the window.

### 4. TCN is the safest choice when missing events is unacceptable
With recall = 0.928 at threshold 0.15, the TCN catches 93 out of every 100 critical events. Its dilated causal convolutions (dilation factors 1, 2, 4, 8, 16) create a very large receptive field that covers the full 6-hour window in a single pass, making it excellent at detecting broad load build-up patterns. The trade-off is lower precision (57%) — more false alarms — which may be acceptable in safety-critical grid operations where missing an event costs far more than an unnecessary alert.

### 5. Class imbalance handling is non-negotiable
Without class-weighted training, all neural networks defaulted to predicting "normal" for most samples (≥ 75% accuracy at zero effort). Applying class weights (0.62 for normal, 2.54 for events) forced the models to properly learn the minority event class. This is a critical design decision for any real-world early-warning system.

### 6. Decision threshold tuning dramatically changes the precision-recall balance
A single model can behave very differently depending on the threshold applied to its output probability:

| Threshold | Effect |
|---|---|
| Low (0.10–0.20) | High recall, lower precision — prioritise catching all events |
| Medium (0.30–0.40) | Balanced — good for general deployment |
| High (0.45–0.55) | High precision, lower recall — prioritise reducing false alarms |

This means the same trained model can be redeployed with a different threshold to suit new operational requirements without retraining.

### 7. LSTM Autoencoder provides unsupervised anomaly detection
An additional LSTM Autoencoder was trained exclusively on normal load patterns. It learns to reconstruct normal sequences with low error; anomalous (high-load) sequences produce high reconstruction error. Using the 95th percentile of training error as a threshold, the autoencoder detects 83 confirmed events in the validation set without ever seeing a labeled event during training — demonstrating that **meaningful load anomalies are structurally distinct from normal patterns**.

---

## Practical Recommendations

| Use Case | Recommended Model | Threshold | Reasoning |
|---|---|---|---|
| **General-purpose deployment** | **Transformer** | 0.40–0.50 | Best F1 — strong balance of precision and recall |
| **Zero-miss safety-critical grid** | **TCN** | 0.10–0.15 | Highest recall (0.928) — catches almost every event |
| **Minimise false alarms / operator fatigue** | **GRU (Tuned)** | 0.35–0.50 | Highest precision (0.796) — fewer unnecessary alerts |
| **Lightweight real-time deployment** | **GRU (Base/Tuned)** | 0.20–0.30 | Good performance, ~15K parameters |
| **Interpretability-first** | **Logistic Regression** | 0.30–0.40 | Fully transparent — useful as a sanity check |

---

## Project Structure

```
Electricity-load-forecasting-DL/
├── Early_Warning_Electricity_Load_Time_SeriesFinal.ipynb   # Full experiment notebook
├── Project_Report.pdf                                       # Detailed written report
├── README.md
```

The entire pipeline — data loading, cleaning, EDA, model definitions, training, threshold tuning, evaluation, and visualisation — is contained in the self-contained Jupyter notebook.

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/95karthik4/Electricity-load-forecasting-DL.git
cd Electricity-load-forecasting-DL

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib keras-tcn

# 4. Download the dataset
# Place LD2011_2014.txt from UCI into the same directory as the notebook
# https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

# 5. Open the notebook
jupyter notebook Early_Warning_Electricity_Load_Time_SeriesFinal.ipynb
```

Run all cells in order. The notebook is fully reproducible (random seed = 42 for NumPy, TensorFlow, and Python's hash).

---

## Future Work

- Expand to the full dataset (370 clients, 2011–2014) for more robust generalization
- Explore advanced Transformer variants designed specifically for time series (Informer, Autoformer, PatchTST)
- Implement ensemble methods combining Transformer (best F1) and TCN (best recall) for a tunable hybrid
- Add real-time streaming inference with a sliding window buffer
- Deploy as a live monitoring dashboard for electricity grid operators

---

## License

This project is licensed under the [MIT License](LICENSE).
