# Electricity Load Forecasting with Deep Learning

> Time-series forecasting for electricity grid spikes using Transformers and BiLSTMs to prevent grid failure.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Electricity grid failures caused by sudden load spikes can have severe economic and societal consequences. This project uses deep learning to accurately forecast short-term electricity demand, giving grid operators advance warning to prevent outages.

Two complementary architectures are implemented and compared:

| Model | Key Strength |
|---|---|
| **Transformer** | Captures long-range temporal dependencies via self-attention |
| **BiLSTM** | Models sequential patterns in both forward and backward directions |

---

## Features

- Multi-step ahead load forecasting (e.g., 1h, 6h, 24h horizons)
- Transformer-based sequence-to-sequence model with positional encoding
- Bidirectional LSTM (BiLSTM) baseline model
- Data preprocessing pipeline (normalization, sliding-window generation)
- Training, validation and test evaluation scripts
- Visualization of predictions vs. ground truth

---

## Model Architecture

### Transformer

```
Input → Positional Encoding → N × Encoder Layers (Multi-Head Attention + FFN) → Linear → Forecast
```

### BiLSTM

```
Input → BiLSTM Layer(s) → Dropout → Fully Connected → Forecast
```

---

## Dataset

The models are designed to work with hourly electricity consumption time-series data. Typical public datasets that can be used include:

- [UCI Electricity Load Diagrams](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
- [Global Energy Forecasting Competition (GEFCom)](http://www.gefcom.com/)
- Any CSV with a datetime index and a numeric load column

Place your dataset in the `data/` directory before running training scripts.

---

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/95karthik4/Electricity-load-forecasting-DL.git
cd Electricity-load-forecasting-DL

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare the data

```bash
python src/preprocess.py --input data/raw/load.csv --output data/processed/
```

### 2. Train a model

```bash
# Transformer
python src/train.py --model transformer --config configs/transformer.yaml

# BiLSTM
python src/train.py --model bilstm --config configs/bilstm.yaml
```

### 3. Evaluate

```bash
python src/evaluate.py --model transformer --checkpoint checkpoints/transformer_best.pt
```

### 4. Visualize predictions

```bash
python src/visualize.py --predictions results/transformer_predictions.csv
```

---

## Project Structure

```
Electricity-load-forecasting-DL/
├── configs/               # YAML configuration files for each model
├── data/
│   ├── raw/               # Original datasets (not tracked by git)
│   └── processed/         # Preprocessed sliding-window tensors
├── src/
│   ├── models/
│   │   ├── transformer.py # Transformer model definition
│   │   └── bilstm.py      # BiLSTM model definition
│   ├── preprocess.py      # Data loading and preprocessing
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Metrics computation (MAE, RMSE, MAPE)
│   └── visualize.py       # Plotting utilities
├── checkpoints/           # Saved model weights
├── results/               # Prediction CSVs and plots
├── requirements.txt
└── README.md
```

---

## Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Transformer | — | — | — |
| BiLSTM | — | — | — |

*Results will be updated as experiments are completed.*

---

## Contributing

Contributions, bug reports and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).
