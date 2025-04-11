# Microsoft Stock Price Prediction using LSTM

## Overview
This project implements a deep learning model using Long Short-Term Memory (LSTM) neural networks to predict Microsoft stock prices. The model uses historical stock data to forecast future price movements and evaluates performance using MAPE and R-squared metrics.

## Features
- Data preprocessing and visualization
- LSTM-based sequential model architecture
- Multiple visualization plots:
  - Open/Close prices over time
  - Trading volume analysis
  - Feature correlation heatmap
  - Prediction vs Actual price comparison
- Performance metrics calculation (MAPE and R-squared)

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── data/
│   └── MicrosoftStock.csv
├── src/
│   └── main.py
├── README.md
└── requirements.txt
```

## Usage
1. Ensure your Microsoft stock data CSV file is in the correct location
2. Run the main script:
```bash
python src/main.py
```

## Model Architecture
- Input LSTM layer (64 units with return sequences)
- Second LSTM layer (64 units)
- Dense layer (128 units with ReLU activation)
- Dropout layer (0.5)
- Output Dense layer (1 unit)

## Results
The model evaluates performance using:
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²) score

## License
MIT License