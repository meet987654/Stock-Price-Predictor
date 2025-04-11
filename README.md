# Microsoft Stock Price Prediction

## Overview
This project implements machine learning models (LSTM and Random Forest) to predict Microsoft stock prices. The models use historical stock data to forecast future price movements and evaluate performance using MAPE and R-squared metrics. The project includes a Streamlit web interface for interactive visualization and prediction.

## Features
- Interactive web interface using Streamlit
- Multiple model options:
  - LSTM-based sequential model
  - Random Forest model
- Data preprocessing and visualization
- Multiple visualization plots:
  - Open/Close prices over time
  - Trading volume analysis
  - Feature correlation heatmap
  - Prediction vs Actual price comparison
- Performance metrics calculation (MAPE and R-squared)

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── data/
│   └── MicrosoftStock.csv
├── model/
│   ├── microsoft_stock_predictor.keras
│   └── scaler.save
├── src/
│   ├── main.py
│   └── app.py
├── README.md
└── requirements.txt
```

## Usage
1. Train the models:
```bash
python src/main.py
```

2. Run the Streamlit application:
```bash
streamlit run src/app.py
```

## Model Architectures

### LSTM Model
- Input LSTM layer (64 units with return sequences)
- Second LSTM layer (64 units)
- Dense layer (128 units with ReLU activation)
- Dropout layer (0.5)
- Output Dense layer (1 unit)

### Random Forest Model
- Number of estimators: 100
- Features: 60-day lookback window
- Random state: 42

## Results
Both models evaluate performance using:
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²) score

## Web Interface
The Streamlit interface provides:
- Data overview with statistics
- Interactive visualizations
- Model selection and predictions
- Performance metrics display

## License
MIT License
