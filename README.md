# Stock Price and Restaurant Price Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Pandas](https://img.shields.io/badge/Pandas-1.x-lightgrey)

A machine learning application that provides two main features:
1. Restaurant price prediction using Zomato dataset
2. Stock price prediction using historical market data

## Features

- Interactive web interface using Streamlit
- Real-time price predictions for both use cases
- Advanced data visualization and insights
- Time-series analysis for stock metrics
- LSTM neural network implementation
- Model performance evaluation (MAPE, R²)
- User-friendly input forms

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone the repository
```bash
git clone https://github.com/meet987654/Stock-Price-Predictor.git
cd Stock-Price-Predictor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run src/app.py
```

## Project Structure
```
Stock-Price-Predictor/
├── src/
│   ├── app.py              # Main Streamlit application
│   └── zomato_predictor.py # Price prediction logic
├── model/                  # Trained models
├── doc/                    # Documentation
└── data/                   # Dataset files
```

## Technologies Used
- Python
- Streamlit
- TensorFlow/Keras
- Pandas
- Plotly
- scikit-learn

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT
