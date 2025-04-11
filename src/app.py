import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from tensorflow import keras
import joblib
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import ta

# Set page configuration
st.set_page_config(
    page_title="Microsoft Stock Price Prediction",
    layout="wide"
)

# Title and description
st.title("Microsoft Stock Price Prediction")
st.markdown("This application predicts Microsoft stock prices")

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "MicrosoftStock.csv")
    return pd.read_csv(data_path, parse_dates=['date'])

data = load_data()

@st.cache_data
def add_technical_indicators(df):
    # Add Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    
    # Add RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    
    # Add MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    
    # Add Volume Features
    df['Volume_MA'] = ta.trend.sma_indicator(df['volume'], window=20)
    df['Volume_STD'] = df['volume'].rolling(window=20).std()
    
    return df

data = add_technical_indicators(data)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Visualizations", "Predictions"])

# Page content
if page == "Data Overview":
    st.header("Data Overview")
    
    st.subheader("Raw Data Sample")
    st.dataframe(data.head())
    
    st.subheader("Data Statistics")
    st.write(data.describe())

elif page == "Visualizations":
    st.header("Data Visualizations")
    
    # Plot 1: Open and Close prices
    st.subheader("Stock Prices Over Time")
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['open'], label='Open Price', color='blue')
    plt.plot(data['date'], data['close'], label='Close Price', color='red')
    plt.title('Open and Close Prices Over Time')
    plt.legend()
    st.pyplot(fig1)
    
    # Plot 2: Trading Volume
    st.subheader("Trading Volume")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['volume'], label='Trading Volume', color='green')
    plt.title('Stock Volume Over Time')
    st.pyplot(fig2)
    
    # Plot 3: Correlation Heatmap
    st.subheader("Feature Correlation")
    numeric_data = data.select_dtypes(include=["int64", "float64"])
    fig3 = plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation HeatMap')
    st.pyplot(fig3)

elif page == "Predictions":
    st.header("Stock Price Predictions")
    
    # Add model selection
    model_type = st.radio("Select Model", ["1. LSTM", "2. Random Forest"])
    
    try:
        # Load data and prepare common variables
        stock_close = data.filter(["close"])
        dataset = stock_close.values
        training_data_len = int(np.ceil(len(dataset) * 0.95))
        
        # Define model_dir before using it
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
        
        if model_type == "1. LSTM":
            try:
                # Prepare data for LSTM
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)
                
                # Create sequences for LSTM
                def create_sequences(data, seq_length=60):
                    X, y = [], []
                    for i in range(seq_length, len(data)):
                        X.append(data[i-seq_length:i, 0])
                        y.append(data[i, 0])
                    return np.array(X), np.array(y)
                
                # Load the model
                model = keras.models.load_model(os.path.join(model_dir, "microsoft_stock_predictor.keras"))
                
                # Prepare test data
                test_data = scaled_data[training_data_len - 60:]
                X_test, y_test = create_sequences(test_data)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                
                # Make predictions
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                
                # Future predictions
                st.subheader("Future Price Prediction")
                prediction_days = st.slider("Select number of days to predict ahead", 1, 30, 1)
                
                # Get last 60 days for future prediction
                last_60_days = scaled_data[-60:]
                future_predictions = []
                current_sequence = last_60_days.reshape((1, 60, 1))
                
                for _ in range(prediction_days):
                    # Get prediction for next day
                    next_pred = model.predict(current_sequence)
                    future_predictions.append(next_pred[0, 0])
                    
                    # Update sequence for next prediction
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[0, -1, 0] = next_pred[0, 0]
                
                # Convert future predictions to original scale
                future_predictions = np.array(future_predictions).reshape(-1, 1)
                future_prices = scaler.inverse_transform(future_predictions)
                
                # Display predictions
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"${data['close'].iloc[-1]:.2f}"
                    )
                with col2:
                    st.metric(
                        label=f"Predicted Price ({prediction_days} days ahead)",
                        value=f"${future_prices[-1][0]:.2f}",
                        delta=f"{((future_prices[-1][0] - data['close'].iloc[-1])/data['close'].iloc[-1])*100:.2f}%"
                    )
                
                # Plot predictions
                st.subheader("Price Predictions")
                fig = plt.figure(figsize=(12, 6))
                last_date = data['date'].max()
                future_dates = pd.date_range(start=last_date, periods=prediction_days + 1, freq='D')[1:]
                
                plt.plot(data['date'][-30:], data['close'][-30:], label='Historical Data')
                plt.plot(future_dates, future_prices, label='Predictions', linestyle='--')
                plt.title('Stock Price Prediction')
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.legend()
                st.pyplot(fig)
                
                # Show metrics for historical predictions
                mape = mean_absolute_percentage_error(
                    data['close'][training_data_len:].values,
                    predictions
                )
                r2 = r2_score(
                    data['close'][training_data_len:].values,
                    predictions
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MAPE", f"{mape * 100:.2f}%")
                with col2:
                    st.metric("R-squared (R²)", f"{r2:.2f}")
                    
            except Exception as e:
                st.error("Error in LSTM prediction process")
                st.exception(e)
        
        else:  # Random Forest
            # Prepare data for Random Forest
            def create_features(data, lookback=60):
                X, y = [], []
                for i in range(lookback, len(data)):
                    X.append(data[i-lookback:i])
                    y.append(data[i])
                return np.array(X), np.array(y)
            
            # Scale the data
            scaler = joblib.load(os.path.join(model_dir, "scaler.save"))
            scaled_data = scaler.transform(dataset)
            
            # Split into train and test
            train_data = scaled_data[:training_data_len]
            test_data = scaled_data[training_data_len - 60:]
            
            # Create features
            X_train, y_train = create_features(train_data)
            X_test, y_test = create_features(test_data)
            
            # Reshape for Random Forest
            X_train_2d = X_train.reshape(X_train.shape[0], -1)
            X_test_2d = X_test.reshape(X_test.shape[0], -1)
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_2d, y_train.ravel())
            
            # Make predictions
            predictions = rf_model.predict(X_test_2d)
            predictions = predictions.reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions)
        
        # Show predictions plot (common for both models)
        train = data[:training_data_len]
        test = data[training_data_len:]
        test['Predictions'] = predictions
        
        fig = plt.figure(figsize=(12, 8))
        plt.plot(train['date'], train['close'], label='Train(Actual Data)', color='blue')
        plt.plot(test['date'], test['close'], label='Test(Actual Data)', color='orange')
        plt.plot(test['date'], test['Predictions'], label='Predictions', color='red')
        plt.title(f'Stock Price Prediction using {model_type.split(".")[1]}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(fig)
        
        # Show metrics
        col1, col2 = st.columns(2)
        with col1:
            mape = mean_absolute_percentage_error(test['close'], test['Predictions'])
            st.metric("MAPE", f"{mape * 100:.2f}%")
        with col2:
            r2 = r2_score(test['close'], test['Predictions'])
            st.metric("R-squared (R²)", f"{r2:.2f}")
            
    except Exception as e:
        st.error(f"Error loading model or making predictions with {model_type}.")
        st.exception(e)
