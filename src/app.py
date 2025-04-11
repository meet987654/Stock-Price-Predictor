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
            # Existing LSTM code
            model = keras.models.load_model(os.path.join(model_dir, "microsoft_stock_predictor.keras"))
            scaler = joblib.load(os.path.join(model_dir, "scaler.save"))
            
            # Scale the data
            scaled_data = scaler.transform(dataset)
            test_data = scaled_data[training_data_len - 60:]
            x_test = []
            
            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, 0])
                
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Make predictions
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            
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