import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import streamlit as st

# Load and preprocess Zomato data
def load_data():
    df = pd.read_csv('zomato.csv')
    return df

def preprocess_data(df):
    # Select relevant features for restaurant price prediction
    features = ['votes', 'average_cost_for_two', 'has_table_booking', 
                'has_online_delivery', 'rating']
    
    # Handle missing values
    df = df[features].fillna(0)
    
    # Convert categorical variables
    df['has_table_booking'] = df['has_table_booking'].map({'Yes': 1, 'No': 0})
    df['has_online_delivery'] = df['has_online_delivery'].map({'Yes': 1, 'No': 0})
    
    return df

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Main prediction function
def predict_price(input_data):
    df = load_data()
    processed_df = preprocess_data(df)
    
    # Prepare training data
    X = processed_df.drop(['average_cost_for_two'], axis=1)
    y = processed_df['average_cost_for_two']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = create_model()
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Process input data and make prediction
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0][0]
