from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, r2_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings

# Update the data loading path
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "MicrosoftStock.csv")
data = pd.read_csv(data_path, parse_dates=['date'])
print(data.columns)  # Debugging step to check column names
print(data.head())
print(data.info())
print(data.describe())

#initial data visualization
#plot 1 - Open and Close prices of time
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['open'], label='Open Price', color='blue')
plt.plot(data['date'], data['close'], label='Close Price', color='red')
plt.title('Open and Close Prices Over Time')
plt.legend()


#plot 2 - Trading Volume (check for outliers)
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['volume'], label='Trading Volume', color='green')
plt.title('Stock Volume Over Time')


#Drop non-numeric columns
numeric_data = data.select_dtypes(include=["int64", "float64"])

#plot 3 - Correlation between features
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation HeatMap')


#convert date into date time then create date filter
data['date'] = pd.to_datetime(data['date'])

prediction = data.loc[
                      (data['date'] > '2020,01,01') &
                      (data['date'] < '2023,01,01')
                      ]


plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['open'], color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Prices Over Time')


#prepare for LSTM Model (Sequential)
stock_close = data.filter(["close"])
dataset = stock_close.values # Convert to numpy array
training_data_len = int(np.ceil( len(dataset) * 0.95 ))

#preprocessing Stages
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len] #95% of all ouur data

X_train, y_train = [], []

#create a sliding window for our stock(60days)
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Build the model
model = keras.Sequential()

#first Layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))

#second Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

#third Layer
model.add(keras.layers.Dense(128, activation='relu'))

#fourth Layer
model.add(keras.layers.Dropout(0.5))

#final output
model.add(keras.layers.Dense(1))

model.summary()

from tensorflow.keras.metrics import RootMeanSquaredError  # type: ignore # Import the metric

model.compile(
    optimizer='adam',
    loss='mae',
    metrics=[RootMeanSquaredError()]  # Use the metric class directly
)

training = model.fit(X_train, y_train, batch_size=32, epochs=20)

# Save the model with .keras extension
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model
model_path = os.path.join(model_dir, "microsoft_stock_predictor.keras")
model.save(model_path)
print(f"Model saved to {model_path}")

# Save the scaler
import joblib
scaler_path = os.path.join(model_dir, "scaler.save")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# To load the model later, you can use:
from tensorflow import keras
import joblib

def load_model_and_scaler(model_dir="model"):
    model = keras.models.load_model(os.path.join(model_dir, "microsoft_stock_predictor"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.save"))
    return model, scaler

#prep test data
test_data = scaled_data[training_data_len - 60:]
x_test,y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

X_test = np.array(x_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

#plotting data
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12, 8))
plt.plot(train['date'], train['close'], label='Train(Actual Data)', color='blue')
plt.plot(test['date'], test['close'], label='Test(Actual Data)', color='orange')
plt.plot(test['date'], test['Predictions'], label='Predictions', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Calculate MAPE
mape = mean_absolute_percentage_error(test['close'], test['Predictions'])
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# Calculate R-squared
r2 = r2_score(test['close'], test['Predictions'])
print(f"R-squared (R2): {r2:.2f}")