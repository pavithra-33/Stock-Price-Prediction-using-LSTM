import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load model and data
model = load_model('lstm_stock_model.h5')
X = np.load('X.npy')
y = np.load('y.npy')

# Predict
predicted = model.predict(X)

# Load and clean original data
df = pd.read_csv('stock_data.csv')

# Ensure 'Close' column is numeric
df = df[['Close']].dropna()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna()

# Fit scaler only on clean numeric data
scaler = MinMaxScaler()
scaler.fit(df.values.reshape(-1, 1))


# Plot
# Inverse transform to get actual price values
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
predicted_prices = scaler.inverse_transform(predicted)

# Plot
plt.figure(figsize=(12,6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

