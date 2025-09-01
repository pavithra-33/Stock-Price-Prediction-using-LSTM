import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load CSV
df = pd.read_csv('stock_data.csv')

# Ensure 'Close' column is numeric
if 'Close' not in df.columns:
    raise ValueError("Column 'Close' not found in CSV.")

# Drop rows with missing or non-numeric 'Close' values
df = df[['Close']].dropna()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna()

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Save preprocessed data
np.save('X.npy', X)
np.save('y.npy', y)
print("✅ Preprocessing complete. Saved X.npy and y.npy.")
