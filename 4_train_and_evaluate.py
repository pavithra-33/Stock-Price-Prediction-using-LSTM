import numpy as np
from sklearn.model_selection import train_test_split
from build_lstm_model import build_model

# Load data
X = np.load('X.npy')
y = np.load('y.npy')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train model
model = build_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Save model
model.save('lstm_stock_model.h5')
print("Model trained and saved.")
