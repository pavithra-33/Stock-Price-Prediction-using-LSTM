## 📈 Stock Price Prediction Using LSTM

### 🔍 Problem Statement
Predicting stock trends can help users make smarter investment decisions. This project uses Long Short-Term Memory (LSTM) neural networks to forecast future stock prices based on historical data.

---

### 🎯 Objective
Build and train an LSTM model to predict stock prices using time-series data from Yahoo Finance, and visualize actual vs predicted trends.

---

### 🧰 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- yfinance

---

### 📁 Project Structure

```
stock_price_prediction/
│
├── 1_fetch_data.py              # Download stock data from Yahoo Finance
├── 2_preprocess_data.py         # Normalize and reshape data for LSTM
├── build_lstm_model.py          # Define LSTM architecture
├── 4_train_and_evaluate.py      # Train model and save it
├── 5_plot_predictions.py        # Visualize actual vs predicted prices
├── stock_data.csv               # Saved historical data
├── X.npy, y.npy                 # Preprocessed training data
├── lstm_stock_model.h5          # Trained model
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

### 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock_price_prediction.git
   cd stock_price_prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### 🚀 How to Run

Run each script in order:

1. **Download stock data**
   ```bash
   python 1_fetch_data.py
   ```

2. **Preprocess data**
   ```bash
   python 2_preprocess_data.py
   ```

3. **Train and save model**
   ```bash
   python 4_train_and_evaluate.py
   ```

4. **Plot predictions**
   ```bash
   python 5_plot_predictions.py
   ```

---

### 📊 Output

- A graph showing actual vs predicted stock prices.
- Model saved as `lstm_stock_model.h5` for reuse or deployment.
