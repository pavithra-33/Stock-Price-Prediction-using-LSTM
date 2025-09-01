import yfinance as yf
import pandas as pd

# Download stock data
ticker = 'AAPL'  # You can change this to any stock symbol
data = yf.download(ticker, start='2015-01-01', end='2023-12-31')

# Save to CSV
data.to_csv('stock_data.csv')
print("Data downloaded and saved as stock_data.csv")
