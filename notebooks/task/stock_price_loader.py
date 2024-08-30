import pandas as pd

import pandas as pd
import yfinance as yf


def fetch_stock_prices(ticker):
    # Fetch the historical stock price data using yfinance
    stock_data = yf.Ticker(ticker).history(period="max")
    
    # Return the fetched stock price data
    return stock_data