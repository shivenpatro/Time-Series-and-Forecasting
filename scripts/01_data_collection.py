import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the ticker symbol and the date range
ticker_symbol = "AAPL"
end_date = datetime.today()
start_date = end_date - timedelta(days=10*365.25) # Approximate 10 years

# Format dates as YYYY-MM-DD
end_date_str = end_date.strftime('%Y-%m-%d')
start_date_str = start_date.strftime('%Y-%m-%d')

print(f"Attempting to download data for {ticker_symbol} from {start_date_str} to {end_date_str}")

try:
    # Download stock data
    aapl_data = yf.download(ticker_symbol, start=start_date_str, end=end_date_str)

    if aapl_data.empty:
        print(f"No data downloaded for {ticker_symbol}. This could be due to an incorrect ticker, date range, or network issues.")
    else:
        print(f"\nSuccessfully downloaded data for {ticker_symbol}:")
        print(aapl_data.head())
        print(f"\nData shape: {aapl_data.shape}")
        print(f"\nData columns: {aapl_data.columns.tolist()}")
        print(f"\nDate range: {aapl_data.index.min()} to {aapl_data.index.max()}")

        # Save the DataFrame to a CSV file
        csv_file_name = f"data/{ticker_symbol}_stock_data.csv" # Updated path
        aapl_data.to_csv(csv_file_name)
        print(f"\nData saved to {csv_file_name}")

except Exception as e:
    print(f"An error occurred: {e}")
