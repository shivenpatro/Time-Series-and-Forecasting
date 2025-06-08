import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose # Added for decomposition
from statsmodels.tsa.stattools import adfuller # Added for ADF test

# --- Load Data (Copied from the end of 02_data_inspection.py logic) ---
csv_file_name = "data/AAPL_stock_data.csv" # Updated path
try:
    df = pd.read_csv(csv_file_name, header=0, skiprows=[1, 2])
    if len(df.columns) > 0:
        actual_date_column_name = df.columns[0]
        df = df.rename(columns={actual_date_column_name: 'Date'})
    else:
        raise ValueError("Loaded DataFrame has no columns after skipping rows.")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    for col in df.columns:
        if col.lower() == 'volume':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("Data loaded successfully for preprocessing and visualization.")
    print(df.head())
    df.info()

except FileNotFoundError:
    print(f"Error: The file {csv_file_name} was not found.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()
# --- End Load Data ---

# Step 2.2: Feature Engineering (Basic)
# We will primarily focus on the 'Close' price for forecasting.
close_prices = df['Close']

print("\nSelected 'Close' price series for analysis:")
print(close_prices.head())

# The 'Date' is already a DatetimeIndex.
print(f"\nIndex type: {type(df.index)}")

# Further steps for visualization and preprocessing will be added below.
print("\nStep 2.1 (Handling Missing Data): Skipped as no missing values were found.")
print("Step 2.2 (Feature Engineering - Basic): 'Close' price selected as target. Date index confirmed.")

# --- Step 2.3: Time Series Visualization ---
print("\nStarting Step 2.3: Time Series Visualization...")

# Plot 1: Close price over time
plt.figure(figsize=(14, 7))
plt.plot(close_prices, label='AAPL Close Price')
plt.title('AAPL Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
print("Prepared Close Price plot.")

# Plot 2: Daily trading Volume over time
plt.figure(figsize=(14, 7))
plt.plot(df['Volume'], label='AAPL Volume', color='orange')
plt.title('AAPL Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
print("Prepared Trading Volume plot.")

# Plot 3: Close price with 50-day and 200-day moving averages
ma50 = close_prices.rolling(window=50).mean()
ma200 = close_prices.rolling(window=200).mean()

plt.figure(figsize=(14, 7))
plt.plot(close_prices, label='AAPL Close Price', alpha=0.7)
plt.plot(ma50, label='50-Day Moving Average', linestyle='--')
plt.plot(ma200, label='200-Day Moving Average', linestyle=':')
plt.title('AAPL Close Price with 50-Day & 200-Day Moving Averages')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
print("Prepared Close Price with Moving Averages plot.")

# Plot 4: Box plot of Close prices by month (for yearly seasonality)
df_for_boxplot = df.copy() 
df_for_boxplot['Month'] = df_for_boxplot.index.strftime('%b') 
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
unique_months_in_data = df_for_boxplot['Month'].unique()
ordered_categories = [m for m in month_order if m in unique_months_in_data] 
if len(ordered_categories) != len(unique_months_in_data): 
    all_plot_months = sorted(list(unique_months_in_data), key=lambda m: month_order.index(m) if m in month_order else -1)
    df_for_boxplot['Month'] = pd.Categorical(df_for_boxplot['Month'], categories=all_plot_months, ordered=True)
else:
    df_for_boxplot['Month'] = pd.Categorical(df_for_boxplot['Month'], categories=ordered_categories, ordered=True)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='Close', data=df_for_boxplot)
plt.title('AAPL Close Price Distribution by Month (Yearly Seasonality Check)')
plt.xlabel('Month')
plt.ylabel('Close Price (USD)')
plt.grid(True)
print("Prepared Close Price by Month box plot.")

print("\nDisplaying plots from Step 2.3. Please close plot windows to continue script execution.")
plt.show()

print("\nStep 2.3: Time Series Visualization completed.")

# --- Step 2.4: Time Series Decomposition ---
print("\nStarting Step 2.4: Time Series Decomposition...")
decomposition_period = 252 
if len(close_prices) >= 2 * decomposition_period:
    result_decompose = seasonal_decompose(close_prices, model='multiplicative', period=decomposition_period)
    plt.figure(figsize=(14, 10)) 
    plt.subplot(411)
    plt.plot(result_decompose.observed, label='Observed')
    plt.legend(loc='upper left')
    plt.title('Time Series Decomposition (Multiplicative, Period=252)')
    plt.subplot(412)
    plt.plot(result_decompose.trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(result_decompose.seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(result_decompose.resid, label='Residual')
    plt.legend(loc='upper left')
    plt.tight_layout() 
    print("Prepared Time Series Decomposition plot.")
    print("\nDisplaying Time Series Decomposition plot. Please close plot window to continue.")
    plt.show()
else:
    print(f"Series is too short (length {len(close_prices)}) for decomposition with period {decomposition_period}. Skipping decomposition plot.")
print("\nStep 2.4: Time Series Decomposition completed.")

# --- Step 2.5: Stationarity Check ---
print("\nStarting Step 2.5: Stationarity Check...")

def perform_adf_test(series, series_name="Series"):
    """Performs ADF test and prints the results."""
    print(f"\nPerforming Augmented Dickey-Fuller test for {series_name}:")
    # Drop NaNs that might be present (e.g., after differencing)
    adf_result = adfuller(series.dropna()) 
    print(f'ADF Statistic for {series_name}: {adf_result[0]}')
    print(f'p-value for {series_name}: {adf_result[1]}')
    print(f'Critical Values for {series_name}:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value}')
    
    if adf_result[1] <= 0.05:
        print(f"Conclusion: {series_name} is likely stationary (p-value <= 0.05). Null hypothesis rejected.")
    else:
        print(f"Conclusion: {series_name} is likely non-stationary (p-value > 0.05). Null hypothesis cannot be rejected.")
    return adf_result[1] # Return p-value

# Perform ADF test on the original 'Close' price series
p_value_orig = perform_adf_test(close_prices, "Original Close Price")

# If non-stationary, try differencing
# We'll store the differenced series for later use with ARIMA/SARIMA
df['Close_diff_1'] = None # Initialize column

if p_value_orig > 0.05:
    print("\nOriginal 'Close' price is non-stationary. Applying first-order differencing.")
    close_prices_diff1 = close_prices.diff().dropna()
    df['Close_diff_1'] = close_prices.diff() # Store with NaNs for index alignment if needed later

    plt.figure(figsize=(14, 7))
    plt.plot(close_prices_diff1, label='First Order Differenced Close Price')
    plt.title('AAPL Close Price (1st Order Differencing)')
    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)
    print("Prepared plot for 1st Order Differenced Close Price.")
    # plt.show() # Show along with other new plots or at the end of this step

    p_value_diff1 = perform_adf_test(close_prices_diff1, "1st Order Differenced Close Price")

    if p_value_diff1 > 0.05:
        print("\nFirst-order differenced series is still non-stationary. Applying second-order differencing.")
        # Note: For stock prices, one difference is usually enough.
        # Excessive differencing can remove useful information.
        close_prices_diff2 = close_prices_diff1.diff().dropna()
        df['Close_diff_2'] = close_prices_diff1.diff()

        plt.figure(figsize=(14, 7))
        plt.plot(close_prices_diff2, label='Second Order Differenced Close Price')
        plt.title('AAPL Close Price (2nd Order Differencing)')
        plt.xlabel('Date')
        plt.ylabel('Difference')
        plt.legend()
        plt.grid(True)
        print("Prepared plot for 2nd Order Differenced Close Price.")
        # plt.show()
        
        perform_adf_test(close_prices_diff2, "2nd Order Differenced Close Price")
else:
    print("\nOriginal 'Close' price series is stationary. No differencing needed for ARIMA/SARIMA based on ADF test.")

print("\nDisplaying plots from Stationarity Check (if any). Please close plot windows to continue.")
plt.show() # Show any new plots (differenced series)

print("\nStep 2.5: Stationarity Check completed.")
# Note: For Prophet and LSTM models, we typically use the original (non-differenced) series,
# as these models can handle non-stationarity internally or through their own preprocessing.
# The differenced series (e.g., close_prices_diff1) will be used for ARIMA/SARIMA.
