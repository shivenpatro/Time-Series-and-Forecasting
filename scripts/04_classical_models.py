import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split # Though for time series, manual split is better
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX # For SARIMA if needed
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm 
from prophet import Prophet # Added for Prophet model

# --- Load Data (Similar to 03_preprocessing_visualization.py) ---
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
    
    for col_name in df.columns: 
        if col_name.lower() == 'volume':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Int64')
        else:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    
    print("Data loaded successfully for classical modeling.")

except FileNotFoundError:
    print(f"Error: The file {csv_file_name} was not found.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()
# --- End Load Data ---

# Prepare the series we need
close_prices = df['Close']
close_prices_diff1 = df['Close'].diff().dropna() 

# --- Step 3.1: Data Splitting ---
print("\nStarting Step 3.1: Data Splitting...")
train_size_percentage = 0.8
train_size_orig = int(len(close_prices) * train_size_percentage)
train_close = close_prices[:train_size_orig]
test_close = close_prices[train_size_orig:]

train_size_diff = int(len(close_prices_diff1) * train_size_percentage)
train_close_diff = close_prices_diff1[:train_size_diff]

print(f"\nOriginal 'Close' price series:")
print(f"Training set size: {len(train_close)} ({train_close.index.min()} to {train_close.index.max()})")
print(f"Test set size: {len(test_close)} ({test_close.index.min()} to {test_close.index.max()})")

plt.figure(figsize=(14, 7))
plt.plot(train_close, label='Training Data (Close Price)')
plt.plot(test_close, label='Test Data (Close Price)')
plt.title('AAPL Close Price - Train/Test Split')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
print("\nPrepared plot for Train/Test Split of Close prices.")
print("Displaying Train/Test Split plot. Please close plot window to continue.")
plt.show()
print("\nStep 3.1: Data Splitting completed.")

# --- Step 3.2: ARIMA Model ---
print("\nStarting Step 3.2: ARIMA Model...")
d = 1 
print(f"Order of differencing (d) = {d}")

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(train_close_diff, ax=axes[0], lags=40, title='ACF for Differenced Training Data')
plot_pacf(train_close_diff, ax=axes[1], lags=40, title='PACF for Differenced Training Data', method='ywm')
plt.tight_layout()
print("Displaying ACF/PACF plots. Please close plot windows to continue.")
plt.show()

print("\nRunning auto_arima to find optimal ARIMA parameters...")
auto_arima_model_fit = pm.auto_arima(train_close, 
                                 start_p=0, start_q=0, max_p=5, max_q=5, 
                                 d=d, seasonal=False, trace=True,       
                                 error_action='ignore', suppress_warnings=True, stepwise=True)    
print("\nAuto ARIMA Model Summary:")
print(auto_arima_model_fit.summary())
arima_order = auto_arima_model_fit.order 
print(f"\nRecommended ARIMA order (p,d,q) from auto_arima: {arima_order}")

print(f"\nFitting ARIMA{arima_order} model using statsmodels...")
model = ARIMA(train_close, order=arima_order)
model_fit = model.fit()
print(model_fit.summary())

forecast_steps = len(test_close)
predictions_arima = model_fit.forecast(steps=forecast_steps)
predictions_arima.index = test_close.index

rmse_arima = np.sqrt(mean_squared_error(test_close, predictions_arima))
mae_arima = mean_absolute_error(test_close, predictions_arima)
print(f"\nARIMA{arima_order} Model Evaluation:")
print(f"  RMSE: {rmse_arima:.4f}")
print(f"  MAE:  {mae_arima:.4f}")
arima_results = {'model': f'ARIMA{arima_order}', 'RMSE': rmse_arima, 'MAE': mae_arima} # Storing results
print(f"Stored results: {arima_results}")

plt.figure(figsize=(14, 7))
plt.plot(train_close, label='Training Data (Close)')
plt.plot(test_close, label='Actual Test Data (Close)', color='blue')
plt.plot(predictions_arima, label=f'ARIMA{arima_order} Predictions', color='red', linestyle='--')
plt.title(f'AAPL Close Price: Actual vs. ARIMA{arima_order} Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
print("\nPrepared plot for Actual vs. ARIMA Predictions.")
print("Displaying Actual vs. ARIMA Predictions plot. Please close plot window to continue.")
plt.show()
print(f"\nStep 3.2: ARIMA Model completed.")

# --- Step 3.3: SARIMA Model ---
print("\nStarting Step 3.3: SARIMA Model...")
print("Attempting to find optimal SARIMA parameters using auto_arima.")
print("This may take some time due to seasonal search with m=252...")
sarima_results = {} # Initialize results for SARIMA

try:
    auto_sarima_model_fit = pm.auto_arima(train_close, 
                                     start_p=0, start_q=0, max_p=3, 
                                     start_P=0, start_Q=0, max_P=2, max_Q=2, 
                                     d=d, seasonal=True, m=252, D=None,           
                                     trace=True, error_action='ignore',  
                                     suppress_warnings=True, stepwise=True, n_jobs=-1)       

    print("\nAuto SARIMA Model Summary:")
    print(auto_sarima_model_fit.summary())
    sarima_order_nons = auto_sarima_model_fit.order 
    seasonal_order_params = auto_sarima_model_fit.seasonal_order
    print(f"\nRecommended SARIMA order (p,d,q)(P,D,Q,m) from auto_arima: {sarima_order_nons}{seasonal_order_params}")

    if seasonal_order_params[0] > 0 or seasonal_order_params[1] > 0 or seasonal_order_params[2] > 0:
        print(f"\nFitting SARIMAX{sarima_order_nons}{seasonal_order_params} model using statsmodels...")
        sarima_model = SARIMAX(train_close, order=sarima_order_nons, seasonal_order=seasonal_order_params,
                               enforce_stationarity=False, enforce_invertibility=False)
        sarima_model_fit_sm = sarima_model.fit(disp=False) 
        print(sarima_model_fit_sm.summary())

        predictions_sarima = sarima_model_fit_sm.forecast(steps=forecast_steps)
        predictions_sarima.index = test_close.index

        rmse_sarima = np.sqrt(mean_squared_error(test_close, predictions_sarima))
        mae_sarima = mean_absolute_error(test_close, predictions_sarima)
        print(f"\nSARIMAX{sarima_order_nons}{seasonal_order_params} Model Evaluation:")
        print(f"  RMSE: {rmse_sarima:.4f}")
        print(f"  MAE:  {mae_sarima:.4f}")
        sarima_results = {'model': f'SARIMAX{sarima_order_nons}{seasonal_order_params}', 'RMSE': rmse_sarima, 'MAE': mae_sarima}
        print(f"Stored results: {sarima_results}")

        plt.figure(figsize=(14, 7))
        plt.plot(train_close, label='Training Data (Close)')
        plt.plot(test_close, label='Actual Test Data (Close)', color='blue')
        plt.plot(predictions_sarima, label=f'SARIMA Predictions', color='green', linestyle='--')
        plt.title(f'AAPL Close Price: Actual vs. SARIMA Predictions')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.legend()
        plt.grid(True)
        print("\nPrepared plot for Actual vs. SARIMA Predictions.")
        print("Displaying Actual vs. SARIMA Predictions plot. Please close plot window to continue.")
        plt.show()
    else:
        print("\nAuto_arima did not find a significant seasonal component (P, D, Q are all zero).")
        print("The SARIMA model collapses to the ARIMA model already fitted.")
        sarima_results = {'model': f'SARIMA (collapses to ARIMA{arima_order})', 'RMSE': rmse_arima, 'MAE': mae_arima} # Use previous ARIMA results

except Exception as e:
    print(f"An error occurred during auto_arima for SARIMA: {e}")
    print("Skipping SARIMA due to error.")
    sarima_results = {'model': 'SARIMA (Error)', 'RMSE': np.nan, 'MAE': np.nan}
print(f"Stored SARIMA results: {sarima_results}")
print(f"\nStep 3.3: SARIMA Model completed.")

# --- Step 3.4: Prophet Model ---
print("\nStarting Step 3.4: Prophet Model...")
prophet_results = {} # Initialize results for Prophet

# Prepare data for Prophet: DataFrame with 'ds' (datestamp) and 'y' (value) columns
# Prophet expects the 'ds' column to be datetime objects. Our index is already datetime.
prophet_train_df = train_close.reset_index() # Convert index to column
prophet_train_df = prophet_train_df.rename(columns={'Date': 'ds', 'Close': 'y'})

# Initialize and fit the Prophet model
# We can add holidays or custom seasonalities later if needed. Start simple.
# Prophet can detect yearly and weekly seasonality by default if data is daily.
# It can also detect daily seasonality if data has timestamps.
model_prophet = Prophet(daily_seasonality=False) # Daily seasonality not relevant for daily stock prices
# yearly_seasonality and weekly_seasonality are True by default.
# We could add changepoint_prior_scale for flexibility if needed.

print("Fitting Prophet model...")
model_prophet.fit(prophet_train_df)

# Create a DataFrame for future dates (for the length of the test set)
future_dates = model_prophet.make_future_dataframe(periods=len(test_close), freq='B') # 'B' for business day frequency
# Check if the last date of future_dates aligns with test_close.index.max()
# If yfinance data skips weekends/holidays, 'B' is appropriate.
# If it includes all days and NaNs them, 'D' might be, but yfinance usually gives trading days.

print(f"Future dates head: {future_dates.head()}")
print(f"Future dates tail: {future_dates.tail()}")
print(f"Test data end date: {test_close.index.max()}")


# Make predictions
forecast_prophet = model_prophet.predict(future_dates)

# Extract the forecast for the test period
# The forecast_prophet DataFrame contains many columns, including 'yhat', 'yhat_lower', 'yhat_upper'
# We need the 'yhat' values that correspond to the test_close period.
# The forecast starts from the beginning of the training data.
predictions_prophet = forecast_prophet['yhat'][-len(test_close):] # Get the last N predictions
predictions_prophet.index = test_close.index # Align index with test_close for evaluation

# Evaluate the model
rmse_prophet = np.sqrt(mean_squared_error(test_close, predictions_prophet))
mae_prophet = mean_absolute_error(test_close, predictions_prophet)
print(f"\nProphet Model Evaluation:")
print(f"  RMSE: {rmse_prophet:.4f}")
print(f"  MAE:  {mae_prophet:.4f}")
prophet_results = {'model': 'Prophet', 'RMSE': rmse_prophet, 'MAE': mae_prophet}
print(f"Stored results: {prophet_results}")

# Plot the forecast
print("\nPlotting Prophet forecast...")
fig1_prophet = model_prophet.plot(forecast_prophet)
plt.title('AAPL Close Price: Prophet Forecast (Train + Test Period)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
# Add actual test data to the plot for comparison
plt.plot(test_close.index, test_close.values, '.r', label='Actual Test Data')
plt.legend()
# plt.show() # Handled by final plt.show()

# Plot forecast components (trend, weekly, yearly seasonality)
fig2_prophet = model_prophet.plot_components(forecast_prophet)
# plt.show() # Handled by final plt.show()
print("Prepared Prophet forecast plots.")

print("\nDisplaying Prophet plots. Please close plot windows to continue.")
plt.show()

print("\nStep 3.4: Prophet Model completed.")

# Store all model results in a list for later comparison
all_model_results = [arima_results, sarima_results, prophet_results]
print("\n--- Current Model Results ---")
for res in all_model_results:
    if res: # Check if dict is not empty
        print(res)
