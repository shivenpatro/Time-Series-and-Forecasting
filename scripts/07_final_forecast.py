import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from datetime import timedelta
import joblib # For saving the scaler
import os # For creating directory if it doesn't exist

# --- Load All Historical Data ---
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
    
    print("Full historical data loaded successfully for final forecasting.")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

except FileNotFoundError:
    print(f"Error: The file {csv_file_name} was not found.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()
# --- End Load Data ---

# Use the 'Close' price from the entire dataset
all_close_prices_df = df['Close'] # Series with DatetimeIndex
all_close_prices_values = all_close_prices_df.values.reshape(-1, 1) # Numpy array for scaling

# --- 1. Data Preparation for LSTM (on entire dataset) ---
print("\nPreparing entire dataset for LSTM training...")

# Scale the entire 'Close' price data
scaler_final = MinMaxScaler(feature_range=(0, 1))
scaled_all_data = scaler_final.fit_transform(all_close_prices_values)

# Create sequences from the entire scaled dataset
sequence_length = 60  # Same as used before

def create_sequences_for_final_model(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

X_full, y_full = create_sequences_for_final_model(scaled_all_data, sequence_length)

# Reshape X_full for LSTM: [samples, timesteps, features]
X_full = np.reshape(X_full, (X_full.shape[0], X_full.shape[1], 1))

print(f"X_full shape (for training on all data): {X_full.shape}")
print(f"y_full shape (for training on all data): {y_full.shape}")

# --- 2. Define and Re-train LSTM Model on Entire Dataset ---
print("\nDefining and re-training LSTM model on the entire dataset...")

model_final_lstm = Sequential()
model_final_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_full.shape[1], 1)))
model_final_lstm.add(Dropout(0.2))
model_final_lstm.add(LSTM(units=50, return_sequences=False))
model_final_lstm.add(Dropout(0.2))
model_final_lstm.add(Dense(units=1))
model_final_lstm.compile(optimizer='adam', loss='mean_squared_error')

model_final_lstm.summary()

print("Training final LSTM model... This may take a few minutes.")
# Fewer epochs might be okay as we have more data, or use similar epochs.
# No validation_split here as we are using all data for training the final model.
model_final_lstm.fit(X_full, y_full, epochs=50, batch_size=32, verbose=1)
print("Final LSTM model training completed.")

# Save the trained model and the scaler
print("\nSaving final LSTM model and scaler...")
model_save_path = "saved_models/lstm_final_model.keras"
scaler_save_path = "saved_models/scaler_final.joblib"

# Ensure the directory exists (it should, as we created it manually)
os.makedirs("saved_models", exist_ok=True)

model_final_lstm.save(model_save_path)
joblib.dump(scaler_final, scaler_save_path)
print(f"Model saved to {model_save_path}")
print(f"Scaler saved to {scaler_save_path}")

# --- 3. Forecast Next M Days ---
M = 30  # Number of future days to forecast
print(f"\nForecasting next {M} days...")

# Last `sequence_length` days from the original scaled data to start forecasting
last_sequence = scaled_all_data[-sequence_length:]
current_batch = last_sequence.reshape((1, sequence_length, 1)) # Reshape for model input

future_predictions_scaled = []

for i in range(M):
    # Get the prediction (next step)
    current_pred_scaled = model_final_lstm.predict(current_batch)[0]
    future_predictions_scaled.append(current_pred_scaled)
    # Update the batch: remove the first element and append the prediction
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred_scaled]], axis=1)

# Inverse transform the scaled predictions
future_predictions_unscaled = scaler_final.inverse_transform(future_predictions_scaled)
future_predictions_unscaled = future_predictions_unscaled.reshape(-1)

print(f"\nForecasted {M} days (unscaled):")
print(future_predictions_unscaled)

# Create future dates for plotting
last_date_historical = all_close_prices_df.index[-1]
future_dates = pd.date_range(start=last_date_historical + timedelta(days=1), periods=M, freq='B') 
# Using 'B' (business day frequency). Adjust if needed based on data's nature.

# --- 4. Plot Historical Data and Future Forecasts ---
plt.figure(figsize=(15, 8))
plt.plot(all_close_prices_df.index, all_close_prices_df.values, label='Historical AAPL Close Price', color='blue')
plt.plot(future_dates, future_predictions_unscaled, label=f'Next {M}-Day LSTM Forecast', color='red', linestyle='--')
plt.title(f'AAPL Close Price: Historical Data and Next {M}-Day Forecast (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.axvline(last_date_historical, color='gray', linestyle=':', lw=2, label=f'Last Historical Data ({last_date_historical.date()})')
plt.legend() # Call legend again to include axvline label
print("\nPrepared plot for historical data and future forecast.")
print("Displaying final forecast plot. Please close plot window to finish.")
plt.show()

print("\nStep 6.1: Final Forecasting with Best Model (LSTM) completed.")
print("Phase 6, Step 6.1 completed.")
