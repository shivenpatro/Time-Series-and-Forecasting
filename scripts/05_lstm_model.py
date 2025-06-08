import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error # For LSTM evaluation

# --- Load Data (Similar to previous scripts) ---
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
    
    print("Data loaded successfully for LSTM modeling.")

except FileNotFoundError:
    print(f"Error: The file {csv_file_name} was not found.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()
# --- End Load Data ---

sequence_length = 60  # Number of past days' prices to use for predicting the next day. Defined earlier now.

# Use the 'Close' price
close_prices_original_df = df['Close'] # Keep original Series with DatetimeIndex for plotting
close_prices_for_scaling = df['Close'].values.reshape(-1, 1) 

# --- Data Splitting (Chronological 80/20) ---
train_size_percentage = 0.8
train_size = int(len(close_prices_for_scaling) * train_size_percentage)

train_data = close_prices_for_scaling[:train_size]
test_data = close_prices_for_scaling[train_size:] # This is the original scale test data for y_true in evaluation

# Get the dates for the test set for plotting.
# These dates correspond to the target values in y_test_scaled / original_y_test_for_eval
# The test data starts after train_size. The actual y_test values start after an additional sequence_length period.
test_dates_start_index = train_size + sequence_length
if test_dates_start_index < len(close_prices_original_df.index):
    test_dates = close_prices_original_df.index[test_dates_start_index:]
else: # Handle case where test set is too small for any y_test values after sequencing
    test_dates = pd.DatetimeIndex([]) 


print(f"\nData Splitting for LSTM:")
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# --- Step 4.1: Data Preparation for LSTM ---
print("\nStarting Step 4.1: Data Preparation for LSTM...")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

print(f"\nScaled training data shape: {scaled_train_data.shape}")
print(f"\nScaled test data shape: {scaled_test_data.shape}")

def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0]) 
        y.append(data[i + sequence_length, 0])     
    return np.array(X), np.array(y)

# sequence_length is already defined globally
X_train, y_train = create_sequences(scaled_train_data, sequence_length)
X_test, y_test_scaled = create_sequences(scaled_test_data, sequence_length) # y_test_scaled are the scaled targets

print(f"\nUsing sequence length (timesteps): {sequence_length}")
print(f"X_train shape: {X_train.shape}") 
print(f"y_train shape: {y_train.shape}")   # These are scaled
print(f"X_test shape: {X_test.shape}")
print(f"y_test_scaled shape: {y_test_scaled.shape}") # These are scaled

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"\nReshaped X_train for LSTM: {X_train.shape}") 
print(f"Reshaped X_test for LSTM: {X_test.shape}")   
print("\nStep 4.1: Data Preparation for LSTM completed.")

# original_y_test_for_eval are the actual unscaled values corresponding to y_test_scaled
# These are the values from test_data starting from index sequence_length
original_y_test_for_eval = test_data[sequence_length:].reshape(-1) # Ensure it's 1D
print(f"Shape of original_y_test_for_eval: {original_y_test_for_eval.shape}")
if len(original_y_test_for_eval) != len(y_test_scaled):
    print("Warning: Length mismatch between original_y_test_for_eval and y_test_scaled!")


# --- Step 4.2: LSTM Model Architecture ---
print("\nStarting Step 4.2: LSTM Model Architecture...")
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(Dropout(0.2)) 
model_lstm.add(LSTM(units=50, return_sequences=False)) 
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1)) 
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
print("\nLSTM Model Architecture:")
model_lstm.summary() 
print("\nStep 4.2: LSTM Model Architecture completed.")

# --- Step 4.3: LSTM Model Training and Prediction ---
print("\nStarting Step 4.3: LSTM Model Training and Prediction...")
print("Training LSTM model... This may take a few minutes.")

# Train the model
# epochs: Number of times the model will cycle through the data.
# batch_size: Number of samples per gradient update.
# validation_split: Fraction of the training data to be used as validation data.
history = model_lstm.fit(X_train, y_train, 
                         epochs=50, # Start with a moderate number, e.g., 50-100
                         batch_size=32, 
                         validation_split=0.1, # Use 10% of training data for validation
                         verbose=1) # Show training progress

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
print("\nPrepared plot for LSTM Training & Validation Loss.")
# plt.show() # Will be shown at the end

# Generate predictions on the test set (X_test)
print("\nGenerating predictions on test data...")
predicted_prices_scaled = model_lstm.predict(X_test)

# Inverse transform the scaled predictions back to original price scale
predicted_prices_lstm = scaler.inverse_transform(predicted_prices_scaled)
predicted_prices_lstm = predicted_prices_lstm.reshape(-1) # Ensure it's 1D for metrics

# The actual values for comparison are original_y_test_for_eval
actual_prices_for_eval = original_y_test_for_eval

# Evaluate the model
rmse_lstm = np.sqrt(mean_squared_error(actual_prices_for_eval, predicted_prices_lstm))
mae_lstm = mean_absolute_error(actual_prices_for_eval, predicted_prices_lstm)

print(f"\nLSTM Model Evaluation:")
print(f"  RMSE: {rmse_lstm:.4f}")
print(f"  MAE:  {mae_lstm:.4f}")
lstm_results = {'model': 'LSTM', 'RMSE': rmse_lstm, 'MAE': mae_lstm}
print(f"Stored results: {lstm_results}")

# Plot actual vs. predicted
plt.figure(figsize=(14, 7))
plt.plot(test_dates, actual_prices_for_eval, label='Actual Test Data (Close)', color='blue', alpha=0.7)
plt.plot(test_dates, predicted_prices_lstm, label='LSTM Predictions', color='red', linestyle='--')
plt.title('AAPL Close Price: Actual vs. LSTM Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
print("\nPrepared plot for Actual vs. LSTM Predictions.")

print("\nDisplaying LSTM related plots. Please close plot windows to continue.")
plt.show()

print("\nStep 4.3: LSTM Model Training and Prediction completed.")

# Store all model results (from this script, assuming others are from previous script)
# This part would typically be in a main script that calls functions or in Phase 5
# For now, just printing the LSTM results.
print("\n--- LSTM Model Results ---")
print(lstm_results)
