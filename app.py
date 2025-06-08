import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta
import os

# --- Configuration and Model/Scaler Loading ---
MODEL_PATH = "saved_models/lstm_final_model.keras"
SCALER_PATH = "saved_models/scaler_final.joblib"
DATA_PATH = "data/AAPL_stock_data.csv"
SEQUENCE_LENGTH = 60  # Must be the same as used during training

# Function to load model and scaler (cached for performance)
@st.cache_resource # Updated from st.cache for newer Streamlit versions
def load_lstm_model_and_scaler():
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.error("Please ensure 'scripts/07_final_forecast.py' has been run successfully to save the model and scaler.")
        return None, None

# Function to load historical data (cached for performance)
@st.cache_data # Updated from st.cache for newer Streamlit versions
def load_historical_data():
    try:
        df = pd.read_csv(DATA_PATH, header=0, skiprows=[1, 2])
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
        return df['Close'] # Return only the 'Close' price Series
    except Exception as e:
        st.error(f"Error loading historical data from {DATA_PATH}: {e}")
        return None

# --- Streamlit App UI ---
st.title("AAPL Stock Price Forecasting with LSTM")

# Load model, scaler, and data
model_lstm, scaler = load_lstm_model_and_scaler()
historical_close_prices = load_historical_data()

if model_lstm is None or scaler is None or historical_close_prices is None:
    st.stop() # Stop execution if essential components failed to load

st.subheader("Historical AAPL Close Prices")
st.line_chart(historical_close_prices)

st.sidebar.header("Forecast Settings")
days_to_forecast = st.sidebar.number_input("Number of future days to forecast:", min_value=1, max_value=365, value=30)

if st.sidebar.button("Generate Forecast"):
    if historical_close_prices.empty:
        st.error("Cannot generate forecast as historical data is empty.")
    else:
        st.subheader(f"LSTM Forecast for the Next {days_to_forecast} Days")
        
        # Prepare last sequence from historical data
        all_data_values = historical_close_prices.values.reshape(-1, 1)
        scaled_all_data = scaler.transform(all_data_values) # Use transform, not fit_transform
        
        last_sequence = scaled_all_data[-SEQUENCE_LENGTH:]
        current_batch = last_sequence.reshape((1, SEQUENCE_LENGTH, 1))
        
        future_predictions_scaled = []
        with st.spinner(f"Generating {days_to_forecast}-day forecast..."):
            for _ in range(days_to_forecast):
                current_pred_scaled = model_lstm.predict(current_batch)[0]
                future_predictions_scaled.append(current_pred_scaled)
                current_batch = np.append(current_batch[:, 1:, :], [[current_pred_scaled]], axis=1)
            
            future_predictions_unscaled = scaler.inverse_transform(future_predictions_scaled)
            future_predictions_unscaled = future_predictions_unscaled.reshape(-1)

        # Create future dates
        last_date_historical = historical_close_prices.index[-1]
        future_dates = pd.date_range(start=last_date_historical + timedelta(days=1), periods=days_to_forecast, freq='B')

        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_predictions_unscaled})
        forecast_df = forecast_df.set_index('Date')

        # Plot historical and forecasted data
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(historical_close_prices.index, historical_close_prices.values, label='Historical AAPL Close Price', color='blue')
        ax.plot(forecast_df.index, forecast_df['Forecast'], label=f'Next {days_to_forecast}-Day LSTM Forecast', color='red', linestyle='--')
        ax.axvline(last_date_historical, color='gray', linestyle=':', lw=2, label=f'Last Historical Data ({last_date_historical.date()})')
        ax.set_title(f'AAPL Close Price: Historical Data and Next {days_to_forecast}-Day Forecast (LSTM)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price (USD)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("Forecasted Values (Next {} Days):".format(days_to_forecast))
        st.dataframe(forecast_df)
else:
    st.sidebar.info("Click 'Generate Forecast' to see the LSTM model's prediction.")

st.markdown("---")
st.markdown("Developed as part of the Time Series Analysis and Forecasting Project.")
