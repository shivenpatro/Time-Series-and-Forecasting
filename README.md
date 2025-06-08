# Time Series Analysis and Forecasting for Stock Market (AAPL)

## Project Objective
This project focuses on analyzing historical stock market data for Apple Inc. (AAPL) and developing various time series forecasting models to predict future price trends. The primary goal is to understand and apply time series concepts, implement different forecasting techniques (ARIMA, Prophet, LSTM), compare their performance, and visualize the insights.

## Dataset Used
- **Stock Ticker:** AAPL (Apple Inc.)
- **Data Source:** Yahoo Finance (downloaded via the `yfinance` library)
- **Period:** Approximately 10 years of daily historical data (e.g., June 2015 - June 2025, actual dates may vary slightly based on download).
- **Features:** 'Open', 'High', 'Low', 'Close', 'Adj Close' (implicitly, as 'Close' is adjusted), 'Volume'. The 'Close' price is the primary target for forecasting.
- **File:** `data/AAPL_stock_data.csv`

## Methodology
The project follows these key phases:
1.  **Environment Setup:** Creation of a Python virtual environment and installation of necessary libraries (`requirements.txt`).
2.  **Data Collection:** Downloading historical stock data for AAPL using `yfinance`.
3.  **Initial Data Inspection:** Understanding the dataset structure, checking for missing values, and basic statistics.
4.  **Data Preprocessing & Visualization:** 
    *   Visualizing 'Close' price, 'Volume', and moving averages.
    *   Performing time series decomposition (trend, seasonality, residuals).
    *   Checking for stationarity using the Augmented Dickey-Fuller (ADF) test and applying differencing if needed.
5.  **Classical Time Series Modeling:**
    *   Splitting data into training (80%) and testing (20%) sets chronologically.
    *   **ARIMA:** Parameter identification (ACF, PACF, `auto_arima`), model fitting, prediction, and evaluation (RMSE, MAE).
    *   **SARIMA:** Attempted with `auto_arima` for yearly seasonality (m=252), but encountered memory constraints.
    *   **Prophet:** Model fitting, prediction, evaluation, and plotting components.
6.  **Deep Learning Modeling:**
    *   **LSTM:** Data preparation (scaling, sequencing), model architecture definition, training, prediction, inverse scaling, and evaluation.
7.  **Model Comparison:** Comparing RMSE and MAE metrics across all implemented models to identify the best performer.
8.  **Final Forecasting:** Using the best model (LSTM) to forecast future stock prices (e.g., next 30 days) after re-training on the entire dataset.

## Tech Stack
- Python 3.x (e.g., 3.12.10 as used in setup)
- Pandas, NumPy (for data manipulation)
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (for preprocessing - MinMaxScaler, and evaluation metrics - MSE, MAE)
- Statsmodels (for ARIMA, SARIMA, seasonal decomposition, ADF test)
- Prophet (by Facebook, for Prophet model)
- TensorFlow/Keras (for LSTM model)
- pmdarima (for `auto_arima`)
- yfinance (for data collection)
- (Optional: Streamlit/Flask for deployment)

## Project Structure
```
.
|-- data/
|   |-- AAPL_stock_data.csv
|-- scripts/
|   |-- 01_data_collection.py
|   |-- 02_data_inspection.py
|   |-- 03_preprocessing_visualization.py
|   |-- 04_classical_models.py
|   |-- 05_lstm_model.py
|   |-- 06_model_comparison.py
|   |-- 07_final_forecast.py
|-- images/  
|   |-- (Saved plots can be placed here)
|-- .gitignore
|-- README.md
|-- requirements.txt 
```

## How to Run
1.  **Clone the repository (if applicable once on GitHub):**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the scripts sequentially from the project root directory:**
    *   `python scripts/01_data_collection.py` (Downloads data to `data/AAPL_stock_data.csv`)
    *   `python scripts/02_data_inspection.py` (Inspects the downloaded data)
    *   `python scripts/03_preprocessing_visualization.py` (Preprocessing, visualizations, decomposition, stationarity)
    *   `python scripts/04_classical_models.py` (ARIMA, SARIMA attempt, Prophet)
    *   `python scripts/05_lstm_model.py` (LSTM model training and evaluation)
    *   `python scripts/06_model_comparison.py` (Prints model comparison table and discussions)
    *   `python scripts/07_final_forecast.py` (Generates final forecast with LSTM)
    *   *(Note: Plot windows will appear during script execution; close them to allow the script to continue/finish.)*

## Key Results/Findings
*(Summarize your key findings here. For example:)*
- The LSTM model demonstrated the best performance on the test set with an RMSE of approximately **8.22** and an MAE of approximately **6.63**.
- The ARIMA(1,1,1) model was the best among the classical approaches, with an RMSE of ~34.30 and MAE of ~26.52.
- Prophet, with default settings, had higher errors (RMSE ~56.19, MAE ~50.52) compared to ARIMA and LSTM for this dataset.
- SARIMA with a yearly seasonal period (m=252) could not be effectively trained due to memory limitations with `auto_arima`.
- The 'Close' price series for AAPL was found to be non-stationary and required first-order differencing to achieve stationarity for ARIMA modeling.

*(You can add more details, such as specific insights from visualizations or model behaviors.)*

## Visualizations
*(You can embed key visualizations here if your GitHub setup allows, or provide links to them, or describe them. Examples:)*
- Plot of historical AAPL 'Close' prices.
- Plot of LSTM predictions vs. actual test data.
- Plot of the final 30-day forecast using the LSTM model.
- (Consider adding plots to the `images/` directory and linking them here)
  ```markdown
  ![LSTM Forecast](images/lstm_forecast_plot.png) 
  ```

## Future Work / Potential Improvements
*(Briefly mention potential next steps or improvements, e.g., from the conceptual tuning discussion in Step 5.3)*
- Exhaustive hyperparameter tuning for the LSTM model.
- Experimenting with different LSTM architectures (e.g., bidirectional, attention).
- Incorporating external regressors or other features (e.g., Volume, technical indicators) into models like Prophet or LSTM.
- Trying different seasonal periods for SARIMA if strong shorter-term seasonality is suspected.
- Deploying the best model as a web application (e.g., using Streamlit or Flask).
