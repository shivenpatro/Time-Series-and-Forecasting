import pandas as pd

# Define the CSV file name
csv_file_name = "data/AAPL_stock_data.csv" # Updated path

try:
    # Read the CSV.
    # header=0 uses the first line of the CSV as column names.
    # skiprows=[1, 2] skips the second and third lines of the CSV file
    # (which are the 'Ticker,AAPL,AAPL...' and 'Date,,,...' metadata lines).
    df = pd.read_csv(csv_file_name, header=0, skiprows=[1, 2])
    print(f"Successfully loaded data from {csv_file_name}, skipping metadata rows.\n")

    # The first column in the loaded df (e.g., df.columns[0], which was 'Price' in the header)
    # now contains the actual date strings. Rename this column to 'Date'.
    if len(df.columns) > 0:
        actual_date_column_name = df.columns[0]
        df = df.rename(columns={actual_date_column_name: 'Date'})
    else:
        raise ValueError("Loaded DataFrame has no columns after skipping rows.")

    # Convert the 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # Set the 'Date' column as the index
    df = df.set_index('Date')

    # Convert all other columns to numeric types.
    # errors='coerce' will turn any non-numeric values (like 'AAPL' if they weren't skipped properly) into NaN.
    for col in df.columns:
        # Volume can be integer, others float.
        # Using pd.NA for missing integers if needed, hence 'Int64'
        if col.lower() == 'volume':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("DataFrame after processing Date column and converting data types:")
    print("Head of processed DataFrame:")
    print(df.head())
    print("\nInfo of processed DataFrame:")
    df.info()
    print("\n")

    # Display the first 5 rows (already shown, but good for consistency)
    print("First 5 rows (head) again:")
    print(df.head())
    print("\n")

    # Display the last 5 rows
    print("Last 5 rows (tail):")
    print(df.tail())
    print("\n")

    # Provide descriptive statistics
    print("Descriptive statistics (describe):")
    print(df.describe(include='all')) # include='all' for non-numeric too if any
    print("\n")

    # Check for missing values in each column
    print("Missing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values)
    print("\nPercentage of missing values per column:")
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print(missing_percentage)
    print("\n")

    print("Current column names after processing:")
    print(df.columns.tolist())

except FileNotFoundError:
    print(f"Error: The file {csv_file_name} was not found. Please ensure it was created in the previous step.")
except Exception as e:
    print(f"An error occurred during data inspection: {e}")
    import traceback
    traceback.print_exc()
