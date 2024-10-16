import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Function to generate a dummy dataset based on the original columns
def generate_dummy_data(start_date, end_date, columns):
    # Create a date range with 1-hour intervals
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate random data for other columns, ensuring similar structure to the original
    n_rows = len(date_range)
    
    dummy_data = {
        'Timestamp': date_range.strftime('%Y/%m/%d %H:%M:%S.00'),
        'Position': np.random.choice([0, 90, 180, 270, 'V'], size=n_rows),
        'Temperature': np.random.uniform(15, 30, size=n_rows),
        'Wiper Count': np.zeros(n_rows),
        '40m CNR (dB)': np.random.uniform(-10, 0, size=n_rows),
        '40m Radial Wind Speed (m/s)': np.random.uniform(-5, 5, size=n_rows),
        '40m Radial Wind Speed Dispersion (m/s)': np.random.uniform(0, 0.1, size=n_rows),
        '40m Wind Speed (m/s)': np.random.uniform(2, 10, size=n_rows),
        '40m Wind Direction (°)': np.random.uniform(0, 360, size=n_rows),
        '40m X-wind (m/s)': np.random.uniform(-10, 10, size=n_rows),
        '40m Y-wind (m/s)': np.random.uniform(-10, 10, size=n_rows),
        '40m Z-wind (m/s)': np.random.uniform(-10, 10, size=n_rows),
        'Unnamed: 12': np.nan,  # Placeholder column
    }
    
    return pd.DataFrame(dummy_data)

# Define the start and end date for 2023
start_date = "2023-01-01"
end_date = "2023-12-31"

df = pd.read_csv("WLS866-104_2024_08_01__00_00_00.rtd", encoding='unicode_escape', skiprows=range(41), sep='\t')

# Use the columns from the original DataFrame
columns = df.columns

# Generate dummy data
dummy_df = generate_dummy_data(start_date, end_date, columns)

# Save the generated dummy data to a new file with the same format
output_file_path = "./dummy_data_2023.rtd"
dummy_df.to_csv(output_file_path, sep='\t', index=False, encoding='unicode_escape')

output_file_path