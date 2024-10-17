import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Function to generate dummy data based on a Weibull distribution for wind-related columns
def generate_dummy_data_weibull(start_date, end_date, columns, k=2, c=10):
    # Create a date range with 1-hour intervals
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Generate random data for other columns, ensuring similar structure to the original
    n_rows = len(date_range)
    
    # Generating random Weibull distributed data for wind speed related columns
    wind_speeds = np.random.weibull(k, n_rows) * c  # Weibull distribution for wind speed

    dummy_data = {
        'Timestamp': date_range.strftime('%Y/%m/%d %H:%M:%S.00'),
        'Position': np.random.choice([0, 90, 180, 270, 'V'], size=n_rows),
        'Temperature': np.random.uniform(15, 30, size=n_rows),
        'Wiper Count': np.zeros(n_rows),
        '40m CNR (dB)': np.random.uniform(-10, 0, size=n_rows),
        '40m Radial Wind Speed (m/s)': np.random.weibull(k, n_rows) * c,  # Weibull distribution for radial wind speed
        '40m Radial Wind Speed Dispersion (m/s)': np.random.uniform(0, 0.1, size=n_rows),
        '40m Wind Speed (m/s)': wind_speeds,  # Weibull distribution for wind speed
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

# Load the original DataFrame to get the columns (if needed)
df = pd.read_csv("WLS866-104_2024_08_01__00_00_00.rtd", encoding='unicode_escape', skiprows=range(41), sep='\t')

# Use the columns from the original DataFrame (optional, could also be removed)
columns = df.columns

# Generate dummy data using Weibull distribution
dummy_df = generate_dummy_data_weibull(start_date, end_date, columns, k=2, c=10)

# Save the generated dummy data to a new file with the same format
output_file_path = "./dummy_data_2023_weibull.rtd"
dummy_df.to_csv(output_file_path, sep='\t', index=False, encoding='unicode_escape')

output_file_path