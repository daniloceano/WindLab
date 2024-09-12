import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from ReadWindCube import WindOperationsAccessor, create_xarray_dataset
import os 

# Function to create Hovmöller Diagram
def create_hovmoller_diagram(data, time, heights, label, title):
    """
    Create a Hovmöller Diagram with given data on the x-axis and height on the y-axis.
    
    Parameters:
    - data: 2D array of data (rows: heights, columns: time steps)
    - time: time array for the x-axis
    - heights: list of heights for the y-axis
    - label: label for the color bar
    - title: title of the plot
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Ensure time and heights arrays are 1D and match the data
    time_grid, height_grid = np.meshgrid(time, heights)
    
    # Plotting the heatmap (Hovmöller diagram)
    plt.contourf(time_grid, height_grid, data, cmap='coolwarm', levels=50)
    
    # Adding color bar
    plt.colorbar(label=label)
    
    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('Height (m)')
    plt.title(title)

    return fig

# Load the data from .rtd into a Pandas DataFrame
file_path = "./WLS866-104_2024_08_01__00_00_00.rtd"
df = pd.read_csv(file_path, encoding='unicode_escape', skiprows=range(41), sep='\t')

# Create the xarray dataset from the DataFrame
ds = create_xarray_dataset(df)

# Now we compute the rolling standard deviation using the WindOperationsAccessor class

# Compute rolling standard deviation for each height
std_40m = ds.wind_cube.compute_std_detrended_data(ds.wind_cube.get_variable(40, 'Wind Speed (m/s)'), window_size=600)
std_60m = ds.wind_cube.compute_std_detrended_data(ds.wind_cube.get_variable(60, 'Wind Speed (m/s)'), window_size=600)
std_70m = ds.wind_cube.compute_std_detrended_data(ds.wind_cube.get_variable(70, 'Wind Speed (m/s)'), window_size=600)
std_100m = ds.wind_cube.compute_std_detrended_data(ds.wind_cube.get_variable(100, 'Wind Speed (m/s)'), window_size=600)
std_120m = ds.wind_cube.compute_std_detrended_data(ds.wind_cube.get_variable(120, 'Wind Speed (m/s)'), window_size=600)
std_150m = ds.wind_cube.compute_std_detrended_data(ds.wind_cube.get_variable(150, 'Wind Speed (m/s)'), window_size=600)

# Combine all wind standard deviations into a 2D array (time steps as columns, heights as rows)
hovmoller_data_std = np.vstack([std_40m.values, std_60m.values, std_70m.values, std_100m.values, std_120m.values, std_150m.values])

# Define the vertical levels for the y-axis (heights)
vertical_levels = [40, 60, 70, 100, 120, 150]

# Time array for the x-axis
time = ds['time'].values

# Call the function to create and show the Hovmöller Diagram for standard deviation
fig = create_hovmoller_diagram(hovmoller_data_std, time, vertical_levels, '10-min Wind Std Dev (m/s)', 'Hovmöller Diagram: Wind Speed Std Dev Across Heights')

# Save the plot
os.makedirs('plots', exist_ok=True)
fig.savefig('plots/hovmoller_std.png')