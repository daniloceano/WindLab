import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from ReadWindCube import WindOperationsAccessor, create_xarray_dataset
import os
from glob import glob

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

# Get the list of all .rtd files
files = sorted(glob("./Data/*.rtd"))

# Initialize an empty list to store DataFrames
df_list = []

# Loop through each file and read it
for file in files:
    df = pd.read_csv(file, encoding='unicode_escape', skiprows=range(41), sep='\t')
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
df_combined = pd.concat(df_list, ignore_index=True)

# Create the xarray dataset from the combined DataFrame
ds = create_xarray_dataset(df_combined)

# Automatically detect the available heights from the dataset
available_heights = ds.coords['height'].values

# Now we compute the rolling standard deviation using the WindOperationsAccessor class for each detected height
hovmoller_data_std = np.vstack([
    ds.wind_cube.compute_std_detrended_data(ds.wind_cube.get_variable(height, 'Wind Speed (m/s)'), window_size=600).values
    for height in available_heights
])

# Time array for the x-axis
time = ds['time'].values

# Call the function to create and show the Hovmöller Diagram for standard deviation
fig = create_hovmoller_diagram(hovmoller_data_std, time, available_heights, '10-min Wind Std Dev (m/s)', 'Hovmöller Diagram: Wind Speed Std Dev Across Heights')

# Save the plot
os.makedirs('plots', exist_ok=True)
fig.savefig('plots/hovmoller_std.png')
