import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
from glob import glob
from ReadWindCube import WindOperationsAccessor, create_xarray_dataset

# Function to create a custom cyclic colormap
def create_custom_cyclic_cmap():
    """
    Create a cyclic colormap that distinguishes wind directions by quadrants.
    Red: North, Blue: South, Green: East, Yellow: West.
    """
    # Define colors for each cardinal direction
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#3498db']
    
    # Create a colormap with specified color stops corresponding to the cardinal directions
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cyclic_cmap', colors, N=20)
    return cmap

# Function to create a Hovmöller Diagram for wind direction or speed
def plot_hovmoller(ax, data, time, heights, cmap, label, title, levels=None):
    """
    Plot a Hovmöller Diagram on the provided Axes object.
    
    Parameters:
    - ax: Matplotlib Axes object to plot the diagram
    - data: 2D array of wind data (rows: heights, columns: time steps)
    - time: time array for the x-axis
    - heights: list of heights for the y-axis
    - cmap: colormap to use (cyclic for wind direction)
    - label: label for the color bar
    - title: title of the plot
    - levels: contour levels to use (optional)
    """
    # Plotting the heatmap (Hovmöller diagram)
    contour = ax.contourf(time, heights, data, cmap=cmap, levels=levels, extend='both')
    
    # Adding color bar
    cbar = plt.colorbar(contour, ax=ax, label=label)
    
    if 'Direction' in title:
        # Customize the color bar to show wind direction labels
        cbar.set_ticks([0, 90, 180, 270, 360])
        cbar.set_ticklabels(['N', 'E', 'S', 'W', 'N'])

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Height (m)')
    ax.set_title(title)
    print(f"Figure for {title} created.")

# Load the data from multiple .rtd files
files = sorted(glob("./Data/*.rtd"))

# Initialize an empty list to store DataFrames
df_list = []

# Loop through each file and read it
for file in files:
    print(f"Reading file: {file}")
    df = pd.read_csv(file, encoding='unicode_escape', skiprows=range(41), sep='\t')
    df_list.append(df)
print(f"Read {len(df_list)} files.")

# Concatenate all DataFrames into a single DataFrame
df_combined = pd.concat(df_list, ignore_index=True)

# Create the xarray dataset from the combined DataFrame
ds = create_xarray_dataset(df_combined)

# Automatically detect the available heights from the dataset
available_heights = ds.coords['height'].values

# Time array for the x-axis (assuming you already have timestamps in the data)
time = pd.to_datetime(df_combined['Timestamp'])
time = time.ffill()  # Forward fill NaN values in timestamps

# Aggregate wind speed data for each height
# Handle NaN values in wind speed data by forward-filling along the time dimension
hovmoller_data_speed = np.vstack([
    ds.wind_cube.get_variable(height, 'Wind Speed (m/s)').ffill(dim='time').values
    for height in available_heights
])
print(f"hovmoller_data_speed shape: {hovmoller_data_speed.shape}")

# Compute the time difference between consecutive time steps
time_diff = time.diff().dropna().median()  # Use median to handle any irregularities

# Convert time difference to seconds (assuming it's in Timedelta format)
time_diff_in_seconds = time_diff.total_seconds()

# Compute the number of time steps for desired window size
total_seconds = 60 # 1 minute
time_steps_for_rolling_std = int(total_seconds / time_diff_in_seconds)

# Compute 1 min rolling standard deviation from wind speed data
hovmoller_data_std = np.vstack([ds.wind_cube.get_variable(height, 'Wind Speed (m/s)').rolling(time=time_steps_for_rolling_std, min_periods=1, center=True).std()
                                 for height in available_heights])

# Compute rolling standard deviation for each height (for wind speed)
# hovmoller_data_std = np.vstack([
#     ds.wind_cube.compute_std_detrended_data(ds.wind_cube.get_variable(height, 'Wind Speed (m/s)'), window_size=time_steps_for_rolling_std).values
#     for height in available_heights
# ])

print(f"hovmoller_data_std shape: {hovmoller_data_std.shape}")

# Aggregate wind direction data for each height
# Handle NaN values in wind direction data by forward-filling along the time dimension
hovmoller_data_dir = np.vstack([
    ds.wind_cube.get_variable(height, 'Wind Direction (°)').ffill(dim='time').values
    for height in available_heights
])
print(f"hovmoller_data_dir shape: {hovmoller_data_dir.shape}")

# Xolormap for wind speed
speed_cmap = 'rainbow'

# Create custom colormap for wind direction
cyclic_cmap = create_custom_cyclic_cmap()

# Colormap for wind speed standard deviation
std_cmap = 'coolwarm'

# Create the figure with 3 rows and shared x-axis
fig, axes = plt.subplots(nrows=3, figsize=(12, 12), sharex=True)

# Plot the wind speed in the first row
plot_hovmoller(axes[0], hovmoller_data_speed, time, available_heights, speed_cmap, 'Wind Speed (m/s)', 'Wind Speed', levels=50)

# Plot the wind direction in the second row
plot_hovmoller(axes[1], hovmoller_data_dir, time, available_heights, cyclic_cmap, 'Wind Direction (°)', 'Wind Direction', levels=np.linspace(0, 360, 361))

# Plot the wind speed standard deviation in the third row
plot_hovmoller(axes[2], hovmoller_data_std, time, available_heights, std_cmap, f'{total_seconds}s Wind Std Dev (m/s)', 'Wind Speed Std Dev', levels=50)

# Save the final figure with all three plots
os.makedirs('plots', exist_ok=True)
filename = f'plots/hovmoller_combined_{total_seconds}s_std.png'
fig.savefig(filename)
print(f"Figure saved to '{filename}'.")