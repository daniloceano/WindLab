import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os 
# Function to create a custom cyclic colormap
def create_custom_cyclic_cmap():
    """
    Create a cyclic colormap that distinguishes wind directions by quadrants.
    Red: North, Blue: South, Green: East, Yellow: West.
    """
    # Define colors for each cardinal direction
    # Red for North (0°), Green for East (90°), Blue for South (180°), Yellow for West (270°)
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#3498db']
    
    # Create a colormap with specified color stops corresponding to the cardinal directions
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cyclic_cmap', colors, N=20)

    return cmap

def create_hovmoller_diagram(data, time, heights, cmap, label, title):
    """
    Create a Hovmöller Diagram with wind direction data on the x-axis and height on the y-axis.
    
    Parameters:
    - data: 2D array of wind direction (rows: heights, columns: time steps)
    - time: time array for the x-axis
    - heights: list of heights for the y-axis
    - cmap: colormap to use (cyclic for wind direction)
    - label: label for the color bar
    - title: title of the plot
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Plotting the heatmap (Hovmöller diagram)
    plt.contourf(time, heights, data, cmap=cmap, levels=np.linspace(0, 360, 361), extend='both')
    
    # Adding color bar
    cbar = plt.colorbar(label=label)
    
    # Customize the color bar to show wind direction labels
    cbar.set_ticks([0, 90, 180, 270, 360])
    cbar.set_ticklabels(['N', 'E', 'S', 'W', 'N'])
    
    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('Height (m)')
    plt.title(title)
    
    return fig

# Load the data (update with your actual data)
df = pd.read_csv("./WLS866-104_2024_08_01__00_00_00.rtd",
                 encoding='unicode_escape',  skiprows=range(41), sep='\t')

# Extract wind direction columns for each height
dir_40m = df['40m Wind Direction (°)']
dir_60m = df['60m Wind Direction (°)']
dir_70m = df['70m Wind Direction (°)']
dir_100m = df['100m Wind Direction (°)']
dir_120m = df['120m Wind Direction (°)']
dir_150m = df['150m Wind Direction (°)']

# Combine all wind direction data into a 2D array (time steps as columns, heights as rows)
hovmoller_data_dir = np.vstack([dir_40m, dir_60m, dir_70m, dir_100m, dir_120m, dir_150m])

# Time array for the x-axis (assuming you already have timestamps in the data)
time = pd.to_datetime(df['Timestamp'])
vertical_levels = [40, 60, 70, 100, 120, 150]  # Heights for the y-axis

# Create custom colormap for wind direction
cyclic_cmap = create_custom_cyclic_cmap()

# Call the function to create and show the Hovmöller Diagram for wind direction
fig = create_hovmoller_diagram(hovmoller_data_dir, time, vertical_levels, cyclic_cmap, 'Wind Direction (°)', 'Hovmöller Diagram: Wind Direction Across Heights')

# Save the plot
os.makedirs('plots', exist_ok=True)
fig.savefig('plots/hovmoller_wind_dir.png')