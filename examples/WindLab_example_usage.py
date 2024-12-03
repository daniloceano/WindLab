from glob import glob
import matplotlib.pyplot as plt
from windlab import WindDataAccessor

### Reading data ###

# Open the data using the WindDataAccessor and specify the reference height
ds = WindDataAccessor.windcube("data/dummy_data_2023.rtd", reference_height=40)
height = float(ds.height[0])

### Plotting ###

# Plot wind speed for a specific height and customize the plot
ax = ds.wind_graph.plot_variable(height=height, variable='Wind Speed (m/s)')
ax.set = ax.set_title(f'Wind Speed at {height}m')
ax.tick_params(axis='x',rotation=45)
plt.show() 

# Plot wind rose without averaging
ax = ds.wind_graph.plot_wind_rose(height)
plt.show()

# Plot wind rose with averaging
ax = ds.wind_graph.plot_wind_rose(height, averaging_window='1h', colormap='coolwarm')
ax.set_title("Wind Rose at 40m", fontsize=16)
plt.show()

# Plot wind rose for 40 meters, filtering for summer (DJF)
ax = ds.wind_graph.plot_wind_rose(height, colormap='coolwarm', period='DJF')
plt.show()

## Tables ###

df_wind_distribution = ds.wind_table.generate_wind_distribution_table(height)
print(df_wind_distribution)

df_wind_distribution = ds.wind_table.generate_wind_distribution_table(height, mode='accumulate')
print('\n\n', df_wind_distribution)

ax, df_data_coverage = ds.wind_table.generate_data_coverage_table(height, plot=True)
print('\n\n', df_data_coverage)

ax, df_average_wind_speed = ds.wind_table.generate_average_wind_speed_table(height, plot=True)
print('\n\n', df_average_wind_speed)

# Compute the maximum wind change for a specific height
# Use a list of original .rtd files as the dummy data contains only hourly data
files = sorted(glob("./Data/WLS*.rtd"))
ds_202408 = WindDataAccessor.windcube(files, reference_height=40)
ax, def_max_wind_change = ds_202408.wind_table.generate_maximum_wind_change_table(height, plot=True)
print('\n\n', def_max_wind_change)