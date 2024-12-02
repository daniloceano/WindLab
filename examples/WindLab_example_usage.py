import matplotlib.pyplot as plt
from windlab import WindDataAccessor

# Create an instance of the accessor and load data from the file
ds = WindDataAccessor.windcube(file_path='data/dummy_data_2023.rtd')

# Access the xarray.Dataset directly
dataset = ds.dataset
print(dataset)

# # Compute the detrended standard deviation for wind speed at 40m
# std_detrended_ws40m = ds_accessor.compute_std_detrended_data(40, 'Wind Speed (m/s)')
# plt.plot(dataset.time, std_detrended_ws40m), plt.title('Detrended Wind Speed std at 40m'), plt.xlabel('Time'), plt.ylabel('Wind Speed (m/s)'), plt.show()

# # Plot wind speed
# ds_accessor.plot_variable(40, 'Wind Speed (m/s)')
# plt.show()

# # Subset the dataset some time steps and plot
# subset_accessor = ds_accessor.isel(time=slice(0, 50))
# subset_accessor.plot_variable(40, 'Wind Speed (m/s)')

# # Subset the dataset for a specific time slice and plot
# subset_accessor = ds_accessor.sel(time=slice('2024-08-01T00:00:00', '2024-08-01T02:00:00'))
# subset_accessor.plot_variable(40, 'Wind Speed (m/s)')

# # Get a pandas DataFrame with wind speed and direction for a specified height
# wind_df = ds_accessor.get_wind_df(40)
# print(wind_df)

# # Plotar a rosa dos ventos para 140 metros no mês de janeiro
# ax = ds_accessor.plot_wind_rose(40, colormap='coolwarm', period='January')
# plt.show()

# # Plot wind rose without averaging
# ds_accessor.plot_wind_rose(40)
# plt.show()

# # Plot wind rose with averaging
# ax = ds_accessor.plot_wind_rose(40, averaging_window='1h', colormap='coolwarm')
# ax.set_title("Wind Rose at 40m", fontsize=16)
# plt.show()

# # Plot wind rose for 40 meters, filtering for summer (DJF)
# ax = ds_accessor.plot_wind_rose(40, colormap='coolwarm', period='DJF')
# plt.show()

# wind_distribution_table = ds_accessor.generate_wind_distribution_table(40)
# print(wind_distribution_table)

# wind_distribution_january = ds_accessor.generate_wind_distribution_table(40, period='January', mode='bins')
# print(wind_distribution_january)

# coverage_table = data_coverage_table = ds_accessor.generate_data_coverage_table(40, plot=True)
# print(coverage_table)
# plt.show()

# average_wind_speed_table = ds_accessor.generate_average_wind_speed_table(40, plot=True)
# print(average_wind_speed_table)
# plt.show()