import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Custom xarray accessor for wind data operations
@xr.register_dataset_accessor("wind_cube")
class WindOperationsAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj  # This is the xarray Dataset

    def get_variable(self, height, variable='Wind Speed (m/s)'):
        """
        Get the specified variable (e.g., wind speed, direction) for the given height.
        
        Parameters:
        height (int): The height for which to get the variable.
        variable (str): The variable to retrieve (default is 'Wind Speed (m/s)').
        
        Returns:
        xarray.DataArray: Data for the given height and variable.
        """
        try:
            return self._obj.sel(height=height)[variable]
        except KeyError:
            raise KeyError(f"Variable '{variable}' not found for height {height}.")
    
    def compute_std_detrended_data(self, data, window_size=600):
        """
        Compute the rolling standard deviation over a specific window, handling NaNs by filling them.
        
        Parameters:
        - data: xarray DataArray of wind speeds
        - window_size: the size of the window (in number of time steps) over which to compute the rolling std.
        
        Returns:
        - rolling_std: the rolling standard deviation for each time step
        """
        # Compute the rolling mean on the filled data
        rolling_mean = data.rolling(time=window_size, min_periods=1, center=True).mean()

        # Subtract the rolling mean to detrend the data
        detrended_data = data - rolling_mean

        # Compute the rolling standard deviation
        rolling_std = detrended_data.rolling(time=window_size, min_periods=1, center=True).std()
        
        return rolling_std


    def plot_variable(self, height, variable='Wind Speed (m/s)'):
        """
        Plot the specified variable for a given height.
        
        Parameters:
        height (int): The height for which to plot the variable.
        variable (str): The variable to plot (default is 'Wind Speed (m/s)').
        """
        data = self.get_variable(height, variable=variable)
        data.plot.line(x="time")
        plt.title(f"{variable} at {height}m")
        plt.xlabel('Time')
        plt.ylabel(variable)
        plt.show()

# Transform the dataset to xarray.Dataset with height as a dimension
def create_xarray_dataset(df):
    """
    Create an xarray.Dataset from the DataFrame for wind speed data at multiple heights.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with wind speed data.
    
    Returns:
    xarray.Dataset: An xarray Dataset containing wind speed, wind direction, and other variables for different heights.
    """
    # Automatically detect heights by extracting the numeric part from the column names
    height_columns = [col for col in df.columns if ' Wind Speed (m/s)' in col]
    heights = sorted(set(int(col.split('m')[0]) for col in height_columns))

    variables = ['Wind Speed (m/s)', 'Wind Direction (°)', 'X-wind (m/s)', 'Y-wind (m/s)', 'Z-wind (m/s)']

    data_vars = {}

    for var in variables:
        # Stack height as a dimension and create DataArrays
        height_data = np.stack([df[f'{height}m {var}'].values for height in heights if f'{height}m {var}' in df.columns])
        data_vars[var] = (['height', 'time'], height_data)

    # Create the xarray dataset
    ds = xr.Dataset(
        data_vars,
        coords={
            'time': pd.to_datetime(df['Timestamp']),
            'height': heights
        }
    )
    
    return ds


if __name__ == "__main__":
    # Example usage
    file_path = "./Data/WLS866-104_2024_08_01__00_00_00.rtd"
    df = pd.read_csv(file_path, encoding='unicode_escape', skiprows=range(41), sep='\t')

    # Create xarray dataset with height as a dimension
    ds = create_xarray_dataset(df)

    # Now we can use the wind_cube accessor methods
    std_detrended_ws40m = ds.wind_cube.compute_std_detrended_data(
        ds.wind_cube.get_variable(40, 'Wind Speed (m/s)')
    )
    std_detrended_ws60m = ds.wind_cube.compute_std_detrended_data(
        ds.wind_cube.get_variable(60, 'Wind Speed (m/s)'))
    std_detrended_ws100m = ds.wind_cube.compute_std_detrended_data(
        ds.wind_cube.get_variable(100, 'Wind Speed (m/s)'))

    # Print standard deviation of detrended data
    print(f"40m: {std_detrended_ws40m.values}, 60m: {std_detrended_ws60m.values}, 100m: {std_detrended_ws100m.values}")

    # Plot wind speed at different heights
    ds.wind_cube.plot_variable(40, 'Wind Speed (m/s)')
    ds.wind_cube.plot_variable(60, 'Wind Speed (m/s)')
    ds.wind_cube.plot_variable(100, 'Wind Speed (m/s)')