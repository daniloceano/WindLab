import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Custom xarray accessor for wind data operations
@xr.register_dataset_accessor("wind_cube")
class ReadWindCubeAccessor:
    """
    This class provides a custom xarray accessor for performing various operations on wind data stored in an xarray.Dataset.
    It supports loading wind data from a .rtd file, subsetting the dataset (using `isel` and `sel` methods), and performing
    calculations such as detrended rolling standard deviations and plotting wind speed variables.

    Attributes:
    -----------
    _path : str
        The path to the .rtd file containing the wind data.
    _obj : xarray.Dataset
        The underlying xarray.Dataset containing the wind data.

    Methods:
    --------
    load_data()
        Loads wind data from a .rtd file and transforms it into an xarray.Dataset.
    
    create_xarray_dataset(df: pd.DataFrame) -> xr.Dataset
        Converts a pandas DataFrame containing wind speed, direction, and other variables into an xarray.Dataset.
    
    dataset() -> xr.Dataset
        Returns the underlying xarray.Dataset object for direct access to the data.
    
    isel(**indexers) -> 'ReadWindCubeAccessor'
        Subsets the dataset using xarray's isel method (for selecting by position/index) and returns a new instance of ReadWindCubeAccessor.
    
    sel(**indexers) -> 'ReadWindCubeAccessor'
        Subsets the dataset using xarray's sel method (for selecting by coordinate labels) and returns a new instance of ReadWindCubeAccessor.
    
    get_variable(height: int, variable: str) -> xr.DataArray
        Retrieves a specific variable (e.g., wind speed) for a given height from the dataset.
    
    compute_std_detrended_data(height: int, variable: str = 'Wind Speed (m/s)', window_size: int = 600) -> xr.DataArray
        Computes the rolling standard deviation of the specified variable for the given height, using a detrending process.
    
    plot_variable(height: int, variable: str = 'Wind Speed (m/s)')
        Plots the specified variable (e.g., wind speed) for the given height.
    
    get_wind_df(height: int) -> pd.DataFrame
        Returns a pandas DataFrame with two columns ('Wind Speed (m/s)' and 'Wind Direction (°)') for the specified height.
    """
    def __init__(self, file_path=None, dataset=None):
        self._path = file_path
        self._obj = dataset  # Allow passing an existing xarray.Dataset

    def load_data(self):
        """
        Loads wind data from a .rtd file and converts it into an xarray.Dataset.
        The data is stored in the '_obj' attribute and becomes accessible through the class methods.

        This method reads the file using pandas, processes it into a structured DataFrame, and then converts
        it into an xarray.Dataset for easier data manipulation.

        Raises:
        -------
        FileNotFoundError: If the specified file path does not exist.
        """
        df = pd.read_csv(self._path, encoding='unicode_escape', skiprows=range(41), sep='\t')
        self._obj = self.create_xarray_dataset(df)

    def create_xarray_dataset(self, df: pd.DataFrame):
        """
        Converts a pandas DataFrame containing wind speed and direction data into an xarray.Dataset with height and time as dimensions.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing wind data (speed, direction, etc.) for multiple heights.

        Returns:
        --------
        xarray.Dataset
            A structured dataset containing wind data variables across different heights and times.
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

    @property
    def dataset(self):
        """
        Provides direct access to the underlying xarray.Dataset.

        Returns:
        --------
        xarray.Dataset
            The dataset containing wind data across different heights and times.
        
        Example:
        --------
        >>> ds_accessor = ReadWindCubeAccessor(<path_to_file>)
        >>> ds_accessor.load_data()
        >>> dataset = ds_accessor.dataset
        >>> print(dataset)
        """
        return self._obj

    def isel(self, **indexers):
        """
        Subsets the dataset using xarray's isel method, which selects by index/position, and returns a new instance of ReadWindCubeAccessor.

        Parameters:
        -----------
        indexers : dict
            The indexers to select along the dataset's dimensions (e.g., time=slice(0, 10)).

        Returns:
        --------
        ReadWindCubeAccessor
            A new instance of ReadWindCubeAccessor with the subsetted dataset.
        
        Example:
        --------
        >>> subset_accessor = ds_accessor.isel(time=slice(0, 10))
        >>> subset_accessor.compute_std_detrended_data(40)
        """
        subsetted_ds = self._obj.isel(**indexers)
        return ReadWindCubeAccessor(dataset=subsetted_ds)
    
    def sel(self, **indexers):
        """
        Subsets the dataset using xarray's sel method, which selects by coordinate labels, and returns a new instance of ReadWindCubeAccessor.

        Parameters:
        -----------
        indexers : dict
            The indexers to select along the dataset's dimensions (e.g., time='2024-08-01').

        Returns:
        --------
        ReadWindCubeAccessor
            A new instance of ReadWindCubeAccessor with the subsetted dataset.
        
        Example:
        --------
        >>> subset_accessor = ds_accessor.sel(time='2024-08-01')
        >>> subset_accessor.plot_variable(40)
        """
        subsetted_ds = self._obj.sel(**indexers)
        return ReadWindCubeAccessor(dataset=subsetted_ds)

    def get_variable(self, height: int, variable: str='Wind Speed (m/s)'):
        """
        Retrieves the specified variable (e.g., wind speed or direction) for a given height.

        Parameters:
        -----------
        height : int
            The height at which to retrieve the variable.
        variable : str, optional
            The name of the variable to retrieve (default is 'Wind Speed (m/s)').

        Returns:
        --------
        xarray.DataArray
            A DataArray containing the values of the specified variable for the given height.
        
        Raises:
        -------
        KeyError
            If the specified variable is not found for the given height.
        """
        try:
            return self._obj.sel(height=height)[variable]
        except KeyError:
            raise KeyError(f"Variable '{variable}' not found for height {height}.")

    def compute_std_detrended_data(self, height: int, variable: str='Wind Speed (m/s)', window_size: int=600):
        """
        Computes the rolling standard deviation for the specified variable at a given height, after detrending the data.

        Parameters:
        -----------
        height : int
            The height at which to compute the rolling standard deviation.
        variable : str, optional
            The variable to compute the standard deviation for (default is 'Wind Speed (m/s)').
        window_size : int, optional
            The size of the rolling window in time steps (default is 600).

        Returns:
        --------
        xarray.DataArray
            The rolling standard deviation for the specified variable and height.
        """
        # Get the variable for the given height
        data = self.get_variable(height, variable)

        # Compute the rolling mean on the filled data
        rolling_mean = data.rolling(time=window_size, min_periods=1, center=True).mean()

        # Subtract the rolling mean to detrend the data
        detrended_data = data - rolling_mean

        # Compute the rolling standard deviation
        rolling_std = detrended_data.rolling(time=window_size, min_periods=1, center=True).std()

        return rolling_std

    def plot_variable(self, height: int, variable: str='Wind Speed (m/s)'):
        """
        Plots the specified variable (e.g., wind speed) for the given height.

        Parameters:
        -----------
        height : int
            The height at which to plot the variable.
        variable : str, optional
            The variable to plot (default is 'Wind Speed (m/s)').

        Example:
        --------
        >>> ds_accessor.plot_variable(40, 'Wind Speed (m/s)')
        """
        data = self.get_variable(height, variable=variable)
        data.plot.line(x="time")
        plt.title(f"{variable} at {height}m")
        plt.xlabel('Time')
        plt.ylabel(variable)
        plt.show()

    def get_wind_df(self, height: int) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing wind speed and direction for a specified height.

        Parameters:
        -----------
        height : int
            The height for which to retrieve wind data.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with columns 'Wind Speed (m/s)' and 'Wind Direction (°)' for the given height.
        
        Example:
        --------
        >>> df = ds_accessor.get_wind_df(40)
        >>> print(df.head())
        """
        wind_speed = self.get_variable(height, variable='Wind Speed (m/s)')
        wind_direction = self.get_variable(height, variable='Wind Direction (°)')

        # Create a DataFrame using time as the index
        wind_df = pd.DataFrame({
            'Wind Speed (m/s)': wind_speed.values,
            'Wind Direction (°)': wind_direction.values
        }, index=wind_speed['time'].values)

        wind_df.index.name = 'Time'
        
        return wind_df

if __name__ == "__main__":
    # Example usage
    # Create an instance of the accessor and load data from the file
    ds_accessor = ReadWindCubeAccessor("./Data/WLS866-104_2024_08_01__00_00_00.rtd")
    ds_accessor.load_data()

    # Access the xarray.Dataset directly
    dataset = ds_accessor.dataset
    print(dataset)

    # Compute the detrended standard deviation for wind speed at 40m
    std_detrended_ws40m = ds_accessor.compute_std_detrended_data(40, 'Wind Speed (m/s)')
    plt.plot(dataset.time, std_detrended_ws40m), plt.title('Detrended Wind Speed std at 40m'), plt.xlabel('Time'), plt.ylabel('Wind Speed (m/s)'), plt.show()

    # Plot wind speed
    ds_accessor.plot_variable(40, 'Wind Speed (m/s)')

    # Subset the dataset some time steps and plot
    subset_accessor = ds_accessor.isel(time=slice(0, 50))
    subset_accessor.plot_variable(40, 'Wind Speed (m/s)')

    # Subset the dataset for a specific time slice and plot
    subset_accessor = ds_accessor.sel(time=slice('2024-08-01T00:00:00', '2024-08-01T02:00:00'))
    subset_accessor.plot_variable(40, 'Wind Speed (m/s)')

    # Get a pandas DataFrame with wind speed and direction for a specified height
    wind_df = ds_accessor.get_wind_df(40)
    print(wind_df)

