import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from windrose import WindroseAxes

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

    def load_data(self, reference_height: int = 0):
        """
        Loads wind data from a .rtd file and converts it into an xarray.Dataset.
        The data is stored in the '_obj' attribute and becomes accessible through the class methods.

        This method reads the file using pandas, processes it into a structured DataFrame, and then converts
        it into an xarray.Dataset for easier data manipulation.

        Parameters:
        -----------
        reference_height : int, optional
            The reference height above sea level in meters to be added to the data heights (default is 0).
        
        Raises:
        -------
        FileNotFoundError: If the specified file path does not exist.
        """
        df = pd.read_csv(self._path, encoding='unicode_escape', skiprows=range(41), sep='\t')
        self._obj = self.create_xarray_dataset(df, reference_height)

    def create_xarray_dataset(self, df: pd.DataFrame, reference_height: int = 0):
        """
        Converts a pandas DataFrame containing wind speed and direction data into an xarray.Dataset with height and time as dimensions.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing wind data (speed, direction, etc.) for multiple heights.
        reference_height : int, optional
            The reference height above sea level in meters to be added to the heights in the data (default is 0).

        Returns:
        --------
        xarray.Dataset
            A structured dataset containing wind data variables across different heights and times.
        """
        # Automatically detect heights by extracting the numeric part from the column names
        height_columns = [col for col in df.columns if ' Wind Speed (m/s)' in col]
        
        # Extract heights and add the reference height
        raw_heights = sorted(set(int(col.split('m')[0]) for col in height_columns))
        heights = [height + reference_height for height in raw_heights]  # Adjust heights with reference height

        variables = ['Wind Speed (m/s)', 'Wind Direction (°)', 'X-wind (m/s)', 'Y-wind (m/s)', 'Z-wind (m/s)']

        data_vars = {}

        for var in variables:
            # Stack height as a dimension and create DataArrays
            height_data = np.stack([df[f'{height}m {var}'].values for height in raw_heights if f'{height}m {var}' in df.columns])
            data_vars[var] = (['height', 'time'], height_data)

        # Create the xarray dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                'time': pd.to_datetime(df['Timestamp']),
                'height': heights  # Adjusted heights
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
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(data['time'], data)
        ax.set_title(f"{variable} at {height}m")
        ax.set_xlabel('Time')
        ax.set_ylabel(variable)
        return ax

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

    def plot_wind_rose(self, height, averaging_window=None, colormap='viridis'):
        """
        Plot a wind rose using wind speed and direction data, with an option to average the data over a specified time window.
        
        Parameters:
        -----------
        height : int
            The height at which to plot the wind rose.
        
        averaging_window : str, optional
            A resampling rule to average the data over a specified time window (e.g., '1H' for 1 hour).
            Default is None, meaning no averaging will be performed.

        colormap : str, optional
            The colormap to use for the wind rose plot. Default is 'viridis'.
        
        Returns:
        --------
        ax : WindroseAxes
            The WindroseAxes instance used for the plot, allowing the user to modify or save the plot.
        
        Example usage:
        --------------
        ax = ds_accessor.plot_wind_rose(40, averaging_window='1H', colormap='coolwarm')
        ax.set_title("Modified Title")  # Example of modifying the plot after it is created
        ax.figure.savefig('windrose_plot.png')  # Example of saving the figure
        """
        # Retrieve wind speed and direction for the specified height
        wind_speed = self.get_variable(height, 'Wind Speed (m/s)')
        wind_direction = self.get_variable(height, 'Wind Direction (°)')

        # Combine wind speed and direction into a DataFrame
        wind_df = pd.DataFrame({
            'Wind Speed (m/s)': wind_speed.values,
            'Wind Direction (°)': wind_direction.values
        }, index=wind_speed['time'].values)

        # If averaging_window is provided, perform resampling and averaging
        if averaging_window:
            wind_df = wind_df.resample(averaging_window).mean()

        # Convert colormap string to a colormap object
        cmap = plt.get_cmap(colormap)

        # Create the wind rose plot
        ax = WindroseAxes.from_ax()
        ax.bar(wind_df['Wind Direction (°)'], wind_df['Wind Speed (m/s)'], normed=True, opening=0.8, edgecolor='white', cmap=cmap)
        
        # Place the legend outside the plot area
        ax.legend()

        # Return the WindroseAxes instance
        return ax
    
    def generate_wind_distribution_table(self, height, speed_thresholds=None, direction_bins=None):
        """
        Generate a cumulative wind distribution table where each row represents the cumulative frequency 
        for wind speeds below a given threshold, and each column represents wind directions ±15° around a central value.

        Parameters:
        -----------
        height : int
            The height at which to calculate the wind distribution.

        speed_thresholds : list, optional
            List of wind speed thresholds (in m/s). Default is [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32].

        direction_bins : list, optional
            List of direction bin edges (in degrees). Default is [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360].

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the wind distribution table, where rows are cumulative wind speed thresholds, 
            columns are wind direction bins (covering ±15°), and values are percentages of occurrence (formatted to 2 decimal places).
        """

        # Default thresholds if none provided
        if speed_thresholds is None:
            speed_thresholds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        if direction_bins is None:
            direction_bins = np.arange(0, 361, 30)  # Centered on 0°, 30°, 60°, etc.

        # Retrieve wind speed and direction for the specified height
        wind_speed = self.get_variable(height, 'Wind Speed (m/s)').values
        wind_direction = self.get_variable(height, 'Wind Direction (°)').values

        # Create a DataFrame with wind speed and direction
        wind_df = pd.DataFrame({
            'Wind Speed (m/s)': wind_speed,
            'Wind Direction (°)': wind_direction
        })

        # Bin wind directions (±15° around the center)
        wind_df['Direction Bin'] = pd.cut(wind_df['Wind Direction (°)'], bins=np.arange(-15, 375, 30), right=False, labels=direction_bins[:-1])

        # Create an empty DataFrame to hold the cumulative wind distribution
        wind_distribution = pd.DataFrame(index=speed_thresholds, columns=direction_bins[:-1])

        # Calculate cumulative frequencies for each threshold
        for threshold in speed_thresholds:
            # Select all data points with wind speed less than or equal to the current threshold
            wind_subset = wind_df[wind_df['Wind Speed (m/s)'] <= threshold]

            # Calculate the frequency distribution for this subset
            direction_distribution = pd.crosstab(wind_subset['Direction Bin'], columns='Frequency', normalize='all') * 100

            # Add the values to the table
            for direction in direction_bins[:-1]:
                wind_distribution.loc[threshold, direction] = direction_distribution.loc[direction, 'Frequency'] if direction in direction_distribution.index else 0

        # Add a column for the "Omni" (all directions) distribution
        wind_distribution['Omni'] = wind_distribution.sum(axis=1)

        # Add total, mean, and maximum rows
        wind_distribution.loc['Total'] = wind_distribution.max(axis=0)

        # Calculate the mean for each column, excluding the "Omni" column
        wind_distribution.loc['Mean'] = wind_distribution.iloc[:, :-1].mean(axis=0)
        wind_distribution.loc['Mean', 'Omni'] = wind_distribution.loc['Mean'].mean()

        # Calculate the maximum for each column, and set the Omni column to the maximum of the maximum values
        wind_distribution.loc['Maximum'] = wind_df.groupby('Direction Bin', observed=False)['Wind Speed (m/s)'].max()
        wind_distribution.loc['Maximum', 'Omni'] = wind_distribution.loc['Maximum'].max()


        # Format the table to have only 2 decimal places
        wind_distribution = wind_distribution.map(lambda x: f'{x:.2f}' if isinstance(x, (float, int)) else x)

        # Rename the index and columns
        wind_distribution.index = [f'< {int(i)}' for i in wind_distribution.index[:-3]] + ['Total', 'Mean', 'Maximum']

        return wind_distribution

if __name__ == "__main__":
    # Example usage
    # Create an instance of the accessor and load data from the file
    ds_accessor = ReadWindCubeAccessor("./dummy_data_2023.rtd")
    ds_accessor.load_data()

    # Access the xarray.Dataset directly
    dataset = ds_accessor.dataset
    print(dataset)

    # # Compute the detrended standard deviation for wind speed at 40m
    # std_detrended_ws40m = ds_accessor.compute_std_detrended_data(40, 'Wind Speed (m/s)')
    # plt.plot(dataset.time, std_detrended_ws40m), plt.title('Detrended Wind Speed std at 40m'), plt.xlabel('Time'), plt.ylabel('Wind Speed (m/s)'), plt.show()

    # # Plot wind speed
    ds_accessor.plot_variable(40, 'Wind Speed (m/s)')

    # # Subset the dataset some time steps and plot
    # subset_accessor = ds_accessor.isel(time=slice(0, 50))
    # subset_accessor.plot_variable(40, 'Wind Speed (m/s)')

    # # Subset the dataset for a specific time slice and plot
    # subset_accessor = ds_accessor.sel(time=slice('2024-08-01T00:00:00', '2024-08-01T02:00:00'))
    # subset_accessor.plot_variable(40, 'Wind Speed (m/s)')

    # # Get a pandas DataFrame with wind speed and direction for a specified height
    # wind_df = ds_accessor.get_wind_df(40)
    # print(wind_df)

    # # Plot wind rose without averaging
    # ds_accessor.plot_wind_rose(40)
    # plt.show()
 
    # # Plot wind rose with averaging
    # ax = ds_accessor.plot_wind_rose(40, averaging_window='1h', colormap='coolwarm')
    # ax.set_title("Wind Rose at 40m", fontsize=16)
    # plt.show()

    wind_distribution_table = ds_accessor.generate_wind_distribution_table(40)
    print(wind_distribution_table)

