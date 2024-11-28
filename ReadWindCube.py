import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import seaborn as sns

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

    def plot_wind_rose(self, height, averaging_window=None, colormap='viridis', period=None):
        """
        Plot a wind rose using wind speed and direction data, with an option to average the data over a specified time window,
        and filter by a specific month or season.

        Parameters:
        -----------
        height : int
            The height at which to plot the wind rose.

        averaging_window : str, optional
            A resampling rule to average the data over a specified time window (e.g., '1H' for 1 hour).
            Default is None, meaning no averaging will be performed.

        colormap : str, optional
            The colormap to use for the wind rose plot. Default is 'viridis'.
        
        period : str, optional
            The specific period to filter by. Can be a month ('January', 'February', etc.) or a season ('DJF', 'JJA', etc.).
            If None, the distribution will be calculated for the entire dataset.

        Returns:
        --------
        ax : WindroseAxes
            The WindroseAxes instance used for the plot, allowing the user to modify or save the plot.

        Example usage:
        --------------
        ax = ds_accessor.plot_wind_rose(40, averaging_window='1H', colormap='coolwarm', period='DJF')
        ax.set_title("Modified Title")  # Example of modifying the plot after it is created
        ax.figure.savefig('windrose_plot.png')  # Example of saving the figure
        """
        # Retrieve wind speed and direction for the specified height
        wind_speed = self.get_variable(height, 'Wind Speed (m/s)')
        wind_direction = self.get_variable(height, 'Wind Direction (°)')
        time = wind_speed['time']

        # Filter data based on the period (month or season)
        if period:
            # Define seasonal periods for meteorological seasons
            seasons = {
                'DJF': [12, 1, 2],
                'MAM': [3, 4, 5],
                'JJA': [6, 7, 8],
                'SON': [9, 10, 11]
            }

            if period in seasons:
                # Filter by season
                wind_speed = wind_speed.sel(time=wind_speed['time.month'].isin(seasons[period]))
                wind_direction = wind_direction.sel(time=wind_direction['time.month'].isin(seasons[period]))
            else:
                # Filter by specific month
                month_num = pd.to_datetime(period, format='%B').month
                wind_speed = wind_speed.sel(time=wind_speed['time.month'] == month_num)
                wind_direction = wind_direction.sel(time=wind_direction['time.month'] == month_num)

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
    
    def generate_wind_distribution_table(self, height, speed_thresholds=None, direction_bins=None, period=None, mode='accumulate'):
        """
        Generate a cumulative or binned wind distribution table.
        
        Parameters:
        -----------
        height : int
            The height at which to calculate the wind distribution.

        speed_thresholds : list, optional
            List of wind speed thresholds (in m/s). Default is [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32].

        direction_bins : list, optional
            List of direction bin edges (in degrees). Default is [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360].

        period : str, optional
            The specific period to filter by. Can be a month ('January', 'February', etc.) or a season ('DJF', 'JJA', etc.).
            If None, the distribution will be calculated for the entire dataset.

        mode : str, optional
            The mode of calculation. Either 'accumulate' for cumulative probabilities or 'bins' for binned probabilities. Default is 'accumulate'.

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the wind distribution table, where rows represent wind speed thresholds or bins, 
            columns are wind direction bins (covering ±15°), and values are percentages of occurrence (formatted to 2 decimal places).
        """
        
        # Default thresholds if none provided
        if speed_thresholds is None and mode == 'accumulate':
            speed_thresholds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        elif speed_thresholds is None and mode == 'bins':
            speed_thresholds = np.arange(0, 33, 1)  # Default bins from 0 to 32 m/s with 1 m/s width
        
        if direction_bins is None:
            direction_bins = np.arange(0, 361, 30)  # Centered on 0°, 30°, 60°, etc.

        # Retrieve wind speed and direction for the specified height
        wind_speed = self.get_variable(height, 'Wind Speed (m/s)')
        wind_direction = self.get_variable(height, 'Wind Direction (°)')
        
        # Filter data based on the period (month or season)
        if period:
            # Define seasonal periods for meteorological seasons
            seasons = {
                'DJF': [12, 1, 2],
                'MAM': [3, 4, 5],
                'JJA': [6, 7, 8],
                'SON': [9, 10, 11]
            }

            if period in seasons:
                # Filter by season
                wind_speed = wind_speed.sel(time=wind_speed['time.month'].isin(seasons[period]))
                wind_direction = wind_direction.sel(time=wind_direction['time.month'].isin(seasons[period]))
            else:
                # Filter by specific month
                month_num = pd.to_datetime(period, format='%B').month
                wind_speed = wind_speed.sel(time=wind_speed['time.month'] == month_num)
                wind_direction = wind_direction.sel(time=wind_direction['time.month'] == month_num)

        # Create a DataFrame with wind speed and direction
        wind_df = pd.DataFrame({
            'Wind Speed (m/s)': wind_speed.values,
            'Wind Direction (°)': wind_direction.values
        })

        # Bin wind directions (±15° around the center)
        wind_df['Direction Bin'] = pd.cut(wind_df['Wind Direction (°)'], bins=np.arange(-15, 375, 30), right=False, labels=direction_bins[:-1])

        if mode == 'accumulate':
            # Create an empty DataFrame to hold the wind distribution
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

        elif mode == 'bins':
            # Create an empty DataFrame to hold the wind distribution
            wind_distribution = pd.DataFrame(index=[f'{start}-{end}' for start, end in zip(speed_thresholds[:-1], speed_thresholds[1:])], columns=direction_bins[:-1])

            # Calculate binned frequencies for each bin range
            for i in range(len(speed_thresholds) - 1):
                # Select all data points within the current bin range
                bin_start = speed_thresholds[i]
                bin_end = speed_thresholds[i + 1]
                wind_subset = wind_df[(wind_df['Wind Speed (m/s)'] >= bin_start) & (wind_df['Wind Speed (m/s)'] < bin_end)]

                # Calculate the frequency distribution for this subset
                direction_distribution = pd.crosstab(wind_subset['Direction Bin'], columns='Frequency', normalize='all') * 100

                # Add the values to the table
                for direction in direction_bins[:-1]:
                    wind_distribution.loc[f'{bin_start}-{bin_end}', direction] = direction_distribution.loc[direction, 'Frequency'] if direction in direction_distribution.index else 0

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
        if mode == 'accumulate':
            wind_distribution.index = [f'< {int(i)}' for i in wind_distribution.index[:-3]] + ['Total', 'Mean', 'Maximum']
        elif mode == 'bins':
            wind_distribution.index = list(wind_distribution.index[:-3]) + ['Total', 'Mean', 'Maximum']

        # Add a multi-level header to the columns to include wind direction labels
        direction_labels = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW", "Omni"]
        wind_distribution.columns = pd.MultiIndex.from_tuples(
            [(label, f"{deg}°") if label != "Omni" else ("Omni", "") for label, deg in zip(direction_labels, list(direction_bins[:-1]) + [None])],
            names=["Direction", "Degrees"]
        )

        return wind_distribution

    def generate_data_coverage_table(self, height, frequency='D', plot=False):
        """
        Generate a data coverage table to check the amount of missing data and optionally plot it.
        
        Parameters:
        -----------
        height : int
            The height at which to calculate the data coverage.

        frequency : str, optional
            The frequency for resampling the data. Default is 'D' for daily frequency.

        plot : bool, optional
            Whether to plot the data coverage table. Default is False.

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the data coverage table, where rows represent months and columns represent days.
        """
        # Retrieve wind speed data for the specified height
        wind_speed = self.get_variable(height, 'Wind Speed (m/s)')

        # Resample the dataset to the specified frequency
        resampled_data = wind_speed.resample(time=frequency).count()

        # Create a DataFrame to hold the coverage information
        coverage_df = resampled_data.to_dataframe().reset_index()
        coverage_df['month'] = coverage_df['time'].dt.to_period('M')
        coverage_df['day'] = coverage_df['time'].dt.day

        # Pivot the table to create a month x day structure
        coverage_table = coverage_df.pivot_table(index='month', columns='day', values='Wind Speed (m/s)', aggfunc='sum')

        # Calculate the percentage of data coverage for each cell
        max_count_per_day = resampled_data.max()
        coverage_table = (coverage_table / float(max_count_per_day)) * 100

        if plot:
            # Plot the coverage table
            plt.figure(figsize=(17, 10))
            sns.heatmap(coverage_table, cmap='RdYlBu', linewidths=0.5, annot=True, fmt=".0f", cbar_kws={'label': 'Cobertura [%]'}, linecolor='black')
            plt.title(f'Cobertura de Dados - Altura: {height} m')
            plt.xlabel('Dia do Mês')
            plt.ylabel('Mês e Ano')

        return coverage_table

    def generate_average_wind_speed_table(self, height, plot=False):
        """
        Generate a table with the average wind speed for each hour of the day, separated by month, and grouped by seasons and globally.
        Optionally, plot the table as a heatmap.
        
        Parameters:
        -----------
        height : int
            The height at which to calculate the average wind speed.

        plot : bool, optional
            Whether to plot the average wind speed table. Default is False.

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the average wind speed table, where rows represent hours of the day, columns represent months, 
            seasons, and a global average, and values are the average wind speed (formatted to 2 decimal places).
        """
        # Retrieve wind speed data for the specified height
        wind_speed = self.get_variable(height, 'Wind Speed (m/s)')

        # Group by hour and month to calculate the average wind speed
        wind_speed_df = wind_speed.to_dataframe().reset_index()
        wind_speed_df['hour'] = wind_speed_df['time'].dt.hour
        wind_speed_df['month'] = wind_speed_df['time'].dt.month

        # Pivot the table to create an hour x month structure
        average_speed_table = wind_speed_df.pivot_table(index='hour', columns='month', values='Wind Speed (m/s)', aggfunc='mean')

        # Add columns for seasonal and global averages
        seasons = {
            'Verão': [12, 1, 2],
            'Outono': [3, 4, 5],
            'Inverno': [6, 7, 8],
            'Primavera': [9, 10, 11]
        }
        for season, months in seasons.items():
            average_speed_table[season] = average_speed_table[months].mean(axis=1)
        average_speed_table['Global'] = average_speed_table.mean(axis=1)

        # Add a row for the monthly average
        monthly_average = average_speed_table.mean(axis=0)
        average_speed_table.loc['Global'] = monthly_average

        # Format the table to have only 2 decimal places
        average_speed_table = average_speed_table.map(lambda x: f'{x:.2f}' if pd.notnull(x) else x)

        if plot:
            # Plot the average wind speed table
            plt.figure(figsize=(18, 10))
            sns.heatmap(average_speed_table.astype(float), cmap='RdYlGn_r', linewidths=0.5, annot=True, fmt=".2f", cbar_kws={'label': 'V/Vmax'}, linecolor='black')
            plt.title(f'Velocidade Média do Vento - Altura: {height} m')
            plt.xlabel('Mês/Estação')
            plt.ylabel('Hora do Dia')

        return average_speed_table

if __name__ == "__main__":
    # Example usage
    # Create an instance of the accessor and load data from the file
    ds_accessor = ReadWindCubeAccessor("./dummy_data_2023.rtd")
    ds_accessor.load_data()

    # Access the xarray.Dataset directly
    dataset = ds_accessor.dataset
    print(dataset)
