import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
from windrose import WindroseAxes

class WindGraphGenerator:

    def __init__(self, dataset: xr.Dataset):
        self.dataset = dataset

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
    
    def plot_max_wind_change_mean_speed(df, second_window=10):
        """
        Plot the maximum wind direction change as a function of the mean wind speed.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame returned by the compute_max_wind_direction_change function.
        second_window : int, optional
            The second window size to use for resampling the data (default is 10).

        Returns:
        --------
        ax : matplotlib.axes.Axes
            The matplotlib axes object containing the plot.
        """
        ax = df.plot(x=f'{second_window}s Mean Speed (m/s)', y=f'{second_window}s Max Direction Change (°)', kind='scatter', figsize=(12, 8))
        ax.set_xlabel('Mean Wind Speed (m/s)')
        ax.set_ylabel('Maximum Wind Direction Change (°)')
        ax.set_title(f'{second_window}s Maximum Wind Direction Change vs. Mean Wind Speed')
        return ax