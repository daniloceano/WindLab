import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class WindTableProcessor:
    def __init__(self, dataset: xr.Dataset):
        self.dataset = dataset

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
    
    def generate_maximum_wind_change_table(df, height, second_window=10):
        """
        Generate a frequency table of maximum changes in wind direction and the corresponding mean wind speed over a specified rolling time window.

        This function generates a table representing the frequency of occurrences of specific wind speed ranges (rows)
        and wind direction change ranges (columns). The purpose of the table is to analyze how often certain levels of
        wind direction change correspond to particular wind speed averages within the given time window.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame returned by the compute_max_wind_direction_change function.
        second_window : int, optional
            The time window (in seconds) over which to calculate the maximum change in wind direction and
            the mean wind speed. Default is 10 seconds.

        Returns:
        --------
        DataFrame
            A DataFrame representing the frequency counts for the different combinations of mean wind speed intervals
            (rows) and direction change intervals (columns).
        
        Notes:
        ------
        - The table is created with wind speed intervals as rows and direction change intervals as columns.
        - The function bins the data into predefined speed and direction change intervals and populates a table
        with the frequency of occurrences.
        """
        # Define bins for wind speed and direction change
        speed_thresholds = np.arange(0, 21, 1)
        direction_bins = np.arange(0, 36, 5)

        # DataFrame with speed intervals on rows and direction change intervals on columns
        max_wind_change_table = pd.DataFrame(
            index=[f'{start}-{end}' for start, end in zip(speed_thresholds[:-1], speed_thresholds[1:])],
            columns=[f'{start}-{end}' for start, end in zip(direction_bins[:-1], direction_bins[1:])]
        )
        
        # Add row for values larger than the last threshold
        max_wind_change_table.loc[f'{speed_thresholds[-1]}+'] = 0

        # Add column for values larger than the last threshold
        max_wind_change_table[f'{direction_bins[-1]}+'] = 0

        # Calculate binned frequencies for each bin range
        for i in range(len(speed_thresholds) - 1):
            # Select all data points within the current speed bin range
            bin_start = speed_thresholds[i]
            bin_end = speed_thresholds[i + 1]
            wind_speed_subset = df[(df[f'{second_window}s Mean Speed (m/s)'] >= bin_start) & (df[f'{second_window}s Mean Speed (m/s)'] < bin_end)]
            
            for j in range(len(direction_bins) - 1):
                # Select all data points within the current direction change bin range
                direction_start = direction_bins[j]
                direction_end = direction_bins[j + 1]
                count = wind_speed_subset[(wind_speed_subset[f'{second_window}s Max Direction Change (°)'] >= direction_start) & 
                                        (wind_speed_subset[f'{second_window}s Max Direction Change (°)'] < direction_end)].shape[0]
                max_wind_change_table.iloc[i, j] = count

        # Populate the last row and column for values greater than the final thresholds
        wind_speed_above = df[df[f'{second_window}s Mean Speed (m/s)'] >= speed_thresholds[-1]]
        for j in range(len(direction_bins) - 1):
            direction_start = direction_bins[j]
            direction_end = direction_bins[j + 1]
            count = wind_speed_above[(wind_speed_above[f'{second_window}s Max Direction Change (°)'] >= direction_start) &
                                    (wind_speed_above[f'{second_window}s Max Direction Change (°)'] < direction_end)].shape[0]
            max_wind_change_table.iloc[-1, j] = count

        direction_above = df[df[f'{second_window}s Max Direction Change (°)'] >= direction_bins[-1]]
        for i in range(len(speed_thresholds) - 1):
            bin_start = speed_thresholds[i]
            bin_end = speed_thresholds[i + 1]
            count = direction_above[(direction_above[f'{second_window}s Mean Speed (m/s)'] >= bin_start) &
                                    (direction_above[f'{second_window}s Mean Speed (m/s)'] < bin_end)].shape[0]
            max_wind_change_table.iloc[i, -1] = count

        # Populate the last cell (bottom-right) for values greater than both thresholds
        count = direction_above[direction_above[f'{second_window}s Mean Speed (m/s)'] >= speed_thresholds[-1]].shape[0]
        max_wind_change_table.iloc[-1, -1] = count

        # Add "total" column and row
        max_wind_change_table.loc['total'] = max_wind_change_table.sum(axis=0)
        max_wind_change_table['total'] = max_wind_change_table.sum(axis=1)

        # Add "percentage" column and row
        max_wind_change_table.loc['percentage'] = (max_wind_change_table.loc['total'] / max_wind_change_table['total'][ 'total']) * 100
        max_wind_change_table['percentage'] = (max_wind_change_table['total'] / max_wind_change_table['total'][ 'total']) * 100
        max_wind_change_table.loc['percentage', 'percentage'] = ''

        return max_wind_change_table