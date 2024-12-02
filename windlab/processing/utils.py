# windlab/processing/utils.py

import pandas as pd
import xarray as xr

def get_wind_df(dataset: xr.Dataset, height: int) -> pd.DataFrame:
    """
    Retorna um DataFrame contendo velocidade e direção do vento para uma altura especificada.

    Parameters:
    -----------
    dataset : xr.Dataset
        O dataset contendo os dados de vento.
    height : int
        A altura para a qual se deseja obter os dados do vento.

    Returns:
    --------
    pandas.DataFrame
        Um DataFrame com as colunas 'Wind Speed (m/s)' e 'Wind Direction (°)' para a altura especificada.
    
    Example:
    --------
    >>> df = get_wind_df(ds, 40)
    >>> print(df.head())
    """
    wind_speed = dataset.sel(height=height)['Wind Speed (m/s)']
    wind_direction = dataset.sel(height=height)['Wind Direction (°)']

    # Criar um DataFrame usando o tempo como índice
    wind_df = pd.DataFrame({
        'Wind Speed (m/s)': wind_speed.values,
        'Wind Direction (°)': wind_direction.values
    }, index=wind_speed['time'].values)

    wind_df.index.name = 'Time'
    
    return wind_df

def compute_max_wind_direction_change(dataset: xr.Dataset, second_window=10):
        """
        Calculate the maximum change in wind direction and the mean wind speed over a specified rolling time window.

        This function retrieves wind direction and speed data for a specified height and computes the maximum change
        in wind direction, as well as the mean wind speed, over a given time window. The values are calculated
        using a rolling window that moves over the data with a specified time duration.

        Parameters:
        -----------
        dataset : xr.Dataset
            The xarray.Dataset containing the wind data.
        second_window : int, optional
            The time window (in seconds) over which to calculate the maximum change in wind direction and
            the mean wind speed. Default is 10 seconds.

        Returns:
        --------
        DataFrame
            A DataFrame with the following columns:
            - 'Wind Direction (°)': Original wind direction data (in degrees).
            - 'Wind Speed (m/s)': Original wind speed data (in meters per second).
            - '{second_window}s Max Direction Change (°)': Maximum change in wind direction over the given time window (in degrees).
            - '{second_window}s Mean Speed (m/s)': Mean wind speed over the given time window (in meters per second).
        
        Notes:
        ------
        - The rolling window uses the specified `second_window` duration to compute statistics for each point.
        - The maximum change in wind direction is calculated as the difference between the maximum and minimum wind direction
        values within the rolling window.
        - The mean wind speed is computed over the same rolling window.
        - The rolling window is inclusive of the current timestamp and moves forward in time.
        - Missing data (NaN values) are dropped prior to applying the rolling calculations to ensure accurate results.

        Example:
        --------
        >>> df_result = compute_max_wind_direction_change(height=100, second_window=10)
        >>> print(df_result.head())
        
        This would return a DataFrame containing the original wind direction and speed data, along with the calculated
        maximum direction change and mean speed for each rolling window of 10 seconds.
        """
        # Retrieve wind direction data for the specified height
        wind_direction = dataset['Wind Direction (°)']
        wind_speed = dataset['Wind Speed (m/s)']

        # Convert to pandas DataFrames
        df_wind_direction = wind_direction.to_dataframe().drop('height', axis=1)
        df_wind_speed = wind_speed.to_dataframe().drop('height', axis=1)
        df = pd.concat([df_wind_direction, df_wind_speed], axis=1)
        df = df.dropna()

        # Compute rolling max and min for the time window specified
        rolling_window = f'{second_window}s'
        
        # Use rolling with a fixed window size determined by the seconds
        max_direction = df['Wind Direction (°)'].rolling(rolling_window, min_periods=1).max()
        min_direction = df['Wind Direction (°)'].rolling(rolling_window, min_periods=1).min()
        mean_speed = df['Wind Speed (m/s)'].rolling(rolling_window, min_periods=1).mean()

        # Calculate the maximum change as the difference between rolling max and min
        df[f'{second_window}s Max Direction Change (°)'] = max_direction - min_direction
        df[f'{second_window}s Mean Speed (m/s)'] = mean_speed

        return df