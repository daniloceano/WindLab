# windlab/processing/utils.py

import pandas as pd
import xarray as xr
import numpy as np
from itertools import combinations
from joblib import Parallel, delayed

def get_wind_df(dataset: xr.Dataset, height: int) -> pd.DataFrame:
    """
    Returns a DataFrame containing wind speed and direction for a specified height.

    Parameters:
    -----------
    dataset : xr.Dataset
        The dataset containing wind data.
    height : int
        The height for which to retrieve wind data.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with columns 'Wind Speed (m/s)' and 'Wind Direction (°)' for the given height.
    
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

def compute_std_detrended_data(dataset: xr.DataArray, window_size: int=600):
        """
        Computes the rolling standard deviation for the specified variable at a given height, after detrending the data.

        Parameters:
        -----------
        dataset : xarray.DataArray
            The xarray.DataArray containing data.
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
        # Compute the rolling mean on the filled data
        rolling_mean = dataset.rolling(time=window_size, min_periods=1, center=True).mean()

        # Subtract the rolling mean to detrend the data
        detrended_data = dataset - rolling_mean

        # Compute the rolling standard deviation
        rolling_std = detrended_data.rolling(time=window_size, min_periods=1, center=True).std()

        return rolling_std

def compute_max_wind_direction_change(dataset: xr.Dataset, second_window=10, n_jobs=-1):
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
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is -1, which means using all processors.

    Returns:
    --------
    DataFrame
        A DataFrame with the following columns:
        - 'Wind Direction (°)': Original wind direction data (in degrees).
        - 'Wind Speed (m/s)': Original wind speed data (in meters per second).
        - '{second_window}s Max Direction Change (°)': Maximum change in wind direction over the given time window (in degrees).
        - '{second_window}s Mean Speed (m/s)': Mean wind speed over the given time window (in meters per second).
        - '{second_window}s Mean Direction Change (°)': Mean change in wind direction over the given time window (in degrees).
    
    Notes:
    ------
    - The rolling window uses the specified second_window duration to compute statistics for each point.
    - The maximum change in wind direction is calculated as the difference between the maximum and minimum wind direction
      values within the rolling window.
    - The mean wind speed is computed over the same rolling window.
    - The rolling window is inclusive of the current timestamp and moves forward in time.
    - Missing data (NaN values) are dropped prior to applying the rolling calculations to ensure accurate results.

    Example:
    --------
    >>> df_result = compute_max_wind_direction_change(dataset, second_window=10)
    >>> print(df_result.head())
    
    This would return a DataFrame containing the original wind direction and speed data, along with the calculated
    maximum direction change and mean speed for each rolling window of 10 seconds.
    """
    # Retrieve wind direction data for the specified height
    wind_direction = dataset['Wind Direction (°)']
    wind_speed = dataset['Wind Speed (m/s)']

    # Convert to pandas DataFrames
    df_wind_direction = wind_direction.to_dataframe()
    
    # Check if 'height' column is present and drop it
    if 'height' in df_wind_direction.columns:
        df_wind_direction = df_wind_direction.drop('height', axis=1)

    df_wind_speed = wind_speed.to_dataframe()
    
    if 'height' in df_wind_speed.columns:
        df_wind_speed = df_wind_speed.drop('height', axis=1)

    df = pd.concat([df_wind_direction, df_wind_speed], axis=1)
    df = df.dropna()

    # Converting wind direction from degrees to radians
    df['Wind Direction (rad)'] = df['Wind Direction (°)'].apply(np.deg2rad)

    # Function to calculate the maximum change in wind direction within the window
    def min_scalar_product(window):
        if len(window) < 2:
            return 1  # Maximum scalar product for aligned vectors is 1, meaning no change if only one value is present
        return min(
            np.cos(a) * np.cos(b) + np.sin(a) * np.sin(b)
            for a, b in combinations(window, 2)
        )

    # Applying the rolling window function to calculate the minimum scalar product using parallel processing
    min_scalar_results = Parallel(n_jobs=n_jobs)(
        delayed(min_scalar_product)(df['Wind Direction (rad)'].iloc[i:i+second_window])
        for i in range(len(df) - second_window + 1)
    )
    df['Min Scalar Product'] = pd.Series(min_scalar_results, index=df.index[second_window - 1:])

    # Calculating the maximum change in wind direction in degrees
    df['Max Change in Direction (°)'] = df['Min Scalar Product'].apply(lambda x: np.rad2deg(np.arccos(x)))

    # Compute maximum and minimum wind speed within the window
    df['Max Wind Speed (m/s)'] = df['Wind Speed (m/s)'].rolling(window=second_window, min_periods=1).max()
    df['Min Wind Speed (m/s)'] = df['Wind Speed (m/s)'].rolling(window=second_window, min_periods=1).min()

    return df