from typing import Union
import numpy as np
import pandas as pd
import xarray as xr
import logging

logging.basicConfig(level=logging.WARNING)

@xr.register_dataset_accessor("wind_data")
class WindDataAccessor:
    @classmethod
    def windcube(cls, file_path: Union[str, list], reference_height: int = 0):
        """
        Reads data from a WindCube LIDAR and converts it to an xarray.Dataset.
        
        Parameters:
        -----------
        file_path : str or list
            Path to the WindCube .rtd file or list of .rtd files.
        reference_height : int, optional
            Reference height to be added to the dataset (default is 0).
        
        Returns:
        --------
        xarray.Dataset
            Dataset containing wind data.
        """
        if isinstance(file_path, str):
            # Read a single file
            df = pd.read_csv(file_path, encoding='unicode_escape', skiprows=range(41), sep='\t')

        elif isinstance(file_path, list):
            # Check that the list is not empty
            if not file_path:
                raise ValueError("The file list is empty.")

            # Read multiple files
            dataframes = []
            for file in file_path:
                try:
                    df = pd.read_csv(file, encoding='unicode_escape', skiprows=range(41), sep='\t')
                    dataframes.append(df)
                except Exception as e:
                    logging.warning(f"Failed to read file {file}: {e}")

            # Check that at least one DataFrame was read successfully
            if dataframes:
                df = pd.concat(dataframes)
            else:
                raise ValueError("None of the provided files could be read.")

        else:
            raise TypeError("file_path must be a string or a list of strings.")

        # Build and return the xarray.Dataset from the loaded DataFrame
        return cls.create_xarray_dataset(df, reference_height)


    @staticmethod
    def create_xarray_dataset(df: pd.DataFrame, reference_height: int = 0):
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