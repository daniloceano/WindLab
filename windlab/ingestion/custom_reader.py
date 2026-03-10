"""
windlab/ingestion/custom_reader.py
-----------------------------------
Reader for CSV and Excel files provided by the user.

The workflow is split into four independent steps to facilitate
validation, testing, and integration with the GUI:

    1. read_file(file_path)
       Loads the raw file and returns the detected column names.

    2. map_columns(column_map, height_value)
       Maps file columns to WindLab standard variable names and records
       the measurement height.

    3. validate()
       Runs consistency checks. Returns warnings (list of str) and raises
       exceptions on critical errors.

    4. to_dataset(reference_height)
       Converts the mapped and validated data to an xarray.Dataset compatible
       with the wind_graph and wind_table accessors.

Compatibility:
    The produced Dataset has dimensions (height, time) and the same variables
    used by WindDataAccessor.windcube(), ensuring that all existing processing
    methods work without modification.

Example usage:
    reader = CustomDataReader()
    columns = reader.read_file("data.csv")
    reader.map_columns({
        "time":              "Timestamp",
        "Wind Speed (m/s)":  "WS_100m",
        "Wind Direction (deg)": "WD_100m",
    }, height_value=100)
    warnings = reader.validate()
    ds = reader.to_dataset(reference_height=0)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# -- WindLab standard variables ----------------------------------------------
REQUIRED_VARIABLES: list[str] = ["Wind Speed (m/s)", "Wind Direction (\u00b0)"]
OPTIONAL_VARIABLES: list[str] = ["X-wind (m/s)", "Y-wind (m/s)", "Z-wind (m/s)"]
ALL_VARIABLES: list[str] = REQUIRED_VARIABLES + OPTIONAL_VARIABLES

# Sentinel used in GUI combo boxes to indicate "no mapping for this variable"
NO_COLUMN_SENTINEL = "— None —"


class CustomDataReader:
    """
    Reader for CSV / Excel files with configurable column mapping.

    Attributes:
    -----------
    raw_df : pd.DataFrame or None
        Raw DataFrame loaded from the file. Available after read_file().
    columns : list[str]
        Column names detected in the file. Available after read_file().
    """

    def __init__(self) -> None:
        self._raw_df: pd.DataFrame | None = None
        self._mapped_df: pd.DataFrame | None = None
        self._column_map: dict[str, str] = {}
        self._height_value: float = 0.0

    # -- Public properties ----------------------------------------------------

    @property
    def raw_df(self) -> pd.DataFrame | None:
        return self._raw_df

    @property
    def columns(self) -> list[str]:
        if self._raw_df is None:
            return []
        return list(self._raw_df.columns)

    # -- Step 1: file reading -------------------------------------------------

    def read_file(self, file_path: str) -> list[str]:
        """
        Read a CSV or Excel file and return the detected column names.

        Parameters:
        -----------
        file_path : str
            Path to a .csv, .xlsx, or .xls file.

        Returns:
        --------
        list of str
            Column names present in the file.

        Raises:
        -------
        ValueError
            If the file extension is not supported.
        IOError
            If the file cannot be read.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in (".xlsx", ".xls"):
            self._raw_df = pd.read_excel(file_path)
        elif suffix == ".csv":
            # Attempt to detect the separator automatically
            self._raw_df = pd.read_csv(file_path, sep=None, engine="python")
        else:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                "Please use .csv, .xlsx, or .xls."
            )

        n_rows, n_cols = self._raw_df.shape
        logger.info("File loaded: %s — %d rows, %d columns.", path.name, n_rows, n_cols)
        # Reset state for subsequent steps
        self._mapped_df = None
        self._column_map = {}
        return self.columns

    # -- Step 2: column mapping -----------------------------------------------

    def map_columns(self, column_map: dict[str, str], height_value: float = 0.0) -> None:
        """
        Map raw file columns to WindLab standard variable names.

        Parameters:
        -----------
        column_map : dict
            Mapping from standard variable keys to file column names.
            Required keys: 'time', 'Wind Speed (m/s)', 'Wind Direction (degree)'.
            Optional keys: 'X-wind (m/s)', 'Y-wind (m/s)', 'Z-wind (m/s)'.
            Use NO_COLUMN_SENTINEL to mark an absent optional column.

            Example::

                {
                    "time":               "DateTime",
                    "Wind Speed (m/s)":   "WS_100m",
                    "Wind Direction (deg)": "WD_100m",
                    "X-wind (m/s)":       "— None —",
                    "Y-wind (m/s)":       "— None —",
                    "Z-wind (m/s)":       "— None —",
                }

        height_value : float, optional
            Measurement height in metres (default 0.0).
            Added to reference_height in to_dataset().

        Raises:
        -------
        RuntimeError
            If read_file() has not been called first.
        KeyError
            If a required column named in the mapping does not exist in the file.
        """
        if self._raw_df is None:
            raise RuntimeError("Call read_file() before map_columns().")

        # Validate that all required keys point to existing columns
        required_keys = ["time"] + REQUIRED_VARIABLES
        for key in required_keys:
            col = column_map.get(key, "")
            if not col or col == NO_COLUMN_SENTINEL:
                raise KeyError(
                    f"Required mapping missing for '{key}'. "
                    "Please provide a valid column name."
                )
            if col not in self._raw_df.columns:
                raise KeyError(
                    f"Column '{col}' (mapped to '{key}') not found in the file. "
                    f"Available columns: {self.columns}"
                )

        self._column_map = column_map
        self._height_value = float(height_value)

        # Build the mapped DataFrame
        df = pd.DataFrame()
        df["time"] = pd.to_datetime(self._raw_df[column_map["time"]], errors="coerce")
        for var in ALL_VARIABLES:
            col = column_map.get(var, NO_COLUMN_SENTINEL)
            if col and col != NO_COLUMN_SENTINEL and col in self._raw_df.columns:
                df[var] = pd.to_numeric(self._raw_df[col], errors="coerce")
            else:
                df[var] = np.nan
        self._mapped_df = df
        logger.info("Column mapping applied. Measurement height: %.1f m.", height_value)

    # -- Step 3: validation ---------------------------------------------------

    def validate(self) -> list[str]:
        """
        Validate the mapped data for consistency.

        Returns:
        --------
        list of str
            List of warning messages (empty if no issues found).

        Raises:
        -------
        RuntimeError
            If map_columns() has not been called first.
        ValueError
            On critical errors (empty data, required columns entirely NaN).
        """
        if self._mapped_df is None:
            raise RuntimeError("Call map_columns() before validate().")

        warnings: list[str] = []
        df = self._mapped_df

        # Check for empty DataFrame
        if df.empty:
            raise ValueError("The file contains no data after reading.")

        # Check for invalid timestamps
        n_invalid_time = df["time"].isna().sum()
        if n_invalid_time == len(df):
            raise ValueError(
                "All timestamps are invalid. "
                "Check the time column format and the column mapping."
            )
        if n_invalid_time > 0:
            warnings.append(f"{n_invalid_time} invalid timestamp(s) will be removed.")

        # Check required variables
        for var in REQUIRED_VARIABLES:
            if df[var].isna().all():
                raise ValueError(
                    f"Required column '{var}' contains only null values. "
                    "Check the column mapping."
                )

        # Warn about missing values in required variables
        for var in REQUIRED_VARIABLES:
            n_missing = int(df[var].isna().sum())
            if n_missing > 0:
                pct = 100.0 * n_missing / len(df)
                warnings.append(
                    f"'{var}': {n_missing} missing value(s) ({pct:.1f}% of records)."
                )

        # Check wind speed range
        valid_speeds = df["Wind Speed (m/s)"].dropna()
        if (valid_speeds < 0).any():
            warnings.append("Negative wind speed values detected. Please verify the data.")
        if (valid_speeds > 100).any():
            warnings.append("Wind speed above 100 m/s detected. Verify the unit is m/s.")

        # Check wind direction range
        valid_dirs = df["Wind Direction (\u00b0)"].dropna()
        if ((valid_dirs < 0) | (valid_dirs > 360)).any():
            warnings.append("Wind direction values outside [0°, 360°] detected.")

        logger.info("Validation complete. %d warning(s) generated.", len(warnings))
        return warnings

    # -- Step 4: conversion to xarray.Dataset ---------------------------------

    def to_dataset(self, reference_height: float = 0.0) -> xr.Dataset:
        """
        Convert the mapped and validated data to an xarray.Dataset.

        The produced Dataset is compatible with the wind_graph and wind_table
        accessors — dimensions (height, time) and the same variables as
        WindDataAccessor.windcube().

        Parameters:
        -----------
        reference_height : float, optional
            Reference height in metres (e.g., altitude of the mast base).
            Added to the measurement height provided in map_columns().
            Default: 0.

        Returns:
        --------
        xarray.Dataset
            Dataset with dimensions (height, time) and WindLab standard variables.

        Raises:
        -------
        RuntimeError
            If map_columns() has not been called first.
        """
        if self._mapped_df is None:
            raise RuntimeError(
                "Call map_columns() before to_dataset(). "
                "Run validate() to check data consistency."
            )

        df = self._mapped_df.copy()

        # Drop rows with invalid timestamps and sort chronologically
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        height = float(self._height_value) + float(reference_height)

        data_vars: dict = {}
        for var in ALL_VARIABLES:
            values = df[var].values.reshape(1, -1)  # shape: (1 height, N times)
            data_vars[var] = (["height", "time"], values)

        ds = xr.Dataset(
            data_vars,
            coords={"time": df["time"].values, "height": [height]},
        )

        logger.info("Dataset created: %d time points, height=%.1f m.", len(df), height)
        return ds
