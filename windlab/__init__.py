from .wind_data_reader import WindDataAccessor
from .processing.graphs import WindGraphGenerator
from .processing.tables import WindTableProcessor
from .processing.utils import compute_max_wind_direction_change, get_wind_df
from .ingestion import CustomDataReader, PRESET_NAMES, AVAILABLE_PRESETS

__version__ = "0.2.0"

__all__ = [
    # Data readers
    "WindDataAccessor",
    "CustomDataReader",
    # Processing (xarray accessors)
    "WindGraphGenerator",
    "WindTableProcessor",
    # Utilities
    "compute_max_wind_direction_change",
    "get_wind_df",
    # Presets
    "PRESET_NAMES",
    "AVAILABLE_PRESETS",
]