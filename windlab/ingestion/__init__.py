"""
windlab.ingestion
-----------------
WindLab data ingestion module.

Responsible for:
- Defining supported format presets (WindCube, Custom, Zephyr).
- Providing CSV and Excel file reading with configurable column mapping.
- Normalising data into the internal xarray.Dataset format, compatible
  with the existing processing pipeline (graphs, tables, utils).

Main exports:
    PRESET_NAMES       – human-readable dict of available presets.
    AVAILABLE_PRESETS  – list of active preset IDs.
    CustomDataReader   – reader for CSV / Excel files with column mapping.
"""

from .presets import PRESET_NAMES, AVAILABLE_PRESETS
from .custom_reader import CustomDataReader

__all__ = [
    "PRESET_NAMES",
    "AVAILABLE_PRESETS",
    "CustomDataReader",
]
