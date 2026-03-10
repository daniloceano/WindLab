"""
windlab/ingestion/presets.py
----------------------------
Definition of supported data format presets for WindLab.

A "preset" encapsulates the reading mode and identifier for a data type.
New presets should be added here without modifying the reading logic
in other modules.

Usage:
    from windlab.ingestion.presets import PRESET_NAMES, AVAILABLE_PRESETS

Structure:
    PRESET_NAMES       - dict {id: human_readable_name} for all known presets.
    AVAILABLE_PRESETS  - list of currently functional IDs (excluding placeholders).
    PRESET_FILE_TYPES  - dict {id: list_of_extensions} for file-type filters.
"""

# -- Full preset map ----------------------------------------------------------
# Key: internal identifier. Value: label shown to the user.
PRESET_NAMES: dict[str, str] = {
    "windcube": "WindCube LIDAR (.rtd)",
    "custom": "Custom (CSV / Excel)",
    "zephyr": "Zephyr LIDAR (coming soon)",  # placeholder - not implemented
}

# -- Active presets -----------------------------------------------------------
# Only these IDs are offered as functional options in the interface.
AVAILABLE_PRESETS: list[str] = ["windcube", "custom"]

# -- Accepted file extensions per preset -------------------------------------
PRESET_FILE_TYPES: dict[str, list[str]] = {
    "windcube": [".rtd"],
    "custom": [".csv", ".xlsx", ".xls"],
    "zephyr": [],  # placeholder
}

# -- Short technical descriptions (for tooltips / documentation) -------------
PRESET_DESCRIPTIONS: dict[str, str] = {
    "windcube": (
        "Reads .rtd files from the WindCube LIDAR (Leosphere). "
        "Heights and variables are detected automatically."
    ),
    "custom": (
        "Reads CSV or Excel files provided by the user. "
        "Requires manual column mapping to WindLab standard variables."
    ),
    "zephyr": (
        "Support for Zephyr LIDAR data. "
        "Planned for future versions."
    ),
}
