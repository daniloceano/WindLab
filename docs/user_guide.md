# User Guide — WindLab

**Version**: 0.2  
**Project**: TC No. 050.0125966.23.9 — PETROBRAS/CENPES & USP/IAG

---

## Table of contents

1. [Installation](#1-installation)
2. [Data input modes](#2-data-input-modes)
3. [Using the graphical interface (GUI)](#3-using-the-graphical-interface-gui)
4. [Using the Python API](#4-using-the-python-api)
5. [WindCube preset](#5-windcube-preset)
6. [Custom mode (CSV / Excel)](#6-custom-mode-csv--excel)
7. [Column mapping — examples](#7-column-mapping--examples)
8. [Available outputs](#8-available-outputs)
9. [Common errors and solutions](#9-common-errors-and-solutions)
10. [Table and figure generation flow](#10-table-and-figure-generation-flow)

---

## 1. Installation

### Prerequisites

- Python 3.10 or higher
- pip (or conda — recommended)

### Steps

```bash
# Clone the repository
git clone https://codigo-externo.petrobras.com.br/tc_usp_iag_renewables/readwindcube.git
cd windlab

# Create the conda environment and install in editable mode (recommended)
bash setup_env.sh

# Or install manually
pip install -e .
```

### Verify installation

```python
import windlab
print(windlab.__version__)  # should print the installed version
```

---

## 2. Data input modes

WindLab supports three input modes:

| Preset | Format | Status |
|---|---|---|
| **WindCube** | `.rtd` files from the WindCube LIDAR (Leosphere) | ✅ Available |
| **Custom** | CSV or Excel with column mapping | ✅ Available |
| **Zephyr** | Zephyr LIDAR files | 🔜 Coming soon |

In both available cases the result is an `xarray.Dataset` with the same dimensions and variables, compatible with all package analysis methods.

---

## 3. Using the graphical interface (GUI)

### Start the interface

After installing the package:

```bash
windlab-gui
```

Or directly via Python:

```bash
python -m windlab.gui.app
```

### Interface flow

#### Step 1 — Select file

Click **"Browse…"** and choose the input file:
- For WindCube: `.rtd` files
- For Custom mode: `.csv`, `.xlsx`, or `.xls` files

#### Step 2 — Choose preset

Select the preset matching the data source:
- **WindCube LIDAR (.rtd)**: no additional configuration.
- **Custom (CSV / Excel)**: the column-mapping panel will appear.

#### Step 3 — (Custom mode only) Map columns

Fill in the combos in the **"Column mapping"** panel:

| Field | Required | Description |
|---|---|---|
| Time column | Yes | Column with timestamps (any format recognised by pandas) |
| Wind speed | Yes | Speed column in m/s |
| Wind direction | Yes | Direction column in degrees (0–360°) |
| X-wind, Y-wind, Z-wind | No | Cartesian wind components (select "— None —" to omit) |
| Measurement height (m) | Yes | Sensor height above ground or mast base |

WindLab attempts to auto-detect common column names (e.g. `timestamp`, `ws`, `wd`). Always review the auto-filled values.

#### Step 4 — Load file

Click **"Load file"**.

- In WindCube mode: the file is read and the dataset is created immediately.
- In Custom mode: the columns are read and the combos are populated. The dataset will be created when you click "Process".

The status log will show the detected heights and the data period.

#### Step 5 — Configure outputs

- **Reference height (m)**: height of the mast base above sea level. Added to the measurement height to obtain the true altitude.
- **Analysis height**: select the height of interest from the combo (automatically populated after loading).
- **Outputs to generate**: tick the boxes for the desired outputs.
- **Figure format / DPI**: choose the image format (PNG, PDF, SVG) and resolution.
- **Output folder**: set where the files will be saved.

#### Step 6 — Process

Click **"Process and generate outputs"**. Progress is shown in the status log. A confirmation message is displayed when finished.

---

## 4. Using the Python API

### Imports

```python
from windlab import WindDataAccessor
from windlab.ingestion import CustomDataReader
```

### Complete example — WindCube

```python
from windlab import WindDataAccessor
import matplotlib.pyplot as plt

# Load data
ds = WindDataAccessor.windcube("data.rtd", reference_height=40)

# Inspect
print(ds)
print("Heights:", ds.height.values)
print("Period:", ds.time.values[[0, -1]])

# Wind rose
ax = ds.wind_graph.plot_wind_rose(height=140, colormap="coolwarm")
ax.set_title("Wind Rose — 140 m")
ax.figure.savefig("wind_rose_140m.png", dpi=150, bbox_inches="tight")

# Distribution table
df = ds.wind_table.generate_wind_distribution_table(height=140, mode="bins")
df.to_csv("distribution_140m.csv")

# Data coverage
df_cov = ds.wind_table.generate_data_coverage_table(height=140, plot=True)
```

### Complete example — Custom mode

```python
from windlab.ingestion import CustomDataReader

reader = CustomDataReader()

# Step 1: read file
columns = reader.read_file("measurements.csv")
print("Columns:", columns)

# Step 2: map columns
reader.map_columns(
    column_map={
        "time":              "DateTime",
        "Wind Speed (m/s)":  "Speed_100m",
        "Wind Direction (°)":"Dir_100m",
        "X-wind (m/s)":      "— None —",
        "Y-wind (m/s)":      "— None —",
        "Z-wind (m/s)":      "— None —",
    },
    height_value=100,
)

# Step 3: validate
warnings = reader.validate()
for w in warnings:
    print(f"[WARNING] {w}")

# Step 4: convert to Dataset
ds = reader.to_dataset(reference_height=0)

# Use normally with the accessors
ax = ds.wind_graph.plot_wind_rose(height=100)
```

---

## 5. WindCube preset

The WindCube reader (`WindDataAccessor.windcube`) reads `.rtd` files produced by the WindCube LIDAR (Leosphere) software.

### Format characteristics

- 41 header lines (automatically skipped).
- Column separator: tab (`\t`).
- Encoding: `unicode_escape`.
- Column names in the format: `{height}m {Variable} ({unit})`, e.g. `100m Wind Speed (m/s)`.
- Time column: `Timestamp`.

### Parameters

```python
ds = WindDataAccessor.windcube(
    file_path,           # str or list[str] — .rtd file(s)
    reference_height=0,  # int — base altitude in metres
)
```

### Multiple files

```python
from glob import glob
files = sorted(glob("testdata/2024/*.rtd"))
ds = WindDataAccessor.windcube(files, reference_height=40)
```

Files that fail to load are skipped with a log warning.

---

## 6. Custom mode (CSV / Excel)

### Supported formats

| Extension | Library | Notes |
|---|---|---|
| `.csv` | `pandas.read_csv` | Separator detected automatically |
| `.xlsx` | `pandas.read_excel` (openpyxl) | Reads the first sheet |
| `.xls` | `pandas.read_excel` | Reads the first sheet |

### Minimum file requirements

- A column with timestamps recognisable by pandas (e.g. `"2024-01-01 00:00:00"`).
- A column with wind speed in **m/s** (numeric values).
- A column with wind direction in **degrees** (0–360°).
- Data from a single measurement height per file.

### Pipeline steps

```python
reader = CustomDataReader()
columns = reader.read_file("file.csv")       # read
reader.map_columns(column_map, height_value) # map
warnings = reader.validate()                 # validate
ds = reader.to_dataset(reference_height)     # normalise
```

### Validations performed

| Check | Action on failure |
|---|---|
| Empty file | Exception (`ValueError`) |
| All timestamps invalid | Exception (`ValueError`) |
| Required column 100% NaN | Exception (`ValueError`) |
| Partially invalid timestamps | Warning, rows removed |
| Negative speed or > 100 m/s | Warning |
| Direction outside [0°, 360°] | Warning |
| Missing values in required columns | Warning with percentage |

---

## 7. Column mapping — examples

### Example 1: simple CSV file

File columns: `Date`, `Time`, `WS_100`, `WD_100`

```python
# Combine date and time into a datetime column first (if needed)
import pandas as pd
df = pd.read_csv("file.csv")
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df.to_csv("file_processed.csv", index=False)

reader = CustomDataReader()
reader.read_file("file_processed.csv")
reader.map_columns({
    "time":              "datetime",
    "Wind Speed (m/s)":  "WS_100",
    "Wind Direction (°)":"WD_100",
    "X-wind (m/s)":      "— None —",
    "Y-wind (m/s)":      "— None —",
    "Z-wind (m/s)":      "— None —",
}, height_value=100)
```

### Example 2: Excel spreadsheet with Cartesian components

Columns: `Timestamp`, `WS`, `WD`, `U`, `V`, `W`

```python
reader = CustomDataReader()
reader.read_file("measurements.xlsx")
reader.map_columns({
    "time":              "Timestamp",
    "Wind Speed (m/s)":  "WS",
    "Wind Direction (°)":"WD",
    "X-wind (m/s)":      "U",
    "Y-wind (m/s)":      "V",
    "Z-wind (m/s)":      "W",
}, height_value=80)
```

### Example 3: file with auto-detected column names

Columns: `timestamp`, `wind_speed`, `wind_direction`

WindLab automatically detects these columns by common names. Simply review the mapping before processing.

---

## 8. Available outputs

All outputs are generated from an `xarray.Dataset` and saved to the configured folder.

| Output | Generated file | Method |
|---|---|---|
| Wind distribution | `wind_distribution_{h}m.csv` | `wind_table.generate_wind_distribution_table` |
| Data coverage | `data_coverage_{h}m.csv` | `wind_table.generate_data_coverage_table` |
| Mean speed | `average_wind_speed_{h}m.csv` | `wind_table.generate_average_wind_speed_table` |
| Wind rose | `wind_rose_{h}m.png` | `wind_graph.plot_wind_rose` |
| Time series | `time_series_{h}m.png` | `wind_graph.plot_variable` |

where `{h}` is the selected analysis height.

### Customisation via API

All methods return `pd.DataFrame` or `matplotlib.Axes` objects, allowing customisation before saving:

```python
# Customise title and save wind rose for the summer season
ax = ds.wind_graph.plot_wind_rose(height=140, period="DJF", colormap="plasma")
ax.set_title("Wind Rose — 140 m — Summer (DJF)", fontsize=14)
ax.figure.savefig("wind_rose_summer_140m.png", dpi=200, bbox_inches="tight")
```

---

## 9. Common errors and solutions

### `ValueError: Unsupported file format`

**Cause**: file extension not recognised by Custom mode.  
**Solution**: use `.csv`, `.xlsx`, or `.xls` files. For other formats, pre-convert with `pandas`.

---

### `KeyError: Column 'X' not found in file`

**Cause**: the mapping points to a column that does not exist in the file.  
**Solution**: click "Load file" first to detect the available columns, then fill in the mapping.

---

### `ValueError: Required column 'Wind Speed (m/s)' contains only null values`

**Cause**: the column mapped to wind speed could not be converted to numeric values.  
**Solution**: check that the column contains numeric values (no text such as "N/A" or extra headers). Use `pd.read_csv()` manually to inspect.

---

### `RuntimeError: Call read_file() before map_columns()`

**Cause**: attempt to map columns without having loaded the file first.  
**Solution**: click "Load file" before "Process".

---

### `ModuleNotFoundError: openpyxl`

**Cause**: dependency for reading `.xlsx` files not installed.  
**Solution**:
```bash
pip install openpyxl
```

---

### Wind rose not generated / empty figure

**Cause**: insufficient data period after seasonal filter, or too many NaN values.  
**Solution**: check data coverage with `generate_data_coverage_table` and consider using `period=None` for the full period.

---

### Heights do not appear in the combo after loading file

**Cause** (Custom mode): the dataset is only created when "Process" is clicked.  
**Solution**: click "Process and generate outputs". The height combo will be updated automatically.

---

## 10. Table and figure generation flow

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT FILE                              │
│         .rtd (WindCube)   or   .csv/.xlsx (Custom)          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │   Reading & validation  │
          │  WindDataAccessor  or   │
          │   CustomDataReader      │
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │     xarray.Dataset      │
          │  dims: time × height    │
          │  vars: Wind Speed, Dir  │
          └────────────┬────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
  ┌──────────────┐         ┌──────────────────┐
  │  wind_graph  │         │   wind_table     │
  ├──────────────┤         ├──────────────────┤
  │ Wind rose    │         │ Wind distribution│
  │              │         │                  │
  │ Time series  │         │ Data coverage    │
  └──────┬───────┘         │                  │
         │                 │ Mean speed       │
         ▼                 │                  │
  .png / .pdf / .svg       │ Max. direction   │
                           │ change           │
                           └────────┬─────────┘
                                    │
                                    ▼
                                  .csv
```

### Recommended sequence for a technical report

1. **Load data** (WindCube or Custom).
2. **Check coverage** with `generate_data_coverage_table(plot=True)` — identify gaps.
3. **Wind distribution table** for each height and period of interest.
4. **Wind rose** for the full period and by season.
5. **Wind speed time series**.
6. **Mean speed table** by hour × month.
7. (If high-frequency data) **Maximum direction change table**.
