# WindLab

**WindLab** is a Python package for manipulating and analysing LIDAR wind data, developed under Cooperation Agreement (TC) No. 050.0125966.23.9 between **PETROBRAS/CENPES** and **USP/IAG**.

The main goal is to automate the generation of tables and figures for use in meteo-oceanographic technical specifications, from LIDAR instrument data (WindCube and others) or from user-supplied files (CSV, Excel).

---

## Features

| Feature | Description |
|---|---|
| **WindCube reader** | Reads `.rtd` files from the WindCube LIDAR (Leosphere), automatically detecting heights and variables. |
| **Custom reader** | Reads CSV or Excel files with manual column mapping to standard variables. |
| **Wind distribution table** | Generates binned or cumulative frequency tables of speed × direction. |
| **Wind rose** | Plots a wind rose for a selected height and period. |
| **Data coverage** | Table and heatmap of temporal data coverage. |
| **Hourly mean speed** | Hour × month table with seasonal and global averages. |
| **Maximum direction change** | Frequency table of direction change × mean speed over a time window. |
| **Graphical interface** | Local tkinter GUI requiring no programming. |

---

## Installation

### Conda environment (recommended)

```bash
# Clone the repository
git clone https://codigo-externo.petrobras.com.br/tc_usp_iag_renewables/readwindcube.git
cd windlab

# Create the 'windlab' environment and install the package in editable mode
bash setup_env.sh

# To install into an existing conda environment (e.g. tc_petrobras):
bash setup_env.sh tc_petrobras
```

The `setup_env.sh` script:
1. Creates (or updates) the conda environment from `environment.yml`.
2. Installs WindLab in editable mode (`pip install -e .`).
3. Verifies the import at the end.

### Manual installation

```bash
conda env create -f environment.yml
conda activate windlab
pip install -e .
```

---

## Quick start

### Via Python API

```python
from windlab import WindDataAccessor

# Load WindCube data
ds = WindDataAccessor.windcube("data.rtd", reference_height=40)

# Generate wind rose
ax = ds.wind_graph.plot_wind_rose(height=140)
ax.figure.savefig("wind_rose.png", dpi=150, bbox_inches="tight")

# Generate wind distribution table
df = ds.wind_table.generate_wind_distribution_table(height=140)
df.to_csv("wind_distribution.csv")
```

### Via Custom mode (CSV / Excel)

```python
from windlab.ingestion import CustomDataReader

reader = CustomDataReader()
reader.read_file("measurements.csv")
reader.map_columns({
    "time":              "Timestamp",
    "Wind Speed (m/s)":  "WS_100m",
    "Wind Direction (°)":"WD_100m",
}, height_value=100)
warnings = reader.validate()
ds = reader.to_dataset(reference_height=0)
# ds is compatible with all wind_graph and wind_table methods
```

### Via graphical interface

```bash
windlab-gui
# or:
python -m windlab.gui.app
```

---

## Repository structure

```
windlab/                        # Main package
│
├── wind_data_reader.py         # WindCube reader (.rtd) → xarray.Dataset
│
├── ingestion/                  # Data ingestion and presets
│   ├── presets.py              # Metadata for supported presets
│   └── custom_reader.py        # CSV/Excel reader with column mapping
│
├── processing/                 # Processing and visualisation
│   ├── graphs.py               # wind_graph accessor (wind rose, time series)
│   ├── tables.py               # wind_table accessor (distribution, coverage, averages)
│   └── utils.py                # Helper functions (direction change, std, etc.)
│
└── gui/                        # Graphical interface (tkinter)
    └── app.py                  # Main WindLabApp window

examples/                       # Example scripts for API usage
docs/                           # Tutorials and user guide
testdata/                       # Test data
```

---

## Usage flow

```
Input file (.rtd / .csv / .xlsx)
         │
         ▼
   Reading and validation
   (WindDataAccessor or CustomDataReader)
         │
         ▼
   xarray.Dataset
   (dimensions: time × height)
         │
         ├──► wind_graph  →  wind rose, time series
         │
         └──► wind_table  →  distribution, coverage, mean speed,
                              direction change
```

---

## Documentation

- **User guide**: [`docs/user_guide.md`](docs/user_guide.md)
- **Code examples**: [`examples/WindLab_example_usage.py`](examples/WindLab_example_usage.py)
- **Tutorial (Jupyter)**: [`docs/WindLab_tutorial.ipynb`](docs/WindLab_tutorial.ipynb)
- **Internal modules**: [`windlab/README.md`](windlab/README.md)

---

## Licence

MIT License. See the `LICENSE` file for details.

## Contact

- **Author**: Danilo Couto de Souza
- **E-mail**: danilo.oceano@gmail.com
- **Institution**: USP/IAG — PETROBRAS/CENPES Partnership (TC No. 050.0125966.23.9)
