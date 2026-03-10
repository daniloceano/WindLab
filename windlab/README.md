# windlab — Core package

This directory contains the central modules of WindLab, organised in four functional layers.

---

## Module structure

```
windlab/
├── __init__.py             # Public package exports
├── wind_data_reader.py     # Native WindCube reader (.rtd)
│
├── ingestion/              # Data ingestion and presets
│   ├── __init__.py
│   ├── presets.py          # Metadata for supported formats
│   └── custom_reader.py    # CSV/Excel reader with column mapping
│
├── processing/             # Processing and output generation
│   ├── graphs.py           # xarray accessor wind_graph
│   ├── tables.py           # xarray accessor wind_table
│   └── utils.py            # Shared helper functions
│
└── gui/                    # Local graphical interface
    ├── __init__.py
    └── app.py              # Main WindLabApp window (tkinter)
```

---

## Layers and responsibilities

### 1. Ingestion (`ingestion/`)

Responsible for loading data from different sources and converting them to the internal format.

**Internal format**: `xarray.Dataset` with dimensions `(height, time)` and variables:
- `Wind Speed (m/s)` — required
- `Wind Direction (°)` — required
- `X-wind (m/s)`, `Y-wind (m/s)`, `Z-wind (m/s)` — optional

**`presets.py`**: Defines metadata (`PRESET_NAMES`, `AVAILABLE_PRESETS`, `PRESET_FILE_TYPES`) for each supported format. Contains no reading logic — configuration only.

**`custom_reader.py`**: Reader for CSV/Excel files. Follows an explicit four-step pipeline:

| Step | Method | Description |
|---|---|---|
| 1 | `read_file(path)` | Loads raw file, returns detected columns |
| 2 | `map_columns(map, height)` | Applies user-supplied column mapping |
| 3 | `validate()` | Checks consistency, returns warnings |
| 4 | `to_dataset(ref_height)` | Produces a compatible `xarray.Dataset` |

**`wind_data_reader.py`**: Native reader for WindCube `.rtd` files. Automatically detects heights and variables using column-name patterns.

#### How to add a new preset

1. Add an entry in `presets.py` (`PRESET_NAMES`, `PRESET_FILE_TYPES`, `PRESET_DESCRIPTIONS`).
2. Create a reading module (e.g. `zephyr_reader.py`) that produces an `xarray.Dataset` with the same schema.
3. Add the dispatch logic in the GUI (`gui/app.py`, method `_load_file`).

---

### 2. Processing (`processing/`)

Analysis modules registered as **xarray accessors** — accessed via `ds.wind_graph.*` and `ds.wind_table.*`.

See [`processing/README.md`](processing/README.md) for details.

---

### 3. Graphical interface (`gui/`)

Local tkinter application. Integrates the ingestion and processing modules without duplicating business rules.

**GUI flow**:
```
Select file → Choose preset → [Map columns] →
Load file → Configure height and outputs → Process
```

Processing runs in a background thread to keep the UI responsive. Figures are saved to disk using the `Agg` backend (no additional windows are opened).

**Entry point** (after `pip install -e .`):
```bash
windlab-gui
```

---

## Module dependencies

```
gui/app.py
    ├── wind_data_reader.WindDataAccessor   (windcube preset)
    ├── ingestion.CustomDataReader          (custom preset)
    └── ingestion.presets                   (preset metadata)

processing/tables.py
    └── processing/utils.compute_max_wind_direction_change
```
