# windlab/processing — Processing modules

This directory contains the analysis and output-generation modules of WindLab.

---

## Modules

### `graphs.py` — Visualisations (`wind_graph`)

Registered as an xarray accessor: `ds.wind_graph.*`

| Method | Description |
|---|---|
| `plot_variable(height, variable)` | Time series of any variable at a given height. |
| `plot_wind_rose(height, averaging_window, colormap, period)` | Wind rose with optional resampling and month/season filter. |

**Dependencies**: `matplotlib`, `windrose`, `utils.compute_max_wind_direction_change`

---

### `tables.py` — Statistical tables (`wind_table`)

Registered as an xarray accessor: `ds.wind_table.*`

| Method | Key parameters | Output |
|---|---|---|
| `generate_wind_distribution_table(height, mode, period)` | `mode='bins'` or `'accumulate'`; `period` for seasonal filtering | `pd.DataFrame` — speed × direction frequency |
| `generate_data_coverage_table(height, frequency, plot)` | `frequency='D'` for daily | `pd.DataFrame` — coverage % by month × day |
| `generate_average_wind_speed_table(height, plot)` | — | `pd.DataFrame` — mean speed by hour × month/season |
| `generate_maximum_wind_change_table(height, second_window, plot)` | `second_window=10` (seconds) | `pd.DataFrame` — direction-change × speed frequency |

**Dependencies**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `utils.compute_max_wind_direction_change`

---

### `utils.py` — Helper functions

Functions reused by `graphs.py` and `tables.py`.

| Function | Description |
|---|---|
| `get_wind_df(dataset, height)` | Returns a `pd.DataFrame` with speed and direction for one height. |
| `compute_std_detrended_data(dataset, window_size)` | Rolling-window standard deviation after detrending. |
| `compute_max_wind_direction_change(dataset, second_window, n_jobs)` | Maximum direction change in a time window (uses `joblib` for parallelism). |

**Dependencies**: `pandas`, `numpy`, `xarray`, `joblib`

---

## xarray accessor pattern

Both `WindGraphGenerator` and `WindTableProcessor` are registered as xarray accessors via the `@xr.register_dataset_accessor` decorator. This enables fluent usage:

```python
# Instead of:
WindGraphGenerator(ds).plot_wind_rose(height=140)

# Use:
ds.wind_graph.plot_wind_rose(height=140)
ds.wind_table.generate_wind_distribution_table(height=140)
```

The input Dataset must have:
- Dimension `height` (float, metres)
- Dimension `time` (datetime64)
- Variable `Wind Speed (m/s)` (required by all methods)
- Variable `Wind Direction (°)` (required by most methods)

---

## Seasonal filters

The `plot_wind_rose` and `generate_wind_distribution_table` methods accept a `period` parameter for temporal filtering:

| Value | Description |
|---|---|
| `None` (default) | Entire available period |
| `'January'`, `'February'`, … | Specific month (full English name) |
| `'DJF'` | Meteorological summer (Dec–Jan–Feb) |
| `'MAM'` | Meteorological autumn (Mar–Apr–May) |
| `'JJA'` | Meteorological winter (Jun–Jul–Aug) |
| `'SON'` | Meteorological spring (Sep–Oct–Nov) |

---

## Extension points

To add a new analysis method:

1. Add the method to `WindGraphGenerator` (in `graphs.py`) or `WindTableProcessor` (in `tables.py`).
2. Use `get_wind_df()` from `utils.py` to extract data for a specific height.
3. Document parameters and return value in the docstring, following the existing pattern.
4. Expose it in `windlab/__init__.py` as a standalone function if needed.
