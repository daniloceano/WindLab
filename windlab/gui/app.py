"""
windlab/gui/app.py
------------------
Simplified local GUI for WindLab (tkinter).

Workflow:
    1. Select input file.
    2. Choose a preset (WindCube or Custom).
    3. In Custom mode: map file columns to WindLab standard variables.
    4. Click "Load file" to read and validate the data.
    5. Select analysis height, desired outputs, figure settings and output folder.
    6. Click "Process & export" to generate all outputs.

To run:
    python -m windlab.gui.app
    # or, after installing the package:
    windlab-gui
"""

from __future__ import annotations

import logging
import os
import threading
import traceback
from pathlib import Path

# Use non-interactive backend so figures are saved without opening windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from windlab.ingestion.custom_reader import (
    ALL_VARIABLES,
    NO_COLUMN_SENTINEL,
    REQUIRED_VARIABLES,
    CustomDataReader,
)
from windlab.ingestion.presets import AVAILABLE_PRESETS, PRESET_NAMES
from windlab.wind_data_reader import WindDataAccessor

import xarray as xr  # noqa: E402

logger = logging.getLogger(__name__)

# -- Figure output options ---------------------------------------------------
FIGURE_FORMATS: list[str] = ["PNG", "PDF", "SVG"]
DPI_OPTIONS: list[str] = ["72", "100", "150", "200", "300"]
DEFAULT_DPI = "150"
DEFAULT_FMT = "PNG"


# -- Helpers -----------------------------------------------------------------

def _float_or(value: str, default: float) -> float:
    """Convert a string to float; return *default* on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# -- Main window -------------------------------------------------------------

class WindLabApp(tk.Tk):
    """Main window of the WindLab GUI."""

    PAD = {"padx": 8, "pady": 4}

    def __init__(self) -> None:
        super().__init__()
        self.title("WindLab")
        self.resizable(True, True)
        self.minsize(700, 660)

        # Internal state
        self._dataset: xr.Dataset | None = None
        self._custom_reader: CustomDataReader | None = None

        self._build_ui()
        self._on_preset_change()   # set correct initial state

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)

        # -- Input file -------------------------------------------------------
        file_frame = ttk.LabelFrame(self, text="Input file")
        file_frame.grid(row=0, column=0, sticky="ew", **self.PAD)
        file_frame.columnconfigure(0, weight=1)

        self._file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self._file_var).grid(
            row=0, column=0, sticky="ew", **self.PAD
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse_file).grid(
            row=0, column=1, **self.PAD
        )

        # -- Preset -----------------------------------------------------------
        preset_frame = ttk.LabelFrame(self, text="Data type (Preset)")
        preset_frame.grid(row=1, column=0, sticky="ew", **self.PAD)

        self._preset_var = tk.StringVar(value="windcube")
        for i, key in enumerate(AVAILABLE_PRESETS):
            ttk.Radiobutton(
                preset_frame,
                text=PRESET_NAMES[key],
                variable=self._preset_var,
                value=key,
                command=self._on_preset_change,
            ).grid(row=0, column=i, padx=12, pady=4, sticky="w")

        # Zephyr — disabled placeholder
        ttk.Radiobutton(
            preset_frame,
            text=PRESET_NAMES["zephyr"],
            variable=self._preset_var,
            value="zephyr",
            state="disabled",
        ).grid(row=0, column=len(AVAILABLE_PRESETS), padx=12, pady=4, sticky="w")

        # -- Column mapping (Custom mode only) --------------------------------
        self._mapping_frame = ttk.LabelFrame(
            self, text="Column mapping (Custom mode)"
        )
        self._mapping_widgets: dict[str, ttk.Combobox] = {}
        self._meas_height_var = tk.StringVar(value="0")
        self._build_mapping_ui()

        # -- Settings ---------------------------------------------------------
        cfg_frame = ttk.LabelFrame(self, text="Settings")
        cfg_frame.grid(row=3, column=0, sticky="ew", **self.PAD)

        ttk.Label(cfg_frame, text="Reference height (m):").grid(
            row=0, column=0, sticky="e", **self.PAD
        )
        self._ref_height_var = tk.StringVar(value="0")
        ttk.Entry(cfg_frame, textvariable=self._ref_height_var, width=8).grid(
            row=0, column=1, sticky="w", **self.PAD
        )

        ttk.Label(cfg_frame, text="Analysis height:").grid(
            row=0, column=2, sticky="e", **self.PAD
        )
        self._analysis_height_var = tk.StringVar()
        self._height_combo = ttk.Combobox(
            cfg_frame, textvariable=self._analysis_height_var, state="disabled", width=10
        )
        self._height_combo.grid(row=0, column=3, sticky="w", **self.PAD)

        # -- Outputs ----------------------------------------------------------
        out_frame = ttk.LabelFrame(self, text="Outputs")
        out_frame.grid(row=4, column=0, sticky="ew", **self.PAD)

        output_options = {
            "wind_dist": "Wind distribution table",
            "coverage":  "Data coverage table",
            "avg_speed": "Average wind speed table",
            "wind_rose": "Wind rose figure",
            "time_series": "Time series figure",
        }
        self._output_vars: dict[str, tk.BooleanVar] = {
            k: tk.BooleanVar(value=True) for k in output_options
        }
        for i, (key, label) in enumerate(output_options.items()):
            row, col = divmod(i, 3)
            ttk.Checkbutton(
                out_frame, text=label, variable=self._output_vars[key]
            ).grid(row=row, column=col, padx=10, pady=2, sticky="w")

        # Figure format and DPI
        fig_row = (len(output_options) + 2) // 3 + 1
        ttk.Label(out_frame, text="Figure format:").grid(
            row=fig_row, column=0, sticky="e", **self.PAD
        )
        self._fig_fmt_var = tk.StringVar(value=DEFAULT_FMT)
        ttk.Combobox(
            out_frame,
            textvariable=self._fig_fmt_var,
            values=FIGURE_FORMATS,
            state="readonly",
            width=6,
        ).grid(row=fig_row, column=1, sticky="w", **self.PAD)

        ttk.Label(out_frame, text="DPI:").grid(
            row=fig_row, column=2, sticky="e", **self.PAD
        )
        self._dpi_var = tk.StringVar(value=DEFAULT_DPI)
        ttk.Combobox(
            out_frame,
            textvariable=self._dpi_var,
            values=DPI_OPTIONS,
            state="readonly",
            width=5,
        ).grid(row=fig_row, column=3, sticky="w", **self.PAD)

        # Output folder
        folder_row = fig_row + 1
        ttk.Label(out_frame, text="Output folder:").grid(
            row=folder_row, column=0, sticky="e", **self.PAD
        )
        self._out_dir_var = tk.StringVar(value=str(Path.cwd() / "output"))
        ttk.Entry(out_frame, textvariable=self._out_dir_var).grid(
            row=folder_row, column=1, columnspan=2, sticky="ew", **self.PAD
        )
        ttk.Button(
            out_frame, text="Browse…", command=self._browse_out_dir
        ).grid(row=folder_row, column=3, **self.PAD)
        out_frame.columnconfigure(1, weight=1)

        # -- Action buttons ---------------------------------------------------
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=5, column=0, sticky="ew", **self.PAD)

        ttk.Button(btn_frame, text="Load file", command=self._load_file).pack(
            side="left", padx=6
        )
        ttk.Button(
            btn_frame, text="Process & export", command=self._start_processing
        ).pack(side="left", padx=6)

        # -- Status log -------------------------------------------------------
        log_frame = ttk.LabelFrame(self, text="Status / Log")
        log_frame.grid(row=6, column=0, sticky="nsew", **self.PAD)
        self.rowconfigure(6, weight=1)

        self._log = scrolledtext.ScrolledText(
            log_frame, height=10, state="disabled", wrap="word"
        )
        self._log.pack(fill="both", expand=True, padx=4, pady=4)

        self._log_msg("WindLab ready. Select a file and a preset.")

    def _build_mapping_ui(self) -> None:
        """Build column mapping widgets inside _mapping_frame."""
        variable_labels = {
            "time":              "Time column  *",
            "Wind Speed (m/s)":  "Wind speed  *",
            "Wind Direction (°)":"Wind direction  *",
            "X-wind (m/s)":      "X-wind  (optional)",
            "Y-wind (m/s)":      "Y-wind  (optional)",
            "Z-wind (m/s)":      "Z-wind  (optional)",
        }

        for i, (var_key, label) in enumerate(variable_labels.items()):
            row = i // 2
            col_base = (i % 2) * 3
            ttk.Label(self._mapping_frame, text=f"{label}:").grid(
                row=row, column=col_base, sticky="e", padx=(12, 4), pady=3
            )
            combo = ttk.Combobox(self._mapping_frame, width=22, state="readonly")
            combo.grid(row=row, column=col_base + 1, sticky="w", padx=(0, 12), pady=3)
            self._mapping_widgets[var_key] = combo

        # Measurement height entry
        n_rows = (len(variable_labels) + 1) // 2
        ttk.Label(self._mapping_frame, text="Measurement height (m):").grid(
            row=n_rows, column=0, sticky="e", padx=(12, 4), pady=3
        )
        ttk.Entry(
            self._mapping_frame, textvariable=self._meas_height_var, width=10
        ).grid(row=n_rows, column=1, sticky="w", padx=(0, 12), pady=3)

    # =========================================================================
    # Event handlers
    # =========================================================================

    def _on_preset_change(self) -> None:
        """Show or hide the column mapping panel based on the selected preset."""
        if self._preset_var.get() == "custom":
            self._mapping_frame.grid(row=2, column=0, sticky="ew", **self.PAD)
        else:
            self._mapping_frame.grid_remove()

    def _browse_file(self) -> None:
        filetypes = [
            ("Supported files", "*.rtd *.csv *.xlsx *.xls"),
            ("WindCube RTD", "*.rtd"),
            ("CSV", "*.csv"),
            ("Excel", "*.xlsx *.xls"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select input file", filetypes=filetypes)
        if path:
            self._file_var.set(path)

    def _browse_out_dir(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._out_dir_var.set(path)

    # =========================================================================
    # File loading
    # =========================================================================

    def _load_file(self) -> None:
        """Load the selected file according to the active preset."""
        file_path = self._file_var.get().strip()
        if not file_path:
            messagebox.showwarning("Warning", "Please select a file before loading.")
            return
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found:\n{file_path}")
            return

        preset = self._preset_var.get()
        try:
            if preset == "windcube":
                self._load_windcube(file_path)
            elif preset == "custom":
                self._load_custom_file(file_path)
            else:
                messagebox.showwarning("Warning", f"Preset '{preset}' is not available.")
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))
            self._log_msg(f"✗ Error: {exc}")
            logger.exception("Error loading file.")

    def _load_windcube(self, file_path: str) -> None:
        """Load a WindCube file and create the dataset."""
        ref_height = _float_or(self._ref_height_var.get(), 0.0)
        self._dataset = WindDataAccessor.windcube(file_path, reference_height=int(ref_height))
        name = Path(file_path).name
        heights = list(self._dataset.height.values)
        t_start = str(self._dataset.time.values[0])[:19]
        t_end   = str(self._dataset.time.values[-1])[:19]
        self._log_msg(f"✓ WindCube file loaded: {name}")
        self._log_msg(f"  Available heights: {heights}")
        self._log_msg(f"  Period: {t_start} → {t_end}")
        self._populate_heights(heights)

    def _load_custom_file(self, file_path: str) -> None:
        """Read the header of a custom file and populate the mapping combo boxes."""
        self._custom_reader = CustomDataReader()
        columns = self._custom_reader.read_file(file_path)
        name = Path(file_path).name
        self._log_msg(f"✓ File read: {name}")
        self._log_msg(f"  Detected columns ({len(columns)}): {', '.join(columns)}")
        self._log_msg("  → Fill in the column mapping and click 'Process & export'.")
        self._fill_mapping_combos(columns)
        # Dataset is not created yet — will be created in _process
        self._dataset = None

    def _fill_mapping_combos(self, columns: list[str]) -> None:
        """
        Populate mapping combos with file columns.
        Attempts to auto-detect common column names.
        """
        optional_choices = [NO_COLUMN_SENTINEL] + columns

        # Heuristics for auto-detection
        auto_hints: dict[str, list[str]] = {
            "time": [
                "time", "timestamp", "date", "datetime",
                "date_time", "date/time",
            ],
            "Wind Speed (m/s)": [
                "wind speed", "ws", "speed",
                "wind_speed", "windspeed",
            ],
            "Wind Direction (°)": [
                "wind direction", "wd", "direction",
                "wind_direction",
            ],
        }
        col_lower_map = {c.lower(): c for c in columns}

        for var_key, combo in self._mapping_widgets.items():
            is_optional = var_key not in (["time"] + REQUIRED_VARIABLES)
            choices = optional_choices if is_optional else columns
            combo.configure(values=choices)

            # Attempt auto-detection
            detected = ""
            for hint in auto_hints.get(var_key, []):
                if hint.lower() in col_lower_map:
                    detected = col_lower_map[hint.lower()]
                    break

            if detected:
                combo.set(detected)
            elif is_optional:
                combo.set(NO_COLUMN_SENTINEL)
            elif columns:
                combo.set(columns[0])

    # =========================================================================
    # Processing (runs in a background thread)
    # =========================================================================

    def _start_processing(self) -> None:
        """Start processing in a background thread to keep the UI responsive."""
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self) -> None:
        try:
            preset = self._preset_var.get()

            # Custom mode: apply mapping and create dataset
            if preset == "custom":
                if self._custom_reader is None:
                    self._log_msg("✗ Load the file before processing.")
                    return
                self._apply_custom_mapping()
                if self._dataset is None:
                    return  # Error already logged in _apply_custom_mapping

            if self._dataset is None:
                self._log_msg("✗ No dataset available. Please load the file first.")
                return

            # Determine analysis height
            height = _float_or(self._analysis_height_var.get(), float("nan"))
            if height != height:  # NaN check
                height = float(self._dataset.height.values[0])
                self._log_msg(f"  Using first available height: {height} m")

            # Prepare output folder
            out_dir = self._out_dir_var.get().strip() or str(Path.cwd() / "output")
            os.makedirs(out_dir, exist_ok=True)
            self._log_msg(f"  Output folder: {out_dir}")

            # Figure settings
            fig_ext = self._fig_fmt_var.get().lower()
            fig_dpi = int(_float_or(self._dpi_var.get(), 150))

            ds = self._dataset
            h_label = f"{int(height)}m"

            # -- Wind distribution table --------------------------------------
            if self._output_vars["wind_dist"].get():
                self._log_msg("  Generating wind distribution table…")
                table = ds.wind_table.generate_wind_distribution_table(height)
                path = os.path.join(out_dir, f"wind_distribution_{h_label}.csv")
                table.to_csv(path)
                self._log_msg(f"  ✓ {os.path.basename(path)}")

            # -- Data coverage table ------------------------------------------
            if self._output_vars["coverage"].get():
                self._log_msg("  Generating data coverage table…")
                table = ds.wind_table.generate_data_coverage_table(height)
                path = os.path.join(out_dir, f"data_coverage_{h_label}.csv")
                table.to_csv(path)
                self._log_msg(f"  ✓ {os.path.basename(path)}")

            # -- Average wind speed table -------------------------------------
            if self._output_vars["avg_speed"].get():
                self._log_msg("  Generating average wind speed table…")
                table = ds.wind_table.generate_average_wind_speed_table(height)
                path = os.path.join(out_dir, f"average_wind_speed_{h_label}.csv")
                table.to_csv(path)
                self._log_msg(f"  ✓ {os.path.basename(path)}")

            # -- Wind rose figure ---------------------------------------------
            if self._output_vars["wind_rose"].get():
                self._log_msg(f"  Generating wind rose ({fig_ext.upper()}, {fig_dpi} dpi)…")
                ax = ds.wind_graph.plot_wind_rose(height)
                path = os.path.join(out_dir, f"wind_rose_{h_label}.{fig_ext}")
                ax.figure.savefig(path, dpi=fig_dpi, bbox_inches="tight")
                plt.close(ax.figure)
                self._log_msg(f"  ✓ {os.path.basename(path)}")

            # -- Time series figure -------------------------------------------
            if self._output_vars["time_series"].get():
                self._log_msg(f"  Generating time series ({fig_ext.upper()}, {fig_dpi} dpi)…")
                ax = ds.wind_graph.plot_variable(height=height)
                path = os.path.join(out_dir, f"time_series_{h_label}.{fig_ext}")
                ax.figure.savefig(path, dpi=fig_dpi, bbox_inches="tight")
                plt.close(ax.figure)
                self._log_msg(f"  ✓ {os.path.basename(path)}")

            self._log_msg(f"\n✓ Processing complete.\n  Files saved to: {out_dir}\n")
            self.after(
                0,
                lambda: messagebox.showinfo(
                    "Done",
                    f"Processing complete.\nFiles saved to:\n{out_dir}",
                ),
            )

        except Exception as exc:
            msg = traceback.format_exc()
            self._log_msg(f"✗ Error during processing:\n{msg}")
            self.after(0, lambda e=str(exc): messagebox.showerror("Error", e))

    def _apply_custom_mapping(self) -> None:
        """Apply column mapping and create the dataset in Custom mode."""
        column_map = {
            var_key: combo.get()
            for var_key, combo in self._mapping_widgets.items()
        }

        meas_height = _float_or(self._meas_height_var.get(), 0.0)
        ref_height  = _float_or(self._ref_height_var.get(), 0.0)

        try:
            self._custom_reader.map_columns(column_map, height_value=meas_height)
        except (KeyError, RuntimeError) as exc:
            self._log_msg(f"✗ Mapping error: {exc}")
            messagebox.showerror("Mapping error", str(exc))
            return

        warnings = self._custom_reader.validate()
        for w in warnings:
            self._log_msg(f"⚠  {w}")

        self._dataset = self._custom_reader.to_dataset(reference_height=ref_height)
        heights = list(self._dataset.height.values)
        n_times = len(self._dataset.time)
        self._log_msg(f"✓ Dataset created: {n_times} records, height(s): {heights}")
        self.after(0, lambda h=heights: self._populate_heights(h))

    # =========================================================================
    # Utilities
    # =========================================================================

    def _populate_heights(self, heights: list) -> None:
        """Fill the height combo box with values available in the dataset."""
        values = [str(h) for h in heights]
        self._height_combo.configure(values=values, state="readonly")
        if values:
            self._analysis_height_var.set(values[0])

    def _log_msg(self, message: str) -> None:
        """Write a line to the status log (thread-safe)."""
        def _write() -> None:
            self._log.configure(state="normal")
            self._log.insert("end", message + "\n")
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _write)


# -- Entry point -------------------------------------------------------------

def run() -> None:
    """Launch the WindLab GUI."""
    app = WindLabApp()
    app.mainloop()


if __name__ == "__main__":
    run()
