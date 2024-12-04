# WindLab

WindLab is a Python package designed for the manipulation and analysis of LIDAR wind data, specifically from WindCube and Zephyr LIDAR devices. This package allows users to read wind data, process it into structured formats, and generate various plots and tables for technical reporting. The project aims to streamline the workflow of analyzing wind measurements, offering tools for data coverage analysis, wind speed plotting, and wind rose visualization.

## Features

- **Wind Data Reader**: Read and process LIDAR data from WindCube and Zephyr devices, transforming it into xarray datasets.
- **Data Coverage Analysis**: Generate data coverage tables and visualize the coverage in a heatmap.
- **Wind Visualizations**: Plot time series of wind speed and direction, as well as wind rose diagrams for specific heights and periods.
- **Flexible and Modular**: Built with modular design, allowing different functionalities like data reading, table generation, and plotting to be extended or customized.

## Installation

To install WindLab, clone the repository and install the package using pip:

```bash
# Clone the repository
git clone https://github.com/seuusuario/windlab.git

# Navigate to the project directory
cd windlab

# Install the package
pip install -e .
```

### Requirements
- Python 3.8+
- numpy
- pandas
- xarray
- matplotlib
- seaborn
- windrose

Dependencies are automatically installed when using `pip install`. Alternatively, you can manually install the requirements by running:

```bash
pip install -r requirements.txt
```

## Usage

WindLab provides tools to read wind data, generate plots, and create data coverage tables. Usage examples are available in the provided Jupyter notebooks within the `docs/` folder.

## Project Structure

- **windlab/**: Main package directory containing core modules for data processing.
  - **wind_data_reader.py**: Functions for reading LIDAR data from WindCube or Zephyr.
  - **processing/**: Modules for processing wind data, including graphs, tables, and utilities.
- **examples/**: Example scripts to demonstrate how to use the package.
- **docs/**: Tutorials and documentation, including Jupyter notebooks with detailed usage examples.
- **tests/**: Unit tests for the package.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes.

### To-Do List
- Expand support for additional LIDAR models.
- Improve the user interface for data visualization.
- Add more unit tests to improve coverage.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For questions or suggestions, please contact:

- **Author**: Seu Nome
- **Email**: seu.email@example.com

