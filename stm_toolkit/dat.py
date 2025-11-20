"""
DAT file handling for Nanonis curve data.

This module provides classes for loading and processing .dat files,
including curve fitting capabilities.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
import matplotlib.pyplot as plt
from .base import BaseFile, BaseFileCollection
from .plotting import BasePlotter
from .utils import find_first_line, get_dat_column
from .analysis import find_max_and_width, get_broadening_by_gaussian_fit


class DATFile(BaseFile):
    """
    Handler for individual .dat files.
    
    Supports loading curves and fitting operations.
    """
    
    def __init__(self, file_path: str | Path):
        """
        Initialize DAT file handler.
        
        Parameters
        ----------
        file_path : str | Path
            Path to the .dat file
        """
        super().__init__(file_path)
        # Only raw data is stored - processing is done on-the-fly when needed
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.data_dict: Dict[str, np.ndarray] = {}  # Dictionary of all columns
        self.column_names: List[str] = []  # List of column names
        
    def _parse_header(self) -> Dict[str, Any]:
        """
        Parse the header section of a .dat file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of header metadata
        """
        header = {}
        
        with open(self.file_path, "r", encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        data_line = find_first_line(str(self.file_path))
        
        # Parse header lines (everything before [DATA])
        for i in range(data_line):
            line = lines[i].strip()
            if "\t" in line:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    header[key] = value
        
        return header
    
    def load(self, x_column: Optional[str] = None, y_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Load raw data from .dat file.
        
        Parameters
        ----------
        x_column : Optional[str]
            Name of column to use as x-axis (default: first column after [DATA])
        y_column : Optional[str]
            Name of column to use as y-axis (default: second column after [DATA])
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'data': Dictionary of all columns {column_name: array}
            - 'x': X-axis data (if x_column specified)
            - 'y': Y-axis data (if y_column specified)
            - 'metadata': Dictionary of file metadata
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if not self.file_path.suffix.lower() == ".dat":
            raise ValueError(f"File must have .dat extension: {self.file_path}")
        
        # Parse header
        self.header = self._parse_header()
        self.metadata = self.header.copy()
        
        # Find data start line
        data_start = find_first_line(str(self.file_path))
        
        # Get column names from header line
        with open(self.file_path, "r", encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            header_line = lines[data_start + 1].strip()
            self.column_names = [col.strip() for col in header_line.split("\t")]
        
        # Load data using numpy (skip header line with column names)
        data = np.genfromtxt(self.file_path, skip_header=data_start + 2, 
                           delimiter="\t", encoding='utf-8')
        
        # Store all columns in dictionary
        if data.ndim == 1:
            # Single row case
            for i, col_name in enumerate(self.column_names):
                if i < len(data):
                    self.data_dict[col_name] = np.array([data[i]])
        else:
            # Multiple rows
            for i, col_name in enumerate(self.column_names):
                if i < data.shape[1]:
                    self.data_dict[col_name] = data[:, i]
        
        # Set default x and y if columns specified
        if x_column is not None:
            if x_column not in self.data_dict:
                raise ValueError(f"Column '{x_column}' not found. Available: {self.column_names}")
            self.x_data = self.data_dict[x_column]
        
        if y_column is not None:
            if y_column not in self.data_dict:
                raise ValueError(f"Column '{y_column}' not found. Available: {self.column_names}")
            self.y_data = self.data_dict[y_column]
        
        # If no columns specified, use first two columns as default
        if self.x_data is None and len(self.column_names) > 0:
            self.x_data = self.data_dict.get(self.column_names[0])
        if self.y_data is None and len(self.column_names) > 1:
            self.y_data = self.data_dict.get(self.column_names[1])
        
        self.raw_data = {
            'data': self.data_dict,
            'x': self.x_data,
            'y': self.y_data,
            'metadata': self.metadata,
            'column_names': self.column_names
        }
        
        return self.raw_data
    
    def get_column(self, column_name: str) -> Optional[np.ndarray]:
        """
        Get a specific column by name.
        
        Parameters
        ----------
        column_name : str
            Name of the column
            
        Returns
        -------
        np.ndarray or None
            Column data
        """
        return self.data_dict.get(column_name)
    
    def get_column_names(self) -> List[str]:
        """Get list of available column names."""
        return self.column_names
    
    def _fit_curve(self, fit_function: Callable, initial_guess: Optional[Dict[str, float]] = None, 
                   fit_method: str = 'least_squares', **kwargs) -> Dict[str, Any]:
        """
        Fit a function to the curve data.
        
        Parameters
        ----------
        fit_function : Callable
            Function to fit (should accept x and parameters)
        initial_guess : Optional[Dict[str, float]]
            Initial guess for fit parameters
        fit_method : str
            Fitting method to use
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of fit parameters
        """
        if self.x_data is None or self.y_data is None:
            raise ValueError("X and Y data required for fitting.")
        
        # TODO: Implement curve fitting
        # Placeholder for implementation
        raise NotImplementedError("Curve fitting not yet implemented.")
    
    def get_curve(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the curve data.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (x_data, y_data)
        """
        return self.x_data, self.y_data
    
    def get_fit(self) -> Optional[Dict[str, Any]]:
        """Get the fit parameters."""
        return self.fit_parameters
    
    def plot(self, x_column: Optional[str] = None, y_column: Optional[str] = None,
             data_source: str = 'raw', **kwargs) -> 'DATPlotter':
        """
        Create a plotter for this DAT file.
        
        Parameters
        ----------
        x_column : Optional[str]
            Column name for x-axis. If None, uses default x_data
        y_column : Optional[str]
            Column name for y-axis. If None, uses default y_data
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        DATPlotter
            Plotter instance
        """
        return DATPlotter(self, x_column=x_column, y_column=y_column, **kwargs)


class DATCollection(BaseFileCollection):
    """
    Collection of DAT files with associated hyperparameters.
    
    Useful for analyzing multiple curves taken at different conditions
    (e.g., different gate voltages).
    """
    
    def __init__(self, file_paths: List[str] | List[Path], hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize DAT collection.
        
        Parameters
        ----------
        file_paths : List[str] | List[Path]
            List of paths to .dat files
        hyperparameters : Optional[Dict[str, Any]]
            Dictionary of hyperparameters (e.g., {'gate_voltage': [0.1, 0.2, ...]})
        """
        super().__init__(file_paths, hyperparameters)
        self.files = [DATFile(path) for path in self.file_paths]
        
    def load_all(self) -> None:
        """Load all DAT files in the collection."""
        for file in self.files:
            file.load()
    
    def get_all_curves(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all curves from the collection.
        
        Returns
        -------
        List[tuple[np.ndarray, np.ndarray]]
            List of (x, y) tuples
        """
        return [file.get_curve() for file in self.files if file.get_curve()[0] is not None]


class DATPlotter(BasePlotter):
    """
    Plotter for DAT file data.
    
    Provides methods for plotting curves and fits.
    """
    
    def __init__(self, dat_file: DATFile, x_column: Optional[str] = None,
                 y_column: Optional[str] = None, data_source: str = 'raw',
                 ax: Optional[plt.Axes] = None, figsize: Optional[Tuple[float, float]] = None, **kwargs):
        """
        Initialize DAT plotter.
        
        Parameters
        ----------
        dat_file : DATFile
            The DAT file to plot
        x_column : Optional[str]
            Column name for x-axis
        y_column : Optional[str]
            Column name for y-axis
        **kwargs
            Additional plotting parameters
        """
        self.dat_file = dat_file
        self.x_column = x_column
        self.y_column = y_column
        
        # Get data
        if x_column is not None:
            self.x_data = dat_file.get_column(x_column)
        else:
            self.x_data = dat_file.x_data
        
        if y_column is not None:
            self.y_data = dat_file.get_column(y_column)
        else:
            self.y_data = dat_file.y_data
        
        super().__init__(dat_file, ax=ax, figsize=figsize, **kwargs)
    
    def _setup_plot(self, **kwargs) -> None:
        """Set up the initial plot."""
        if self.x_data is None or self.y_data is None:
            raise ValueError("X and Y data required for plotting.")
        
        # If axes already exists (from ax parameter), use it; otherwise create new figure
        if self.ax is None:
            # Use unified figure size from BasePlotter
            figsize = self._get_figsize(self.DEFAULT_FIGSIZE_1D, **kwargs)
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111)
        
        self.line_plot, = self.ax.plot(self.x_data, self.y_data, 
                                       marker=kwargs.get('marker', ''),
                                       linestyle=kwargs.get('linestyle', '-'),
                                       linewidth=kwargs.get('linewidth', 1.5),
                                       label=kwargs.get('label', None))
        
        xlabel = kwargs.get('xlabel', self.x_column or 'X')
        ylabel = kwargs.get('ylabel', self.y_column or 'Y')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
        # Only set title if not already set (for subplot compatibility)
        if kwargs.get('title') is not None:
            self.ax.set_title(kwargs.get('title'))
        elif not hasattr(self.ax, '_title_set'):
            self.ax.set_title(kwargs.get('title', f'{self.dat_file.file_path.name}'))
            self.ax._title_set = True
        
        self.ax.grid(True, alpha=0.3)
        
        if kwargs.get('label'):
            self.ax.legend()
    
    def add_fit(self, fit_function: Callable, fit_params: Dict[str, float]) -> None:
        """
        Add fitted curve to the plot.
        
        Parameters
        ----------
        fit_function : Callable
            Fitting function
        fit_params : Dict[str, float]
            Fit parameters
        """
        if self.x_data is None:
            return
        
        x_fit = np.linspace(self.x_data.min(), self.x_data.max(), len(self.x_data) * 10)
        y_fit = fit_function(x_fit, **fit_params)
        self.ax.plot(x_fit, y_fit, 'r--', label='Fit', linewidth=2)
        self.ax.legend()
    
    def add_max_and_width(self, xrange: Optional[Tuple[float, float]] = None,
                          n_fit: Optional[int] = None, store_results: bool = True,
                          **kwargs) -> Tuple[float, float, float, float, float, float]:
        """
        Find and plot maximum and width (FWHM) on the curve.
        
        This method creates the plot when called if it hasn't been created yet.
        
        Parameters
        ----------
        xrange : Optional[Tuple[float, float]]
            Range to search within
        n_fit : Optional[int]
            Number of points for polynomial fitting
        store_results : bool
            If True, store results in dat_file.processed_data
        **kwargs
            Additional parameters for find_max_and_width
            
        Returns
        -------
        Tuple[float, float, float, float, float, float]
            (x_max, y_max, x_half_left, y_half, x_half_right, y_half)
        """
        self._ensure_plot()
        from .analysis import find_max_and_width
        
        result = find_max_and_width(self.x_data, self.y_data, xrange, 
                                   n_fit=n_fit, **kwargs)
        x0, y0, xhl, yhl, xhr, yhr = result
        
        # Store results in data class
        if store_results:
            self.dat_file.fwhm_results = result
            self.dat_file.processed_data['fwhm'] = {
                'x_max': x0,
                'y_max': y0,
                'x_half_left': xhl,
                'y_half': yhl,
                'x_half_right': xhr,
                'fwhm': xhr - xhl
            }
        
        # Plot markers
        self.ax.axvline(x0, color='r', ls='--', alpha=0.7, label='Max')
        self.ax.axhline(y0, color='r', ls='--', alpha=0.7)
        self.ax.axvline(xhl, color='g', ls=':', alpha=0.7, label='FWHM')
        self.ax.axvline(xhr, color='g', ls=':', alpha=0.7)
        self.ax.axhline(yhl, color='g', ls=':', alpha=0.7)
        self.ax.plot(x0, y0, 'ro', markersize=8)
        self.ax.legend()
        
        return result
    
    def add_gaussian_fit(self, xrange: Optional[Tuple[float, float]] = None,
                        initial_guess: Optional[Dict[str, float]] = None,
                        store_results: bool = True) -> Dict[str, Any]:
        """
        Fit a Gaussian with linear background to the curve and plot it.
        
        This method creates the plot when called if it hasn't been created yet.
        
        Parameters
        ----------
        xrange : Optional[Tuple[float, float]]
            Range to fit within
        initial_guess : Optional[Dict[str, float]]
            Initial guess for fit parameters
        store_results : bool
            If True, store fit parameters in dat_file.processed_data
            
        Returns
        -------
        Dict[str, Any]
            Fit results including FWHM
        """
        self._ensure_plot()
        """
        Fit a Gaussian with linear background to the curve and plot it.
        
        Parameters
        ----------
        xrange : Optional[Tuple[float, float]]
            Range to fit within
        initial_guess : Optional[Dict[str, float]]
            Initial guess for fit parameters
        store_results : bool
            If True, store fit parameters in dat_file.processed_data
            
        Returns
        -------
        Dict[str, Any]
            Fit results including FWHM
        """
        from .analysis import get_broadening_by_gaussian_fit, gaussian_with_linear_background
        
        result = get_broadening_by_gaussian_fit(
            self.x_data, self.y_data,
            xrange=xrange,
            initial_guess=initial_guess
        )
        
        # Store results in data class
        if store_results:
            self.dat_file.fit_parameters = result
            self.dat_file.processed_data['gaussian_fit'] = result
        
        # Plot the fit
        x_fit = np.linspace(self.x_data.min(), self.x_data.max(), len(self.x_data) * 10)
        y_fit = gaussian_with_linear_background(x_fit, *result['fit_params'])
        self.ax.plot(x_fit, y_fit, 'r--', label='Gaussian fit', linewidth=2)
        
        # Add FWHM markers
        center = result['center']
        fwhm = result['fwhm']
        y_center = gaussian_with_linear_background(center, *result['fit_params'])
        y_half = y_center / 2
        
        self.ax.axvline(center, color='r', ls='--', alpha=0.7, label=f'Center: {center:.4f}')
        self.ax.axvline(center - fwhm/2, color='g', ls=':', alpha=0.7, label=f'FWHM: {fwhm:.4f}')
        self.ax.axvline(center + fwhm/2, color='g', ls=':', alpha=0.7)
        self.ax.axhline(y_half, color='g', ls=':', alpha=0.7)
        self.ax.plot(center, y_center, 'ro', markersize=8)
        self.ax.legend()
        
        return result

