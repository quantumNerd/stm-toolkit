"""
QCoDes database file handling.

This module provides classes for loading and processing QCoDes database files.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from .base import BaseFile, BaseFileCollection

try:
    from qcodes.dataset.data_export import reshape_2D_data
    from qcodes.dataset import initialise_or_create_database_at, load_or_create_experiment
except ImportError:
    reshape_2D_data = None
    initialise_or_create_database_at = None
    load_or_create_experiment = None


class QCoDesDatabase(BaseFile):
    """
    Handler for QCoDes database files.
    
    Supports loading and processing data from QCoDes measurement databases.
    """
    
    def __init__(self, db_path: str | Path, experiment_name: str, sample_name: str):
        """
        Initialize QCoDes database handler.
        
        Parameters
        ----------
        db_path : str | Path
            Path to the QCoDes database file
        experiment_name : str
            Name of the experiment
        sample_name : str
            Name of the sample
        """
        super().__init__(db_path)
        self.db_path = Path(db_path)
        self.experiment_name = experiment_name
        self.sample_name = sample_name
        
        # Initialize database and experiment
        if initialise_or_create_database_at is None:
            raise ImportError("QCoDes is required. Please install qcodes.")
        
        initialise_or_create_database_at(str(self.db_path))
        self.exp0 = load_or_create_experiment(
            experiment_name=self.experiment_name,
            sample_name=self.sample_name
        )
        
        # Data storage
        self.x: Optional[np.ndarray] = None  # Gate (V)
        self.y: Optional[np.ndarray] = None  # Bias (V)
        self.z: Optional[np.ndarray] = None  # dI/dV or other signal
        self.run_id: Optional[int] = None
        self.raw_data: Optional[Dict[str, Any]] = None
        
    def load_run(self, run_id: int, signal_key: str = "S86", 
                 x_key: str = "Gate", y_key: str = "Bias", z_key: str = "S86") -> Dict[str, Any]:
        """
        Load data from a specific run_id.
        
        Parameters
        ----------
        run_id : int
            Run ID to load
        signal_key : str
            Key for the signal parameter (default: "S86")
        x_key : str
            Key for x-axis data (default: "Gate")
        y_key : str
            Key for y-axis data (default: "Bias")
        z_key : str
            Key for z-axis data (default: "S86")
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'x': X-axis data (Gate)
            - 'y': Y-axis data (Bias)
            - 'z': Z-axis data (dI/dV)
            - 'run_id': Run ID
        """
        if reshape_2D_data is None:
            raise ImportError("qcodes.dataset.data_export.reshape_2D_data is required. "
                             "Please ensure QCoDes is properly installed.")
        
        # Load data from QCoDes
        dat0 = self.exp0.data_set(run_id).get_parameter_data()
        
        # Extract data
        signal_data = dat0[signal_key]
        
        x_raw = signal_data[x_key]
        y_raw = signal_data[y_key]
        z_raw = signal_data[z_key]
        
        # Reshape to 2D
        x, y, z = reshape_2D_data(x_raw, y_raw, z_raw)
        
        # Store in class
        self.x = x
        self.y = y
        self.z = z
        self.run_id = run_id
        
        self.raw_data = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'run_id': self.run_id
        }
        
        return self.raw_data
    
    def load(self) -> Dict[str, Any]:
        """
        Legacy method - use load_run() instead.
        """
        raise NotImplementedError("Use load_run(run_id) to load data from a specific run.")
    
    def process(self, filter_measurements: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Process the loaded database data.
        
        Parameters
        ----------
        filter_measurements : bool
            Whether to filter measurements based on criteria
        **kwargs
            Additional processing parameters:
            - filter_criteria: Dictionary of filter criteria
            - aggregate: Whether to aggregate measurements
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed data
        """
        if self.raw_data is None:
            raise ValueError("Data must be loaded before processing. Call load() first.")
        
        processed = {
            'measurements': self.measurements.copy() if self.measurements is not None else None
        }
        
        # Filter measurements
        if filter_measurements:
            filter_criteria = kwargs.get('filter_criteria', {})
            processed['measurements'] = self._filter_measurements(processed['measurements'], filter_criteria)
        
        # Aggregation
        if kwargs.get('aggregate', False):
            processed['aggregated'] = self._aggregate_measurements(processed['measurements'], **kwargs)
        
        self.processed_data = processed
        return processed
    
    def _filter_measurements(self, measurements: List[Dict[str, Any]], 
                            criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter measurements based on criteria.
        
        Parameters
        ----------
        measurements : List[Dict[str, Any]]
            List of measurement dictionaries
        criteria : Dict[str, Any]
            Filter criteria
            
        Returns
        -------
        List[Dict[str, Any]]
            Filtered list of measurements
        """
        # TODO: Implement measurement filtering
        raise NotImplementedError("Measurement filtering not yet implemented.")
    
    def _aggregate_measurements(self, measurements: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Aggregate measurements.
        
        Parameters
        ----------
        measurements : List[Dict[str, Any]]
            List of measurement dictionaries
        **kwargs
            Aggregation parameters
            
        Returns
        -------
        Dict[str, Any]
            Aggregated data
        """
        # TODO: Implement measurement aggregation
        raise NotImplementedError("Measurement aggregation not yet implemented.")
    
    def get_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the loaded data.
        
        Returns
        -------
        Dict[str, Any] or None
            Dictionary containing x, y, z data and run_id
        """
        return self.raw_data
    
    def plot(self, ax=None, plot_title: str = "", norm_const: float = 10, **kwargs):
        """
        Plot the loaded data as a 2D colormap.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure
        plot_title : str
            Title for the plot
        norm_const : float
            Normalization constant for colormap (default: 10)
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the plot
        """
        if self.x is None or self.y is None or self.z is None:
            raise ValueError("No data loaded. Call load_run(run_id) first.")
        
        # Create figure if needed (using default 2D figure size)
        if ax is None:
            from .plotting import BasePlotter
            default_figsize = BasePlotter.DEFAULT_FIGSIZE_2D
            figsize = kwargs.get('figsize', default_figsize)
            fig, ax = plt.subplots(figsize=figsize)
        
        # Set up colormap
        cmap = plt.cm.RdYlBu_r
        
        # Create plot
        if norm_const is not None:
            norm = plt.Normalize(0, norm_const)
            im = ax.pcolormesh(
                self.x,  # Gate (V)
                self.y * 1e3,  # Bias (mV)
                self.z / 1e-3 * 1e9,  # dI/dV (nS)
                cmap=cmap,
                norm=norm,
                rasterized=True,
                shading='nearest'
            )
        else:
            im = ax.pcolormesh(
                self.x,  # Gate (V)
                self.y * 1e3,  # Bias (mV)
                self.z / 1e-3 * 1e9,  # dI/dV (nS)
                cmap=cmap,
                rasterized=True,
                shading='nearest'
            )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$dI/dV$ (nS)')
        
        # Set labels
        ax.set_xlabel(r'$V_{\mathrm{g}}$ (V)')
        ax.set_ylabel(r'$V_{\mathrm{b}}$ (mV)')
        title = plot_title if plot_title else f'Run ID: {self.run_id}'
        ax.set_title(title)
        
        return ax


class QCoDesCollection(BaseFileCollection):
    """
    Collection of QCoDes database files with associated hyperparameters.
    
    Useful for analyzing multiple QCoDes databases.
    """
    
    def __init__(self, db_path: str | Path, experiment_name: str, sample_name: str,
                 hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize QCoDes collection.
        
        Parameters
        ----------
        db_path : str | Path
            Path to the QCoDes database file
        experiment_name : str
            Name of the experiment
        sample_name : str
            Name of the sample
        hyperparameters : Optional[Dict[str, Any]]
            Dictionary of hyperparameters
        """
        # For collection, we use a single database with multiple runs
        super().__init__([db_path], hyperparameters)
        self.db = QCoDesDatabase(db_path, experiment_name, sample_name)
        self.files = [self.db]  # Single database instance
        
    def load_all(self) -> None:
        """Load all QCoDes database files in the collection."""
        for file in self.files:
            file.load()
    
    def process_all(self, filter_measurements: bool = False, **kwargs) -> None:
        """
        Process all files in the collection.
        
        Parameters
        ----------
        filter_measurements : bool
            Whether to filter measurements
        **kwargs
            Additional processing parameters
        """
        for file in self.files:
            file.process(filter_measurements=filter_measurements, **kwargs)
    
    def get_all_measurements(self) -> List[List[Dict[str, Any]]]:
        """
        Get all measurements from all databases in the collection.
        
        Returns
        -------
        List[List[Dict[str, Any]]]
            List of measurement lists (one per database)
        """
        return [file.get_measurements() for file in self.files if file.get_measurements() is not None]


def simple_plot_dts(exp0, run_id: int, ax=None, plot_title: str = "", 
                    norm_const: float = 10, **kwargs):
    """
    Simple function to load and plot QCoDes data from one experiment and one run-id.
    
    This function loads data using exp0.data_set(run_id).get_parameter_data() and
    plots it as a 2D colormap (dI/dV vs Gate and Bias).
    
    Parameters
    ----------
    exp0
        QCoDes experiment object
    run_id : int
        Run ID to load
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure
    plot_title : str
        Title for the plot
    norm_const : float
        Normalization constant for colormap (default: 10)
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    if reshape_2D_data is None:
        raise ImportError("qcodes.dataset.data_export.reshape_2D_data is required. "
                         "Please ensure QCoDes is properly installed.")
    
    # Load data
    dat0 = exp0.data_set(run_id).get_parameter_data()
    
    # Extract data using default keys (as in original notebook)
    # keys=["S86","Gate","Bias","S86"] means:
    # - Get "S86" parameter data
    # - Use "Gate" as x-axis
    # - Use "Bias" as y-axis  
    # - Use "S86" values as z (color)
    didv = dat0["S86"]  # get the x component of lock-in data
    
    x = didv["Gate"]
    y = didv["Bias"]
    z = didv["S86"]
    
    # Reshape to 2D (as in original notebook)
    x, y, z = reshape_2D_data(x, y, z)
    didv = z
    
    # Create figure if needed (using default 2D figure size)
    if ax is None:
        from .plotting import BasePlotter
        default_figsize = BasePlotter.DEFAULT_FIGSIZE_2D
        figsize = kwargs.get('figsize', default_figsize)
        fig, ax = plt.subplots(figsize=figsize)
    
    # Set up colormap
    cmap = plt.cm.RdYlBu_r
    
    # Create plot
    if norm_const is not None:
        norm = plt.Normalize(0, norm_const)
        im = ax.pcolormesh(
            x,  # Gate (V)
            y * 1e3,  # Bias (mV)
            didv / 1e-3 * 1e9,  # dI/dV (nS)
            cmap=cmap,
            norm=norm,
            rasterized=True,
            shading='nearest'
        )
    else:
        im = ax.pcolormesh(
            x,  # Gate (V)
            y * 1e3,  # Bias (mV)
            didv / 1e-3 * 1e9,  # dI/dV (nS)
            cmap=cmap,
            rasterized=True,
            shading='nearest'
        )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$dI/dV$ (nS)')
    
    # Set labels
    ax.set_xlabel(r'$V_{\mathrm{g}}$ (V)')
    ax.set_ylabel(r'$V_{\mathrm{b}}$ (mV)')
    ax.set_title(plot_title if plot_title else f'Run ID: {run_id}')
    
    return ax

