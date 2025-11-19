"""
DAT file handling for Nanonis curve data.

This module provides classes for loading and processing .dat files,
including curve fitting capabilities.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from .base import BaseFile, BaseFileCollection


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
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.fit_parameters: Optional[Dict[str, Any]] = None
        self.fit_function: Optional[Callable] = None
        
    def load(self) -> Dict[str, Any]:
        """
        Load raw data from .dat file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'x': X-axis data (e.g., bias voltage)
            - 'y': Y-axis data (e.g., current)
            - 'metadata': Dictionary of file metadata
        """
        # TODO: Implement actual .dat file loading
        # This is a placeholder that will be implemented when sample files are provided
        raise NotImplementedError("DAT file loading not yet implemented. Waiting for sample files.")
        
        # Placeholder structure:
        # self.x_data = ...
        # self.y_data = ...
        # self.metadata = ...
        # self.raw_data = {
        #     'x': self.x_data,
        #     'y': self.y_data,
        #     'metadata': self.metadata
        # }
        # return self.raw_data
    
    def process(self, fit: bool = False, fit_function: Optional[Callable] = None, **kwargs) -> Dict[str, Any]:
        """
        Process the loaded curve data.
        
        Parameters
        ----------
        fit : bool
            Whether to perform curve fitting
        fit_function : Optional[Callable]
            Function to fit to the data
        **kwargs
            Additional processing parameters:
            - initial_guess: Initial guess for fit parameters
            - fit_method: Fitting method ('least_squares', 'minimize', etc.)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed data
        """
        if self.raw_data is None:
            raise ValueError("Data must be loaded before processing. Call load() first.")
        
        processed = {
            'x': self.x_data,
            'y': self.y_data
        }
        
        # Curve fitting
        if fit and fit_function is not None:
            self.fit_parameters = self._fit_curve(fit_function, **kwargs)
            processed['fit_parameters'] = self.fit_parameters
            processed['fit_function'] = fit_function
            if self.x_data is not None:
                processed['y_fit'] = fit_function(self.x_data, **self.fit_parameters)
        
        self.processed_data = processed
        return processed
    
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
    
    def process_all(self, fit: bool = False, fit_function: Optional[Callable] = None, **kwargs) -> None:
        """
        Process all files in the collection.
        
        Parameters
        ----------
        fit : bool
            Whether to perform curve fitting
        fit_function : Optional[Callable]
            Function to fit to the data
        **kwargs
            Additional processing parameters
        """
        for file in self.files:
            file.process(fit=fit, fit_function=fit_function, **kwargs)
    
    def get_all_curves(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all curves from the collection.
        
        Returns
        -------
        List[tuple[np.ndarray, np.ndarray]]
            List of (x, y) tuples
        """
        return [file.get_curve() for file in self.files if file.get_curve()[0] is not None]

