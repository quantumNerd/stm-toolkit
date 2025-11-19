"""
Grid spectroscopy file handling for Nanonis .3ds files.

This module provides classes for loading and processing .3ds files,
which contain grid spectroscopy data (I-V or dI/dV maps).
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from .base import BaseFile, BaseFileCollection


class GridSpectroscopyFile(BaseFile):
    """
    Handler for individual .3ds files.
    
    Supports loading grid spectroscopy data and processing operations.
    """
    
    def __init__(self, file_path: str | Path):
        """
        Initialize grid spectroscopy file handler.
        
        Parameters
        ----------
        file_path : str | Path
            Path to the .3ds file
        """
        super().__init__(file_path)
        self.spectroscopy_data: Optional[np.ndarray] = None  # 3D array: [x, y, bias]
        self.bias_voltage: Optional[np.ndarray] = None
        self.x_positions: Optional[np.ndarray] = None
        self.y_positions: Optional[np.ndarray] = None
        
    def load(self) -> Dict[str, Any]:
        """
        Load raw data from .3ds file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'data': 3D numpy array of spectroscopy data [x, y, bias]
            - 'bias_voltage': Array of bias voltages
            - 'x_positions': Array of X positions
            - 'y_positions': Array of Y positions
            - 'metadata': Dictionary of file metadata
        """
        # TODO: Implement actual .3ds file loading
        # This is a placeholder that will be implemented when sample files are provided
        raise NotImplementedError("Grid spectroscopy file loading not yet implemented. Waiting for sample files.")
        
        # Placeholder structure:
        # self.spectroscopy_data = ...  # 3D array
        # self.bias_voltage = ...
        # self.x_positions = ...
        # self.y_positions = ...
        # self.metadata = ...
        # self.raw_data = {
        #     'data': self.spectroscopy_data,
        #     'bias_voltage': self.bias_voltage,
        #     'x_positions': self.x_positions,
        #     'y_positions': self.y_positions,
        #     'metadata': self.metadata
        # }
        # return self.raw_data
    
    def process(self, normalize: bool = False, smooth: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Process the loaded grid spectroscopy data.
        
        Parameters
        ----------
        normalize : bool
            Whether to normalize the data
        smooth : bool
            Whether to smooth the data
        **kwargs
            Additional processing parameters:
            - smooth_method: Smoothing method ('gaussian', 'moving_average', etc.)
            - smooth_sigma: Sigma for Gaussian smoothing (if applicable)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed data
        """
        if self.raw_data is None:
            raise ValueError("Data must be loaded before processing. Call load() first.")
        
        processed = {
            'data': self.spectroscopy_data.copy() if self.spectroscopy_data is not None else None,
            'bias_voltage': self.bias_voltage,
            'x_positions': self.x_positions,
            'y_positions': self.y_positions
        }
        
        # Normalization
        if normalize and processed['data'] is not None:
            processed['data'] = self._normalize(processed['data'], **kwargs)
        
        # Smoothing
        if smooth and processed['data'] is not None:
            processed['data'] = self._smooth(processed['data'], **kwargs)
        
        self.processed_data = processed
        return processed
    
    def _normalize(self, data: np.ndarray, method: str = 'z_score', **kwargs) -> np.ndarray:
        """
        Normalize the spectroscopy data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        method : str
            Normalization method ('z_score', 'min_max', etc.)
            
        Returns
        -------
        np.ndarray
            Normalized data
        """
        # TODO: Implement normalization methods
        raise NotImplementedError("Normalization not yet implemented.")
    
    def _smooth(self, data: np.ndarray, smooth_method: str = 'gaussian', smooth_sigma: float = 1.0, **kwargs) -> np.ndarray:
        """
        Smooth the spectroscopy data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        smooth_method : str
            Smoothing method ('gaussian', 'moving_average')
        smooth_sigma : float
            Sigma for Gaussian smoothing
            
        Returns
        -------
        np.ndarray
            Smoothed data
        """
        # TODO: Implement smoothing methods
        raise NotImplementedError("Smoothing not yet implemented.")
    
    def get_spectrum_at_position(self, x: int, y: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get a single spectrum at a specific grid position.
        
        Parameters
        ----------
        x : int
            X grid index
        y : int
            Y grid index
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (bias_voltage, spectrum_data)
        """
        if self.spectroscopy_data is None or self.bias_voltage is None:
            return None, None
        
        spectrum = self.spectroscopy_data[x, y, :]
        return self.bias_voltage, spectrum
    
    def get_map_at_bias(self, bias_index: int) -> Optional[np.ndarray]:
        """
        Get a 2D map at a specific bias voltage.
        
        Parameters
        ----------
        bias_index : int
            Index of the bias voltage
            
        Returns
        -------
        np.ndarray
            2D map at the specified bias
        """
        if self.spectroscopy_data is None:
            return None
        
        return self.spectroscopy_data[:, :, bias_index]


class GridSpectroscopyCollection(BaseFileCollection):
    """
    Collection of grid spectroscopy files with associated hyperparameters.
    
    Useful for analyzing multiple grid spectroscopy datasets taken at different conditions.
    """
    
    def __init__(self, file_paths: List[str] | List[Path], hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize grid spectroscopy collection.
        
        Parameters
        ----------
        file_paths : List[str] | List[Path]
            List of paths to .3ds files
        hyperparameters : Optional[Dict[str, Any]]
            Dictionary of hyperparameters (e.g., {'gate_voltage': [0.1, 0.2, ...]})
        """
        super().__init__(file_paths, hyperparameters)
        self.files = [GridSpectroscopyFile(path) for path in self.file_paths]
        
    def load_all(self) -> None:
        """Load all grid spectroscopy files in the collection."""
        for file in self.files:
            file.load()
    
    def process_all(self, normalize: bool = False, smooth: bool = False, **kwargs) -> None:
        """
        Process all files in the collection.
        
        Parameters
        ----------
        normalize : bool
            Whether to normalize the data
        smooth : bool
            Whether to smooth the data
        **kwargs
            Additional processing parameters
        """
        for file in self.files:
            file.process(normalize=normalize, smooth=smooth, **kwargs)
    
    def get_all_data(self) -> List[np.ndarray]:
        """
        Get all spectroscopy data arrays from the collection.
        
        Returns
        -------
        List[np.ndarray]
            List of 3D data arrays
        """
        return [file.spectroscopy_data for file in self.files if file.spectroscopy_data is not None]

