"""
Grid spectroscopy file handling for Nanonis .3ds files.

This module provides classes for loading and processing .3ds files,
which contain grid spectroscopy data (I-V or dI/dV maps).
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import re
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
        
    def _parse_header(self) -> Tuple[Dict[str, Any], int]:
        """
        Parse the header section of a .3ds file.
        Based on nanonis_load implementation.
        
        Returns
        -------
        Tuple[Dict[str, Any], int]
            Header dictionary and byte position where data starts
        """
        header = {}
        
        with open(self.file_path, "rb") as f:
            file_bytes = f.read()
        
        # Find header end marker
        header_text = ""
        idx = 0
        while True:
            try:
                header_text += chr(file_bytes[idx])  # Python 3
            except (TypeError, IndexError):
                break
            idx += 1
            if ":HEADER_END:" in header_text:
                break
        
        # Parse header lines
        header_text = header_text.split("\r\n")[:-1]
        for entry in header_text:
            if "=" not in entry:
                continue
            entry_array = entry.split("=", 1)
            key = entry_array[0].strip()
            value = entry_array[1].strip() if len(entry_array) > 1 else ""
            if key == "Comment":
                # Comment may contain "=" so join the rest
                value = "=".join(entry_array[1:])
            header[key] = value
        
        # Data starts after ":HEADER_END:" marker (idx points to end of marker)
        data_start = idx + 2  # +2 for \r\n after :HEADER_END:
        
        return header, data_start
    
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
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if not self.file_path.suffix.lower() == ".3ds":
            raise ValueError(f"File must have .3ds extension: {self.file_path}")
        
        # Parse header
        self.header, data_start = self._parse_header()
        self.metadata = self.header.copy()
        
        # Extract grid dimensions (based on nanonis_load)
        temp = re.split(' |"', self.header.get("Grid dim", '"1 1"'))
        self.nx = int(temp[1])
        self.ny = int(temp[3])
        
        # Extract grid settings
        temp = re.split(";|=", self.header.get("Grid settings", "0;0;0;0;0"))
        self.header["x_center (nm)"] = float(temp[0]) * 1e9
        self.header["y_center (nm)"] = float(temp[1]) * 1e9
        self.header["x_size (nm)"] = float(temp[2]) * 1e9
        self.header["y_size (nm)"] = float(temp[3]) * 1e9
        self.header["angle"] = float(temp[4])
        
        # Extract number of parameters and points
        self.n_params = int(self.header.get("# Parameters (4 byte)", 13))
        self.n_points = int(self.header.get("Points", 241))
        
        # Extract channels
        channels_str = self.header.get("Channels", '"Current (A)"')
        channels = re.split('"|;', channels_str)[1:-1]
        self.channels = [ch for ch in channels if ch]  # Remove empty strings
        n_channels = len(self.channels)
        
        # Read binary data
        with open(self.file_path, "rb") as f:
            file_bytes = f.read()
        
        raw_data = file_bytes[data_start:]
        bpp = self.n_points * n_channels + self.n_params  # bytes per point
        data_pts = self.nx * self.ny * bpp
        
        # Read numerical data (big-endian float32)
        numerical_data = np.frombuffer(raw_data, dtype=">f")
        
        # Extract start and end bias from first two values
        if len(numerical_data) >= 2:
            self.header["Start Bias (V)"] = float(numerical_data[0])
            self.header["End Bias (V)"] = float(numerical_data[1])
        else:
            # Fallback to header values
            sweep_start = float(self.header.get("Bias Spectroscopy>Sweep Start (V)", -0.1))
            sweep_end = float(self.header.get("Bias Spectroscopy>Sweep End (V)", 0.1))
            self.header["Start Bias (V)"] = sweep_start
            self.header["End Bias (V)"] = sweep_end
        
        # Extract parameters
        self.parameters = {}
        self.parameter_list = []
        
        fixed_params = self.header.get("Fixed parameters", "").strip('"').split(";")
        exp_params = self.header.get("Experiment parameters", "").strip('"').split(";")
        
        for param_name in fixed_params:
            if param_name:
                self.parameters[param_name] = []
                self.parameter_list.append(param_name)
        for param_name in exp_params:
            if param_name:
                self.parameters[param_name] = []
                self.parameter_list.append(param_name)
        
        # Organize data into predata structure (as in nanonis_load)
        predata = [[{} for y in range(self.ny)] for x in range(self.nx)]
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(n_channels):
                    start_index = (
                        (i * self.ny + j) * bpp
                        + self.n_params
                        + k * self.n_points
                    )
                    end_index = start_index + self.n_points
                    
                    if end_index <= len(numerical_data):
                        predata[i][j][self.channels[k]] = numerical_data[start_index:end_index]
                        
                        # Extract parameters (only for first channel)
                        if k == 0:
                            for idx, param_name in enumerate(self.parameter_list):
                                param_idx = (
                                    (i * self.ny + j) * bpp
                                    + idx
                                )
                                if param_idx < len(numerical_data):
                                    self.parameters[param_name].append(numerical_data[param_idx])
                                else:
                                    self.parameters[param_name].append(0.0)
                    else:
                        predata[i][j][self.channels[k]] = np.zeros(self.n_points)
                        if k == 0:
                            for param_name in self.parameters:
                                self.parameters[param_name].append(0.0)
        
        # Create bias array
        self.bias_voltage = np.linspace(
            self.header["Start Bias (V)"],
            self.header["End Bias (V)"],
            self.n_points
        )
        
        # Organize data into dictionary (as in nanonis_load)
        self.data = {}
        for channel in self.channels:
            self.data[channel] = np.array(
                [
                    [predata[x][y][channel] for y in range(self.ny)]
                    for x in range(self.nx)
                ]
            )
        
        # Store spectroscopy data in 3D format [x, y, bias] for first channel
        if self.channels:
            self.spectroscopy_data = self.data[self.channels[0]]
        
        # Extract positions from parameters if available
        if "X (m)" in self.parameters and "Y (m)" in self.parameters:
            self.x_positions = np.array(self.parameters["X (m)"]) * 1e9  # Convert to nm
            self.y_positions = np.array(self.parameters["Y (m)"]) * 1e9  # Convert to nm
            # Reshape to match grid
            if len(self.x_positions) == self.nx * self.ny:
                self.x_positions = self.x_positions.reshape(self.nx, self.ny)
                self.y_positions = self.y_positions.reshape(self.nx, self.ny)
        else:
            # Create grid positions
            x_center = self.header.get("x_center (nm)", 0)
            y_center = self.header.get("y_center (nm)", 0)
            x_size = self.header.get("x_size (nm)", self.nx * 1.0)
            y_size = self.header.get("y_size (nm)", self.ny * 1.0)
            
            x_pos = np.linspace(x_center - x_size/2, x_center + x_size/2, self.nx)
            y_pos = np.linspace(y_center - y_size/2, y_center + y_size/2, self.ny)
            self.x_positions, self.y_positions = np.meshgrid(x_pos, y_pos, indexing='ij')
        
        self.raw_data = {
            'data': self.data,  # Dictionary of all channels
            'spectroscopy_data': self.spectroscopy_data,  # First channel as 3D array
            'bias_voltage': self.bias_voltage,
            'x_positions': self.x_positions,
            'y_positions': self.y_positions,
            'metadata': self.metadata,
            'channels': self.channels,
            'parameters': self.parameters
        }
        
        return self.raw_data
    
    def get_channel(self, channel_name: str) -> Optional[np.ndarray]:
        """
        Get data for a specific channel.
        
        Parameters
        ----------
        channel_name : str
            Name of the channel
            
        Returns
        -------
        np.ndarray or None
            3D array [x, y, bias] for the channel
        """
        return self.data.get(channel_name)
    
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
    
    def get_all_data(self) -> List[np.ndarray]:
        """
        Get all spectroscopy data arrays from the collection.
        
        Returns
        -------
        List[np.ndarray]
            List of 3D data arrays
        """
        return [file.spectroscopy_data for file in self.files if file.spectroscopy_data is not None]

