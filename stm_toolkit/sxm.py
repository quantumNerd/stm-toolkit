"""
SXM file handling for Nanonis 2D image data.

This module provides classes for loading and processing .sxm files,
including background subtraction and FFT analysis.

Based on nanonis_load implementation:
https://github.com/dilwong/nanonis_load/blob/master/nanonis_load/sxm.py
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import scipy.signal
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from matplotlib import colors
from .base import BaseFile, BaseFileCollection
from .plotting import BasePlotter


class SXMFile(BaseFile):
    """
    Handler for individual .sxm files.
    
    Supports loading 2D images and processing operations like:
    - Background subtraction
    - FFT analysis
    """
    
    def __init__(self, file_path: str | Path,
                 gate_voltage_key: Optional[str] = None,
                 bias_voltage_key: Optional[str] = None,
                 source_drain_voltage_key: Optional[str] = None,
                 multipass_config_index: int = -2,
                 multipass_bias_index: int = -4):
        """
        Initialize SXM file handler.
        
        Parameters
        ----------
        file_path : str | Path
            Path to the .sxm file
        gate_voltage_key : Optional[str]
            Header key for gate voltage. If None, tries multiple defaults:
            - ":Outputs>Output 3 Value:" (machine-specific)
            - ":Ext. VI 1>Gate voltage (V):" (fallback)
            - From comment if available
        bias_voltage_key : Optional[str]
            Header key for bias voltage. If None, tries multiple defaults:
            - ":Multipass-Config:" (machine-specific, uses multipass_config_index and multipass_bias_index)
            - ":BIAS:" (fallback)
        source_drain_voltage_key : Optional[str]
            Header key for source-drain voltage. If None, uses:
            - ":Outputs>Output 2 Value:" (machine-specific)
        multipass_config_index : int
            Index in ":Multipass-Config:" list to use for bias extraction. Default: -2
        multipass_bias_index : int
            Index in the tab-split line from multipass_config_index. Default: -4
        """
        super().__init__(file_path)
        # Original data storage
        self.image_data: Optional[np.ndarray] = None  # Raw image data
        self.data: Dict[str, List[np.ndarray]] = {}  # Channel data: {channel_name: [forward, backward]}
        self.header: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}  # Store metadata separately
        
        # Only raw data is stored - processing is done on-the-fly when needed
        
        # Image properties
        self.x_pixels: Optional[int] = None
        self.y_pixels: Optional[int] = None
        self.x_range: Optional[float] = None  # in nm
        self.y_range: Optional[float] = None  # in nm
        
        # Store configuration for voltage extraction
        self.gate_voltage_key = gate_voltage_key
        self.bias_voltage_key = bias_voltage_key
        self.source_drain_voltage_key = source_drain_voltage_key
        self.multipass_config_index = multipass_config_index
        self.multipass_bias_index = multipass_bias_index
        
    def _parse_header(self, file_bytes: bytes) -> Tuple[Dict[str, Any], int]:
        """
        Parse the header section of an .sxm file.
        
        Parameters
        ----------
        file_bytes : bytes
            The file content as bytes
            
        Returns
        -------
        tuple[Dict[str, Any], int]
            Header dictionary and index where header ends
        """
        header = {}
        header_text = ""
        idx = 0
        
        # Read header until SCANIT_END
        while idx < len(file_bytes):
            try:
                header_text += chr(file_bytes[idx])
            except (ValueError, TypeError):
                # Handle binary data that can't be decoded
                break
            idx += 1
            if ":SCANIT_END:" in header_text:
                break
        
        # Parse header lines
        header_lines = header_text.split("\n")
        prev_header = None
        
        for line in header_lines:
            line = line.strip()
            if line.startswith(":") and line.endswith(":"):
                prev_header = line
                header[line] = []
            elif prev_header is not None:
                header[prev_header].append(line)
        
        # Extract common header fields
        if ":SCAN_PIXELS:" in header:
            temp = header[":SCAN_PIXELS:"][0].strip().split()
            header["x_pixels"] = int(temp[0])
            header["y_pixels"] = int(temp[1])
            self.x_pixels = header["x_pixels"]
            self.y_pixels = header["y_pixels"]
        
        if ":SCAN_RANGE:" in header:
            temp = header[":SCAN_RANGE:"][0].strip().split()
            header["x_range (nm)"] = float(temp[0]) * 1e9
            header["y_range (nm)"] = float(temp[1]) * 1e9
            self.x_range = header["x_range (nm)"]
            self.y_range = header["y_range (nm)"]
        
        if ":SCAN_OFFSET:" in header:
            temp = header[":SCAN_OFFSET:"][0].strip().split()
            header["x_center (nm)"] = float(temp[0]) * 1e9
            header["y_center (nm)"] = float(temp[1]) * 1e9
        
        if ":SCAN_ANGLE:" in header:
            temp = header[":SCAN_ANGLE:"][0].strip().split()
            header["angle"] = float(temp[0])
        
        if ":SCAN_DIR:" in header:
            header["direction"] = header[":SCAN_DIR:"][0]
        
        if ":DATA_INFO:" in header:
            # DATA_INFO format: Channel\tName\tUnit\tDirection\tCalibration\tOffset
            # After strip(), the leading tab is removed, so indices are:
            # 0: Channel number, 1: Name, 2: Unit, 3: Direction
            temp = [chnls.split("\t") for chnls in header[":DATA_INFO:"][1:-1]]
            header["channels"] = [
                chnls[1].replace("_", " ") + " (" + chnls[2] + ")" for chnls in temp
            ]
        
        # Extract gate voltage (configurable, machine-specific)
        header["gate_voltage"] = 0.0
        gate_voltage_found = False
        
        # Try user-specified key first
        if self.gate_voltage_key and self.gate_voltage_key in header:
            try:
                header["gate_voltage"] = float(header[self.gate_voltage_key][0])
                gate_voltage_found = True
            except (ValueError, IndexError):
                pass
        
        # Try default machine-specific key: ":Outputs>Output 3 Value:"
        if not gate_voltage_found:
            try:
                header["gate_voltage"] = float(header[":Outputs>Output 3 Value:"][0])
                gate_voltage_found = True
            except (KeyError, ValueError, IndexError):
                pass
        
        # Try fallback: ":Ext. VI 1>Gate voltage (V):"
        if not gate_voltage_found:
            try:
                header["gate_voltage"] = float(header[":Ext. VI 1>Gate voltage (V):"][0])
                gate_voltage_found = True
            except (KeyError, ValueError, IndexError):
                pass
        
        # Try extracting from comment as last resort
        if not gate_voltage_found:
            try:
                if ":COMMENT:" in header and len(header[":COMMENT:"]) > 0:
                    split_comment = header[":COMMENT:"][0].split()
                    if "V_g" in split_comment:
                        header["gate_voltage"] = float(split_comment[split_comment.index("V_g") + 2])
                        gate_voltage_found = True
            except (ValueError, IndexError):
                pass
        
        # Extract bias voltage (configurable, machine-specific)
        header["bias"] = 0.0
        bias_voltage_found = False
        
        # Try user-specified key first
        if self.bias_voltage_key and self.bias_voltage_key in header:
            try:
                header["bias"] = float(header[self.bias_voltage_key][0])
                bias_voltage_found = True
            except (ValueError, IndexError):
                pass
        
        # Try default machine-specific method: ":Multipass-Config:"
        if not bias_voltage_found:
            try:
                multipass_config = header[":Multipass-Config:"]
                if len(multipass_config) > abs(self.multipass_config_index):
                    config_line = multipass_config[self.multipass_config_index]
                    config_parts = config_line.split("\t")
                    if len(config_parts) > abs(self.multipass_bias_index):
                        header["bias"] = float(config_parts[self.multipass_bias_index])
                        bias_voltage_found = True
            except (KeyError, ValueError, IndexError):
                pass
        
        # Try fallback: ":BIAS:"
        if not bias_voltage_found:
            try:
                header["bias"] = float(header[":BIAS:"][0])
                bias_voltage_found = True
            except (KeyError, ValueError, IndexError):
                pass
        
        # Extract source-drain voltage (configurable, machine-specific)
        header["source_drain_voltage"] = 0.0
        
        # Try user-specified key first
        if self.source_drain_voltage_key and self.source_drain_voltage_key in header:
            try:
                header["source_drain_voltage"] = float(header[self.source_drain_voltage_key][0])
            except (ValueError, IndexError):
                pass
        
        # Try default machine-specific key: ":Outputs>Output 2 Value:"
        if header["source_drain_voltage"] == 0.0:
            try:
                header["source_drain_voltage"] = float(header[":Outputs>Output 2 Value:"][0])
            except (KeyError, ValueError, IndexError):
                pass
        
        # Extract Z-controller info
        if ":Z-CONTROLLER:" in header and len(header[":Z-CONTROLLER:"]) > 1:
            z_ctrl = header[":Z-CONTROLLER:"][1].split("\t")
            if len(z_ctrl) > 3:
                header["setpoint_current"] = float(z_ctrl[3].split()[0]) * 1e12  # Convert to pA
                if len(z_ctrl) > 4:
                    header["p_gain"] = float(z_ctrl[4].split()[0]) * 1e12  # Convert to pm
        
        return header, idx
    
    def load(self) -> Dict[str, Any]:
        """
        Load raw data from .sxm file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'data': Dictionary of channel data {channel_name: [forward, backward]}
            - 'header': Dictionary of file metadata
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if not self.file_path.suffix.lower() == ".sxm":
            raise ValueError(f"File must have .sxm extension: {self.file_path}")
        
        # Read file as binary
        with open(self.file_path, "rb") as f:
            file_bytes = f.read()
        
        # Parse header
        self.header, header_end_idx = self._parse_header(file_bytes)
        self.metadata = self.header.copy()
        
        # Find the actual start of binary data (after ":SCANIT_END:")
        data_start = file_bytes.find(b":SCANIT_END:")
        if data_start == -1:
            raise ValueError("Could not find :SCANIT_END: marker in file")
        
        # Skip ":SCANIT_END:" and any whitespace/newlines (typically 5 bytes)
        data_start = data_start + len(":SCANIT_END:") + 5
        raw_data = file_bytes[data_start:]
        
        # Parse binary data
        size = self.header["x_pixels"] * self.header["y_pixels"]
        raw_data_array = np.frombuffer(raw_data, dtype=">f")  # Big-endian float32
        
        num_channels = len(self.header["channels"])
        direction = self.header.get("direction", "up")
        
        # Load data for each channel
        for idx, channel_name in enumerate(self.header["channels"]):
            channel_data = raw_data_array[idx * size * 2 : (idx + 1) * size * 2]
            
            if direction == "down":
                # Forward pass
                forward = np.nan_to_num(
                    np.flipud(
                        channel_data[0:size].reshape(
                            self.header["y_pixels"], self.header["x_pixels"]
                        )
                    )
                )
                # Backward pass
                backward = np.nan_to_num(
                    np.flip(
                        channel_data[size : 2 * size].reshape(
                            self.header["y_pixels"], self.header["x_pixels"]
                        ),
                        axis=(0, 1),
                    )
                )
            else:  # direction == "up"
                # Forward pass
                forward = np.nan_to_num(
                    channel_data[0:size].reshape(
                        self.header["y_pixels"], self.header["x_pixels"]
                    )
                )
                # Backward pass
                backward = np.nan_to_num(
                    np.fliplr(
                        channel_data[size : 2 * size].reshape(
                            self.header["y_pixels"], self.header["x_pixels"]
                        )
                    )
                )
            
            self.data[channel_name] = [forward, backward]
        
        # Set default image_data to first channel, forward direction
        if len(self.data) > 0:
            first_channel = list(self.data.keys())[0]
            self.image_data = self.data[first_channel][0]
        
        self.raw_data = {
            'data': self.data,
            'header': self.header,
            'channels': self.header["channels"]
        }
        
        return self.raw_data
    
    def process(self, background_subtract: bool = False, fft: bool = False, 
                channel: Optional[str] = None, direction: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Process the loaded image data and return results (no storage).
        
        This method computes processing on-the-fly and returns results without storing them.
        For plotting, processing is done automatically when needed.
        
        Parameters
        ----------
        background_subtract : bool
            Whether to perform background subtraction
        fft : bool
            Whether to compute FFT
        channel : Optional[str]
            Channel name to process. If None, uses default image_data
        direction : int
            Direction (0=forward, 1=backward)
        **kwargs
            Additional processing parameters:
            - background_method: Method for background subtraction ('polynomial', 'plane', 'line')
            - background_order: Order for polynomial background (if applicable)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed data (not stored in class)
        """
        if self.raw_data is None:
            raise ValueError("Data must be loaded before processing. Call load() first.")
        
        # Get the image to process
        if channel is not None and channel in self.data:
            image_to_process = self.data[channel][direction]
        else:
            image_to_process = self.image_data
        
        if image_to_process is None:
            raise ValueError("No image data available for processing.")
        
        processed = {}
        
        # Background subtraction
        if background_subtract:
            # Extract background_method from kwargs to avoid duplicate argument
            # Make a copy of kwargs to avoid modifying the original
            bg_kwargs = kwargs.copy()
            background_method = bg_kwargs.pop('background_method', 'plane')
            processed_img = self._subtract_background(
                background_method=background_method,
                channel=channel,
                direction=direction,
                **bg_kwargs
            )
            processed['image'] = processed_img
        else:
            processed['image'] = image_to_process.copy()
        
        # FFT analysis
        if fft:
            if channel is not None:
                # Compute FFT for specified channel
                fft_result = self.compute_fft(image=processed.get('image'), channel=channel, 
                                             direction=direction, **kwargs)
            else:
                # Compute FFT for all channels
                fft_results = self.compute_fft_all_channels(direction=direction, 
                                                           image=processed.get('image'), **kwargs)
                fft_result = fft_results
            processed['fft'] = fft_result
        
        return processed
    
    def compute_fft(self, image: Optional[np.ndarray] = None, channel: Optional[str] = None,
                   direction: int = 0, window_function: Optional[str] = None,
                   background_subtract: bool = False, background_method: str = 'plane',
                   **kwargs) -> Dict[str, Any]:
        """
        Compute 2D FFT of the image and return results (no storage).
        
        This function computes FFT on-the-fly without storing results in the class.
        Useful for analysis without plotting.
        
        Parameters
        ----------
        image : Optional[np.ndarray]
            Image to compute FFT for. If None, uses raw channel data or image_data.
        channel : Optional[str]
            Channel name to use. If None, uses default image_data.
        direction : int
            Direction (0=forward, 1=backward) if using channel data
        window_function : Optional[str]
            Window function name ('blackman', etc.)
        background_subtract : bool
            If True, subtract background before computing FFT
        background_method : str
            Background subtraction method ('plane', 'line') if background_subtract=True
        **kwargs
            Additional parameters for background subtraction
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'fft_magnitude': Magnitude of FFT
            - 'FX': Frequency meshgrid for x-axis (nm⁻¹, 2D array)
            - 'FY': Frequency meshgrid for y-axis (nm⁻¹, 2D array)
            - 'ft': Full FFT array (complex)
            - 'channel': Channel name used
            - 'direction': Direction used
        """
        from .analysis import fft2d
        
        # Determine which image to use
        if image is None:
            if channel is not None and channel in self.data:
                image = self.data[channel][direction].copy()
            elif self.image_data is not None:
                image = self.image_data.copy()
            else:
                raise ValueError("No image data available for FFT computation.")
        else:
            image = image.copy()
        
        # Apply background subtraction if requested
        if background_subtract:
            if background_method == 'plane':
                image = self.subtract_plane(image=image, channel=channel, direction=direction)
            elif background_method == 'line':
                image = self.subtract_linear_by_line(image=image, channel=channel, direction=direction)
        
        # Apply windowing if requested
        image_for_fft = image.copy()
        if window_function and window_function.lower() == "blackman":
            window = np.outer(
                np.blackman(image_for_fft.shape[0]),
                np.blackman(image_for_fft.shape[1])
            )
            image_for_fft = np.fft.fftshift(window * np.fft.fftshift(image_for_fft))
        
        # Calculate pixel spacing for frequency conversion
        x_range = self.x_range or 100.0
        y_range = self.y_range or 100.0
        x_pixels = self.x_pixels or image_for_fft.shape[1]
        y_pixels = self.y_pixels or image_for_fft.shape[0]
        
        sxm_x = np.linspace(-x_range / 2, x_range / 2, x_pixels)
        sxm_y = np.linspace(-y_range / 2, y_range / 2, y_pixels)
        dx = sxm_x[1] - sxm_x[0]  # nm
        dy = sxm_y[1] - sxm_y[0]  # nm
        
        # Compute FFT using analysis module (FX, FY will be in nm⁻¹, already 2D meshgrids)
        ft_magnitude, FX, FY = fft2d(image_for_fft, dx=dx, dy=dy, shift=True)
        
        # Return results (no storage)
        channel_key = channel if channel is not None else 'default'
        return {
            'fft_magnitude': ft_magnitude,
            'FX': FX,  # 2D meshgrid
            'FY': FY,  # 2D meshgrid
            'ft': np.fft.fftshift(np.fft.fft2(image_for_fft)),
            'channel': channel_key,
            'direction': direction,
            'window_function': window_function
        }
    
    def compute_fft_all_channels(self, direction: int = 0, window_function: Optional[str] = None,
                                background_subtract: bool = False, background_method: str = 'plane',
                                **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Compute 2D FFT for all available channels and return results (no storage).
        
        Parameters
        ----------
        direction : int
            Direction (0=forward, 1=backward) to use for all channels
        window_function : Optional[str]
            Window function name ('blackman', etc.)
        background_subtract : bool
            If True, subtract background before computing FFT
        background_method : str
            Background subtraction method ('plane', 'line') if background_subtract=True
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of FFT results per channel: {channel_name: {FX, FY, ...}}
        """
        if not self.data:
            # No channels available, compute for default image
            return {'default': self.compute_fft(
                channel=None, direction=direction, 
                window_function=window_function,
                background_subtract=background_subtract,
                background_method=background_method,
                **kwargs
            )}
        
        # Compute FFT for each channel
        results = {}
        for channel_name in self.data.keys():
            results[channel_name] = self.compute_fft(
                channel=channel_name, direction=direction,
                window_function=window_function,
                background_subtract=background_subtract,
                background_method=background_method,
                **kwargs
            )
        
        return results
    
    def _subtract_background(self, background_method: str = 'plane', background_order: int = 1, **kwargs) -> np.ndarray:
        """
        Subtract background from image.
        
        Parameters
        ----------
        background_method : str
            Method for background subtraction ('plane', 'polynomial', 'line')
        background_order : int
            Order for polynomial background (if method is 'polynomial')
            
        Returns
        -------
        np.ndarray
            Image with background subtracted
        """
        if self.image_data is None:
            raise ValueError("Image data not loaded.")
        
        image = self.image_data.copy()
        
        if background_method == 'plane':
            # Fit and subtract a 2D plane
            return self.subtract_plane(image, **kwargs)
        elif background_method == 'line':
            # Subtract linear fit from each line
            return self.subtract_linear_by_line(image, **kwargs)
        elif background_method == 'polynomial':
            # TODO: Implement polynomial background subtraction
            raise NotImplementedError("Polynomial background subtraction not yet implemented.")
        else:
            raise ValueError(f"Unknown background method: {background_method}")
    
    def subtract_plane(self, image: Optional[np.ndarray] = None, channel: Optional[str] = None, direction: int = 0) -> np.ndarray:
        """
        Fit and subtract a 2D plane from the image.
        
        Parameters
        ----------
        image : Optional[np.ndarray]
            Image to process. If None, uses self.image_data
        channel : Optional[str]
            Channel name. If provided, uses channel data instead of image
        direction : int
            Direction (0=forward, 1=backward). Used if channel is provided
            
        Returns
        -------
        np.ndarray
            Image with plane subtracted
        """
        if image is None:
            if channel is not None and channel in self.data:
                image = self.data[channel][direction]
            else:
                image = self.image_data
        
        if image is None:
            raise ValueError("No image data available.")
        
        # Create coordinate grids
        y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        
        # Flatten arrays for fitting
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = image.flatten()
        
        # Fit plane: z = a*x + b*y + c
        A = np.vstack([x_flat, y_flat, np.ones(len(x_flat))]).T
        coeffs = np.linalg.lstsq(A, z_flat, rcond=None)[0]
        
        # Calculate plane
        plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        
        return image - plane
    
    def subtract_linear_by_line(self, image: Optional[np.ndarray] = None, channel: Optional[str] = None, direction: int = 0) -> np.ndarray:
        """
        Subtract a linear fit from each fast-scan line.
        
        Parameters
        ----------
        image : Optional[np.ndarray]
            Image to process. If None, uses self.image_data
        channel : Optional[str]
            Channel name. If provided, uses channel data instead of image
        direction : int
            Direction (0=forward, 1=backward). Used if channel is provided
            
        Returns
        -------
        np.ndarray
            Image with line-by-line background subtracted
        """
        if image is None:
            if channel is not None and channel in self.data:
                image = self.data[channel][direction]
            else:
                image = self.image_data
        
        if image is None:
            raise ValueError("No image data available.")
        
        # Use scipy.signal.detrend which does line-by-line detrending
        return scipy.signal.detrend(image, axis=1)
    
    
    def get_image(self, processed: bool = False, channel: Optional[str] = None, direction: int = 0) -> Optional[np.ndarray]:
        """
        Get the image data.
        
        Parameters
        ----------
        processed : bool
            If True, return processed image; if False, return raw image
        channel : Optional[str]
            Channel name. If None, uses default image_data
        direction : int
            Direction (0=forward, 1=backward)
            
        Returns
        -------
        np.ndarray or None
            Image data
        """
        if processed:
            return self.processed_image
        
        if channel is not None and channel in self.data:
            return self.data[channel][direction]
        
        return self.image_data
    
    def get_channel_names(self) -> List[str]:
        """Get list of available channel names."""
        return list(self.data.keys())
    
    def get_gate_voltage(self) -> float:
        """
        Get gate voltage in V.
        
        Returns
        -------
        float
            Gate voltage in volts
        """
        return self.header.get("gate_voltage", 0.0)
    
    def get_bias(self) -> float:
        """
        Get sample bias voltage in V.
        
        Returns
        -------
        float
            Bias voltage in volts
        """
        return self.header.get("bias", 0.0)
    
    def get_source_drain_voltage(self) -> float:
        """
        Get source-drain voltage in V.
        
        Returns
        -------
        float
            Source-drain voltage in volts
        """
        return self.header.get("source_drain_voltage", 0.0)
    
    def plot(self, channel: Optional[str] = None, direction: int = 0, 
             data_source: str = 'raw', ax: Optional[plt.Axes] = None, **kwargs) -> 'SXMPlotter':
        """
        Create a plotter for this SXM file.
        
        Parameters
        ----------
        channel : Optional[str]
            Channel name to plot
        direction : int
            Direction (0=forward, 1=backward)
        data_source : str
            What to plot: 'raw' (original data), 'processed' (background-subtracted),
            or 'fft' (FFT results). Default: 'raw'
        ax : Optional[matplotlib.axes.Axes]
            Axes to plot on. If None, creates a new figure.
        **kwargs
            Additional plotting parameters (e.g., figsize, cmap, etc.)
            
        Returns
        -------
        SXMPlotter
            Plotter instance
        """
        return SXMPlotter(self, channel=channel, direction=direction, data_source=data_source, ax=ax, **kwargs)


class SXMPlotter(BasePlotter):
    """
    Plotter for SXM file data.
    
    Provides methods for plotting 2D images and FFT analysis.
    """
    
    def __init__(self, sxm_file: SXMFile, channel: Optional[str] = None, 
                 direction: int = 0, data_source: str = 'raw',
                 flatten: bool = False, subtract_plane: bool = False,
                 subtract_line: bool = False, cmap: Union[str, Any] = 'Blues_r',
                 filter_current: Optional[int] = None, map_color_std: Optional[float] = None,
                 ax: Optional[plt.Axes] = None, figsize: Optional[Tuple[float, float]] = None, **kwargs):
        """
        Initialize SXM plotter.
        
        Parameters
        ----------
        sxm_file : SXMFile
            The SXM file to plot
        channel : Optional[str]
            Channel name to plot
        direction : int
            Direction (0=forward, 1=backward)
        processed : bool
            Whether to plot processed data
        flatten : bool
            If True, subtract linear fit from every fast-scan line
        subtract_plane : bool
            If True, fit and subtract a 2D plane
        subtract_line : bool
            If True, subtract line-by-line background (same as flatten)
        cmap : str or matplotlib.colors.Colormap
            Colormap name or colormap object. Default is 'Blues_r' (reversed Blues).
            Common options: 'Blues_r', 'RdYlBu_r', 'viridis', 'plasma', 'inferno', etc.
        filter_current : int
            Savitzky-Golay filter window size (0 = no filtering)
        map_color_std : Optional[float]
            Color mapping standard deviation multiplier (None = no clipping)
        **kwargs
            Additional plotting parameters
        """
        self.sxm_file = sxm_file
        self.channel = channel
        self.direction = direction
        self.data_source = data_source  # 'raw', 'processed', or 'fft'
        self.flatten = flatten or subtract_line
        self.subtract_plane = subtract_plane
        # Convert string colormap names to actual colormap objects
        if isinstance(cmap, str):
            try:
                self.cmap = plt.get_cmap(cmap)
            except ValueError:
                # Fallback to default if colormap not found
                self.cmap = cm.Blues_r
        else:
            self.cmap = cmap
        self.filter_current = filter_current
        self.map_color_std = map_color_std
        super().__init__(sxm_file, ax=ax, figsize=figsize, **kwargs)
    
    def _setup_plot(self, **kwargs) -> None:
        """Set up the initial plot."""
        # If FFT plotting, skip the rest (handled in fft() method)
        if self.data_source == 'fft':
            # Create figure if needed
            if self.ax is None:
                figsize = self._get_figsize(self.DEFAULT_FIGSIZE_2D, **kwargs)
                self.fig = plt.figure(figsize=figsize)
                self.ax = self.fig.add_subplot(111)
            return
        
        # Get raw image data
        image_data = self.sxm_file.get_image(
            processed=False,
            channel=self.channel,
            direction=self.direction
        )
        
        if image_data is None:
            raise ValueError("No image data available to plot.")
        
        # Process on-the-fly based on data_source
        if self.data_source == 'processed':
            # Apply background subtraction on-the-fly
            if self.subtract_plane:
                image_data = self.sxm_file.subtract_plane(
                    image=image_data,
                    channel=self.channel,
                    direction=self.direction
                )
            elif self.flatten:
                image_data = self.sxm_file.subtract_linear_by_line(
                    image=image_data,
                    channel=self.channel,
                    direction=self.direction
                )
            else:
                # Default: subtract plane
                image_data = self.sxm_file.subtract_plane(
                    image=image_data,
                    channel=self.channel,
                    direction=self.direction
                )
        
        # Apply filtering if requested (Savitzky-Golay filter, as in notebook)
        if self.filter_current is not None and self.filter_current > 0:
            image_data = savgol_filter(image_data, self.filter_current, 1, axis=-1)
        
        # Create figure if axes not provided
        if self.ax is None:
            figsize = self._get_figsize(self.DEFAULT_FIGSIZE_2D, **kwargs)
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111)
        
        # Handle NaN values
        avg_dat = image_data[~np.isnan(image_data)].mean()
        image_data[np.isnan(image_data)] = avg_dat
        
        # Apply color mapping normalization (as in notebook)
        if self.map_color_std is not None and self.map_color_std > 0:
            vcenter = np.median(image_data)
            it_std = np.std(image_data)
            image_data = np.clip(
                image_data,
                a_min=vcenter - it_std * self.map_color_std,
                a_max=vcenter + it_std * self.map_color_std
            )
        
        # Only create figure if axes doesn't exist (not provided via plot_on_axes)
        if self.ax is None:
            # Create figure using unified figure size from BasePlotter
            figsize = self._get_figsize(self.DEFAULT_FIGSIZE_2D, **kwargs)
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111)
        
        # Get spatial ranges and create coordinate arrays (centered at 0)
        x_range = self.sxm_file.x_range or 100.0  # Default to 100 nm
        y_range = self.sxm_file.y_range or 100.0
        x_pixels = self.sxm_file.x_pixels or image_data.shape[1]
        y_pixels = self.sxm_file.y_pixels or image_data.shape[0]
        
        # Create coordinate arrays centered at 0 (like in the notebook)
        sxm_x = np.linspace(-x_range / 2, x_range / 2, x_pixels)
        sxm_y = np.linspace(-y_range / 2, y_range / 2, y_pixels)
        
        # Use pcolormesh for better coordinate handling (as in notebook)
        X, Y = np.meshgrid(sxm_x, sxm_y)
        self.im_plot = self.ax.pcolormesh(
            X, Y, image_data,
            cmap=self.cmap,
            rasterized=kwargs.get('rasterized', True),
            shading='nearest'
        )
        
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (nm)")
        self.ax.set_ylabel("Y (nm)")
        
        # Add colorbar
        self.fig.colorbar(self.im_plot, ax=self.ax)
        
        # Store image data for FFT
        self.image_data = image_data
        
        # Set title
        channel_name = self.channel or "Default"
        direction_str = "Forward" if self.direction == 0 else "Backward"
        title = f"{channel_name} ({direction_str})"
        vg = self.sxm_file.get_gate_voltage()
        vb = self.sxm_file.get_bias()
        vd = self.sxm_file.get_source_drain_voltage()
        if vg != 0.0 or vb != 0.0 or vd != 0.0:
            voltage_parts = []
            if vg != 0.0:
                voltage_parts.append(f"Vg = {vg:.3f} V")
            if vb != 0.0:
                voltage_parts.append(f"Vb = {vb:.3f} V")
            if vd != 0.0:
                voltage_parts.append(f"Vd = {vd:.3f} V")
            if voltage_parts:
                title += " | " + ", ".join(voltage_parts)
        self.ax.set_title(title)
    
    def xlim(self, x_min: float, x_max: float) -> None:
        """Set x-axis limits."""
        if self.ax is not None:
            self.ax.set_xlim(x_min, x_max)
    
    def ylim(self, y_min: float, y_max: float) -> None:
        """Set y-axis limits."""
        if self.ax is not None:
            self.ax.set_ylim(y_min, y_max)
    
    def clim(self, c_min: float, c_max: float) -> None:
        """Set color axis limits."""
        if self.im_plot is not None:
            self.im_plot.set_clim(c_min, c_max)
    
    def colormap(self, cmap: str) -> None:
        """Change the colormap."""
        if self.im_plot is not None:
            self.im_plot.set_cmap(cmap)
    
    def fft(self, k_circle: float = -1, k_range: float = -1, 
            show_radial: bool = False, enhance_peaks: bool = False,
            fft_cmap: Optional[Union[str, Any]] = None, **kwargs) -> None:
        """
        Plot the Fourier transform using stored FFT results.
        
        This method creates the FFT plot when called.
        
        Parameters
        ----------
        k_circle : float
            Radius of circle to overlay on FFT (in nm⁻¹). If <= 0, no circle is drawn.
        k_range : float
            Range for kx and ky axes (in nm⁻¹). If <= 0, auto-determined.
        show_radial : bool
            If True, also plot radial distribution function
        enhance_peaks : bool
            If True, plot sqrt of FFT magnitude to enhance peaks. Default is False (plot magnitude).
        fft_cmap : Optional[str or matplotlib.colors.Colormap]
            Colormap for FFT plot. If None, uses 'Blues_r' (default). 
            Common options: 'Blues_r', 'viridis', 'plasma', 'inferno', etc.
        **kwargs
            Additional plotting parameters (e.g., figsize)
        """
        # Ensure main plot is created first (for data_source='fft', this will be skipped)
        if self.data_source != 'fft':
            self._ensure_plot()
        
        # Compute FFT on-the-fly (no storage)
        # Get image to compute FFT from
        if self.data_source == 'processed':
            # Get processed image
            image = self.sxm_file.get_image(processed=False, channel=self.channel, direction=self.direction)
            if self.subtract_plane:
                image = self.sxm_file.subtract_plane(image=image, channel=self.channel, direction=self.direction)
            elif self.flatten:
                image = self.sxm_file.subtract_linear_by_line(image=image, channel=self.channel, direction=self.direction)
            else:
                image = self.sxm_file.subtract_plane(image=image, channel=self.channel, direction=self.direction)
        else:
            # Use raw image
            image = self.sxm_file.get_image(processed=False, channel=self.channel, direction=self.direction)
        
        # Compute FFT on-the-fly
        fft_result = self.sxm_file.compute_fft(
            image=image,
            channel=self.channel,
            direction=self.direction,
            window_function=kwargs.get('window_function', None),
            background_subtract=False  # Already processed if needed
        )
        
        from .analysis import radial_distribution
        
        ft_magnitude = fft_result['fft_magnitude']
        FX = fft_result['FX']  # Already 2D meshgrid
        FY = fft_result['FY']  # Already 2D meshgrid
        
        # Apply sqrt enhancement if requested (as in notebook)
        if enhance_peaks:
            ft_plot = np.sqrt(ft_magnitude)
        else:
            ft_plot = ft_magnitude
        
        max_fft = np.max(ft_plot[1:-1, 1:-1])
        
        # FX and FY are already in nm⁻¹ from fft2d (with dx, dy provided) and are 2D meshgrids
        
        # Get actual frequency ranges
        FX_max = np.abs(FX).max()
        FY_max = np.abs(FY).max()
        
        # Get spatial ranges for determining k_range
        x_range = self.sxm_file.x_range or 100.0
        y_range = self.sxm_file.y_range or 100.0
        
        # Determine k_range
        if k_range <= 0:
            # Auto-determine range
            if abs(x_range - y_range) < 1e-6:  # Ranges are equal - use smaller for square plot
                # Use the smaller frequency range to ensure square plot
                # This makes sense because the lower resolution direction limits what we can see
                k_range = min(FX_max, FY_max)
            else:
                # Different ranges - use smaller of the two
                k_range = min(FX_max, FY_max)
        
        # For square plots when ranges are equal, ensure we use the same range for both axes
        if abs(x_range - y_range) < 1e-6:  # Ranges are equal
            # Use the determined k_range (which should be the smaller frequency)
            pass  # k_range already set above
        
        # Create figure(s) using unified figure sizes
        if show_radial:
            figsize = self._get_figsize(self.DEFAULT_FIGSIZE_MULTI, **kwargs)
            self.fft_fig, axes = plt.subplots(1, 2, figsize=figsize)
            self.fft_ax = axes[0]
            self.radial_ax = axes[1]
        else:
            figsize = self._get_figsize(self.DEFAULT_FIGSIZE_FFT, **kwargs)
            self.fft_fig = plt.figure(figsize=figsize)
            self.fft_ax = self.fft_fig.add_subplot(111)
            self.radial_ax = None
        
        # FX and FY are already 2D meshgrids, use them directly for plotting
        
        # Determine FFT colormap (use fft_cmap if provided, otherwise use default Blues_r)
        if fft_cmap is None:
            fft_cmap = cm.Blues_r
        elif isinstance(fft_cmap, str):
            try:
                fft_cmap = plt.get_cmap(fft_cmap)
            except ValueError:
                fft_cmap = cm.Blues_r
        
        # Plot FFT (using enhanced version if requested)
        self.fft_plot = self.fft_ax.pcolormesh(
            FX, FY, ft_plot,
            shading='auto',
            cmap=fft_cmap
        )
        
        # Add k-circle if requested
        if k_circle > 0:
            circle = Circle((0, 0), radius=k_circle, edgecolor='k', 
                          facecolor='none', ls='--', linewidth=1.5)
            self.fft_ax.add_patch(circle)
        
        self.fft_ax.set_xlabel("kx (nm⁻¹)")
        self.fft_ax.set_ylabel("ky (nm⁻¹)")
        # Update title to indicate sqrt if used
        if enhance_peaks:
            self.fft_ax.set_title("FFT (sqrt|FFT|)")
        else:
            self.fft_ax.set_title("FFT")
        self.fft_ax.set_xlim(-k_range, k_range)
        self.fft_ax.set_ylim(-k_range, k_range)
        self.fft_ax.set_aspect("equal")
        
        # Update colorbar label to indicate sqrt if used
        if enhance_peaks:
            colorbar_label = 'sqrt|FFT| intensity'
        else:
            colorbar_label = 'FFT intensity'
        self.fft_fig.colorbar(self.fft_plot, ax=self.fft_ax, label=colorbar_label)
        self.fft_clim(0, max_fft)
        
        # Radial distribution function (using analysis module)
        if show_radial and self.radial_ax is not None:
            # Use radial_distribution from analysis module
            k_radial, mean_fft = radial_distribution(
                ft_plot, FX, FY,  # FX, FY are 2D meshgrids
                k_max=k_range if k_range > 0 else None,
                n_bins=1000
            )
            
            self.radial_ax.plot(k_radial, mean_fft, 'b-', linewidth=1.5)
            if k_circle > 0:
                self.radial_ax.axvline(k_circle, color='k', ls='--', linewidth=1.5)
            self.radial_ax.set_xlim(0, k_range)
            self.radial_ax.set_xlabel("kr (nm⁻¹)")
            # Update ylabel and title to indicate sqrt if used
            if enhance_peaks:
                self.radial_ax.set_ylabel("mean sqrt|FFT|")
                self.radial_ax.set_title("Radial distribution of sqrt|FFT| intensity")
            else:
                self.radial_ax.set_ylabel("mean |FFT|")
                self.radial_ax.set_title("Radial distribution of FFT intensity")
            self.radial_ax.grid(True, alpha=0.3)
    
    def fft_clim(self, c_min: float, c_max: float) -> None:
        """Set color axis limits on the Fourier transform."""
        if hasattr(self, 'fft_plot') and self.fft_plot is not None:
            self.fft_plot.set_clim(c_min, c_max)
    
    def fft_colormap(self, cmap: str) -> None:
        """Change the colormap for the Fourier transform."""
        if hasattr(self, 'fft_plot') and self.fft_plot is not None:
            self.fft_plot.set_cmap(cmap)


class SXMCollection(BaseFileCollection):
    """
    Collection of SXM files with associated hyperparameters.
    
    Useful for analyzing multiple images taken at different conditions
    (e.g., different gate voltages).
    """
    
    def __init__(self, file_paths: List[str] | List[Path], hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize SXM collection.
        
        Parameters
        ----------
        file_paths : List[str] | List[Path]
            List of paths to .sxm files
        hyperparameters : Optional[Dict[str, Any]]
            Dictionary of hyperparameters (e.g., {'gate_voltage': [0.1, 0.2, ...]})
        """
        super().__init__(file_paths, hyperparameters)
        self.files = [SXMFile(path) for path in self.file_paths]
        
    def load_all(self) -> None:
        """Load all SXM files in the collection."""
        for file in self.files:
            file.load()
    
    def process_all(self, background_subtract: bool = False, fft: bool = False, **kwargs) -> None:
        """
        Process all files in the collection.
        
        Parameters
        ----------
        background_subtract : bool
            Whether to perform background subtraction
        fft : bool
            Whether to compute FFT
        **kwargs
            Additional processing parameters
        """
        for file in self.files:
            file.process(background_subtract=background_subtract, fft=fft, **kwargs)
    
    def get_all_images(self, processed: bool = False) -> List[np.ndarray]:
        """
        Get all images from the collection.
        
        Parameters
        ----------
        processed : bool
            If True, return processed images; if False, return raw images
            
        Returns
        -------
        List[np.ndarray]
            List of image arrays
        """
        return [file.get_image(processed=processed) for file in self.files if file.get_image(processed=processed) is not None]

