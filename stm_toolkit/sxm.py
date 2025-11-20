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
                 multipass_bias_index: int = -4,
                 auto_load: bool = True):
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
        auto_load : bool
            If True, automatically load the file on initialization (default: True).
            If False, call load() manually.
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
        
        # Auto-load if requested
        if auto_load:
            try:
                self.load()
            except (FileNotFoundError, ValueError) as e:
                # Store error but don't raise - allows object creation even if file doesn't exist
                # User can call load() manually later
                self._load_error = e
                self._load_attempted = True
            else:
                self._load_error = None
                self._load_attempted = True
        else:
            self._load_error = None
            self._load_attempted = False
        
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
    
    # Deprecated processing methods removed - use calc() to create ProcessedSXMFile instead
    # All processing functions are now in analysis.py and called by ProcessedSXMFile
    
    def get_image(self, channel: Optional[str] = None, direction: int = 0) -> Optional[np.ndarray]:
        """
        Get the raw image data.
        
        Parameters
        ----------
        channel : Optional[str]
            Channel name. If None, uses default image_data
        direction : int
            Direction (0=forward, 1=backward)
            
        Returns
        -------
        np.ndarray or None
            Raw image data
        """
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
             ax: Optional[plt.Axes] = None, **kwargs) -> 'SXMPlotter':
        """
        Create a plotter for this SXM file.
        
        Parameters
        ----------
        channel : Optional[str]
            Channel name to plot
        direction : int
            Direction (0=forward, 1=backward)
        ax : Optional[matplotlib.axes.Axes]
            Axes to plot on. If None, creates a new figure.
        **kwargs
            Additional plotting parameters (e.g., figsize, cmap, subtract_plane, flatten, etc.)
            
        Returns
        -------
        SXMPlotter
            Plotter instance
        """
        return SXMPlotter(self, channel=channel, direction=direction, ax=ax, **kwargs)
    
    def calc(self, steps: Optional[List[Dict[str, Any]]] = None,
             subtract_plane: bool = False, flatten: bool = False,
             filter_current: Optional[int] = None, fft: bool = False,
             channel: Optional[str] = None, direction: int = 0,
             window_function: Optional[str] = None, 
             processing_order: Optional[List[str]] = None, **kwargs) -> 'ProcessedSXMFile':
        """
        Apply processing operations and return a ProcessedSXMFile.
        
        This method creates a processed copy of the file with processing applied.
        The processed file stores processed data instead of raw data.
        
        **New flexible API (recommended):**
            Use `steps` parameter to specify processing steps with parameters:
            
            ```python
            processed = sxm_file.calc(steps=[
                {'type': 'subtract_plane'},
                {'type': 'filter', 'window_size': 16},
                {'type': 'filter', 'window_size': 32},  # Second filter
                {'type': 'fft', 'window_function': 'hanning'},
                {'type': 'ifft'},  # Inverse FFT to go back to real space
            ])
            ```
        
        **Legacy API (still supported):**
            Use boolean flags and processing_order:
            - Processing steps are applied in the order specified by processing_order,
              or in the default order: subtract_plane/flatten -> filter -> fft
        
        Parameters
        ----------
        steps : Optional[List[Dict[str, Any]]]
            List of processing steps with parameters. Each step is a dict with:
            - 'type': Step type ('subtract_plane', 'flatten', 'filter', 'fft', 'ifft')
            - Additional parameters for each step type:
              * 'filter': 'window_size' (int), 'polyorder' (int, default=1), 'axis' (int, default=-1)
              * 'fft': 'window_function' (str, optional), 'kaiser_beta' (float, default=5.0)
                - window_function: 'blackman', 'hanning' (or 'hann'), 'hamming', 'bartlett', 'kaiser'
                - kaiser_beta: Only used if window_function='kaiser'
              * 'ifft': No parameters needed - uses stored complex data
            If None, uses legacy API with boolean flags.
        subtract_plane : bool
            [Legacy API] If True, subtract a 2D plane from the image
        flatten : bool
            [Legacy API] If True, subtract linear fit from every fast-scan line
        filter_current : Optional[int]
            [Legacy API] Savitzky-Golay filter window size (None = no filtering)
        fft : bool
            [Legacy API] If True, compute and store FFT results
        channel : Optional[str]
            Channel to process. If None, processes default image_data
        direction : int
            Direction (0=forward, 1=backward)
        window_function : Optional[str]
            [Legacy API] Window function for FFT ('blackman', etc.)
        processing_order : Optional[List[str]]
            [Legacy API] Order of processing steps: ['subtract_plane', 'filter', 'fft'] or 
            ['flatten', 'filter', 'fft']. If None, uses default order based on 
            which flags are True.
        **kwargs
            Additional processing parameters
            
        Returns
        -------
        ProcessedSXMFile
            A processed copy of this file with processing applied
        """
        return ProcessedSXMFile.from_sxm_file(
            self, steps=steps, subtract_plane=subtract_plane, flatten=flatten,
            filter_current=filter_current, fft=fft,
            channel=channel, direction=direction,
            window_function=window_function, 
            processing_order=processing_order, **kwargs
        )


class ProcessedSXMFile(SXMFile):
    """
    Processed version of SXMFile with processing operations applied.
    
    This class stores processed data instead of raw data, while maintaining
    all metadata and the ability to access the original raw data.
    """
    
    def __init__(self, original_file: SXMFile, processed_data: Dict[str, Any],
                 processing_steps: List[str]):
        """
        Initialize processed SXM file.
        
        Parameters
        ----------
        original_file : SXMFile
            The original SXMFile this was processed from
        processed_data : Dict[str, Any]
            Dictionary containing processed data:
            - 'image_data': Processed image (replaces raw image_data)
            - 'data': Processed channel data (replaces raw data)
            - 'fft': Optional FFT results
        processing_steps : List[str]
            List of processing steps applied (e.g., ['subtract_plane', 'filter'])
        """
        # Initialize with original file path and settings
        super().__init__(
            original_file.file_path,
            gate_voltage_key=original_file.gate_voltage_key,
            bias_voltage_key=original_file.bias_voltage_key,
            source_drain_voltage_key=original_file.source_drain_voltage_key,
            multipass_config_index=original_file.multipass_config_index,
            multipass_bias_index=original_file.multipass_bias_index,
            auto_load=False  # Don't auto-load, we'll set data manually
        )
        
        # Store reference to original file
        self.original_file = original_file
        
        # Copy all metadata and properties from original
        self.header = original_file.header.copy()
        self.metadata = original_file.metadata.copy()
        self.x_pixels = original_file.x_pixels
        self.y_pixels = original_file.y_pixels
        
        # Check if FFT was applied at any point (changes coordinate system)
        self.is_fft_data = processed_data.get('fft_data', False)
        
        if self.is_fft_data:
            # FFT was applied, so data is in k-space and axes should be in nm⁻¹
            # Store original ranges for reference
            self.original_x_range = processed_data.get('original_x_range', original_file.x_range)
            self.original_y_range = processed_data.get('original_y_range', original_file.y_range)
            
            # Get k-space ranges from FX and FY
            FX = processed_data.get('fft_FX')
            FY = processed_data.get('fft_FY')
            if FX is not None and FY is not None:
                # FX and FY are 2D arrays with values in nm⁻¹
                # Get the maximum absolute value to determine the full range
                kx_max = np.abs(FX).max()
                ky_max = np.abs(FY).max()
                self.x_range = kx_max * 2  # Full range from -max to +max (in nm⁻¹)
                self.y_range = ky_max * 2
                self._fft_FX = FX  # Store for plotting (already 2D meshgrid)
                self._fft_FY = FY
            else:
                # Fallback to original ranges if FX/FY not available
                self.x_range = original_file.x_range
                self.y_range = original_file.y_range
                self._fft_FX = None
                self._fft_FY = None
        else:
            # Regular processed data (real-space in nm)
            # Use original ranges from processed_data if stored, otherwise from original file
            if 'original_x_range' in processed_data and processed_data['original_x_range'] is not None:
                self.x_range = processed_data['original_x_range']
                self.y_range = processed_data['original_y_range']
            else:
                self.x_range = original_file.x_range
                self.y_range = original_file.y_range
            self._fft_FX = None
            self._fft_FY = None
            self.original_x_range = processed_data.get('original_x_range', original_file.x_range)
            self.original_y_range = processed_data.get('original_y_range', original_file.y_range)
        
        # Replace raw data with processed data
        self.image_data = processed_data.get('image_data', original_file.image_data)
        if 'data' in processed_data:
            # Deep copy to avoid modifying original
            self.data = {k: [v[0].copy(), v[1].copy()] for k, v in processed_data['data'].items()}
        else:
            # Deep copy channel data from original
            self.data = {k: [v[0].copy(), v[1].copy()] for k, v in original_file.data.items()}
            # Update with processed image_data if it's the default channel
            if self.image_data is not None and self.image_data is not original_file.image_data:
                # Find which channel corresponds to image_data and update it
                for ch_name, ch_data in self.data.items():
                    if (ch_data[0] is original_file.image_data or 
                        (ch_data[0] is not None and original_file.image_data is not None and 
                         np.array_equal(ch_data[0], original_file.image_data))):
                        ch_data[0] = self.image_data
                        break
        
        # Store processing information
        self.processing_steps = processing_steps
        self.processed_data = processed_data
        
        # Store which channel and direction were processed
        self.processed_channel = processed_data.get('processed_channel', None)
        self.processed_direction = processed_data.get('processed_direction', 0)
        
        # Store FFT if computed
        if 'fft' in processed_data:
            self._fft_results = processed_data['fft']
        else:
            self._fft_results = None
    
    @classmethod
    def from_sxm_file(cls, sxm_file: SXMFile, steps: Optional[List[Dict[str, Any]]] = None,
                     subtract_plane: bool = False, flatten: bool = False,
                     filter_current: Optional[int] = None, fft: bool = False,
                     channel: Optional[str] = None, direction: int = 0,
                     window_function: Optional[str] = None,
                     processing_order: Optional[List[str]] = None, **kwargs) -> 'ProcessedSXMFile':
        """
        Create a ProcessedSXMFile from an SXMFile by applying processing operations.
        
        Processing steps are applied sequentially in the order specified.
        
        Parameters
        ----------
        sxm_file : SXMFile
            The original SXMFile to process
        subtract_plane : bool
            If True, subtract a 2D plane
        flatten : bool
            If True, subtract linear fit from every fast-scan line
        filter_current : Optional[int]
            Savitzky-Golay filter window size
        fft : bool
            If True, compute FFT
        channel : Optional[str]
            Channel to process
        direction : int
            Direction (0=forward, 1=backward)
        window_function : Optional[str]
            Window function for FFT
        processing_order : Optional[List[str]]
            Order of processing steps: ['subtract_plane', 'filter', 'fft'] or 
            ['flatten', 'filter', 'fft']. If None, uses default order based on 
            which flags are True: background -> filter -> fft
        **kwargs
            Additional processing parameters
            
        Returns
        -------
        ProcessedSXMFile
            Processed file with operations applied
        """
        if sxm_file.image_data is None and not sxm_file.data:
            raise ValueError("File must be loaded before processing. Call load() first.")
        
        # Get the image to process
        if channel is not None and channel in sxm_file.data:
            image = sxm_file.data[channel][direction].copy()
            process_channel = channel
        else:
            image = sxm_file.image_data.copy()
            process_channel = None
        
        processing_steps = []
        processed_data = {}
        
        # Track coordinate system state
        fft_applied = False
        fft_FX = None
        fft_FY = None
        original_x_range = sxm_file.x_range
        original_y_range = sxm_file.y_range
        
        # Use new flexible API if steps provided
        if steps is not None:
            # Apply processing steps sequentially
            # Each step transforms the data, which becomes the input for the next step
            for step_def in steps:
                if not isinstance(step_def, dict):
                    raise ValueError(f"Each step must be a dict with 'type' key. Got: {step_def}")
                
                step_type = step_def.get('type')
                if step_type is None:
                    raise ValueError(f"Step definition must have 'type' key. Got: {step_def}")
                
                step_params = {k: v for k, v in step_def.items() if k != 'type'}
                
                if step_type == 'subtract_plane':
                    from .analysis import subtract_plane_2d
                    image = subtract_plane_2d(image)
                    processing_steps.append('subtract_plane')
                    
                elif step_type == 'flatten':
                    from .analysis import subtract_linear_by_line_2d
                    image = subtract_linear_by_line_2d(image)
                    processing_steps.append('flatten')
                    
                elif step_type == 'filter':
                    from scipy.signal import savgol_filter
                    window_size = step_params.get('window_size')
                    if window_size is None:
                        raise ValueError("'filter' step requires 'window_size' parameter")
                    polyorder = step_params.get('polyorder', 1)
                    axis = step_params.get('axis', -1)
                    
                    # Handle complex data: filter real and imaginary parts separately
                    is_complex = np.iscomplexobj(image)
                    if is_complex:
                        real_filtered = savgol_filter(np.real(image), window_size, polyorder, axis=axis)
                        imag_filtered = savgol_filter(np.imag(image), window_size, polyorder, axis=axis)
                        image = real_filtered + 1j * imag_filtered
                    else:
                        image = savgol_filter(image, window_size, polyorder, axis=axis)
                    
                    processing_steps.append(f'filter_{window_size}')
                    
                elif step_type == 'fft':
                    # FFT transforms data AND changes coordinate system
                    from .analysis import compute_fft_2d
                    window_func = step_params.get('window_function')
                    kaiser_beta = step_params.get('kaiser_beta', 5.0)
                    
                    fft_result = compute_fft_2d(
                        image=image,
                        x_range=sxm_file.x_range or 100.0,
                        y_range=sxm_file.y_range or 100.0,
                        x_pixels=sxm_file.x_pixels,
                        y_pixels=sxm_file.y_pixels,
                        window_function=window_func,
                        kaiser_beta=kaiser_beta
                    )
                    
                    # Store complex FFT data (not magnitude) so operations can work on complex
                    image = fft_result['ft']  # Complex FFT
                    
                    fft_FX = fft_result['FX']  # Store k-space coordinates
                    fft_FY = fft_result['FY']
                    if not fft_applied:
                        # Store original ranges before switching to k-space (only once)
                        original_x_range = sxm_file.x_range
                        original_y_range = sxm_file.y_range
                    fft_applied = True  # Mark that coordinate system has changed
                    processing_steps.append('fft')
                    
                elif step_type == 'ifft':
                    # Inverse FFT: go back to real space
                    # Following the pattern from ideal_low_pass_filter_2d:
                    # Forward: fft2 -> fftshift (stored FFT is already shifted)
                    # Inverse: ifftshift -> ifft2
                    if not fft_applied:
                        raise ValueError("Cannot apply inverse FFT: not in k-space. Apply FFT first.")
                    
                    # Image should already be complex FFT (stored as complex throughout)
                    # The stored FFT from compute_fft is already shifted (fftshift applied)
                    if np.iscomplexobj(image):
                        image_fft_shifted = image.copy()
                    else:
                        # If somehow we have magnitude, convert to complex (zero phase)
                        # This shouldn't happen with new approach, but handle gracefully
                        image_fft_shifted = image.astype(complex)
                    
                    # Shift the zero frequency component back to the corner (unshift)
                    # This reverses the fftshift from the forward FFT
                    image_fft_unshifted = np.fft.ifftshift(image_fft_shifted)
                    
                    # Perform the inverse FFT
                    image_real = np.fft.ifft2(image_fft_unshifted)
                    
                    # Take real part (should be real anyway, but numerical errors can add small imaginary parts)
                    image = np.real(image_real)
                    
                    # Switch back to real-space coordinate system
                    fft_applied = False  # No longer in k-space after ifft
                    fft_FX = None
                    fft_FY = None
                    # Ranges will be restored when stored in processed_data
                    # Don't modify original file's ranges!
                    
                    processing_steps.append('ifft')
                    
                else:
                    raise ValueError(f"Unknown step type '{step_type}'. "
                                   f"Valid types: 'subtract_plane', 'flatten', 'filter', 'fft', 'ifft'")
        
        else:
            # Legacy API: use boolean flags and processing_order
            # Determine processing order
            if processing_order is None:
                # Default order based on what's enabled
                processing_order = []
                if subtract_plane:
                    processing_order.append('subtract_plane')
                elif flatten:
                    processing_order.append('flatten')
                if filter_current is not None and filter_current > 0:
                    processing_order.append('filter')
                if fft:
                    processing_order.append('fft')
            else:
                # Validate processing_order contains only valid steps
                valid_steps = ['subtract_plane', 'flatten', 'filter', 'fft']
                for step in processing_order:
                    if step not in valid_steps:
                        raise ValueError(f"Invalid processing step '{step}'. Valid steps: {valid_steps}")
            
            # Apply processing steps sequentially in order
            # Each step transforms the data, which becomes the input for the next step
            for step in processing_order:
                if step == 'subtract_plane' and subtract_plane:
                    from .analysis import subtract_plane_2d
                    image = subtract_plane_2d(image)
                    processing_steps.append('subtract_plane')
                elif step == 'flatten' and flatten:
                    from .analysis import subtract_linear_by_line_2d
                    image = subtract_linear_by_line_2d(image)
                    processing_steps.append('flatten')
                elif step == 'filter' and filter_current is not None and filter_current > 0:
                    from scipy.signal import savgol_filter
                    # Handle complex data: filter real and imaginary parts separately
                    is_complex = np.iscomplexobj(image)
                    if is_complex:
                        real_filtered = savgol_filter(np.real(image), filter_current, 1, axis=-1)
                        imag_filtered = savgol_filter(np.imag(image), filter_current, 1, axis=-1)
                        image = real_filtered + 1j * imag_filtered
                    else:
                        image = savgol_filter(image, filter_current, 1, axis=-1)
                    processing_steps.append(f'filter_{filter_current}')
                elif step == 'fft' and fft:
                    # FFT transforms data AND changes coordinate system
                    from .analysis import compute_fft_2d
                    fft_result = compute_fft_2d(
                        image=image,
                        x_range=sxm_file.x_range or 100.0,
                        y_range=sxm_file.y_range or 100.0,
                        x_pixels=sxm_file.x_pixels,
                        y_pixels=sxm_file.y_pixels,
                        window_function=window_function
                    )
                    # Store complex FFT data (not magnitude) so operations can work on complex
                    image = fft_result['ft']  # Complex FFT
                    fft_FX = fft_result['FX']  # Store k-space coordinates
                    fft_FY = fft_result['FY']
                    if not fft_applied:
                        # Store original ranges before switching to k-space (only once)
                        original_x_range = sxm_file.x_range
                        original_y_range = sxm_file.y_range
                    fft_applied = True  # Mark that coordinate system has changed
                    processing_steps.append('fft')
        
        # Store the final processed data (result of last processing step)
        if process_channel:
            # Process specific channel
            processed_data['data'] = {k: [v[0].copy(), v[1].copy()] for k, v in sxm_file.data.items()}
            processed_data['data'][channel][direction] = image
            # image_data should point to the processed data for the requested direction
            processed_data['image_data'] = processed_data['data'][channel][direction]
            processed_data['processed_channel'] = channel
            processed_data['processed_direction'] = direction
        else:
            # Process default image
            processed_data['image_data'] = image
            processed_data['data'] = {k: [v[0].copy(), v[1].copy()] for k, v in sxm_file.data.items()}
            # Update default image_data in data if it matches
            for ch_name, ch_data in processed_data['data'].items():
                if (ch_data[0] is sxm_file.image_data or 
                    (ch_data[0] is not None and sxm_file.image_data is not None and 
                     np.array_equal(ch_data[0], sxm_file.image_data))):
                    ch_data[0] = image
                    processed_data['processed_channel'] = ch_name  # Store which channel was the default
                    break
            if 'processed_channel' not in processed_data:
                # If we couldn't find the channel, store None (default image_data)
                processed_data['processed_channel'] = None
            processed_data['processed_direction'] = direction
        
        # Store FFT information if FFT was applied (regardless of position in chain)
        # After ifft, fft_applied becomes False, so data is back in real-space
        processed_data['fft_data'] = fft_applied
        if fft_applied:
            processed_data['fft_FX'] = fft_FX  # Store k-space coordinates for plotting
            processed_data['fft_FY'] = fft_FY
            processed_data['original_x_range'] = original_x_range  # Store original ranges for reference
            processed_data['original_y_range'] = original_y_range
        else:
            # If not in k-space (e.g., after ifft), ensure ranges are set to original
            # This handles the case where ifft was the last step
            processed_data['original_x_range'] = original_x_range
            processed_data['original_y_range'] = original_y_range
        
        return cls(sxm_file, processed_data, processing_steps)
    
    def get_original_file(self) -> SXMFile:
        """
        Get the original SXMFile containing the raw data.
        
        This method returns the original file object in case it was lost in the code.
        
        Returns
        -------
        SXMFile
            The original SXMFile with raw data
        """
        return self.original_file
    
    def get_raw_image(self, channel: Optional[str] = None, direction: int = 0) -> Optional[np.ndarray]:
        """
        Get the original raw image from the original file.
        
        Parameters
        ----------
        channel : Optional[str]
            Channel name
        direction : int
            Direction (0=forward, 1=backward)
            
        Returns
        -------
        Optional[np.ndarray]
            Raw image data from original file
        """
        return self.original_file.get_image(channel=channel, direction=direction)
    
    def get_image(self, processed: bool = True, channel: Optional[str] = None, 
                  direction: int = 0) -> Optional[np.ndarray]:
        """
        Get image data. For ProcessedSXMFile, returns processed data by default.
        
        Parameters
        ----------
        processed : bool
            If True, return processed image; if False, return raw from original file
        channel : Optional[str]
            Channel name
        direction : int
            Direction (0=forward, 1=backward)
            
        Returns
        -------
        Optional[np.ndarray]
            Image data
        """
        if not processed:
            return self.original_file.get_image(channel=channel, direction=direction)
        
        # Return processed data
        if channel is not None and channel in self.data:
            return self.data[channel][direction]
        
        return self.image_data
    
    def get_fft(self, channel: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get stored FFT results if available.
        
        Parameters
        ----------
        channel : Optional[str]
            Channel name
            
        Returns
        -------
        Optional[Dict[str, Any]]
            FFT results if computed during processing
        """
        return self._fft_results
    
    def compute_fft(self, image: Optional[np.ndarray] = None, channel: Optional[str] = None,
                   direction: int = 0, window_function: Optional[str] = None,
                   kaiser_beta: float = 5.0, **kwargs) -> Dict[str, Any]:
        """
        Compute FFT from processed data.
        
        This method computes FFT from the already-processed image data.
        Uses analysis module functions.
        """
        from .analysis import compute_fft_2d
        
        if image is None:
            image = self.get_image(processed=True, channel=channel, direction=direction)
        else:
            image = image.copy()
        
        if image is None:
            raise ValueError("No image data available for FFT computation.")
        
        # Get ranges from original file
        original_file = self.get_original_file()
        return compute_fft_2d(
            image=image,
            x_range=original_file.x_range or 100.0,
            y_range=original_file.y_range or 100.0,
            x_pixels=original_file.x_pixels,
            y_pixels=original_file.y_pixels,
            window_function=window_function,
            kaiser_beta=kaiser_beta
        )


class SXMPlotter(BasePlotter):
    """
    Plotter for SXM file data.
    
    Provides methods for plotting 2D images and FFT analysis.
    """
    
    def __init__(self, sxm_file: SXMFile, channel: Optional[str] = None, 
                 direction: int = 0,
                 flatten: bool = False, subtract_plane: bool = False,
                 subtract_line: bool = False, cmap: Union[str, Any] = 'Blues_r',
                 filter_current: Optional[int] = None, map_color_std: Optional[float] = None,
                 enhance_peaks: bool = False,
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
        enhance_peaks : bool
            If True, apply sqrt transformation to FFT data to enhance peaks.
            Only applies when plotting FFT data (ProcessedSXMFile with is_fft_data=True).
            Default is False.
        **kwargs
            Additional plotting parameters
        """
        self.sxm_file = sxm_file
        self.channel = channel
        self.direction = direction
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
        self.enhance_peaks = enhance_peaks
        super().__init__(sxm_file, ax=ax, figsize=figsize, **kwargs)
    
    def _setup_plot(self, **kwargs) -> None:
        """Set up the initial plot."""
        # FFT is computed on-the-fly when fft() is called, no special handling needed here
        
        # Check if this is a ProcessedSXMFile (data is already processed)
        is_processed_file = isinstance(self.sxm_file, ProcessedSXMFile)
        
        # For ProcessedSXMFile, use the processed channel if no channel specified
        # or if the specified channel matches the processed channel
        if is_processed_file:
            processed_channel = self.sxm_file.processed_channel
            processed_direction = self.sxm_file.processed_direction
            
            # If no channel specified, use the processed channel
            if self.channel is None and processed_channel is not None:
                self.channel = processed_channel
                self.direction = processed_direction
            # If channel matches processed channel, use processed direction
            elif self.channel == processed_channel:
                self.direction = processed_direction
        
        # Get image data
        # For ProcessedSXMFile, get_image() returns processed data by default
        # For regular SXMFile, get_image() returns raw data
        if is_processed_file:
            # ProcessedSXMFile.get_image() accepts processed parameter
            image_data = self.sxm_file.get_image(
                processed=True,  # Get processed data for ProcessedSXMFile
                channel=self.channel,
                direction=self.direction
            )
        else:
            # SXMFile.get_image() does not accept processed parameter
            image_data = self.sxm_file.get_image(
                channel=self.channel,
                direction=self.direction
            )
        
        if image_data is None:
            raise ValueError("No image data available to plot.")
        
        # Check if this is FFT data (needed before processing)
        is_fft_data = isinstance(self.sxm_file, ProcessedSXMFile) and self.sxm_file.is_fft_data
        
        # If data is complex, take magnitude for plotting (but keep original for processing)
        is_complex = np.iscomplexobj(image_data)
        if is_complex:
            # Store complex data for potential later use
            image_data_complex = image_data.copy()
            # Use magnitude for plotting
            image_data = np.abs(image_data)
        else:
            image_data_complex = None
        
        # Apply sqrt enhancement if requested and data is FFT
        if self.enhance_peaks and is_fft_data:
            image_data = np.sqrt(image_data)
        
        # Only apply processing if this is NOT a ProcessedSXMFile
        # (ProcessedSXMFile already has processed data stored)
        if not is_processed_file:
            # Apply background subtraction if requested
            from .analysis import subtract_plane_2d, subtract_linear_by_line_2d
            if self.subtract_plane:
                # If we had complex data, use it for processing
                if image_data_complex is not None:
                    image_data_complex = subtract_plane_2d(image_data_complex)
                    image_data = np.abs(image_data_complex)  # Update magnitude
                else:
                    image_data = subtract_plane_2d(image_data)
            elif self.flatten:
                # If we had complex data, use it for processing
                if image_data_complex is not None:
                    image_data_complex = subtract_linear_by_line_2d(image_data_complex)
                    image_data = np.abs(image_data_complex)  # Update magnitude
                else:
                    image_data = subtract_linear_by_line_2d(image_data)
            
            # Apply filtering if requested (Savitzky-Golay filter, as in notebook)
            if self.filter_current is not None and self.filter_current > 0:
                from scipy.signal import savgol_filter
                # If we had complex data, filter it
                if image_data_complex is not None:
                    real_filtered = savgol_filter(np.real(image_data_complex), self.filter_current, 1, axis=-1)
                    imag_filtered = savgol_filter(np.imag(image_data_complex), self.filter_current, 1, axis=-1)
                    image_data_complex = real_filtered + 1j * imag_filtered
                    image_data = np.abs(image_data_complex)  # Update magnitude
                else:
                    image_data = savgol_filter(image_data, self.filter_current, 1, axis=-1)
        
        # Create figure if axes not provided
        if self.ax is None:
            figsize = self._get_figsize(self.DEFAULT_FIGSIZE_2D, **kwargs)
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111)
        
        # Handle NaN values (image_data is now magnitude if complex)
        avg_dat = image_data[~np.isnan(image_data)].mean()
        image_data[np.isnan(image_data)] = avg_dat
        
        # Apply color mapping normalization (as in notebook)
        # image_data is already magnitude at this point
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
        
        # Check if this is FFT data (from ProcessedSXMFile with fft=True)
        is_fft_data = isinstance(self.sxm_file, ProcessedSXMFile) and self.sxm_file.is_fft_data
        
        if is_fft_data:
            # For FFT data, use k-space coordinates (nm⁻¹)
            FX = self.sxm_file._fft_FX
            FY = self.sxm_file._fft_FY
            
            if FX is None or FY is None:
                # Fallback: compute coordinates from ranges
                x_range = self.sxm_file.x_range or 100.0  # in nm⁻¹
                y_range = self.sxm_file.y_range or 100.0
                x_pixels = self.sxm_file.x_pixels or image_data.shape[1]
                y_pixels = self.sxm_file.y_pixels or image_data.shape[0]
                
                # Create k-space coordinate arrays centered at 0
                kx = np.linspace(-x_range / 2, x_range / 2, x_pixels)
                ky = np.linspace(-y_range / 2, y_range / 2, y_pixels)
                X, Y = np.meshgrid(kx, ky)
            else:
                # Use stored FX, FY directly (already 2D meshgrids with correct shape)
                X, Y = FX, FY
            
            self.im_plot = self.ax.pcolormesh(
                X, Y, image_data,
                cmap=self.cmap,
                rasterized=kwargs.get('rasterized', True),
                shading='nearest'
            )
            
            self.ax.set_aspect("equal")
            self.ax.set_xlabel("kx (nm⁻¹)")
            self.ax.set_ylabel("ky (nm⁻¹)")
        else:
            # Regular image data, use real-space coordinates (nm)
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
        
        # Store image data (magnitude if complex, original if real)
        # For complex data, we plot magnitude but may need complex data for further operations
        if image_data_complex is not None:
            self.image_data = image_data_complex  # Store complex version
        else:
            self.image_data = image_data  # Store real version
        
        # Set title
        channel_name = self.channel or "Default"
        direction_str = "Forward" if self.direction == 0 else "Backward"
        
        # is_fft_data was already determined earlier in the method
        if is_fft_data:
            if self.enhance_peaks:
                title = f"FFT (sqrt): {channel_name} ({direction_str})"
            else:
                title = f"FFT: {channel_name} ({direction_str})"
        else:
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
    
    # Removed fft() method - use ProcessedSXMFile.plot() instead:
    # processed = sxm_file.calc(steps=[{'type': 'fft'}])
    # processed.plot()


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
    
    def get_all_images(self) -> List[np.ndarray]:
        """
        Get all raw images from all files in the collection.
        
        Returns
        -------
        List[np.ndarray]
            List of raw image arrays
        """
        return [file.get_image() for file in self.files if file.get_image() is not None]

