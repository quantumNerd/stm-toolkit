"""
SXM file handling for Nanonis 2D image data.

This module provides classes for loading and processing .sxm files,
including background subtraction and FFT analysis.

Based on nanonis_load implementation:
https://github.com/dilwong/nanonis_load/blob/master/nanonis_load/sxm.py
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import scipy.signal
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
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
    
    def __init__(self, file_path: str | Path):
        """
        Initialize SXM file handler.
        
        Parameters
        ----------
        file_path : str | Path
            Path to the .sxm file
        """
        super().__init__(file_path)
        self.image_data: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.fft_data: Optional[np.ndarray] = None
        self.data: Dict[str, List[np.ndarray]] = {}  # Channel data: {channel_name: [forward, backward]}
        self.header: Dict[str, Any] = {}
        self.x_pixels: Optional[int] = None
        self.y_pixels: Optional[int] = None
        self.x_range: Optional[float] = None  # in nm
        self.y_range: Optional[float] = None  # in nm
        
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
        
        # Try to extract gate voltage
        try:
            header["gate_voltage"] = float(header[":Ext. VI 1>Gate voltage (V):"][0])
        except (KeyError, ValueError, IndexError):
            try:
                if ":COMMENT:" in header and len(header[":COMMENT:"]) > 0:
                    split_comment = header[":COMMENT:"][0].split()
                    if "V_g" in split_comment:
                        header["gate_voltage"] = float(split_comment[split_comment.index("V_g") + 2])
                    else:
                        header["gate_voltage"] = 0.0
                else:
                    header["gate_voltage"] = 0.0
            except (ValueError, IndexError):
                header["gate_voltage"] = 0.0
        
        # Extract bias voltage
        if ":BIAS:" in header:
            header["bias"] = float(header[":BIAS:"][0])
        
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
        Process the loaded image data.
        
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
            Dictionary containing processed data
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
            if channel is not None:
                self.processed_image = self._subtract_background(
                    background_method=kwargs.get('background_method', 'plane'),
                    channel=channel,
                    direction=direction,
                    **kwargs
                )
            else:
                self.processed_image = self._subtract_background(**kwargs)
            processed['image'] = self.processed_image
        else:
            self.processed_image = image_to_process.copy()
            processed['image'] = self.processed_image
        
        # FFT analysis
        if fft:
            self.fft_data = self._compute_fft(self.processed_image)
            processed['fft'] = self.fft_data
            processed['fft_magnitude'] = np.abs(self.fft_data)
            processed['fft_phase'] = np.angle(self.fft_data)
        
        self.processed_data = processed
        return processed
    
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
    
    def _compute_fft(self, image: np.ndarray) -> np.ndarray:
        """
        Compute FFT of the image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image data
            
        Returns
        -------
        np.ndarray
            FFT of the image
        """
        if image is None:
            raise ValueError("Image data required for FFT.")
        
        return np.fft.fft2(image)
    
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
        """Get gate voltage in V."""
        return self.header.get("gate_voltage", 0.0)
    
    def get_bias(self) -> float:
        """Get sample bias in V."""
        return self.header.get("bias", 0.0)
    
    def plot(self, channel: Optional[str] = None, direction: int = 0, 
             processed: bool = False, **kwargs) -> 'SXMPlotter':
        """
        Create a plotter for this SXM file.
        
        Parameters
        ----------
        channel : Optional[str]
            Channel name to plot
        direction : int
            Direction (0=forward, 1=backward)
        processed : bool
            Whether to plot processed data
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        SXMPlotter
            Plotter instance
        """
        return SXMPlotter(self, channel=channel, direction=direction, processed=processed, **kwargs)


class SXMPlotter(BasePlotter):
    """
    Plotter for SXM file data.
    
    Provides methods for plotting 2D images and FFT analysis.
    """
    
    def __init__(self, sxm_file: SXMFile, channel: Optional[str] = None, 
                 direction: int = 0, processed: bool = False,
                 flatten: bool = False, subtract_plane: bool = False,
                 subtract_line: bool = False, cmap: str = 'gray',
                 filter_current: int = 0, map_color_std: Optional[float] = None,
                 **kwargs):
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
        cmap : str
            Colormap name
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
        self.processed = processed
        self.flatten = flatten or subtract_line
        self.subtract_plane = subtract_plane
        self.cmap = cmap
        self.filter_current = filter_current
        self.map_color_std = map_color_std
        super().__init__(sxm_file, **kwargs)
    
    def _setup_plot(self, **kwargs) -> None:
        """Set up the initial plot."""
        # Get image data
        if self.processed and self.sxm_file.processed_image is not None:
            image_data = self.sxm_file.processed_image.copy()
        else:
            image_data = self.sxm_file.get_image(
                processed=False,
                channel=self.channel,
                direction=self.direction
            )
        
        if image_data is None:
            raise ValueError("No image data available to plot.")
        
        # Apply filtering if requested (Savitzky-Golay filter, as in notebook)
        if self.filter_current > 0:
            image_data = savgol_filter(image_data, self.filter_current, 1, axis=-1)
        
        # Apply processing if requested
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
        
        # Create figure
        self.fig = plt.figure(figsize=kwargs.get('figsize', (8, 8)))
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
        if self.sxm_file.get_gate_voltage() != 0.0:
            title += f" | Vg = {self.sxm_file.get_gate_voltage():.3f} V"
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
    
    def fft(self, window_function: Optional[str] = None, k_circle: float = -1, 
            k_range: float = -1, show_radial: bool = False) -> None:
        """
        Plot the Fourier transform.
        
        Parameters
        ----------
        window_function : Optional[str]
            Window function name ('blackman', etc.)
        k_circle : float
            Radius of circle to overlay on FFT (in nm⁻¹). If <= 0, no circle is drawn.
        k_range : float
            Range for kx and ky axes (in nm⁻¹). If <= 0, auto-determined.
        show_radial : bool
            If True, also plot radial distribution function
        """
        if self.image_data is None:
            raise ValueError("No image data available for FFT.")
        
        def correct_fft2d(image_data: np.ndarray, window_function: Optional[str] = None) -> np.ndarray:
            """Compute corrected 2D FFT with optional windowing."""
            window = np.ones(image_data.shape)
            if window_function and window_function.lower() == "blackman":
                window = np.outer(
                    np.blackman(image_data.shape[0]),
                    np.blackman(image_data.shape[1])
                )
            return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(window * image_data)))
        
        ft = correct_fft2d(self.image_data, window_function)
        ft_magnitude = np.abs(ft)
        max_fft = np.max(ft_magnitude[1:-1, 1:-1])
        
        # Calculate FFT spatial frequencies using np.fft.fftfreq (as in notebook)
        x_range = self.sxm_file.x_range or 100.0
        y_range = self.sxm_file.y_range or 100.0
        x_pixels = self.sxm_file.x_pixels or self.image_data.shape[1]
        y_pixels = self.sxm_file.y_pixels or self.image_data.shape[0]
        
        # Create coordinate arrays (centered at 0)
        sxm_x = np.linspace(-x_range / 2, x_range / 2, x_pixels)
        sxm_y = np.linspace(-y_range / 2, y_range / 2, y_pixels)
        
        # Calculate pixel spacing
        dx = sxm_x[1] - sxm_x[0]  # nm
        dy = sxm_y[1] - sxm_y[0]  # nm
        
        # Use np.fft.fftfreq for accurate frequency calculation (as in notebook)
        ny, nx = self.image_data.shape
        fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))  # nm⁻¹
        fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))  # nm⁻¹
        
        # Determine k_range
        if k_range <= 0:
            # Auto-determine range (use 0.1 nm⁻¹ as default, or max frequency if smaller)
            k_range = min(0.1, max(np.abs(fx).max(), np.abs(fy).max()))
        
        # For square plots when ranges are equal, use the same frequency range
        if abs(x_range - y_range) < 1e-6:  # Ranges are equal
            k_range = max(np.abs(fx).max(), np.abs(fy).max())
        
        # Create figure(s)
        if show_radial:
            self.fft_fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            self.fft_ax = axes[0]
            self.radial_ax = axes[1]
        else:
            self.fft_fig = plt.figure(figsize=(8, 8))
            self.fft_ax = self.fft_fig.add_subplot(111)
            self.radial_ax = None
        
        # Create meshgrid for pcolormesh
        FX, FY = np.meshgrid(fx, fy)
        
        # Plot FFT
        self.fft_plot = self.fft_ax.pcolormesh(
            FX, FY, ft_magnitude,
            shading='auto',
            cmap=self.cmap
        )
        
        # Add k-circle if requested
        if k_circle > 0:
            circle = Circle((0, 0), radius=k_circle, edgecolor='k', 
                          facecolor='none', ls='--', linewidth=1.5)
            self.fft_ax.add_patch(circle)
        
        self.fft_ax.set_xlabel("kx (nm⁻¹)")
        self.fft_ax.set_ylabel("ky (nm⁻¹)")
        self.fft_ax.set_title("FFT")
        self.fft_ax.set_xlim(-k_range, k_range)
        self.fft_ax.set_ylim(-k_range, k_range)
        self.fft_ax.set_aspect("equal")
        
        self.fft_fig.colorbar(self.fft_plot, ax=self.fft_ax, label='FFT intensity')
        self.fft_clim(0, max_fft)
        
        # Radial distribution function (as in notebook)
        if show_radial and self.radial_ax is not None:
            R = np.sqrt(FX**2 + FY**2)
            R_flat = R.ravel()
            FT_flat = ft_magnitude.ravel()
            
            r_bins = np.linspace(0, R.max(), 1000)
            r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
            radial_profile = np.zeros_like(r_centers)
            counts = np.zeros_like(r_centers)
            
            bin_indices = np.digitize(R_flat, r_bins)
            for i in range(len(r_centers)):
                in_bin = bin_indices == i
                if np.any(in_bin):
                    radial_profile[i] = FT_flat[in_bin].mean()
                    counts[i] = in_bin.sum()
            
            self.radial_ax.plot(r_centers, radial_profile, 'b-', linewidth=1.5)
            if k_circle > 0:
                self.radial_ax.axvline(k_circle, color='k', ls='--', linewidth=1.5)
            self.radial_ax.set_xlim(0, k_range)
            self.radial_ax.set_xlabel("kr (nm⁻¹)")
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

