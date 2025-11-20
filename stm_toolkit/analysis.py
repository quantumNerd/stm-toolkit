"""
Mathematical analysis functions for STM data.

This module provides general-purpose mathematical operations that can be
applied to different data types:
- 1D analysis: FWHM, Gaussian fitting
- 2D analysis: 2D FFT, radial distribution
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter


def find_max_and_width(x: np.ndarray, y: np.ndarray, xrange: Optional[Tuple[float, float]] = None,
                      interp_times: int = 20, n_fit: Optional[int] = None) -> Tuple[float, float, float, float, float, float]:
    """
    Find the maximum value and its width (FWHM) in a 1D curve.
    
    Simple FWHM method: finds maximum and half-maximum points.
    
    Parameters
    ----------
    x : np.ndarray
        X values
    y : np.ndarray
        Y values
    xrange : Optional[Tuple[float, float]]
        Range to search within (x_min, x_max)
    interp_times : int
        Interpolation factor for higher resolution
    n_fit : Optional[int]
        Number of points to use for polynomial fitting around max and half-max
        If None, uses simple interpolation
        
    Returns
    -------
    Tuple[float, float, float, float, float, float]
        (x_max, y_max, x_half_left, y_half, x_half_right, y_half)
    """
    from .utils import find_first_occurrence
    
    x = np.array(x)
    y = np.array(y)
    
    # Reverse if needed
    if len(x) > 1 and x[1] < x[0]:
        x = np.flip(x)
        y = np.flip(y)
    
    # Interpolate for higher resolution
    x_new = np.linspace(np.min(x), np.max(x), len(x) * interp_times)
    y_new = np.interp(x_new, x, y)
    x = x_new
    y = y_new
    
    # Apply range filter if specified
    if xrange is not None and len(xrange) == 2:
        idx = (x > xrange[0]) & (x < xrange[1])
        x = x[idx]
        y = y[idx]
    
    # Find maximum
    idx_max = np.argmax(y)
    x0 = x[idx_max]
    y0 = y[idx_max]
    
    # Fit polynomial around maximum if requested
    if n_fit is not None and n_fit > 0:
        i1 = max(0, idx_max - n_fit)
        i2 = min(len(x), idx_max + n_fit)
        coefficients = np.polyfit(x[i1:i2], y[i1:i2], 2)
        y0 = coefficients[2] - coefficients[1]**2 / (4 * coefficients[0])
        x0 = -coefficients[1] / (2 * coefficients[0])
    
    # Find half-maximum points
    y_half = y0 / 2
    idx_target_left = find_first_occurrence(x, x0, y, y_half, reverse_direction=True)
    idx_target_right = find_first_occurrence(x, x0, y, y_half, reverse_direction=False)
    
    # Fit lines around half-maximum points if requested
    if n_fit is not None and n_fit > 0:
        # Fit left side
        i1 = max(0, idx_target_left - n_fit)
        i2 = min(len(x), idx_target_left + n_fit)
        coefficients = np.polyfit(x[i1:i2], y[i1:i2], 1)
        k = coefficients[0]  # slope
        b = coefficients[1]  # intercept
        xhl = (y_half - b) / k
        
        # Fit right side
        i1 = max(0, idx_target_right - n_fit)
        i2 = min(len(x), idx_target_right + n_fit)
        coefficients = np.polyfit(x[i1:i2], y[i1:i2], 1)
        k = coefficients[0]  # slope
        b = coefficients[1]  # intercept
        xhr = (y_half - b) / k
        
        return x0, y0, xhl, y_half, xhr, y_half
    else:
        return x0, y0, x[idx_target_left], y[idx_target_left], x[idx_target_right], y[idx_target_right]


def gaussian_with_linear_background(x: np.ndarray, amplitude: float, center: float, 
                                    sigma: float, slope: float, intercept: float) -> np.ndarray:
    """
    Gaussian function with linear background.
    
    Parameters
    ----------
    x : np.ndarray
        X values
    amplitude : float
        Amplitude of Gaussian
    center : float
        Center position of Gaussian
    sigma : float
        Standard deviation (width) of Gaussian
    slope : float
        Slope of linear background
    intercept : float
        Intercept of linear background
        
    Returns
    -------
    np.ndarray
        y = amplitude * exp(-(x-center)^2 / (2*sigma^2)) + slope*x + intercept
    """
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    linear = slope * x + intercept
    return gaussian + linear


def get_broadening_by_gaussian_fit(x: np.ndarray, y: np.ndarray, 
                                   xrange: Optional[Tuple[float, float]] = None,
                                   initial_guess: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Get peak broadening (FWHM) by fitting a Gaussian with linear background.
    
    Parameters
    ----------
    x : np.ndarray
        X values
    y : np.ndarray
        Y values
    xrange : Optional[Tuple[float, float]]
        Range to fit within (x_min, x_max)
    initial_guess : Optional[Dict[str, float]]
        Initial guess for fit parameters: {'amplitude', 'center', 'sigma', 'slope', 'intercept'}
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'amplitude': Gaussian amplitude
        - 'center': Peak center position
        - 'sigma': Gaussian standard deviation
        - 'fwhm': Full width at half maximum (2.355 * sigma)
        - 'slope': Linear background slope
        - 'intercept': Linear background intercept
        - 'fit_params': Full fit parameters array
        - 'fit_cov': Fit covariance matrix
    """
    x = np.array(x)
    y = np.array(y)
    
    # Apply range filter if specified
    if xrange is not None and len(xrange) == 2:
        idx = (x >= xrange[0]) & (x <= xrange[1])
        x = x[idx]
        y = y[idx]
    
    # Default initial guess
    if initial_guess is None:
        idx_max = np.argmax(y)
        y_max = y[idx_max]
        x_max = x[idx_max]
        y_min = np.min(y)
        y_range = y_max - y_min
        
        # Estimate sigma from peak width at half max
        y_half = (y_max + y_min) / 2
        half_max_indices = np.where(y > y_half)[0]
        if len(half_max_indices) > 0:
            sigma_est = (x[half_max_indices[-1]] - x[half_max_indices[0]]) / 2.355
        else:
            sigma_est = (x[-1] - x[0]) / 10
        
        # Estimate linear background
        slope_est = (y[-1] - y[0]) / (x[-1] - x[0]) if len(x) > 1 else 0
        intercept_est = y[0] - slope_est * x[0]
        
        initial_guess = {
            'amplitude': y_range,
            'center': x_max,
            'sigma': max(sigma_est, (x[-1] - x[0]) / 100),  # Ensure positive
            'slope': slope_est,
            'intercept': intercept_est
        }
    
    # Prepare initial guess array
    p0 = [
        initial_guess.get('amplitude', np.max(y) - np.min(y)),
        initial_guess.get('center', x[np.argmax(y)]),
        initial_guess.get('sigma', (x[-1] - x[0]) / 10),
        initial_guess.get('slope', 0),
        initial_guess.get('intercept', np.min(y))
    ]
    
    # Perform fit
    try:
        popt, pcov = curve_fit(
            gaussian_with_linear_background,
            x, y,
            p0=p0,
            maxfev=10000
        )
        
        amplitude, center, sigma, slope, intercept = popt
        
        # Calculate FWHM (Full Width at Half Maximum)
        # For Gaussian: FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        fwhm = 2.355 * abs(sigma)
        
        return {
            'amplitude': amplitude,
            'center': center,
            'sigma': sigma,
            'fwhm': fwhm,
            'slope': slope,
            'intercept': intercept,
            'fit_params': popt,
            'fit_cov': pcov
        }
    except Exception as e:
        raise RuntimeError(f"Gaussian fit failed: {e}")


def fft2d(image: np.ndarray, dx: float = 1.0, dy: float = 1.0, shift: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D FFT of an image.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array
    dx : float
        Pixel spacing in x-direction (for frequency units). Default: 1.0
    dy : float
        Pixel spacing in y-direction (for frequency units). Default: 1.0
    shift : bool
        If True, shift zero frequency to center (default: True)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (fft_magnitude, FX, FY) where:
        - fft_magnitude: Magnitude of FFT (2D array)
        - FX: Frequency meshgrid for x-axis (2D array, in units of 1/dx)
        - FY: Frequency meshgrid for y-axis (2D array, in units of 1/dy)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got {image.ndim}D")
    
    # Compute FFT
    ft = np.fft.fft2(image)
    
    # Shift zero frequency to center if requested
    if shift:
        ft = np.fft.fftshift(ft)
    
    # Calculate magnitude
    ft_magnitude = np.abs(ft)
    
    # Generate frequency arrays (in units of cycles/pixel)
    ny, nx = image.shape
    fx_1d = np.fft.fftfreq(nx, d=dx)  # cycles per unit (e.g., nm⁻¹ if dx in nm)
    fy_1d = np.fft.fftfreq(ny, d=dy)  # cycles per unit
    
    if shift:
        fx_1d = np.fft.fftshift(fx_1d)
        fy_1d = np.fft.fftshift(fy_1d)
    
    # Create meshgrid (2D arrays)
    FY, FX = np.meshgrid(fy_1d, fx_1d, indexing='ij')
    
    return ft_magnitude, FX, FY


def radial_distribution(fft_magnitude: np.ndarray, FX: np.ndarray, FY: np.ndarray,
                       k_max: Optional[float] = None, n_bins: int = 1000,
                       smooth: bool = True, smooth_sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial distribution of FFT magnitude.
    
    This function computes the mean FFT magnitude as a function of radial k-vector.
    Uses histogram binning for efficiency and optionally applies Gaussian smoothing
    for a smoother appearance.
    
    Parameters
    ----------
    fft_magnitude : np.ndarray
        Magnitude of 2D FFT (2D array)
    FX : np.ndarray
        Frequency meshgrid for x-axis (2D array)
    FY : np.ndarray
        Frequency meshgrid for y-axis (2D array)
    k_max : Optional[float]
        Maximum k value to include (None = use all)
    n_bins : int
        Number of radial bins (default: 1000 for smooth appearance)
    smooth : bool
        If True, apply Gaussian smoothing to the result (default: True)
    smooth_sigma : float
        Sigma parameter for Gaussian smoothing (default: 1.0)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (k_radial, mean_fft) where:
        - k_radial: Radial k values (1D array, bin centers)
        - mean_fft: Mean FFT magnitude at each k (1D array)
    """
    # Calculate radial distance (flatten 2D arrays)
    k_radial_2d = np.sqrt(FX**2 + FY**2)
    k_radial_flat = k_radial_2d.flatten()
    fft_magnitude_flat = fft_magnitude.flatten()
    
    # Apply k_max filter if specified
    if k_max is not None:
        mask = k_radial_flat <= k_max
        k_radial_flat = k_radial_flat[mask]
        fft_magnitude_flat = fft_magnitude_flat[mask]
    
    # Determine bin range
    k_min = 0.0
    k_max_actual = k_radial_flat.max()
    
    # Use np.histogram for efficient binning with weighted mean
    # Create bin edges
    bin_edges = np.linspace(k_min, k_max_actual, n_bins + 1)
    k_radial = (bin_edges[1:] + bin_edges[:-1]) / 2  # Bin centers
    
    # Compute histogram (counts) and weighted histogram (sum of values)
    counts, _ = np.histogram(k_radial_flat, bins=bin_edges)
    weighted_sum, _ = np.histogram(k_radial_flat, bins=bin_edges, weights=fft_magnitude_flat)
    
    # Compute mean (avoid division by zero)
    mean_fft = np.where(counts > 0, weighted_sum / counts, 0.0)
    
    # Apply Gaussian smoothing if requested
    if smooth and len(mean_fft) > 1:
        mean_fft = gaussian_filter(mean_fft, sigma=smooth_sigma)
    
    return k_radial, mean_fft


def subtract_plane_2d(image: np.ndarray) -> np.ndarray:
    """
    Fit and subtract a 2D plane from an image.
    
    This is a pure mathematical operation: image in, image out.
    Works with both real and complex images.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array (can be real or complex)
        
    Returns
    -------
    np.ndarray
        Image with plane subtracted (same dtype as input)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got {image.ndim}D")
    
    # Create coordinate grids
    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    
    # Handle complex data: fit plane to real part, but subtract from complex
    is_complex = np.iscomplexobj(image)
    if is_complex:
        image_real = np.real(image)
    else:
        image_real = image
    
    # Flatten arrays for fitting
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = image_real.flatten()
    
    # Fit plane: z = a*x + b*y + c
    A = np.vstack([x_flat, y_flat, np.ones(len(x_flat))]).T
    coeffs = np.linalg.lstsq(A, z_flat, rcond=None)[0]
    
    # Calculate plane
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    
    # Subtract plane from original (works for both real and complex)
    return image - plane


def subtract_linear_by_line_2d(image: np.ndarray) -> np.ndarray:
    """
    Subtract a linear fit from each fast-scan line (row) of an image.
    
    This is a pure mathematical operation: image in, image out.
    Works with both real and complex images.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array (can be real or complex)
        
    Returns
    -------
    np.ndarray
        Image with line-by-line background subtracted (same dtype as input)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got {image.ndim}D")
    
    import scipy.signal
    
    # Handle complex data: detrend real and imaginary parts separately
    is_complex = np.iscomplexobj(image)
    if is_complex:
        # Detrend real and imaginary parts separately
        real_detrended = scipy.signal.detrend(np.real(image), axis=1)
        imag_detrended = scipy.signal.detrend(np.imag(image), axis=1)
        return real_detrended + 1j * imag_detrended
    else:
        # Use scipy.signal.detrend which does line-by-line detrending
        return scipy.signal.detrend(image, axis=1)


def compute_fft_2d(image: np.ndarray, x_range: float, y_range: float,
                   x_pixels: Optional[int] = None, y_pixels: Optional[int] = None,
                   window_function: Optional[str] = None, kaiser_beta: float = 5.0) -> Dict[str, Any]:
    """
    Compute 2D FFT of an image with optional windowing.
    
    This is a pure mathematical operation: image in, FFT results out.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array (can be real or complex)
    x_range : float
        Real-space range in x-direction (nm)
    y_range : float
        Real-space range in y-direction (nm)
    x_pixels : Optional[int]
        Number of pixels in x-direction. If None, uses image.shape[1]
    y_pixels : Optional[int]
        Number of pixels in y-direction. If None, uses image.shape[0]
    window_function : Optional[str]
        Window function name. Supported: 'blackman', 'hanning' (or 'hann'), 
        'hamming', 'bartlett', 'kaiser'. If None, no windowing is applied.
    kaiser_beta : float
        Beta parameter for Kaiser window (default: 5.0). Only used if window_function='kaiser'.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'fft_magnitude': Magnitude of FFT
        - 'FX': Frequency meshgrid for x-axis (nm⁻¹, 2D array)
        - 'FY': Frequency meshgrid for y-axis (nm⁻¹, 2D array)
        - 'ft': Full FFT array (complex, shifted)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got {image.ndim}D")
    
    # Apply windowing if requested
    image_for_fft = image.copy()
    if window_function:
        window_name = window_function.lower()
        if window_name == "blackman":
            window = np.outer(
                np.blackman(image_for_fft.shape[0]),
                np.blackman(image_for_fft.shape[1])
            )
        elif window_name == "hanning" or window_name == "hann":
            window = np.outer(
                np.hanning(image_for_fft.shape[0]),
                np.hanning(image_for_fft.shape[1])
            )
        elif window_name == "hamming":
            window = np.outer(
                np.hamming(image_for_fft.shape[0]),
                np.hamming(image_for_fft.shape[1])
            )
        elif window_name == "bartlett":
            window = np.outer(
                np.bartlett(image_for_fft.shape[0]),
                np.bartlett(image_for_fft.shape[1])
            )
        elif window_name == "kaiser":
            window = np.outer(
                np.kaiser(image_for_fft.shape[0], kaiser_beta),
                np.kaiser(image_for_fft.shape[1], kaiser_beta)
            )
        else:
            raise ValueError(
                f"Unknown window function '{window_function}'. "
                f"Supported windows: 'blackman', 'hanning', 'hann', 'hamming', 'bartlett', 'kaiser'"
            )
        
        # Apply window directly to the image in real space
        image_for_fft = window * image_for_fft
    
    # Calculate pixel spacing for frequency conversion
    x_pixels = x_pixels or image_for_fft.shape[1]
    y_pixels = y_pixels or image_for_fft.shape[0]
    
    sxm_x = np.linspace(-x_range / 2, x_range / 2, x_pixels)
    sxm_y = np.linspace(-y_range / 2, y_range / 2, y_pixels)
    dx = sxm_x[1] - sxm_x[0]  # nm
    dy = sxm_y[1] - sxm_y[0]  # nm
    
    # Compute FFT using fft2d function
    ft_magnitude, FX, FY = fft2d(image_for_fft, dx=dx, dy=dy, shift=True)
    
    # Return results
    return {
        'fft_magnitude': ft_magnitude,
        'FX': FX,  # 2D meshgrid
        'FY': FY,  # 2D meshgrid
        'ft': np.fft.fftshift(np.fft.fft2(image_for_fft)),  # Complex FFT, shifted
    }

