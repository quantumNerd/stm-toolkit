"""
Utility functions for STM data analysis.

This module provides helper functions used across the toolkit,
including functions for data processing, analysis, and visualization.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


def find_first_line(fname: str) -> int:
    """
    Find the line index where "[DATA]" appears in a .dat file.
    
    Parameters
    ----------
    fname : str
        Path to the .dat file
        
    Returns
    -------
    int
        Line index where "[DATA]" appears
    """
    with open(fname, "r", encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if "[DATA]" in line:
            return i
    
    raise ValueError(f"Could not find [DATA] marker in file: {fname}")


def get_dat_column(fname: str, header_name: str) -> int:
    """
    Get the column index for a given header name in a .dat file.
    
    Parameters
    ----------
    fname : str
        Path to the .dat file
    header_name : str
        Name of the column header to find
        
    Returns
    -------
    int
        Column index (0-based)
    """
    data_line = find_first_line(fname)
    
    with open(fname, "r", encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
    
    # The header line is right after [DATA]
    header_line = lines[data_line + 1].strip()
    headers = header_line.split("\t")
    
    try:
        return headers.index(header_name)
    except ValueError:
        raise ValueError(f"Column '{header_name}' not found in file {fname}. Available columns: {headers}")


def find_crossing(y: np.ndarray, y0: float, bool_last: bool = False) -> List[int]:
    """
    Find indices where y crosses y0.
    
    Parameters
    ----------
    y : np.ndarray
        Array of y values
    y0 : float
        Threshold value
    bool_last : bool
        If True, return only the last crossing; if False, return all crossings
        
    Returns
    -------
    List[int]
        List of indices where crossing occurs
    """
    idx_arr = []
    for i in range(len(y) - 1):
        if y[i] < y0 and y[i + 1] > y0:
            idx_arr.append(i)
        if y[i] > y0 and y[i + 1] < y0:
            idx_arr.append(i)
    
    if bool_last:
        if len(idx_arr) > 0:
            return [idx_arr[-1]]
        else:
            return []
    else:
        return idx_arr


def find_first_occurrence(x: np.ndarray, x0: float, y: np.ndarray, y_target: float,
                         reverse_direction: bool = False) -> int:
    """
    Find the first occurrence where y crosses y_target, starting from x0.
    
    Parameters
    ----------
    x : np.ndarray
        X values
    x0 : float
        Starting x value
    y : np.ndarray
        Y values
    y_target : float
        Target y value to find
    reverse_direction : bool
        If True, search backwards from x0; if False, search forwards
        
    Returns
    -------
    int
        Index of first occurrence
    """
    x = np.array(x)
    y = np.array(y)
    idx = np.argmin(np.abs(x - x0))
    y0 = y[idx]
    i = idx
    
    if not reverse_direction:
        rg = range(idx, len(x), 1)
    else:
        rg = range(idx, -1, -1)
    
    for i in rg:
        idx_target = i
        if np.sign(y[i] - y_target) == -1 * np.sign(y0 - y_target):
            break
    
    return idx_target


# find_max_and_width moved to analysis.py
# For backward compatibility, we'll import it when needed to avoid circular imports
def find_max_and_width(*args, **kwargs):
    """Backward compatibility wrapper - redirects to analysis.find_max_and_width"""
    from .analysis import find_max_and_width as _find_max_and_width
    return _find_max_and_width(*args, **kwargs)


def get_thresh_interp(bias: np.ndarray, current: np.ndarray, thresh: float, npts_mul: int = 10) -> np.ndarray:
    """
    Find threshold bias values where current crosses threshold.
    
    Parameters
    ----------
    bias : np.ndarray
        Bias voltage array
    current : np.ndarray
        Current array (can be 2D with multiple curves)
    thresh : float
        Threshold current value
    npts_mul : int
        Interpolation multiplier for higher resolution
        
    Returns
    -------
    np.ndarray
        Array of threshold bias values
    """
    threshold_bias = []
    current_2d = np.atleast_2d(current)
    
    for row in current_2d:
        bias_interp = np.linspace(np.min(bias), np.max(bias), num=len(bias) * npts_mul)
        row_interp = np.interp(bias_interp, bias, row)
        threshold_bias.append(bias_interp[np.abs(row_interp - thresh).argmin()])
    
    return np.array(threshold_bias)


def get_unique(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get unique values and their first occurrence indices.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (unique_values, unique_indices)
    """
    unique_x = [x[0]]
    unique_x_idx = [0]
    for i in range(len(x) - 1):
        if not x[i + 1] == x[i]:
            unique_x.append(x[i + 1])
            unique_x_idx.append(i + 1)
    return np.array(unique_x), np.array(unique_x_idx)

