"""
Base plotting classes for STM data visualization.

This module provides abstract base classes for plotting functionality
that can be extended by specific file type plotters. It unifies common
plotting behaviors like figure size, saving, and styling across all
1D and 2D plotters.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class BasePlotter(ABC):
    """
    Abstract base class for plotting STM data.
    
    Provides a common interface for all plotter classes with unified
    figure size management, saving functionality, and styling options.
    """
    
    # Default figure sizes for different plot types
    DEFAULT_FIGSIZE_1D: Tuple[float, float] = (8, 6)  # Width, Height in inches
    DEFAULT_FIGSIZE_2D: Tuple[float, float] = (8, 8)  # Square for 2D images
    DEFAULT_FIGSIZE_FFT: Tuple[float, float] = (8, 8)  # Square for FFT
    DEFAULT_FIGSIZE_MULTI: Tuple[float, float] = (12, 5)  # For multi-panel plots
    
    # Default save settings
    DEFAULT_DPI: int = 300
    DEFAULT_FORMAT: str = 'png'
    
    def __init__(self, data_source: Any, ax: Optional[plt.Axes] = None,
                 figsize: Optional[Tuple[float, float]] = None,
                 dpi: int = DEFAULT_DPI, **kwargs):
        """
        Initialize the plotter.
        
        Parameters
        ----------
        data_source : Any
            The data source object (e.g., SXMFile, DATFile)
        ax : Optional[matplotlib.axes.Axes]
            Axes to plot on. If None, creates a new figure.
        figsize : Optional[Tuple[float, float]]
            Figure size (width, height) in inches. If None, uses default for plot type.
            Only used if ax is None.
        dpi : int
            Resolution in dots per inch for saving (default: 300)
        **kwargs
            Additional plotting parameters
        """
        self.data_source = data_source
        self.ax: Optional[plt.Axes] = ax
        self.fig: Optional[plt.Figure] = ax.figure if ax is not None else None
        self.figsize = figsize
        self.dpi = dpi
        self._plot_kwargs = kwargs  # Store kwargs for lazy initialization
        self._plot_initialized = False  # Flag to track if plot has been created
    
    @abstractmethod
    def _setup_plot(self, **kwargs) -> None:
        """
        Set up the initial plot.
        
        Parameters
        ----------
        **kwargs
            Plotting parameters
        """
        pass
    
    def _ensure_plot(self) -> None:
        """
        Ensure the plot has been created. Creates it if it hasn't been initialized yet.
        """
        if not self._plot_initialized:
            self._setup_plot(**self._plot_kwargs)
            self._plot_initialized = True
    
    def plot(self) -> 'BasePlotter':
        """
        Create and return the plotter. This triggers plot creation.
        
        Returns
        -------
        BasePlotter
            Self (for method chaining)
        """
        self._ensure_plot()
        return self
    
    def _get_figsize(self, default: Tuple[float, float], **kwargs) -> Tuple[float, float]:
        """
        Get figure size, using provided figsize or default.
        
        Parameters
        ----------
        default : Tuple[float, float]
            Default figure size for this plot type
        **kwargs
            May contain 'figsize' key
            
        Returns
        -------
        Tuple[float, float]
            Figure size (width, height) in inches
        """
        # Priority: 1) self.figsize (from __init__), 2) kwargs['figsize'], 3) default
        if self.figsize is not None:
            return self.figsize
        return kwargs.get('figsize', default)
    
    def show(self) -> None:
        """
        Display the plot.
        
        This will show the figure using plt.show(). In Jupyter notebooks,
        the figure will be displayed inline.
        """
        self._ensure_plot()
        if self.fig is not None:
            plt.show()
    
    def save(self, filename: Union[str, Path], dpi: Optional[int] = None,
             format: Optional[str] = None, bbox_inches: str = 'tight',
             **kwargs) -> None:
        """
        Save the plot to a file.
        
        Parameters
        ----------
        filename : str or Path
            Output filename. Format is inferred from extension if not specified.
        dpi : Optional[int]
            Resolution in dots per inch. If None, uses self.dpi (default: 300)
        format : Optional[str]
            File format ('png', 'pdf', 'svg', 'eps', etc.). If None, inferred from filename.
        bbox_inches : str
            Bounding box in inches. Default is 'tight' to remove extra whitespace.
        **kwargs
            Additional arguments passed to savefig (e.g., facecolor, edgecolor, transparent)
            
        Examples
        --------
        >>> plotter.save('output.png')
        >>> plotter.save('output.pdf', dpi=600)
        >>> plotter.save('output.png', transparent=True, facecolor='white')
        """
        self._ensure_plot()
        if self.fig is None:
            raise ValueError("No figure to save. Plot must be created first.")
        
        filename = Path(filename)
        
        # Determine format
        if format is None:
            format = filename.suffix[1:] if filename.suffix else self.DEFAULT_FORMAT
        
        # Use provided dpi or default
        save_dpi = dpi if dpi is not None else self.dpi
        
        # Save with common defaults
        self.fig.savefig(
            filename,
            dpi=save_dpi,
            format=format,
            bbox_inches=bbox_inches,
            **kwargs
        )
    
    def close(self) -> None:
        """Close the figure and free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def set_title(self, title: str, **kwargs) -> None:
        """
        Set the plot title.
        
        Parameters
        ----------
        title : str
            Title text
        **kwargs
            Additional arguments passed to ax.set_title (e.g., fontsize, pad)
        """
        self._ensure_plot()
        if self.ax is not None:
            self.ax.set_title(title, **kwargs)
    
    def set_xlabel(self, label: str, **kwargs) -> None:
        """
        Set the x-axis label.
        
        Parameters
        ----------
        label : str
            Label text
        **kwargs
            Additional arguments passed to ax.set_xlabel (e.g., fontsize)
        """
        self._ensure_plot()
        if self.ax is not None:
            self.ax.set_xlabel(label, **kwargs)
    
    def set_ylabel(self, label: str, **kwargs) -> None:
        """
        Set the y-axis label.
        
        Parameters
        ----------
        label : str
            Label text
        **kwargs
            Additional arguments passed to ax.set_ylabel (e.g., fontsize)
        """
        self._ensure_plot()
        if self.ax is not None:
            self.ax.set_ylabel(label, **kwargs)
    
    def set_xlim(self, left: Optional[float] = None, right: Optional[float] = None, **kwargs) -> None:
        """
        Set the x-axis limits.
        
        Parameters
        ----------
        left : Optional[float]
            Left limit. If None, uses current limit.
        right : Optional[float]
            Right limit. If None, uses current limit.
        **kwargs
            Additional arguments passed to ax.set_xlim (e.g., emit, auto)
        """
        self._ensure_plot()
        if self.ax is not None:
            self.ax.set_xlim(left, right, **kwargs)
    
    def set_ylim(self, bottom: Optional[float] = None, top: Optional[float] = None, **kwargs) -> None:
        """
        Set the y-axis limits.
        
        Parameters
        ----------
        bottom : Optional[float]
            Bottom limit. If None, uses current limit.
        top : Optional[float]
            Top limit. If None, uses current limit.
        **kwargs
            Additional arguments passed to ax.set_ylim (e.g., emit, auto)
        """
        self._ensure_plot()
        if self.ax is not None:
            self.ax.set_ylim(bottom, top, **kwargs)
    
    def set_xscale(self, value: str, **kwargs) -> None:
        """
        Set the x-axis scale.
        
        Parameters
        ----------
        value : str
            Scale type ('linear', 'log', 'symlog', 'logit', etc.)
        **kwargs
            Additional arguments passed to ax.set_xscale
        """
        self._ensure_plot()
        if self.ax is not None:
            self.ax.set_xscale(value, **kwargs)
    
    def set_yscale(self, value: str, **kwargs) -> None:
        """
        Set the y-axis scale.
        
        Parameters
        ----------
        value : str
            Scale type ('linear', 'log', 'symlog', 'logit', etc.)
        **kwargs
            Additional arguments passed to ax.set_yscale
        """
        self._ensure_plot()
        if self.ax is not None:
            self.ax.set_yscale(value, **kwargs)
    
    def grid(self, visible: bool = True, which: str = 'major', **kwargs) -> None:
        """
        Configure the grid lines.
        
        Parameters
        ----------
        visible : bool
            Whether to show grid lines (default: True)
        which : str
            Which grid lines to show ('major', 'minor', 'both')
        **kwargs
            Additional arguments passed to ax.grid (e.g., alpha, linestyle, color)
        """
        self._ensure_plot()
        if self.ax is not None:
            self.ax.grid(visible=visible, which=which, **kwargs)
    
    def legend(self, *args, **kwargs) -> None:
        """
        Place a legend on the axes.
        
        Parameters
        ----------
        *args
            Positional arguments passed to ax.legend
        **kwargs
            Keyword arguments passed to ax.legend (e.g., loc, labels, title)
        """
        self._ensure_plot()
        if self.ax is not None:
            self.ax.legend(*args, **kwargs)
    
    def tight_layout(self, **kwargs) -> None:
        """
        Adjust subplot parameters to give specified padding.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to fig.tight_layout (e.g., pad, w_pad, h_pad)
        """
        self._ensure_plot()
        if self.fig is not None:
            self.fig.tight_layout(**kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate unknown attributes to the axes object.
        
        This makes the plotter behave like a matplotlib Axes object,
        automatically forwarding all axes methods and properties that aren't
        explicitly defined on the plotter. This includes:
        - set_xticks, set_yticks
        - set_xticklabels, set_yticklabels
        - tick_params
        - plot, scatter, imshow, pcolormesh, etc.
        - All other matplotlib axes methods and properties
        
        Parameters
        ----------
        name : str
            Attribute name
            
        Returns
        -------
        Any
            The attribute from self.ax (wrapped if callable to ensure plot initialization)
            
        Raises
        ------
        AttributeError
            If the attribute doesn't exist on the axes object
        """
        # Ensure plot is initialized before delegating
        self._ensure_plot()
        
        if self.ax is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}' "
                f"and no axes available"
            )
        
        if not hasattr(self.ax, name):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}' "
                f"and axes object has no attribute '{name}'"
            )
        
        attr = getattr(self.ax, name)
        
        # If it's a callable, wrap it to ensure plot is initialized (redundant but safe)
        if callable(attr):
            def wrapper(*args, **kwargs):
                self._ensure_plot()
                return attr(*args, **kwargs)
            # Preserve the original function's metadata
            wrapper.__name__ = attr.__name__
            wrapper.__doc__ = attr.__doc__
            return wrapper
        
        return attr
    

