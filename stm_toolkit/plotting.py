"""
Base plotting classes for STM data visualization.

This module provides abstract base classes for plotting functionality
that can be extended by specific file type plotters.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import matplotlib.pyplot as plt
import numpy as np


class BasePlotter(ABC):
    """
    Abstract base class for plotting STM data.
    
    Provides a common interface for all plotter classes.
    """
    
    def __init__(self, data_source: Any, **kwargs):
        """
        Initialize the plotter.
        
        Parameters
        ----------
        data_source : Any
            The data source object (e.g., SXMFile, DATFile)
        **kwargs
            Additional plotting parameters
        """
        self.data_source = data_source
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self._setup_plot(**kwargs)
    
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
    
    def show(self) -> None:
        """Display the plot."""
        if self.fig is not None:
            plt.show()
    
    def save(self, filename: str, dpi: int = 300, **kwargs) -> None:
        """
        Save the plot to a file.
        
        Parameters
        ----------
        filename : str
            Output filename
        dpi : int
            Resolution in dots per inch
        **kwargs
            Additional arguments passed to savefig
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, **kwargs)
    
    def close(self) -> None:
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def set_title(self, title: str) -> None:
        """Set the plot title."""
        if self.ax is not None:
            self.ax.set_title(title)
    
    def set_xlabel(self, label: str) -> None:
        """Set the x-axis label."""
        if self.ax is not None:
            self.ax.set_xlabel(label)
    
    def set_ylabel(self, label: str) -> None:
        """Set the y-axis label."""
        if self.ax is not None:
            self.ax.set_ylabel(label)

