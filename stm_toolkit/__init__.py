"""
STM Toolkit - A Python package for analyzing Nanonis STM data.

This package provides utilities for loading and processing:
- .sxm files (2D images)
- .dat files (curves)
- .3ds files (grid spectroscopy)
- QCoDes database files
"""

from .base import BaseFile, BaseFileCollection
from .plotting import BasePlotter
from .sxm import SXMFile, SXMCollection, SXMPlotter, ProcessedSXMFile
from .dat import DATFile, DATCollection, DATPlotter
from .grid_spectroscopy import GridSpectroscopyFile, GridSpectroscopyCollection
from .qcodes import QCoDesDatabase, QCoDesCollection, simple_plot_dts
from . import utils
from . import analysis

__version__ = "0.1.0"

__all__ = [
    "BaseFile",
    "BaseFileCollection",
    "BasePlotter",
    "SXMFile",
    "SXMCollection",
    "SXMPlotter",
    "ProcessedSXMFile",
    "DATFile",
    "DATCollection",
    "DATPlotter",
    "GridSpectroscopyFile",
    "GridSpectroscopyCollection",
    "QCoDesDatabase",
    "QCoDesCollection",
    "simple_plot_dts",
    "utils",
    "analysis",
]

