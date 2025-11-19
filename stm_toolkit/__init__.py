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
from .sxm import SXMFile, SXMCollection, SXMPlotter
from .dat import DATFile, DATCollection
from .grid_spectroscopy import GridSpectroscopyFile, GridSpectroscopyCollection
from .qcodes import QCoDesDatabase, QCoDesCollection

__version__ = "0.1.0"

__all__ = [
    "BaseFile",
    "BaseFileCollection",
    "BasePlotter",
    "SXMFile",
    "SXMCollection",
    "SXMPlotter",
    "DATFile",
    "DATCollection",
    "GridSpectroscopyFile",
    "GridSpectroscopyCollection",
    "QCoDesDatabase",
    "QCoDesCollection",
]

