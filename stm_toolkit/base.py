"""
Base classes for STM data file handling.

This module provides abstract base classes that define the interface
for individual files and collections of files.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path


class BaseFile(ABC):
    """
    Abstract base class for individual STM data files.
    
    Each file class should inherit from this and implement:
    - load(): Load raw data from file
    - process(): Process the raw data
    """
    
    def __init__(self, file_path: str | Path):
        """
        Initialize the file handler.
        
        Parameters
        ----------
        file_path : str | Path
            Path to the data file
        """
        self.file_path = Path(file_path)
        self.raw_data: Optional[Dict[str, Any]] = None
        self.processed_data: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None
        
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """
        Load raw data from the file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the raw data
        """
        pass
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Process the raw data.
        
        Parameters
        ----------
        **kwargs
            Processing parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the processed data
        """
        pass
    
    def get_raw_data(self) -> Optional[Dict[str, Any]]:
        """Get the raw data."""
        return self.raw_data
    
    def get_processed_data(self) -> Optional[Dict[str, Any]]:
        """Get the processed data."""
        return self.processed_data
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get the file metadata."""
        return self.metadata


class BaseFileCollection(ABC):
    """
    Abstract base class for collections of STM data files.
    
    This class handles multiple files of the same type with associated
    hyperparameters (e.g., gate voltage, temperature).
    """
    
    def __init__(self, file_paths: List[str] | List[Path], hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the file collection.
        
        Parameters
        ----------
        file_paths : List[str] | List[Path]
            List of paths to data files
        hyperparameters : Optional[Dict[str, Any]]
            Dictionary of hyperparameters (e.g., {'gate_voltage': [0.1, 0.2, ...]})
        """
        self.file_paths = [Path(p) for p in file_paths]
        self.hyperparameters = hyperparameters or {}
        self.files: List[BaseFile] = []
        
    @abstractmethod
    def load_all(self) -> None:
        """Load all files in the collection."""
        pass
    
    @abstractmethod
    def process_all(self, **kwargs) -> None:
        """
        Process all files in the collection.
        
        Parameters
        ----------
        **kwargs
            Processing parameters
        """
        pass
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get the hyperparameters dictionary."""
        return self.hyperparameters
    
    def set_hyperparameter(self, key: str, value: Any) -> None:
        """Set a hyperparameter value."""
        self.hyperparameters[key] = value
    
    def get_file(self, index: int) -> BaseFile:
        """Get a specific file by index."""
        return self.files[index]
    
    def __len__(self) -> int:
        """Return the number of files in the collection."""
        return len(self.files)
    
    def __getitem__(self, index: int) -> BaseFile:
        """Allow indexing into the collection."""
        return self.files[index]

