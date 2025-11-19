# STM Toolkit

A Python package for analyzing Nanonis STM (Scanning Tunneling Microscopy) data.

## Overview

This toolkit provides utilities for loading and processing various file types from Nanonis STM measurements:

- **.sxm files**: 2D image loading and processing (background subtraction, FFT)
- **.dat files**: Curve loading and fitting
- **.3ds files**: Grid spectroscopy data loading and processing
- **QCoDes database files**: Database file handling from QCoDes processing

## Project Structure

```
stm-toolkit/
├── stm_toolkit/
│   ├── __init__.py          # Package initialization and exports
│   ├── base.py              # Abstract base classes (BaseFile, BaseFileCollection)
│   ├── sxm.py               # SXM file handling (SXMFile, SXMCollection)
│   ├── dat.py               # DAT file handling (DATFile, DATCollection)
│   ├── grid_spectroscopy.py # 3DS file handling (GridSpectroscopyFile, GridSpectroscopyCollection)
│   └── qcodes.py            # QCoDes database handling (QCoDesDatabase, QCoDesCollection)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Architecture

### Base Classes

- **`BaseFile`**: Abstract base class for individual file handlers
  - Stores raw and processed data separately
  - Defines interface: `load()`, `process()`
  
- **`BaseFileCollection`**: Abstract base class for collections of files
  - Handles multiple files with associated hyperparameters (e.g., gate voltage)
  - Defines interface: `load_all()`, `process_all()`

### File Type Classes

Each file type has two classes:

1. **Individual File Handler** (e.g., `SXMFile`):
   - Loads and processes a single file
   - Maintains raw and processed data
   - Provides type-specific processing methods

2. **Collection Handler** (e.g., `SXMCollection`):
   - Manages multiple files of the same type
   - Associates files with hyperparameters
   - Provides batch processing capabilities

## Installation

```bash
pip install -r requirements.txt
```

## Usage Example

```python
from stm_toolkit import SXMFile, SXMCollection

# Single file
sxm = SXMFile("path/to/file.sxm")
sxm.load()
sxm.process(background_subtract=True, fft=True)

# Collection with hyperparameters
file_paths = ["file1.sxm", "file2.sxm", "file3.sxm"]
gate_voltages = [0.1, 0.2, 0.3]
collection = SXMCollection(
    file_paths,
    hyperparameters={"gate_voltage": gate_voltages}
)
collection.load_all()
collection.process_all(background_subtract=True)
```

## Status

This project is currently in development. File loading implementations are placeholders waiting for sample files to be provided.

## License

[Add your license here]