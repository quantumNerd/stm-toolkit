# STM Toolkit - Project Review & Suggestions

## Overall Assessment

The project is well-structured with a clear object-oriented design. The separation of concerns (base classes, file handlers, plotters, analysis) is good. However, there are several areas for improvement.

---

## 1. Code Quality & Consistency

### ✅ Strengths
- Clean OOP architecture with abstract base classes
- Good separation of raw data and processing
- Consistent naming conventions
- Type hints are used throughout

### 🔧 Suggestions

#### 1.1 Inconsistent Return Types
- **Issue**: `BaseFile.process()` returns `Dict[str, Any]`, but some implementations may not return consistent structures
- **Suggestion**: Define a `TypedDict` or dataclass for return types to ensure consistency

```python
from typing import TypedDict

class ProcessedData(TypedDict):
    image: Optional[np.ndarray]
    fft: Optional[Dict[str, Any]]
    # ... other fields
```

#### 1.2 Error Handling
- **Issue**: Some file loading methods have basic error handling, but could be more robust
- **Suggestion**: 
  - Add custom exception classes (`STMFileError`, `STMProcessingError`)
  - Provide more context in error messages
  - Add validation for file formats before processing

```python
class STMFileError(Exception):
    """Base exception for STM file operations"""
    pass

class STMFileNotFoundError(STMFileError):
    """Raised when file is not found"""
    pass

class STMFileFormatError(STMFileError):
    """Raised when file format is invalid"""
    pass
```

#### 1.3 Magic Numbers
- **Issue**: Hard-coded values scattered throughout (e.g., `+5` for data offset, `1e12` for unit conversion)
- **Suggestion**: Define constants at module level

```python
# In sxm.py
SCANIT_END_OFFSET = 5  # Bytes after :SCANIT_END: marker
PA_TO_A = 1e-12  # Picoampere to Ampere conversion
```

---

## 2. Architecture & Design

### ✅ Strengths
- Good use of abstract base classes
- Separation of data loading, processing, and plotting
- On-the-fly processing (no storage) is memory-efficient

### 🔧 Suggestions

#### 2.1 BaseFile Interface
- **Issue**: `BaseFile` has `processed_data` attribute, but implementations don't store processed data anymore
- **Suggestion**: Remove `processed_data` from `BaseFile` or make it optional/clear that it's not used

#### 2.2 Collection Processing
- **Issue**: `process_all()` returns results but doesn't store them - this might be confusing
- **Suggestion**: 
  - Either store results in collection, or
  - Rename to `process_all_and_return()` to be explicit
  - Add `process_all_and_store()` if storage is needed

#### 2.3 Plotter Initialization
- **Issue**: Lazy plotting is good, but the `_plot_initialized` flag management could be clearer
- **Suggestion**: Consider using a property decorator or context manager pattern

#### 2.4 Circular Import Risk
- **Issue**: `utils.py` has a backward compatibility wrapper that imports from `analysis.py`
- **Suggestion**: Remove the wrapper or document why it's needed. Consider moving all analysis functions to `analysis.py` and deprecating `utils.find_max_and_width`

---

## 3. Documentation

### ✅ Strengths
- Good docstrings for classes and methods
- README provides overview

### 🔧 Suggestions

#### 3.1 README Updates
- **Issue**: README mentions "processed data storage" but code now does on-the-fly processing
- **Suggestion**: Update README to reflect current architecture:
  - Remove references to stored processed data
  - Add examples of on-the-fly processing
  - Document the `ax` parameter for subplots
  - Add API reference section

#### 3.2 Docstring Consistency
- **Issue**: Some docstrings are more detailed than others
- **Suggestion**: 
  - Use consistent format (NumPy or Google style)
  - Add "Raises" sections for exceptions
  - Add "Examples" sections for complex methods

#### 3.3 Type Hints in Docstrings
- **Issue**: Some docstrings duplicate type information from type hints
- **Suggestion**: Use type hints primarily, docstrings for descriptions

#### 3.4 Missing Documentation
- **Issue**: No documentation for:
  - Configuration options (voltage key mappings)
  - Processing parameters
  - Plotting customization
- **Suggestion**: Add a "User Guide" section to README or create separate docs

---

## 4. Testing

### ⚠️ Critical Gap
- **Issue**: No visible test suite
- **Suggestion**: 
  - Add `tests/` directory with unit tests
  - Use `pytest` for testing
  - Test file loading, processing, and plotting
  - Add sample test files (small, synthetic data)
  - Test error cases (missing files, invalid formats)

```python
# tests/test_sxm.py
import pytest
from stm_toolkit import SXMFile

def test_sxm_file_not_found():
    with pytest.raises(FileNotFoundError):
        sxm = SXMFile("nonexistent.sxm")
        sxm.load()

def test_sxm_load_valid_file():
    # Test with sample file
    pass
```

---

## 5. Performance

### ✅ Strengths
- On-the-fly processing saves memory
- Efficient use of numpy operations

### 🔧 Suggestions

#### 5.1 Large File Handling
- **Issue**: No chunking or memory-mapped file support for very large files
- **Suggestion**: Add optional memory-mapped file support for large `.sxm` files

#### 5.2 FFT Computation
- **Issue**: FFT is recomputed every time `fft()` is called
- **Suggestion**: Consider optional caching with a flag (but keep default as on-the-fly)

#### 5.3 Collection Processing
- **Issue**: `process_all()` processes sequentially
- **Suggestion**: Add optional parallel processing for collections

```python
from concurrent.futures import ThreadPoolExecutor

def process_all(self, parallel: bool = False, **kwargs):
    if parallel:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda f: f.process(**kwargs), 
                self.files
            ))
    else:
        # Sequential processing
```

---

## 6. Code Organization

### 🔧 Suggestions

#### 6.1 Module Size
- **Issue**: `sxm.py` is very large (~1200 lines)
- **Suggestion**: Consider splitting into:
  - `sxm_file.py` - File loading
  - `sxm_processing.py` - Background subtraction, FFT
  - `sxm_plotter.py` - Plotting (already separate but could be cleaner)

#### 6.2 Unused Imports
- **Issue**: Some imports may not be used (e.g., `scipy.signal` vs `savgol_filter`)
- **Suggestion**: Run `pylint` or `flake8` to find unused imports

#### 6.3 Test Files in Root
- **Issue**: Test files (`test_*.py`) are in root directory
- **Suggestion**: Move to `tests/` directory or add to `.gitignore` if they're temporary

---

## 7. Missing Features / TODOs

### High Priority
1. **Polynomial background subtraction** - Currently raises `NotImplementedError`
2. **Curve fitting in DATFile** - Currently raises `NotImplementedError`
3. **Normalization/smoothing in GridSpectroscopy** - Currently raises `NotImplementedError`

### Medium Priority
1. **Measurement filtering in QCoDes** - Currently raises `NotImplementedError`
2. **Measurement aggregation in QCoDes** - Currently raises `NotImplementedError`

### Low Priority
1. **Unit tests** - Critical but separate from features
2. **Performance optimizations** - Can be done incrementally

---

## 8. Best Practices

### 🔧 Suggestions

#### 8.1 Logging
- **Issue**: No logging system
- **Suggestion**: Add `logging` module for debugging and user feedback

```python
import logging
logger = logging.getLogger(__name__)

def load(self):
    logger.debug(f"Loading file: {self.file_path}")
    # ...
```

#### 8.2 Configuration
- **Issue**: Hard-coded defaults (e.g., voltage keys, multipass indices)
- **Suggestion**: Create a configuration class or use `dataclasses` for defaults

```python
@dataclass
class SXMConfig:
    gate_voltage_key: Optional[str] = None
    bias_voltage_key: Optional[str] = None
    multipass_config_index: int = -2
    # ...
```

#### 8.3 Validation
- **Issue**: Limited input validation
- **Suggestion**: Add validation decorators or methods

```python
def validate_file_path(func):
    def wrapper(self, *args, **kwargs):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        return func(self, *args, **kwargs)
    return wrapper
```

#### 8.4 Version Management
- **Issue**: Version is hard-coded in `__init__.py`
- **Suggestion**: Consider using `setuptools_scm` or reading from `pyproject.toml`

---

## 9. Dependencies

### ✅ Good
- Minimal dependencies (numpy, scipy, matplotlib, qcodes)
- Version constraints are reasonable

### 🔧 Suggestions
- Add `pytest` to `requirements-dev.txt` for development
- Consider adding `typing-extensions` for better type hint support on older Python versions
- Document Python version requirements (3.8+?)

---

## 10. Immediate Action Items

### High Priority
1. ✅ **Update README** - Reflect current architecture (on-the-fly processing)
2. ✅ **Add unit tests** - At least basic file loading tests
3. ✅ **Fix/Remove TODOs** - Either implement or document why not implemented
4. ✅ **Add error classes** - Custom exceptions for better error handling

### Medium Priority
5. ✅ **Add logging** - For debugging and user feedback
6. ✅ **Documentation** - API reference, user guide
7. ✅ **Code organization** - Split large files if needed

### Low Priority
8. ✅ **Performance** - Parallel processing, caching options
9. ✅ **Configuration** - Centralized config management
10. ✅ **Validation** - Input validation decorators

---

## Summary

The project has a solid foundation with good architecture. The main areas for improvement are:
1. **Testing** - Critical gap
2. **Documentation** - Needs updates to reflect current implementation
3. **Error handling** - Could be more robust
4. **Code organization** - Some files are very large
5. **Missing features** - Several NotImplementedError cases

The codebase is maintainable and well-structured, but would benefit from the suggestions above to make it production-ready.

