# Installation Instructions for tinygrad

## Prerequisites

Before installing tinygrad, ensure you have the following prerequisites:

- Python 3.10 or higher
- pip (Python package installer)

## Installation Steps

1. **Clone the Repository** (optional)
   If you want to work with the latest version of tinygrad, you can clone the repository:
   ```
   git clone https://github.com/geohot/tinygrad.git
   cd tinygrad
   ```

2. **Install tinygrad**
   You can install tinygrad directly from PyPI using pip:
   ```
   pip install tinygrad
   ```

   Alternatively, if you cloned the repository, you can install it in editable mode:
   ```
   pip install -e .
   ```

3. **Install Optional Dependencies**
   Depending on your needs, you may want to install optional dependencies. For example:
   - For LLVM support:
     ```
     pip install llvmlite
     ```
   - For ARM support:
     ```
     pip install unicorn
     ```
   - For Triton support:
     ```
     pip install triton-nightly>=2.1.0.dev20231014192330
     ```

## Verification

To verify that tinygrad is installed correctly, you can run the following command in your Python environment:
```python
import tinygrad
print(tinygrad.__version__)
```

This should display the version of tinygrad you have installed.