#!/usr/bin/env python3

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='tinygrad',
      version='0.6.0',
      description='You like pytorch? You like micrograd? You love tinygrad! <3',
      author='George Hotz',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['tinygrad', 'tinygrad.codegen', 'tinygrad.nn', 'tinygrad.runtime', 'tinygrad.shape'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy', 'requests', 'pillow', 'tqdm', 'networkx', 'pyopencl'],
      python_requires='>=3.8',
      extras_require={
        'llvm': ["llvmlite"],
        'cuda': ["pycuda"],
        'triton': ["triton>=2.0.0.dev20221202"],
        'metal': ["pyobjc-framework-Metal", "pyobjc-framework-Cocoa", "pyobjc-framework-libdispatch"],
        'linting': [
            "flake8",
            "pylint",
            "mypy",
            "pre-commit",
        ],
        'testing': [
            "torch",
            "pytest",
            "pytest-xdist",
            "onnx",
            "onnx2torch",
            "opencv-python",
            "tabulate",
            "safetensors",
        ],
      },
      include_package_data=True)
