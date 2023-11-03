#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(name='tinygrad',
      version='0.7.0',
      description='You like pytorch? You like micrograd? You love tinygrad! <3',
      author='George Hotz',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['tinygrad', 'tinygrad.codegen', 'tinygrad.nn', 'tinygrad.renderer', 'tinygrad.runtime', 'tinygrad.shape', 'tinygrad.features'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=["numpy", "requests", "pillow", "tqdm", "networkx", "pyopencl", "PyYAML",
                        "pyobjc-framework-Metal; platform_system=='Darwin'",
                        "pyobjc-framework-Cocoa; platform_system=='Darwin'",
                        "pyobjc-framework-libdispatch; platform_system=='Darwin'"],
      python_requires='>=3.8',
      extras_require={
        'llvm': ["llvmlite"],
        'cuda': ["pycuda"],
        'arm': ["unicorn"],
        'triton': ["triton-nightly>=2.1.0.dev20231014192330", "pycuda"],
        'webgpu': ["wgpu"],
        'linting': [
            "flake8",
            "pylint",
            "mypy",
            "typing-extensions",
            "pre-commit",
            "ruff",
            "types-PyYAML",
            "types-tqdm",
        ],
        'testing': [
            "torch",
            "pytest",
            "pytest-xdist",
            "onnx==1.14.1",
            "onnx2torch",
            "opencv-python",
            "tabulate",
            "safetensors",
            "cloudpickle",
            "transformers",
            "sentencepiece",
            "tiktoken",
            "librosa",
        ]
      },
      include_package_data=True)
