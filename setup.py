#!/usr/bin/env python3

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='tinygrad',
      version='0.4.0',
      description='You like pytorch? You like micrograd? You love tinygrad! heart',
      author='George Hotz',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['tinygrad', 'tinygrad.llops', 'tinygrad.nn'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy', 'requests', 'pillow', 'networkx'],
      python_requires='>=3.8',
      extras_require={
        'gpu': ["pyopencl", "six"],
        'llvm': ["llvmlite"],
        'cuda': ["pycuda"],
        'triton': ["triton>=2.0.0.dev20221202"],
        'testing': [
            "pytest",
            "torch~=1.11.0",
            "tqdm",
            "protobuf~=3.19.0",
            "onnx",
            "onnx2torch",
            "mypy",
            "pylint",
            "pre-commit",
        ],
      },
      include_package_data=True)
