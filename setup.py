#!/usr/bin/env python3

import pathlib
from pathlib import Path

import pkg_resources
import setuptools
from setuptools import setup

def load_requirements(fn: str) -> list:
  with pathlib.Path(fn).open(encoding='utf-8') as f:
    return [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(f)
    ]

def load_file(fn: str) -> list:
  with pathlib.Path(fn).open(encoding='utf-8') as f:
    return f.read()

directory = Path(__file__).resolve().parent

install_requires = load_requirements('requirements.txt')
install_testing_requires = load_requirements('requirements-testing.txt')
long_description = load_file(directory / 'README.md')


setup(name='tinygrad',
      version='0.8.0',
      description='You like pytorch? You like micrograd? You love tinygrad! <3',
      author='George Hotz',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['tinygrad', 'tinygrad.codegen', 'tinygrad.nn', 'tinygrad.renderer',
                  'tinygrad.runtime', 'tinygrad.runtime.graph', 'tinygrad.shape', 'tinygrad.features'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=install_requires,
      python_requires='>=3.8',
      extras_require={
        'llvm': ["llvmlite"],
        'arm': ["unicorn"],
        'triton': ["triton-nightly>=2.1.0.dev20231014192330"],
        'webgpu': ["wgpu>=v0.12.0"],
        'linting': [
            "pylint",
            "mypy",
            "typing-extensions",
            "pre-commit",
            "ruff",
            "types-tqdm",
        ],
        'testing': install_testing_requires
      },
      include_package_data=True)
