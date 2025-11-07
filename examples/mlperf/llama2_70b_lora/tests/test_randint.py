#!/usr/bin/env python3
"""
Tensor.randint signature validation and method testing.

This module systematically tests different calling conventions for Tensor.randint
to identify the correct signature and validate parameter passing methods.
Used for debugging input tensor creation issues in the MLPerf LoRA implementation.
"""

import sys
from pathlib import Path
from typing import Callable, Final, List, Tuple

sys.path.insert(0, str(Path(__file__).parents[4]))

from tinygrad import Tensor


BATCH_SIZE: Final[int] = 2
SEQUENCE_LENGTH: Final[int] = 8
VOCABULARY_SIZE: Final[int] = 100

SUCCESS_PREFIX: Final[str] = "PASS"
FAILURE_PREFIX: Final[str] = "FAIL"


def _create_tuple_method() -> Tensor:
  """Create tensor using tuple shape parameter.
  
  Returns:
    Tensor created with tuple shape specification
  """
  return Tensor.randint(0, VOCABULARY_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH))


def _create_args_method() -> Tensor:
  """Create tensor using variadic shape arguments.
  
  Returns:
    Tensor created with variadic shape arguments
  """
  return Tensor.randint(0, VOCABULARY_SIZE, BATCH_SIZE, SEQUENCE_LENGTH)


def _create_shape_args_method() -> Tensor:
  """Create tensor using unpacked shape arguments with explicit parameters.
  
  Returns:
    Tensor created with unpacked shape arguments
  """
  return Tensor.randint(low=0, high=VOCABULARY_SIZE, *[BATCH_SIZE, SEQUENCE_LENGTH])


def _create_explicit_method() -> Tensor:
  """Create tensor using explicit keyword shape parameter.
  
  Returns:
    Tensor created with explicit keyword shape parameter
  """
  return Tensor.randint(low=0, high=VOCABULARY_SIZE, shape=(BATCH_SIZE, SEQUENCE_LENGTH))


def test_randint_signatures() -> None:
  """
  Test various Tensor.randint calling conventions to identify working signatures.
  
  Validates different parameter passing methods and reports success/failure
  for each approach. This helps identify the correct API usage pattern
  for tensor creation in production code.
  """
  print("Testing Tensor.randint signature validation...")
  
  test_methods: List[Tuple[str, Callable[[], Tensor]]] = [
    ("tuple", _create_tuple_method),
    ("args", _create_args_method),
    ("shape_args", _create_shape_args_method),
    ("explicit", _create_explicit_method),
  ]
  
  for method_name, method_func in test_methods:
    try:
      result: Tensor = method_func()
      print(f"{SUCCESS_PREFIX} {method_name}: {result.shape}")
    except Exception as e:
      print(f"{FAILURE_PREFIX} {method_name}: {e}")
  
  print(f"\nTensor.randint signature: {Tensor.randint.__doc__}")
  help(Tensor.randint)


if __name__ == "__main__":
  test_randint_signatures()