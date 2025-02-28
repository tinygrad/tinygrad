#!/usr/bin/env python3
import os
# Set TYPED=1 to enable runtime type checking
os.environ["TYPED"] = "1"

# Ensure numpy is imported
import numpy as np
from typing import Optional

# Import tinygrad after setting TYPED=1
from tinygrad import Tensor, typechecked
from tinygrad.helpers import fetch, getenv, Context

@typechecked
def example_with_type_annotations(x: Tensor, scale: float = 1.0) -> Tensor:
    return x * scale

@typechecked
def function_with_multiple_types(x: Tensor, y: list, z: dict, flag: bool = True) -> Tensor:
    if flag:
        return x * len(y) if z else x
    return x

@typechecked
def function_with_optional(x: Tensor, y: Optional[float] = None) -> Tensor:
    if y is not None:
        return x * y
    return x

def test_valid_types():
    # Basic valid type test
    x = Tensor([1, 2, 3, 4])
    y = example_with_type_annotations(x, 2.0)
    print("Basic valid type test passed:", y.shape, y.dtype)
    
    # Multiple valid types test
    y = function_with_multiple_types(x, [1, 2, 3], {"key": "value"}, True)
    print("Multiple valid types test passed:", y.shape, y.dtype)
    
    # Optional parameter test
    y = function_with_optional(x)
    print("Optional parameter test passed:", y.shape, y.dtype)
    y = function_with_optional(x, 2.5)
    print("Optional parameter with value test passed:", y.shape, y.dtype)

def test_invalid_types():
    # Basic invalid type test
    try:
        x = "not a tensor"
        y = example_with_type_annotations(x, 2.0)  # Should fail with TYPED=1
        print("TYPED is not working: Basic test should have failed!")
    except Exception as e:
        print(f"Basic type check caught error as expected: {type(e).__name__}")
    
    # Invalid float type
    try:
        x = Tensor([1, 2, 3, 4])
        y = example_with_type_annotations(x, "not a float")  # Should fail
        print("TYPED is not working: Float type test should have failed!")
    except Exception as e:
        print(f"Float type check caught error as expected: {type(e).__name__}")
    
    # Invalid list type
    try:
        x = Tensor([1, 2, 3, 4])
        y = function_with_multiple_types(x, "not a list", {"key": "value"}, True)
        print("TYPED is not working: List type test should have failed!")
    except Exception as e:
        print(f"List type check caught error as expected: {type(e).__name__}")
    
    # Invalid dict type
    try:
        x = Tensor([1, 2, 3, 4])
        y = function_with_multiple_types(x, [1, 2, 3], "not a dict", True)
        print("TYPED is not working: Dict type test should have failed!")
    except Exception as e:
        print(f"Dict type check caught error as expected: {type(e).__name__}")
    
    # Invalid boolean flag
    try:
        x = Tensor([1, 2, 3, 4])
        y = function_with_multiple_types(x, [1, 2, 3], {"key": "value"}, "not a boolean")
        print("TYPED is not working: Boolean type test should have failed!")
    except Exception as e:
        print(f"Boolean type check caught error as expected: {type(e).__name__}")

if __name__ == "__main__":
    print(f"Running with TYPED={os.getenv('TYPED')}")
    print("=== Testing Valid Types ===")
    test_valid_types()
    print("\n=== Testing Invalid Types ===")
    test_invalid_types()
    print("\n=== All Type Checking Tests Completed ===") 