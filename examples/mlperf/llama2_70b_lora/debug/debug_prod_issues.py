#!/usr/bin/env python3
"""
Debug production readiness test failures.

This module systematically tests and debugs three critical production issues:
1. Optimizer scope problems with LoRA parameter extraction
2. Serialization failures with safetensors and device management
3. End-to-end input shape mismatches in forward pass and loss computation

Each test isolates specific failure modes and attempts multiple resolution
strategies to identify the root cause and working solutions.
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Final

sys.path.insert(0, str(Path(__file__).parents[3]))

from tinygrad import Tensor, Device
from tinygrad.nn.state import get_parameters, safe_save, safe_load
from tinygrad.nn.optim import AdamW
from examples.mlperf.llama2_70b_lora.lora import (
    LoRALinear, LoRAConfig, apply_lora_to_model, LoRAParameterManager
)
from extra.models.llama import Transformer


SEPARATOR: Final[str] = "=" * 60
TEST_MODEL_CONFIG: Final[Dict[str, Any]] = {
    "dim": 128,
    "hidden_dim": 256,
    "n_heads": 4,
    "n_layers": 2,
    "norm_eps": 1e-5,
    "vocab_size": 1000,
    "max_context": 256,
    "jit": False
}

MINIMAL_MODEL_CONFIG: Final[Dict[str, Any]] = {
    "dim": 64,
    "hidden_dim": 128,
    "n_heads": 2,
    "n_layers": 1,
    "norm_eps": 1e-5,
    "vocab_size": 100,
    "max_context": 32,
    "jit": False
}

LORA_TEST_CONFIG: Final[LoRAConfig] = LoRAConfig(
    r=4, 
    alpha=8.0, 
    target_modules=["wq", "wv"]
)


def debug_optimizer_scope() -> None:
    """
    Debug optimizer scope issues with LoRA parameter extraction.
    
    Tests the complete pipeline from model creation through LoRA application
    to parameter extraction and optimizer instantiation. This addresses
    common scope and parameter access failures in production environments.
    """
    print("\nDEBUGGING OPTIMIZER SCOPE")
    
    try:
        model = Transformer(**TEST_MODEL_CONFIG)
        initial_param_count = len(get_parameters(model))
        print(f"Model created with {initial_param_count} parameters")
        
        apply_lora_to_model(model=model, config=LORA_TEST_CONFIG)
        print("LoRA adapters applied successfully")
        
        LoRAParameterManager.freeze_base_model(model=model)
        lora_params = LoRAParameterManager.get_lora_parameters(model=model)
        print(f"Extracted {len(lora_params)} LoRA parameters")
        
        optimizer = AdamW(lora_params, lr=1e-4)
        print("Optimizer created successfully")
        
    except Exception as e:
        print(f"Optimizer scope failed: {e}")
        import traceback
        traceback.print_exc()


def debug_serialization() -> None:
    """
    Debug serialization failures with multiple resolution strategies.
    
    Tests various approaches to handle device placement and tensor state
    issues that commonly cause safetensors serialization to fail. Attempts
    direct serialization, explicit CPU movement, realization, and numpy
    conversion strategies.
    """
    print("\nDEBUGGING SERIALIZATION")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test.safetensors"
            
            original_device = Device.DEFAULT
            Device.DEFAULT = "CPU"
            
            lora_layer = LoRALinear(64, 32, r=4, alpha=8.0)
            
            print(f"LoRA layer created on {Device.DEFAULT}")
            print(f"  linear.weight device: {lora_layer.linear.weight.device}")
            print(f"  lora_A.weight device: {lora_layer.lora_A.weight.device}")
            
            serialization_strategies = [
                ("direct", _create_direct_state_dict),
                ("to_cpu", _create_cpu_state_dict),
                ("realize_first", _create_realized_state_dict),
                ("numpy_detour", _create_numpy_state_dict)
            ]
            
            for strategy_name, strategy_func in serialization_strategies:
                if _test_serialization_strategy(
                    strategy_name, strategy_func, lora_layer, checkpoint_path
                ):
                    break
            
            Device.DEFAULT = original_device
            
    except Exception as e:
        print(f"Serialization debug failed: {e}")


def debug_e2e_input_shapes() -> None:
    """
    Debug end-to-end input shape mismatches in forward pass and loss computation.
    
    Tests different input tensor creation methods and validates the complete
    forward pass pipeline including logit computation, label shifting, and
    loss calculation. Addresses common dtype and shape compatibility issues.
    """
    print("\nDEBUGGING E2E INPUT SHAPES")
    
    try:
        model = Transformer(**MINIMAL_MODEL_CONFIG)
        batch_size, seq_len, vocab_size = 2, 8, 100
        
        input_creation_methods = [
            ("randint", _create_randint_inputs),
            ("manual", _create_manual_inputs),
            ("dtype_explicit", _create_explicit_dtype_inputs)
        ]
        
        for method_name, method_func in input_creation_methods:
            if _test_input_method(
                method_name, method_func, model, batch_size, seq_len, vocab_size
            ):
                break
                
    except Exception as e:
        print(f"E2E debug failed: {e}")


def _create_direct_state_dict(lora_layer: LoRALinear) -> Dict[str, Tensor]:
    """Create state dict with direct tensor references.
    
    Args:
      lora_layer: LoRA layer to create state dict from
      
    Returns:
      State dict with direct tensor references
    """
    return {
        "w": lora_layer.linear.weight,
        "a": lora_layer.lora_A.weight,
        "b": lora_layer.lora_B.weight
    }


def _create_cpu_state_dict(lora_layer: LoRALinear) -> Dict[str, Tensor]:
    """Create state dict with explicit CPU device placement.
    
    Args:
      lora_layer: LoRA layer to create state dict from
      
    Returns:
      State dict with explicit CPU device placement
    """
    return {
        "w": lora_layer.linear.weight.to("CPU"),
        "a": lora_layer.lora_A.weight.to("CPU"),
        "b": lora_layer.lora_B.weight.to("CPU")
    }


def _create_realized_state_dict(lora_layer: LoRALinear) -> Dict[str, Tensor]:
    """Create state dict with realized tensors moved to CPU.
    
    Args:
      lora_layer: LoRA layer to create state dict from
      
    Returns:
      State dict with realized tensors moved to CPU
    """
    return {
        "w": lora_layer.linear.weight.realize().to("CPU"),
        "a": lora_layer.lora_A.weight.realize().to("CPU"),
        "b": lora_layer.lora_B.weight.realize().to("CPU")
    }


def _create_numpy_state_dict(lora_layer: LoRALinear) -> Dict[str, Tensor]:
    """Create state dict using numpy conversion to avoid device issues.
    
    Args:
      lora_layer: LoRA layer to create state dict from
      
    Returns:
      State dict using numpy conversion to avoid device issues
    """
    return {
        "w": Tensor(lora_layer.linear.weight.numpy()),
        "a": Tensor(lora_layer.lora_A.weight.numpy()),
        "b": Tensor(lora_layer.lora_B.weight.numpy())
    }


def _test_serialization_strategy(
    strategy_name: str,
    strategy_func: callable,
    lora_layer: LoRALinear,
    checkpoint_path: Path
) -> bool:
    """
    Test a specific serialization strategy.
    
    Args:
        strategy_name: Human-readable name for the strategy
        strategy_func: Function that creates the state dict
        lora_layer: LoRA layer to serialize
        checkpoint_path: Path for temporary checkpoint file
        
    Returns:
        True if strategy succeeded, False otherwise
    """
    try:
        print(f"  Testing {strategy_name} approach...")
        state_dict = strategy_func(lora_layer)
        safe_save(state_dict, checkpoint_path)
        loaded = safe_load(checkpoint_path)
        print(f"    {strategy_name} approach successful")
        return True
    except Exception as e:
        print(f"    {strategy_name} failed: {e}")
        return False


def _create_randint_inputs(batch_size: int, seq_len: int, vocab_size: int) -> tuple:
    """Create inputs using Tensor.randint.
    
    Args:
      batch_size: Batch size for input tensors
      seq_len: Sequence length for input tensors
      vocab_size: Vocabulary size for input tensors
      
    Returns:
      Input tensors using Tensor.randint
    """
    input_ids = Tensor.randint(0, vocab_size, (batch_size, seq_len))
    labels = Tensor.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels


def _create_manual_inputs(batch_size: int, seq_len: int, vocab_size: int) -> tuple:
    """Create inputs using manual numpy array construction.
    
    Args:
      batch_size: Batch size for input tensors
      seq_len: Sequence length for input tensors
      vocab_size: Vocabulary size for input tensors
      
    Returns:
      Input tensors using manual numpy array construction
    """
    import numpy as np
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32))
    labels = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32))
    return input_ids, labels


def _create_explicit_dtype_inputs(batch_size: int, seq_len: int, vocab_size: int) -> tuple:
    """Create inputs with explicit dtype casting.

    Args:
      batch_size: Batch size for input tensors
      seq_len: Sequence length for input tensors
      vocab_size: Vocabulary size for input tensors
      
    Returns:
      Input tensors with explicit dtype casting
    """
    input_ids = Tensor.randint(0, vocab_size, (batch_size, seq_len)).cast("int32")
    labels = Tensor.randint(0, vocab_size, (batch_size, seq_len)).cast("int32")
    return input_ids, labels


def _test_input_method(
    method_name: str,
    method_func: callable,
    model: Transformer,
    batch_size: int,
    seq_len: int,
    vocab_size: int
) -> bool:
    """
    Test a specific input creation method with full forward pass.
    
    Args:
        method_name: Human-readable name for the method
        method_func: Function that creates input tensors
        model: Transformer model for testing
        batch_size: Batch size for input tensors
        seq_len: Sequence length for input tensors
        vocab_size: Vocabulary size for input tensors
        
    Returns:
        True if method succeeded, False otherwise
    """
    try:
        print(f"  Testing {method_name} input creation...")
        
        input_ids, labels = method_func(batch_size, seq_len, vocab_size)
        
        print(f"    input_ids.shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        print(f"    labels.shape: {labels.shape}, dtype: {labels.dtype}")
        
        logits = model.forward(input_ids, start_pos=0, temperature=float('nan'))
        print(f"    logits.shape: {logits.shape}")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        print(f"    shift_logits.shape: {shift_logits.shape}")
        print(f"    shift_labels.shape: {shift_labels.shape}")
        
        shift_logits_flat = shift_logits.reshape(-1, shift_logits.shape[-1])
        shift_labels_flat = shift_labels.reshape(-1)
        
        print(f"    flattened: {shift_logits_flat.shape}, {shift_labels_flat.shape}")
        
        loss = shift_logits_flat.sparse_categorical_crossentropy(shift_labels_flat)
        print(f"    {method_name} method successful! Loss: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"    {method_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Execute all debug tests systematically."""
    print("DEBUGGING PRODUCTION READINESS ISSUES")
    print(SEPARATOR)
    
    debug_optimizer_scope()
    debug_serialization()
    debug_e2e_input_shapes()
    
    print(f"\n{SEPARATOR}")
    print("DEBUG COMPLETE")


if __name__ == "__main__":
    main()