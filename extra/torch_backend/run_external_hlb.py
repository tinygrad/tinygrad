#!/usr/bin/env python3
"""
Runner for external hlb-CIFAR10 PyTorch code with tinygrad torch backend.

This script allows running the original hlb-CIFAR10 implementation:
    https://github.com/tysam-code/hlb-CIFAR10

with tinygrad as the underlying computation engine.

Usage:
    # Clone the repository first
    git clone https://github.com/tysam-code/hlb-CIFAR10.git

    # Run with tinygrad backend
    python extra/torch_backend/run_external_hlb.py --hlb-path ./hlb-CIFAR10/main.py

Environment variables:
    CUDA_VISIBLE_DEVICES=0  # Select GPU if tinygrad supports it
    DEBUG=1                 # Enable debug output
"""

# CRITICAL: Import tinygrad torch backend BEFORE loading any PyTorch code
import extra.torch_backend.backend  # noqa: F401

import sys
import torch
import importlib.util
import argparse
from pathlib import Path

# Set device to tinygrad's custom device
torch.set_default_device("tiny")


def load_external_module(filepath: str):
    """Dynamically load an external Python module."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    spec = importlib.util.spec_from_file_location("external_hlb", filepath)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["external_hlb"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Run external hlb-CIFAR10 with tinygrad torch backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download hlb-CIFAR10 first
  git clone https://github.com/tysam-code/hlb-CIFAR10.git

  # Run with tinygrad backend
  python extra/torch_backend/run_external_hlb.py --hlb-path ./hlb-CIFAR10/main.py

  # With custom hyperparameters
  python extra/torch_backend/run_external_hlb.py --hlb-path ./hlb-CIFAR10/main.py --steps 1000
        """
    )

    parser.add_argument(
        "--hlb-path",
        type=str,
        required=True,
        help="Path to hlb-CIFAR10 main.py script"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of training steps (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size (default: 512)"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate every N steps"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("hlb-CIFAR10 with tinygrad Torch Backend")
    print("=" * 70)
    print(f"Device: tiny (tinygrad)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"HLB script: {args.hlb_path}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    # Load the external module
    print(f"\nLoading: {args.hlb_path}")
    try:
        hlb_module = load_external_module(args.hlb_path)
    except Exception as e:
        print(f"Error loading module: {e}", file=sys.stderr)
        sys.exit(1)

    # Look for train function
    if hasattr(hlb_module, "train_cifar"):
        print("Found train_cifar() function, starting training...")
        try:
            hlb_module.train_cifar()
        except Exception as e:
            print(f"Error during training: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif hasattr(hlb_module, "main"):
        print("Found main() function, starting...")
        try:
            hlb_module.main()
        except Exception as e:
            print(f"Error during execution: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Error: Could not find train_cifar() or main() in the loaded module", file=sys.stderr)
        print(f"Available functions: {[x for x in dir(hlb_module) if not x.startswith('_')]}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
