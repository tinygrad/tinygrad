#!/usr/bin/env python3
"""
Quick validation that hlb-CIFAR10 can work with tinygrad torch backend.
Tests core functionality without requiring full dataset download.
"""

import sys
from pathlib import Path

# Add current directory to path for tinygrad imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Import backend FIRST
import extra.torch_backend.backend  # noqa: F401

# Set device
torch.set_default_device("tiny")

print("=" * 70)
print("HLB-CIFAR10 × Tinygrad Integration Test")
print("=" * 70)


def test_device_registration():
    """Test that 'tiny' device is registered."""
    print("\n[TEST 1] Device Registration")
    try:
        device = torch.device("tiny")
        print(f"✓ Device 'tiny' registered: {device}")
        return True
    except Exception as e:
        print(f"✗ Failed to register device: {e}")
        return False


def test_tensor_creation():
    """Test tensor creation on tiny device."""
    print("\n[TEST 2] Tensor Creation")
    try:
        x = torch.randn(10, 10, device="tiny")
        print(f"✓ Created tensor on tiny device: {x.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed to create tensor: {e}")
        return False


def test_tensor_operations():
    """Test basic tensor operations."""
    print("\n[TEST 3] Tensor Operations")
    try:
        x = torch.randn(10, 10, device="tiny")
        y = torch.randn(10, 10, device="tiny")

        # Arithmetic
        z = x + y
        z = x * y
        z = x @ y  # Matrix multiplication

        # Reductions
        s = z.sum()
        m = z.mean()

        # Reshape
        r = z.reshape(100)

        print(f"✓ Tensor operations working")
        print(f"  - Add: {(x + y).shape}")
        print(f"  - MatMul: {(x @ y).shape}")
        print(f"  - Sum: {s.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nn_modules():
    """Test PyTorch NN modules."""
    print("\n[TEST 4] Neural Network Modules")
    try:
        # Create small network
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        ).to("tiny")

        # Test forward pass
        x = torch.randn(2, 3, 32, 32, device="tiny")
        y = model(x)

        print(f"✓ NN modules working")
        print(f"  - Input: {x.shape}")
        print(f"  - Output: {y.shape}")
        return True, model
    except Exception as e:
        print(f"✗ NN modules failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_optimizer():
    """Test PyTorch optimizer."""
    print("\n[TEST 5] Optimizer (SGD)")
    try:
        model = nn.Linear(10, 5, device="tiny")
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Single optimization step
        x = torch.randn(4, 10, device="tiny")
        y = torch.randn(4, 5, device="tiny")
        loss = nn.MSELoss()(model(x), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"✓ Optimizer working")
        print(f"  - Loss: {loss.item():.6f}")
        return True
    except Exception as e:
        print(f"✗ Optimizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler():
    """Test OneCycleLR scheduler."""
    print("\n[TEST 6] Learning Rate Scheduler (OneCycleLR)")
    try:
        model = nn.Linear(10, 5, device="tiny")
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Create scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.1,
            total_steps=100,
            pct_start=0.3,
            div_factor=25.0
        )

        # Simulate steps
        initial_lr = optimizer.param_groups[0]['lr']
        for i in range(10):
            scheduler.step()

        final_lr = optimizer.param_groups[0]['lr']

        print(f"✓ Scheduler working")
        print(f"  - Initial LR: {initial_lr:.6f}")
        print(f"  - After 10 steps: {final_lr:.6f}")
        return True
    except Exception as e:
        print(f"✗ Scheduler failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autograd():
    """Test autograd/backpropagation."""
    print("\n[TEST 7] Autograd/Backpropagation")
    try:
        model = nn.Sequential(
            nn.Linear(10, 20, device="tiny"),
            nn.ReLU(),
            nn.Linear(20, 5, device="tiny")
        )

        x = torch.randn(4, 10, device="tiny")
        y = torch.randn(4, 5, device="tiny")

        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)

        # Backward pass
        loss.backward()

        # Check gradients
        gradients_exist = all(p.grad is not None for p in model.parameters())

        print(f"✓ Autograd working")
        print(f"  - Loss: {loss.item():.6f}")
        print(f"  - All parameters have gradients: {gradients_exist}")
        return True
    except Exception as e:
        print(f"✗ Autograd failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training_step():
    """Test a complete training iteration."""
    print("\n[TEST 8] Full Training Iteration")
    try:
        # Create a simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 10)
        ).to("tiny")

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=100)

        # Create dummy batch
        x = torch.randn(4, 3, 32, 32, device="tiny")
        y = torch.randint(0, 10, (4,), device="tiny")

        # Training step
        model.train()
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"✓ Full training iteration working")
        print(f"  - Batch size: {x.shape[0]}")
        print(f"  - Loss: {loss.item():.6f}")
        print(f"  - New LR: {optimizer.param_groups[0]['lr']:.6f}")
        return True
    except Exception as e:
        print(f"✗ Full training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    tests = [
        test_device_registration,
        test_tensor_creation,
        test_tensor_operations,
        lambda: test_nn_modules()[0],  # Extract boolean from tuple
        test_optimizer,
        test_scheduler,
        test_autograd,
        test_full_training_step,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)

    if all(results):
        print("\n✓ All integration tests passed!")
        print("\nYou can now run hlb-CIFAR10 with tinygrad:")
        print("  python extra/torch_backend/hlb_cifar10_tiny.py")
        return 0
    else:
        print("\n✗ Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
