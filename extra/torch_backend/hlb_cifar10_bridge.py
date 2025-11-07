"""
Bridge to run hlb-CIFAR10 PyTorch code with tinygrad torch backend.
Handles PyTorch OneCycleLR scheduler compatibility.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Optional

# Must import backend BEFORE creating any torch tensors
import extra.torch_backend.backend  # noqa: F401

# Set default device to tiny
torch.set_default_device("tiny")

class TinyOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    """Wrapper for OneCycleLR that works properly with tinygrad torch backend."""

    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, anneal_strategy='linear',
                 cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                 div_factor=25.0, final_div_factor=10000.0, last_epoch=-1, verbose=False):
        """
        Args compatible with both PyTorch and work with tinygrad backend.
        """
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            last_epoch=last_epoch,
            verbose=verbose
        )

class TinyOptimizerGroup:
    """Wrapper for using multiple optimizers with OneCycleLR schedulers."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

def setup_tiny_training():
    """Setup function to initialize tinygrad torch backend for training."""
    torch.set_default_device("tiny")
    # Disable profiling overhead
    torch.autograd.grad_mode.set_multithreading_enabled(False)

def get_cifar10_loaders(batch_size=512, num_workers=0):
    """Get CIFAR10 data loaders. Downloads automatically."""
    from torchvision import datasets, transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

__all__ = [
    'TinyOneCycleLR',
    'TinyOptimizerGroup',
    'setup_tiny_training',
    'get_cifar10_loaders',
    'torch',
    'nn',
    'optim',
]
