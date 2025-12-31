# NN (Neural Networks)

The `tinygrad/nn/` directory contains neural network primitives and utilities.

## `__init__.py`

Defines high-level layers:
- **`Linear`**: Fully connected layer.
- **`Conv2d`**, **`ConvTranspose2d`**: Convolutional layers.
- **`BatchNorm2d`**, **`GroupNorm`**, **`LayerNorm`**, **`InstanceNorm`**: Normalization layers.
- **`Embedding`**: Embedding layer.
- **`LSTM`**, **`GRU`**, **`RNN`**: Recurrent layers.

These layers store their parameters as `Tensor`s and implement the forward pass.

## `optim.py`

Implements optimization algorithms:
- **`SGD`**: Stochastic Gradient Descent.
- **`Adam`**, **`AdamW`**: Adaptive Moment Estimation.
- **`RMSprop`**.
- **`LAMB`**.
- **`LARS`**.

Optimizers take a list of parameters and update them based on their gradients.

## `state.py`

Utilities for saving and loading model state.
- **`get_parameters(obj)`**: Recursively finds all `Tensor`s in an object (that require grad).
- **`get_state_dict(obj)`**: Returns a dictionary of parameters.
- **`load_state_dict(obj, state_dict)`**: Loads parameters into a model.
- **`safe_save`**, **`safe_load`**: Save/load weights using the safetensors format.
- **`torch_load`**: Load PyTorch checkpoints.

## `datasets.py`

Helper functions to load common datasets (MNIST, CIFAR).

## `onnx.py`

ONNX model loader. Converts ONNX graphs to tinygrad code/models.
