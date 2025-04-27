# Using the Tinygrad Backend for hlb-cifar10

This document outlines how to run the `hlb-cifar10` training script using the `tinygrad` backend instead of the default PyTorch backend.

## Approach

The first attempt of integration aims to leverage `tinygrad` for the core model computation while still utilizing PyTorch for data loading and some initial preprocessing steps. The key modifications in `hlb-CIFAR10/main.py` are:

1.  **Conditional Import:** The `tinygrad.frontend.torch` module is imported only when the environment variable `TINY_BACKEND` is set to a non-empty value (e.g., `1`).
    ```python
    if getenv("TINY_BACKEND"):
        import tinygrad.frontend.torch
    ```
2.  **Model Initialization and Device Transfer:** The `SpeedyConvNet` model is initially constructed using standard PyTorch layers (`nn.Module`, `nn.Conv2d`, etc.). This includes the whitening layer initialization, which performs calculations using PyTorch tensors based on the training data. After the model is fully constructed and initialized (including the fixed whitening layer weights), it's transferred to the `tinygrad` device.
    ```python
    net = make_net()
    # ... (whitening initialization using torch.no_grad() and PyTorch tensors) ...
    net.to("cpu").to("tiny")
    ```
3.  **Data Handling:**
    *   The dataset is loaded and preprocessed (normalization, padding) using PyTorch and kept as PyTorch tensors initially.
    *   Inside the training (`main` function) and evaluation loops, batches of input images and target labels are explicitly moved from CPU PyTorch tensors to `tinygrad` tensors just before being fed into the model.
    ```python
      # Inside the training loop:
      for epoch_step, (inputs, targets) in enumerate(get_batches(...)):
          inputs = inputs.cpu().to("tiny")
          targets = targets.cpu().to("tiny")
          outputs = net(inputs)
          # ... rest of training step ...

      # Inside the evaluation loop:
      with torch.no_grad():
          for inputs, targets in get_batches(data, key='eval', ...):
              inputs = inputs.cpu().to("tiny")
              targets = targets.cpu().to("tiny")
              # ... get outputs ...
    ```
    This approach allows PyTorch's efficient dataloading and initial transforms to be used while offloading the main network computations to `tinygrad`.

## Running the Script

To run the training with the `tinygrad` backend, set the `TINY_BACKEND` environment variable:

```bash
TINY_BACKEND=1 python hlb-CIFAR10/main.py
```

You can also increase the debug level to see the kernels being executed by `tinygrad`:

```bash
TINY_BACKEND=1 DEBUG=2 python hlb-CIFAR10/main.py
```

## Results Comparison

Here are the results obtained using the standard Torch backend and the `tinygrad` backend.

**Torch Backend**
```
--------------------------------------------------------------------------------------------------------
|  epoch  |  train_loss  |  val_loss  |  train_acc  |  val_acc  |  ema_val_acc  |  total_time_seconds  |
--------------------------------------------------------------------------------------------------------
|      0  |      2.3027  |    1.4484  |     0.1084  |   0.7020  |               |              1.1964  |
|      1  |      1.4385  |    1.3256  |     0.6943  |   0.7711  |               |              2.1656  |
|      2  |      1.2812  |    1.2950  |     0.7891  |   0.7922  |       0.7926  |              3.1322  |
|      3  |      1.2100  |    1.2480  |     0.8457  |   0.8123  |       0.8007  |              4.0993  |
|      4  |      1.1836  |    1.1818  |     0.8447  |   0.8458  |       0.8457  |              5.0662  |
|      5  |      1.1016  |    1.1435  |     0.8965  |   0.8657  |       0.8786  |              6.0334  |
|      6  |      1.0693  |    1.1337  |     0.9219  |   0.8768  |       0.8809  |              7.0001  |
|      7  |      1.0986  |    1.1056  |     0.9092  |   0.8895  |       0.9047  |              7.9710  |
|      8  |      1.0703  |    1.0987  |     0.9229  |   0.8841  |       0.9178  |              8.9409  |
|      9  |      1.0283  |    1.0225  |     0.9395  |   0.9286  |       0.9283  |              9.9118  |
|     10  |      0.9858  |    1.0192  |     0.9658  |   0.9310  |       0.9346  |             10.8817  |
|     11  |      0.9805  |    0.9998  |     0.9707  |   0.9414  |       0.9410  |             11.8521  |
# Note: Displaying results from a single run with 12 epochs for brevity.
```

**Tinygrad Backend**
```
--------------------------------------------------------------------------------------------------------
|  epoch  |  train_loss  |  val_loss  |  train_acc  |  val_acc  |  ema_val_acc  |  total_time_seconds  |
--------------------------------------------------------------------------------------------------------
|      0  |      2.3164  |    1.5548  |     0.0859  |   0.6412  |               |             52.2511  |
|      1  |      1.4736  |    1.3771  |     0.6719  |   0.7370  |               |            101.6236  |
|      2  |      1.2939  |    1.4186  |     0.7881  |   0.7146  |       0.8257  |            153.7861  |
|      3  |      1.2637  |    1.2745  |     0.8027  |   0.7934  |       0.7904  |            276.2520  |
|      4  |      1.1436  |    1.1898  |     0.8770  |   0.8407  |       0.8407  |            399.6414  |
|      5  |      1.1406  |    1.1403  |     0.8691  |   0.8725  |       0.8737  |            522.1047  |
|      6  |      1.0947  |    1.1148  |     0.8984  |   0.8834  |       0.8700  |            646.6541  |
|      7  |      1.1191  |    1.1038  |     0.8848  |   0.8932  |       0.9029  |            770.2137  |
|      8  |      1.0732  |    1.0755  |     0.9219  |   0.9019  |       0.9169  |            894.6566  |
|      9  |      1.0264  |    1.0291  |     0.9502  |   0.9266  |       0.9265  |           1020.3546  |
|     10  |      0.9878  |    1.0228  |     0.9727  |   0.9281  |       0.9370  |           1145.5348  |
|     11  |      0.9619  |    1.0020  |     0.9854  |   0.9395  |       0.9393  |           1272.4826  |
|     12  |      0.9502  |    1.0017  |     0.9893  |   0.9398  |       0.9392  |           1284.9792  |
--------------------------------------------------------------------------------------------------------
# Note: Displaying results from a single run with 12 epochs for brevity.
```

**Debug Output Snippet (`DEBUG=2`)**

Enabling `DEBUG=2` shows the `tinygrad` kernels being compiled and executed, confirming the backend is active. Example output:
```
*** CUDA    1698 r_65536_8_3_3_2_2                         arg  2 mem  0.87 GB tm     58.43us/  1726.54ms (   484.52 GFLOPS  807.5|807.5   GB/s) ['max_pool2d']
*** CUDA    1699 E_4_32_4                                  arg  2 mem  0.87 GB tm     10.75us/  1726.56ms (     0.00 GFLOPS    0.3|0.3     GB/s) ['__mul__', 'unsqueeze']
*** CUDA    1700 E_4_32_4                                  arg  2 mem  0.87 GB tm      8.26us/  1726.56ms (     0.00 GFLOPS    0.4|0.4     GB/s) ['__add__', 'unsqueeze']
*** CUDA    1701 E_36864_32_4                              arg  2 mem  0.88 GB tm     32.38us/  1726.60ms (     0.00 GFLOPS  582.8|582.8   GB/s) ['contiguous', 'max_pool2d']
*** CUDA    1702 r_512_16_64_9                             arg  2 mem  0.88 GB tm     42.11us/  1726.64ms (   115.36 GFLOPS  224.1|238.1   GB/s) ['var_mean', 'cast']
*** CUDA    1703 r_512_16_64_9n1                           arg  3 mem  0.88 GB tm     41.44us/  1726.68ms (   345.55 GFLOPS  227.8|242.8   GB/s) ['rsqrt', '__add__', 'var_mean', 'cast', '__sub__']
*** CUDA    1704 E_512_32_2_16_3_3                         arg  6 mem  0.90 GB tm     32.10us/  1726.71ms (   588.06 GFLOPS  882.3|1666.2  GB/s) ['contiguous', '__add__', '__mul__', '__sub__', 'max_pool2d']
*** CUDA    1705 E_36864_32_4n1                            arg  2 mem  0.91 GB tm     31.68us/  1726.74ms (     0.00 GFLOPS  893.7|893.7   GB/s) ['contiguous', 'cast']
*** CUDA    1706 E_49152_32_3                              arg  2 mem  0.93 GB tm     38.94us/  1726.78ms (  3150.25 GFLOPS  727.0|727.0   GB/s) ['contiguous', '__mul__', '__add__', 'cast', 'erf']
*** CUDA    1707 E_36864_32_4n1                            arg  2 mem  0.93 GB tm     30.85us/  1726.81ms (     0.00 GFLOPS  917.8|917.8   GB/s) ['cast']
*** CUDA    1708 r_128_8_8_16_512_3_3_4_3_3                arg  3 mem  0.92 GB tm   3264.42us/  1730.08ms (  8058.62 GFLOPS    7.2|1853.1  GB/s) ['conv2d']
*** CUDA    1709 xfer    27648,    CUDA <- CUDA            arg  2 mem  0.92 GB tm    100.43us/  1730.18ms (     0.00 GFLOPS    0.3|0.3     GB/s)
*** CUDA    1710 E_144_32_3n1                              arg  2 mem  0.92 GB tm     23.97us/  1730.20ms (     0.58 GFLOPS    2.3|2.3     GB/s) ['contiguous', '__rmul__']
*** CUDA    1711 E_108_32_4                                arg  2 mem  0.92 GB tm      9.98us/  1730.21ms (     0.00 GFLOPS    5.5|5.5     GB/s)
*** CUDA    1712 xfer    73728,    CUDA <- CUDA            arg  2 mem  0.92 GB tm     38.53us/  1730.25ms (     0.00 GFLOPS    1.9|1.9     GB/s)
*** CUDA    1713 E_384_32_3n1                              arg  2 mem  0.92 GB tm     15.68us/  1730.27ms (     2.35 GFLOPS    9.4|9.4     GB/s) ['contiguous', '__rmul__']
*** CUDA    1714 E_288_32_4                                arg  2 mem  0.92 GB tm     10.78us/  1730.28ms (     0.00 GFLOPS   13.7|13.7    GB/s)
*** CUDA    1715 xfer   294912,    CUDA <- CUDA            arg  2 mem  0.92 GB tm     37.16us/  1730.31ms (     0.00 GFLOPS    7.9|7.9     GB/s)
*** CUDA    1716 E_1536_32_3n1                             arg  2 mem  0.92 GB tm     17.57us/  1730.33ms (     8.39 GFLOPS   33.6|33.6    GB/s) ['contiguous', '__rmul__']
*** CUDA    1717 E_1152_32_4                               arg  2 mem  0.92 GB tm     10.69us/  1730.34ms (     0.00 GFLOPS   55.2|55.2    GB/s)
*** CUDA    1718 xfer    1.18M,    CUDA <- CUDA            arg  2 mem  0.92 GB tm     37.80us/  1730.38ms (     0.00 GFLOPS   31.2|31.2    GB/s)
*** CUDA    1719 E_6144_32_3n1                             arg  2 mem  0.92 GB tm     17.92us/  1730.40ms (    32.91 GFLOPS  131.7|131.7   GB/s) ['contiguous', '__rmul__']
*** CUDA    1720 E_4608_32_4                               arg  2 mem  0.92 GB tm     12.48us/  1730.41ms (     0.00 GFLOPS  189.0|189.0   GB/s)
# ... (rest of the debug log truncated)
```


This is just the first step in integrating the `tinygrad` backend. Future improvements will focus on transitioning the model initialization and data preprocessing entirely to `tinygrad`.
