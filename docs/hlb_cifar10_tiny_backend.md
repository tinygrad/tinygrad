# Using the Tinygrad Backend for hlb-cifar10

This document outlines how to run the `hlb-cifar10` training script using the `tinygrad` backend instead of the default PyTorch backend.

## Approach

The current integration with `tinygrad` supports core model computations, data loading, and certain initial preprocessing steps. However, model weight initialization is not yet supported due to the missing `unfold` operation (refer to PR: https://github.com/tinygrad/tinygrad/pull/9919) and issues with the `batch_crop` function. The primary changes made in `hlb-CIFAR10/main.py` are as follows:

1.  **Conditional Import:** The `tinygrad.frontend.torch` module is imported only when the environment variable `TINY_BACKEND` is set to a non-empty value (e.g., `1`).
    ```python
    if getenv("TINY_BACKEND"):
        import tinygrad.frontend.torch
        hyp["misc"]["device"] = "tiny" # Device set for tinygrad
    ```

2.  **Model Initialization:**
    Currently, we can switch to the `TINY_BACKEND` without altering any code to build the `SpeedyConvNet` model. However, model weight initialization is not fully supported yet due to the missing `unfold` operation in tinygrad, which I am actively working on resolving.

    For eigenvalue decomposition, tinygrad doesn't yet implement `linalg.eigh`, so we had to use NumPy's implementation as a workaround:

    ```python
    def get_whitening_parameters(patches):
        ...
        est_covariance = torch.cov(patches.view(n, c*h*w).t())
        # eigenvalues, eigenvectors = torch.linalg.eigh(est_covariance, UPLO='U') # this is the same as saying we want our eigenvectors, with the specification that the matrix be an upper triangular matrix (instead of a lower-triangular matrix)
        eigenvalues,eigenvectors = np.linalg.eigh(est_covariance.cpu().numpy(), UPLO="U")
        eigenvalues = torch.from_numpy(eigenvalues).to(patches.device)
        eigenvectors = torch.from_numpy(eigenvectors).to(patches.device)
        return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.t().reshape(c*h*w,c,h,w).flip(0)
    ```

3.  **Data Handling:**
    Dataset loading and preprocessing steps such as normalization and padding are fully compatible with the tinygrad backend without requiring any code modifications. However, minor adjustments are necessary in the original hlb-cifar10 repository due to hardcoded device specifications in the `batch_cutmix` and `get_batches` functions, which default to "cuda". We have updated these functions to dynamically select the device based on the `TINY_BACKEND` environment variable as shown below:

    ```python
    def get_batches(data_dict, key, batchsize, epoch_fraction=1., cutmix_size=None):
        num_epoch_examples = len(data_dict[key]['images'])
        device = "tiny" if getenv("TINY_BACKEND") else "cuda"
        shuffled = torch.from_numpy(np.random.permutation(num_epoch_examples)).to(device=device)
        ...

    def batch_cutmix(inputs, targets, patch_size):
        with torch.no_grad():
            device = "tiny" if getenv("TINY_BACKEND") else "cuda"
            batch_permuted = torch.randperm(inputs.shape[0], device=device)
            ...
    ```

    In the `get_batches` function, the `shuffled` tensor is first created on the CPU to ensure eager evaluation and avoid issues with lazy execution on the tinygrad backend. After creation, it is moved to the target device (`tiny` or `cuda`). This prevents potential infinite loops or hangs that can occur if the tensor is created directly on the `tiny` device.

    The `batch_crop` function has been modified to work around a recursion depth issue in the tinygrad backend. The key changes are:

    1. Moving mask generation and selection operations to CPU/numpy to avoid tinygrad backend issues
    2. Maintaining the same random crop behavior (not switching to center crop)
    3. Ensuring proper device handling by moving the final result back to the original device

    Original implementation:
    ```python
    def batch_crop(inputs, crop_size):
        with torch.no_grad():
            crop_mask_batch = make_random_square_masks(inputs, crop_size)
            crop_mask_batch = crop_mask_batch.expand((-1, 3, -1, -1))
            cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)
            return cropped_batch
    ```

    Modified version:
    ```python
    def batch_crop(inputs, crop_size):
        with torch.no_grad():
            crop_mask_batch = make_random_square_masks(inputs, crop_size)
            crop_mask_batch = crop_mask_batch.expand((-1, 3, -1, -1)).cpu().numpy()
            cropped_batch = torch.from_numpy(inputs.cpu().numpy()[crop_mask_batch]).view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)
            return cropped_batch.to(inputs.device)
    ```

    The modification maintains the same functionality while working around current tinygrad limitations.

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
|      0  |      2.3145  |    1.8262  |     0.1016  |   0.4531  |               |             27.0005  |
|      1  |      1.6055  |    1.5401  |     0.5986  |   0.6423  |               |             51.8716  |
|      2  |      1.3662  |    1.3735  |     0.7490  |   0.7377  |       0.6844  |             78.1916  |
|      3  |      1.2588  |    1.2815  |     0.8154  |   0.7949  |       0.7746  |            106.3516  |
|      4  |      1.2070  |    1.2498  |     0.8389  |   0.8070  |       0.8073  |            132.8662  |
|      5  |      1.1289  |    3.9064  |     0.8779  |   0.1419  |       0.2091  |            160.9472  |
|      6  |      1.8477  |    1.2162  |     0.4697  |   0.8316  |       0.8409  |            189.1839  |
|      7  |      1.1602  |    1.2195  |     0.8721  |   0.8281  |       0.8383  |            217.4564  |
|      8  |      1.1221  |    1.1332  |     0.8916  |   0.8811  |       0.8842  |            245.9936  |
|      9  |      1.0684  |    1.0850  |     0.9365  |   0.9007  |       0.9022  |            275.9708  |
|     10  |      0.9878  |    1.0772  |     0.9746  |   0.9069  |       0.9120  |            305.6327  |
|     11  |      0.9609  |    1.0630  |     0.9883  |   0.9153  |       0.9144  |            334.0995  |
|     12  |      0.9404  |    1.0629  |     0.9932  |   0.9143  |       0.9147  |            340.1850  |
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
