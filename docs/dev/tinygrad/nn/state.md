# State Implementation Details

`tinygrad/nn/state.py` handles saving/loading weights.

## 1. SafeTensors (`safe_load`, `safe_save`)

Tinygrad uses the SafeTensors format as its native checkpoint format.

### 1.1 `safe_load`
1.  **Read Header**: Reads the first 8 bytes (length of header).
2.  **Parse Header**: Reads JSON header. Contains map of `tensor_name -> {dtype, shape, offsets}`.
3.  **Lazy Load**:
    *   Creates a `Tensor` from the file (`mmap` via `pathlib.Path`).
    *   Slices this giant tensor using the `offsets` from the header.
    *   `bitcast`s the slice to the correct dtype.
    *   `reshape`s to correct shape.
    *   *Result*: Zero-copy load. Data is only paged in when used.

### 1.2 `safe_save`
1.  **Metadata**: Constructs header JSON.
2.  **Padding**: Aligns header to 8 bytes.
3.  **Write**:
    *   Writes header length (int64).
    *   Writes header JSON.
    *   Appends tensor data (contiguous).

## 2. PyTorch Compatibility (`torch_load`)

Capable of loading `.pth` files (pickle) and converting them to tinygrad tensors on the fly.

### 2.1 `TorchPickle`
A custom `pickle.Unpickler` subclass.
*   **`find_class`**: Intercepts `torch.*Storage` and redirects to tinygrad types.
*   **`persistent_load`**: Handles the mapping of storage IDs to file offsets.

### 2.2 Zip/Tar Support
PyTorch saves new checkpoints as Zip files and old ones as Tar.
*   The code inspects the file header (magic bytes) to detect format.
*   **Zip**: Reads `data.pkl` and maps storage keys to `data/` files in the zip.
*   **Tar**: Reads `pickle` file and maps to `storages/` files.

### 2.3 `_rebuild_tensor_v2`
The hook called by pickle to reconstruct a tensor.
*   **Inputs**: Storage, offset, size, stride.
*   **Action**:
    *   Slices the raw storage tensor.
    *   `bitcast` to correct type.
    *   `reshape`.
    *   **Permutation**: If strides indicate a permutation (e.g., `(1, 2)` instead of `(2, 1)` for shape `(3, 3)`), it applies `permute` to the tinygrad tensor.

## 3. GGUF (`gguf_load`)

Support for loading quantized models (Llama.cpp format).
*   **Reader**: Parses GGUF binary format (KV pairs, Tensor infos).
*   **De-quantization (`ggml_data_to_tensor`)**:
    *   GGUF stores blocks of quantized data.
    *   This function implements the decode logic using tinygrad `Tensor` ops (vectorized).
    *   Supported: Q4_0, Q4_1, Q8_0, Q4_K, Q6_K.
    *   This allows running GGUF models by decoding weights on the fly (or converting them).

## 4. `get_parameters` / `load_state_dict`

*   **`get_state_dict`**: Recursively traverses object `__dict__` to find `Tensor`s.
*   **`load_state_dict`**:
    *   Matches keys.
    *   Checks shapes.
    *   `v.replace(state_dict[k])`: Updates the model weights in-place by swapping the UOp.
    *   `v.realize()`: Triggers the load (IO).
