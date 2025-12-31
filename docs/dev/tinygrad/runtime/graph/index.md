# Runtime Graph Implementation Details

`tinygrad/runtime/graph/` contains implementations for hardware-accelerated graph execution. These allow launching a sequence of kernels with a single driver call.

## 1. `cuda.py` (`CUDAGraph`)

Implements CUDA Graphs.

### 1.1 Capture
*   **`__init__`**:
    *   Starts capture stream (`cudaStreamBeginCapture`).
    *   Replays the `jit_cache`.
    *   Ends capture (`cudaStreamEndCapture`).
    *   Instantiates the graph (`cudaGraphInstantiate`).

### 1.2 Execution (`__call__`)
*   **Update**:
    *   Iterates through `input_replace` map.
    *   Updates kernel parameters in the executable graph using `cudaGraphExecKernelNodeSetParams`.
    *   Updates variable values (symbolic shapes).
*   **Launch**: `cudaGraphLaunch`.

## 2. `metal.py` (`MetalGraph`)

Implements Metal Indirect Command Buffers (ICB).

### 2.1 Construction
*   Creates an `ICB` object.
*   Encodes all kernel commands (set pipeline, set buffers, dispatch) into the ICB.

### 2.2 Execution
*   Updates buffer pointers in the ICB if input tensors change (though typically tinygrad tries to reuse buffers).
*   Encodes an `executeCommandsInBuffer` command on the main command buffer.

## 3. `hcq.py` (`HCQGraph`)

Hardware Command Queue. A generic graph implementation for devices that support command queues (like AMD GPU via HSA/ROCm).

*   **Logic**:
    *   Builds a packet list (AQL packets for AMD).
    *   Writes packets to a ring buffer (queue).
    *   Updates signals/doorbells to trigger execution.
    *   Supports dependency signals (barriers) between queues.
