# Apps Implementation Details

`tinygrad/apps/` contains example applications and reusable components built on top of tinygrad.

## 1. `llm.py` (LLM Inference)

This file implements a complete LLaMA/Transformer inference engine with KV caching, quantization support, and an OpenAI-compatible API server.

### 1.1 `Transformer` Class
The main model class.
*   **Initialization**: Configures layers based on arguments (dim, n_heads, etc.).
*   **`forward`**:
    *   Embeds tokens.
    *   Iterates through `TransformerBlock`s.
    *   Applies Norm and Output projection.
*   **`__call__`**:
    *   Dispatches to `forward_jit` (TinyJit) if `T=1` (decoding phase) and `JIT=1`.
    *   Otherwise calls standard `forward` (prefill phase).
*   **`generate`**:
    *   The decoding loop.
    *   Maintains `start_pos`.
    *   Uses a symbolic variable `v_start_pos` for JIT compatibility (binds the current step).

### 1.2 `TransformerBlock`
*   **Attention**:
    *   Uses `nn.Linear` for Q/K/V.
    *   Applies RoPE (`apply_rope`).
    *   **KV Cache**:
        *   Lazy allocation (`self.cache_kv`) on first run.
        *   Updates cache at `start_pos`.
        *   Uses `scaled_dot_product_attention` (supports Flash Attention / SDPA).
*   **FeedForward**: Supports both dense (MLP) and MoE (Mixture of Experts).

### 1.3 `SimpleTokenizer`
*   Implements BPE tokenization logic in pure Python.
*   Handles regex splitting, byte fallback, and special tokens.
*   Compatible with GGUF vocabulary.

### 1.4 GGUF Support (`from_gguf`)
*   Loads weights from GGUF files.
*   Handles quantization (dequantization happens in `nn.state.ggml_data_to_tensor`).
*   Configures model hyperparameters from GGUF metadata.

### 1.5 Server
*   Implements a basic `http.server`.
*   Exposes `/v1/chat/completions`.
*   Streams responses (SSE).
