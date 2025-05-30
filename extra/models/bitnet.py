import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import sys
import re
import mmap
import json

from tinygrad import Tensor, Device, dtypes
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.state import load_state_dict, get_state_dict
from tinygrad.helpers import getenv, DEBUG

# ────────────────────────────────────────────────────────────
# Configuration and Constants
# ────────────────────────────────────────────────────────────

VALUES_PER_ITEM = 4  # 4 ternary values packed per uint8 byte
DEBUG_PRINT = getenv("DEBUG_BITNET", False)

def debug(msg: str) -> None:
    if DEBUG_PRINT:
        print(f"[DEBUG] {msg}")
        sys.stdout.flush()

class BitNetConfig:
    """BitNet model configuration matching HuggingFace implementation."""
    def __init__(self):
        self.hidden_size = 2560
        self.intermediate_size = 6912
        self.num_attention_heads = 20
        self.num_key_value_heads = 5
        self.num_hidden_layers = 30
        self.rms_norm_eps = 1e-05
        self.vocab_size = 128256
        self.max_position_embeddings = 4096
        self.hidden_act = "relu2"
        self.rope_theta = 500000.0
        self.bos_token_id = 128000
        self.eos_token_id = 128001
        self.pad_token_id = None
        self.attention_dropout = 0.0
        self.use_bias = False
        
        # Quantization config
        self.quant_method = "bitnet"
        self.linear_class = "autobitlinear"
        self.quantization_mode = "offline"
    
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

# ────────────────────────────────────────────────────────────
# Weight Packing/Unpacking (Exact HF Implementation)
# ────────────────────────────────────────────────────────────

def unpack_weights(packed: Tensor, target_dtype=dtypes.float32) -> Tensor:
    """Unpack weights directly on GPU without CPU conversion"""
    # Get the packed data as tinygrad tensor (stay on current device)
    if packed.dtype != dtypes.uint8:
        packed = packed.cast(dtypes.uint8)
    
    # Unpack using tinygrad operations instead of numpy
    # This avoids the CPU conversion that triggers compilation
    flat = packed.flatten()
    
    # Extract 2-bit values directly in tinygrad
    val0 = (flat & 0x03).cast(dtypes.int8) - 1  # bits 0-1
    val1 = ((flat >> 2) & 0x03).cast(dtypes.int8) - 1  # bits 2-3  
    val2 = ((flat >> 4) & 0x03).cast(dtypes.int8) - 1  # bits 4-5
    val3 = ((flat >> 6) & 0x03).cast(dtypes.int8) - 1  # bits 6-7
    
    # Stack and reshape
    unpacked = Tensor.stack([val0, val1, val2, val3], dim=1).flatten()
    
    # Reshape to target shape
    original_shape = list(packed.shape)
    original_shape[0] *= 4  # Expand first dimension by 4
    
    return unpacked.reshape(original_shape).cast(target_dtype)


def pack_weights(quantized_weights: Tensor) -> Tensor:
    """
    Pack ternary weights into 2-bit format.
    
    Args:
        quantized_weights: Tensor with ternary values {-1, 0, 1}
        
    Returns:
        Packed tensor with 4 values per uint8 byte
    """
    original_shape = quantized_weights.shape
    row_dim = (original_shape[0] + VALUES_PER_ITEM - 1) // VALUES_PER_ITEM
    
    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])
    
    # Shift to {0, 1, 2} range for packing
    shifted = (quantized_weights + 1).to("CPU").numpy().astype(np.uint8)
    packed = np.zeros(packed_tensor_shape, dtype=np.uint8)
    
    for i in range(VALUES_PER_ITEM):
        start = i * row_dim
        end = min(start + row_dim, original_shape[0])
        if start < original_shape[0]:
            packed[:(end-start)] |= shifted[start:end] << (2 * i)
    
    return Tensor(packed, dtype=dtypes.uint8, device=quantized_weights.device, requires_grad=False)

# ────────────────────────────────────────────────────────────
# Core BitNet Layers
# ────────────────────────────────────────────────────────────

class BitLinear:
    """
    BitNet Linear layer with 2-bit packed weights and int8 activations.
    Follows HuggingFace implementation exactly.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=dtypes.float32):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device or Device.DEFAULT
        
        # Packed weight tensor: (out_features//4, in_features)
        self.weight = Tensor.zeros(
            (out_features // VALUES_PER_ITEM, in_features),
            dtype=dtypes.uint8,
            device=self.device,
            requires_grad=False
        )
        
        # Weight scale (single scalar)
        self.weight_scale = Tensor.ones(
            (1,),
            dtype=dtypes.float32,
            device=self.device,
            requires_grad=False
        )
        
        # Optional bias
        if bias:
            self.bias = Tensor.zeros(
                (out_features,),
                dtype=dtype,
                device=self.device,
                requires_grad=False
            )
        else:
            self.bias = None
        
        # Cache for unpacked weights
        self._unpacked_weights = None
        self._weight_hash = None
    
    def _get_unpacked_weights(self) -> Tensor:
        """Get unpacked weights with caching."""
        # Simple hash for cache invalidation
        current_hash = hash(str(self.weight.lazydata))
        
        if self._unpacked_weights is None or self._weight_hash != current_hash:
            debug(f"BitLinear: Unpacking weights for {self.in_features}x{self.out_features}")
            self._unpacked_weights = unpack_weights(self.weight, self.dtype)
            self._weight_hash = current_hash
        
        return self._unpacked_weights
    
    def activation_quant(self, x: Tensor, num_bits: int = 8) -> Tuple[Tensor, Tensor]:
        """
        Per-token symmetric quantization following HF implementation.
        
        Args:
            x: Input tensor [..., features]
            num_bits: Number of bits for quantization
            
        Returns:
            (quantized_tensor, scale_tensor)
        """
        Qn = -(2 ** (num_bits - 1))  # -128
        Qp = 2 ** (num_bits - 1) - 1  # 127
        
        # Per-token scaling (max across last dimension)
        scale = Qp / x.abs().max(axis=-1, keepdim=True).maximum(1e-5)
        
        # Quantize and clamp
        result = (x * scale).round().clip(Qn, Qp)
        
        return result, scale
    
    def post_quant_process(self, y: Tensor, input_scale: Tensor, weight_scale: Tensor) -> Tensor:
        """Dequantize output using both scales."""
        return y / (input_scale * weight_scale)
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass with quantized computation.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # Get unpacked ternary weights
        w_unpacked = self._get_unpacked_weights()
        
        # Quantize activations
        x_quant, x_scale = self.activation_quant(x)
        
        # Matrix multiplication with quantized inputs and ternary weights
        y = x_quant @ w_unpacked.T
        
        # Dequantize using scales
        y = self.post_quant_process(y, x_scale, self.weight_scale)
        
        # Add bias if present
        if self.bias is not None:
            y = y + self.bias
        
        return y

class BitNetRMSNorm:
    """RMS Normalization following HuggingFace implementation."""
    
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        self.eps = eps
        self.weight = Tensor.ones(
            (dim,),
            dtype=dtypes.float32,
            device=device or Device.DEFAULT,
            requires_grad=False
        )
    
    def __call__(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        h = x.cast(dtypes.float32)
        variance = h.pow(2).mean(axis=-1, keepdim=True)
        h = h * (variance + self.eps).rsqrt()
        return (self.weight * h).cast(input_dtype)

class BitNetRotaryEmbedding:
    """Rotary Position Embedding."""
    
    def __init__(self, config: BitNetConfig, device=None):
        self.dim = config.head_dim()
        self.device = device or Device.DEFAULT
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (config.rope_theta ** (
            Tensor.arange(0, self.dim, 2, dtype=dtypes.float32, device=self.device) / self.dim
        ))
        self.inv_freq = inv_freq
    
    def __call__(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generate cos/sin for rotary embeddings.
        
        Args:
            x: Input tensor (for dtype/device reference)
            position_ids: Position indices [batch, seq_len]
            
        Returns:
            (cos, sin) tensors for rotary embedding
        """
        # Generate frequencies
        freqs = position_ids.unsqueeze(-1).float() @ self.inv_freq.unsqueeze(0)
        
        # Duplicate for real/imaginary parts
        emb = freqs.repeat_interleave(2, dim=-1)
        
        cos = emb.cos().cast(x.dtype)
        sin = emb.sin().cast(x.dtype)
        
        return cos, sin

def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dimensions."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return (-x2).cat(x1, dim=-1)

def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # Ensure cos/sin have correct dimensions for broadcasting
    if cos.ndim == 3:  # [batch, seq, dim]
        cos = cos.unsqueeze(1)  # [batch, 1, seq, dim]
        sin = sin.unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat key/value states for grouped query attention."""
    if n_rep == 1:
        return hidden_states
    
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states.unsqueeze(2).expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

# ────────────────────────────────────────────────────────────
# Model Architecture
# ────────────────────────────────────────────────────────────

class BitNetAttention:
    """Multi-head attention with BitNet quantization."""
    
    def __init__(self, config: BitNetConfig, layer_idx: int, device=None):
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.device = device or Device.DEFAULT
        
        # Projection layers (BitLinear)
        self.q_proj = BitLinear(config.hidden_size, self.num_heads * self.head_dim, device=self.device)
        self.k_proj = BitLinear(config.hidden_size, self.num_key_value_heads * self.head_dim, device=self.device)
        self.v_proj = BitLinear(config.hidden_size, self.num_key_value_heads * self.head_dim, device=self.device)
        self.o_proj = BitLinear(self.num_heads * self.head_dim, config.hidden_size, device=self.device)
        
        # Attention sub-normalization (BitNet specific)
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=self.device)
    
    def __call__(self, 
                 hidden_states: Tensor,
                 position_embeddings: Tuple[Tensor, Tensor],
                 attention_mask: Optional[Tensor] = None,
                 past_key_value: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for attention.
        
        Returns:
            (attention_output, (present_key, present_value))
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Update cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = past_key.cat(key_states, dim=2)
            value_states = past_value.cat(value_states, dim=2)
        
        present_key_value = (key_states, value_states)
        
        # Repeat K/V for grouped query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Scaled dot-product attention
        attn_weights = query_states @ key_states.transpose(2, 3) * self.scaling
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = attn_weights.softmax()
        attn_output = attn_weights @ value_states
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        # Apply sub-normalization (BitNet specific)
        attn_output = self.attn_sub_norm(attn_output)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value

class BitNetMLP:
    """BitNet MLP with quantized linear layers."""
    
    def __init__(self, config: BitNetConfig, device=None):
        self.device = device or Device.DEFAULT
        
        # Three linear projections
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size, device=self.device)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size, device=self.device)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size, device=self.device)
        
        # Sub-normalization
        self.ffn_sub_norm = BitNetRMSNorm(config.intermediate_size, eps=config.rms_norm_eps, device=self.device)
    
    def __call__(self, x: Tensor) -> Tensor:
        # SwiGLU-like activation: gate_proj(x) * relu²(up_proj(x))
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # ReLU²(x) = (ReLU(x))²
        activated = (up.relu() ** 2) * gate
        
        # Apply sub-normalization
        activated = self.ffn_sub_norm(activated)
        
        # Final projection
        return self.down_proj(activated)

class BitNetDecoderLayer:
    """Single BitNet decoder layer."""
    
    def __init__(self, config: BitNetConfig, layer_idx: int, device=None):
        self.device = device or Device.DEFAULT
        
        # Components
        self.self_attn = BitNetAttention(config, layer_idx, device=self.device)
        self.mlp = BitNetMLP(config, device=self.device)
        
        # Layer normalization
        self.input_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=self.device)
        self.post_attention_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=self.device)
    
    def __call__(self,
                 hidden_states: Tensor,
                 position_embeddings: Tuple[Tensor, Tensor],
                 attention_mask: Optional[Tensor] = None,
                 past_key_value: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for decoder layer.
        
        Returns:
            (hidden_states, present_key_value)
        """
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        
        hidden_states = residual + hidden_states
        
        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value

class BitNetModel:
    """Core BitNet transformer model."""
    
    def __init__(self, config: BitNetConfig, device=None):
        self.config = config
        self.device = device or Device.DEFAULT
        
        # Token embeddings
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        if self.device != "CPU":
            self.embed_tokens.weight = self.embed_tokens.weight.to(self.device)
        
        # Decoder layers
        self.layers = [
            BitNetDecoderLayer(config, layer_idx, device=self.device)
            for layer_idx in range(config.num_hidden_layers)
        ]
        
        # Final normalization
        self.norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=self.device)
        
        # Rotary embeddings
        self.rotary_emb = BitNetRotaryEmbedding(config, device=self.device)
    
    def __call__(self, 
                 input_ids: Tensor,
                 past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Forward pass.
        
        Returns:
            (hidden_states, present_key_values)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Position embeddings
        if past_key_values is not None and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[2]
            position_ids = Tensor.arange(past_length, past_length + seq_len, device=self.device).unsqueeze(0)
        else:
            position_ids = Tensor.arange(seq_len, device=self.device).unsqueeze(0)
        
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Process through layers
        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_key_value = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value
            )
            
            present_key_values.append(present_key_value)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, present_key_values

class BitNetForCausalLM:
    """BitNet model for causal language modeling."""
    
    def __init__(self, config: BitNetConfig, device=None):
        self.config = config
        self.device = device or Device.DEFAULT
        
        # Core model
        self.model = BitNetModel(config, device=self.device)
        
        # Language modeling head (tied to embeddings)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.device != "CPU":
            self.lm_head.weight = self.lm_head.weight.to(self.device)
        
        # Tie weights
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def __call__(self, 
                 input_ids: Tensor,
                 past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
                 *sample_args) -> Tuple[Any, List[Tuple[Tensor, Tensor]]]:
        """
        Forward pass with optional sampling.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Cached key-value states
            *sample_args: Sampling parameters (temp, top_k, top_p, etc.)
            
        Returns:
            If sample_args provided: (sampled_token, present_key_values, logits)
            Else: (logits, present_key_values)
        """
        # Forward through model
        hidden_states, present_key_values = self.model(input_ids, past_key_values)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # If sampling parameters provided, sample next token
        if sample_args:
            # Get logits for last token
            last_token_logits = logits[0, -1, :]  # [vocab_size]
            token = sample(last_token_logits, *sample_args)
            return token, present_key_values, logits
        else:
            return logits, present_key_values

# ────────────────────────────────────────────────────────────
# Sampling Functions
# ────────────────────────────────────────────────────────────

def sample(logits: Tensor, 
          temperature: float = 0.0,
          top_k: int = 0,
          top_p: float = 0.0,
          alpha_frequency: float = 0.0,
          alpha_presence: float = 0.0) -> int:
    """
    Sample next token from logits.
    
    Args:
        logits: 1D tensor of logits [vocab_size]
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        alpha_frequency: Frequency penalty
        alpha_presence: Presence penalty
        
    Returns:
        Sampled token ID
    """
    debug(f"sample: logits.shape={logits.shape}, temp={temperature}")
    
    # Greedy sampling for temperature near 0
    if temperature < 1e-6:
        token = int(logits.argmax().detach().to("CPU").numpy())
        debug(f"sample: greedy token={token}")
        return token
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply penalties (simplified)
    # TODO: Implement frequency and presence penalties
    
    # Top-k filtering
    if top_k > 0:
        values, indices = logits.topk(top_k)
        logits_filtered = Tensor.full_like(logits, float('-inf'))
        # TODO: Implement proper top-k filtering
    
    # Convert to probabilities
    probs = logits.softmax()
    
    # Sample
    try:
        probs_np = probs.detach().to("CPU").numpy()
        # Filter special tokens (keep only first 128000 tokens)
        if len(probs_np) > 128000:
            probs_np = probs_np[:128000]
            probs_np = probs_np / probs_np.sum()
        
        token = int(np.random.choice(len(probs_np), p=probs_np))
        debug(f"sample: sampled token={token}")
        return token
    except Exception as e:
        debug(f"sample: error={e}, falling back to argmax")
        return int(logits.argmax().detach().to("CPU").numpy())

# ────────────────────────────────────────────────────────────
# Weight Loading
# ────────────────────────────────────────────────────────────

def load_safetensors_weights(file_path: Path) -> Dict[str, np.ndarray]:
    """Load weights from safetensors file."""
    debug(f"Loading weights from {file_path}")
    
    with open(file_path, 'rb') as f:
        # Read header
        header_size = int.from_bytes(f.read(8), 'little')
        metadata_bytes = f.read(header_size)
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        # Memory map for efficient loading
        data_start = 8 + header_size
        f.seek(0)
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        weights = {}
        for key, info in metadata.items():
            if key == "__metadata__":
                continue
            
            dtype_str = info['dtype']
            shape = info['shape']
            offset_start, offset_end = info['data_offsets']
            
            # Map dtypes
            dtype_map = {
                'U8': np.uint8, 'I8': np.int8,
                'F32': np.float32, 'F16': np.float16,
                'BF16': np.float16  # Approximate
            }
            np_dtype = dtype_map.get(dtype_str, np.float32)
            
            # Read data
            raw_bytes = mm[data_start + offset_start:data_start + offset_end]
            
            if dtype_str == 'BF16':
                # Convert BF16 to float32
                uint16_arr = np.frombuffer(raw_bytes, dtype=np.uint16)
                float32_bytes = np.zeros(uint16_arr.size * 4, dtype=np.uint8)
                float32_view = float32_bytes.view(np.uint32)
                float32_view[:] = uint16_arr.astype(np.uint32) << 16
                arr = float32_bytes.view(np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)
            
            weights[key] = arr
        
        mm.close()
    
    debug(f"Loaded {len(weights)} weight tensors")
    return weights

def convert_and_load_weights(model: BitNetForCausalLM, weights: Dict[str, np.ndarray]):
    """Convert HF weights to tinygrad format and load into model."""
    debug("Converting and loading weights")
    
    # Convert weights to tensors
    tensor_weights = {}
    
    for key, weight_np in weights.items():
        if key.endswith('.weight_scale'):
            # Scale tensors
            tensor_weights[key] = Tensor(
                weight_np.astype(np.float32),
                dtype=dtypes.float32,
                device=model.device,
                requires_grad=False
            )
        elif key.endswith('.weight') and any(proj in key for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            # BitLinear weights - keep as packed uint8
            if weight_np.dtype == np.int8:
                weight_np = weight_np.astype(np.uint8)
            tensor_weights[key] = Tensor(
                weight_np,
                dtype=dtypes.uint8,
                device=model.device,
                requires_grad=False
            )
        else:
            # Regular weights (embeddings, norms)
            if weight_np.dtype == np.float16:
                weight_np = weight_np.astype(np.float32)
            tensor_weights[key] = Tensor(
                weight_np,
                dtype=dtypes.float32,
                device=model.device,
                requires_grad=False
            )
    
    # Load into model
    try:
        load_state_dict(model, tensor_weights, strict=False, realize=False)
        debug("Weights loaded successfully")
    except Exception as e:
        debug(f"Error loading weights: {e}")
        raise

def build_transformer(model_path: Path, load_weights: bool = True) -> Tuple[BitNetForCausalLM, Optional[Dict]]:
    """Build and optionally load BitNet transformer."""
    debug(f"Building transformer from {model_path}")
    
    config = BitNetConfig()
    model = BitNetForCausalLM(config, device=Device.DEFAULT)
    
    if not load_weights:
        return model, None
    
    # Find weights file
    if model_path.is_file():
        weights_file = model_path
    else:
        weights_file = model_path / "model.safetensors"
    
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")
    
    # Load and convert weights
    raw_weights = load_safetensors_weights(weights_file)
    convert_and_load_weights(model, raw_weights)
    
    debug("Transformer built successfully")
    return model, raw_weights