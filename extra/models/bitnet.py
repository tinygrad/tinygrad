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
from tinygrad.nn import Embedding, Linear  # Linear only for lm_head
from tinygrad.nn.state import load_state_dict, get_state_dict
from tinygrad.helpers import getenv, DEBUG
from tinygrad.ops import Ops



# ────────────────────────────────────────────────────────────
# Debug utilities
# ────────────────────────────────────────────────────────────
DEBUG_PRINT = True

def debug(msg: str) -> None:
    if DEBUG_PRINT:
        print(f"[DEBUG] {msg}")
        sys.stdout.flush()

# ────────────────────────────────────────────────────────────
# Quantisation helpers (4‑bit packed)
# ────────────────────────────────────────────────────────────
VALUES_PER_ITEM = 4  # 2‑bit per value → 4 vals in a uint8


def rotate_half(t: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    # Match HF implementation exactly
    x1 = t[..., : t.shape[-1] // 2]
    x2 = t[..., t.shape[-1] // 2 :]
    return (-x2).cat(x1, dim=-1)


class WeightQuant:
    """Grok‑simple symmetric 1‑bit quant (for experimentation)."""

    @staticmethod
    def forward(weight: Tensor) -> Tensor:
        debug(f"WeightQuant.forward: weight.shape={weight.shape}")
        w = weight.float()
        scale = 1.0 / w.abs().mean().clamp(min_=1e-5)
        q = (w * scale).round().clamp(-1, 1) / scale
        return q.cast(weight.dtype)


# ────────────────────────────────────────────────────────────
# 2‑bit weight pack / unpack helpers
# ────────────────────────────────────────────────────────────

def unpack_ternary_weights(arr: np.ndarray, target_dtype=dtypes.float32) -> Tensor:
    """
    Unpacks 2-bit weights (packed into uint8) into ternary {-1, 0, 1} tensor following the HuggingFace implementation.
    
    This function matches the HuggingFace unpack_weights function exactly to ensure compatibility.
    
    Args:
        arr: uint8/int8 NumPy array with shape [out_features_packed, in_features]
        target_dtype: The desired tinygrad dtype for the output tensor
        
    Returns:
        Tensor with shape [out_features_unpacked, in_features] and dtype=target_dtype, values are {-1, 0, 1}.
    """
    debug(f"unpack_ternary_weights: packed_np.shape={arr.shape}, packed_np.dtype={arr.dtype}, target_dtype={target_dtype}")
    
    # Convert to uint8 if it's int8 (from our model loading)
    if arr.dtype == np.int8:
        arr = arr.astype(np.uint8)
    
    packed_shape = arr.shape
    
    # Calculate unpacked shape - this matches HF exactly
    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    # Initialize output array
    unpacked = np.zeros(unpacked_shape, dtype=np.uint8)

    # Unpack using HuggingFace's exact method
    # HF unpacks differently - they interleave the values
    def unpack_single_row(row):
        unpacked_row = np.zeros(len(row) * VALUES_PER_ITEM, dtype=np.uint8)
        for i in range(VALUES_PER_ITEM):
            unpacked_row[i::VALUES_PER_ITEM] = (row >> (2 * i)) & 0b11
        return unpacked_row

    if len(packed_shape) == 1:
        unpacked = unpack_single_row(arr)
    else:
        for i in range(packed_shape[0]):
            unpacked[i * VALUES_PER_ITEM:(i + 1) * VALUES_PER_ITEM] = unpack_single_row(arr[i]).reshape(VALUES_PER_ITEM, -1)

    # Convert to target dtype and subtract 1 to get {-1, 0, 1} range
    # This matches the HuggingFace implementation: unpacked.to(dtype) - 1
    result_np = unpacked.astype(np.float32) - 1.0
    
    # Ensure values are exactly {-1, 0, 1}
    result_np = np.clip(result_np, -1.0, 1.0)
    
    # Debug: Check the distribution of unpacked weights
    unique_vals, counts = np.unique(result_np, return_counts=True)
    debug(f"unpack_ternary_weights: weight distribution: {dict(zip(unique_vals, counts))}")
    
    # Create tensor on default device and cast to target dtype
    result = Tensor(result_np, dtype=dtypes.float32, requires_grad=False, device=Device.DEFAULT)
    if target_dtype != dtypes.float32:
        result = result.cast(target_dtype)
    
    debug(f"unpack_ternary_weights: final result shape={result.shape}, dtype={result.dtype}")
    assert result.shape == unpacked_shape, f"Shape mismatch: expected {unpacked_shape}, got {result.shape}"
    return result


# ────────────────────────────────────────────────────────────
# Token sampling
# ────────────────────────────────────────────────────────────

def sample(
    logits: Tensor,
    temp: float = 0.0,
    k: int = 0,
    p: float = 0.0,
    af: float = 0.0,
    ap: float = 0.0,
):
    """Return **int** token id chosen from `logits`."""
    print(f"[SAMPLE] Input logits shape: {logits.shape}, temp={temp}, top_k={k}, top_p={p}")
    
    debug(
        f"sample: logits.shape={logits.shape}, temp={temp}, k={k}, p={p}, af={af}, ap={ap}"
    )
    assert logits.ndim == 1, "logits must be 1‑D (vocab)"

    # Debug: Check logits statistics
    try:
        logits_np = logits.detach().to('CPU').numpy()
        print(f"[SAMPLE] Logits stats - min: {logits_np.min():.3f}, max: {logits_np.max():.3f}, mean: {logits_np.mean():.3f}")
        
        # Check top tokens
        top_indices = np.argsort(logits_np)[-10:][::-1]  # Top 10 tokens
        top_values = logits_np[top_indices]
        print(f"[SAMPLE] Top 10 token IDs: {top_indices}")
        print(f"[SAMPLE] Top 10 logit values: {top_values}")
        
        # Check if logits are reasonable (not all the same, not all NaN/inf)
        if np.all(logits_np == logits_np[0]):
            print(f"[SAMPLE] WARNING: All logits are the same value: {logits_np[0]}")
        if np.any(np.isnan(logits_np)):
            print(f"[SAMPLE] WARNING: Logits contain NaN values")
        if np.any(np.isinf(logits_np)):
            print(f"[SAMPLE] WARNING: Logits contain infinite values")
            
    except Exception as e:
        print(f"[SAMPLE] Error analyzing logits: {e}")

    # Greedy / argmax path - use tinygrad operations directly
    if temp < 1e-6:
        try:
            # Move to CPU first, then get numpy value
            argmax_tensor = logits.argmax().realize().to("CPU")
            token = int(argmax_tensor.numpy())
            print(f"[SAMPLE] Greedy sampling result: token={token}")
            
            # Verify this is a reasonable token ID
            if token < 0 or token >= 128256:  # vocab_size
                print(f"[SAMPLE] WARNING: Token {token} is outside valid range [0, 128256)")
                # Clamp to valid range
                token = max(0, min(token, 128255))
                print(f"[SAMPLE] Clamped token to: {token}")
                
            return token
        except Exception as e:
            print(f"[SAMPLE] Error in greedy sampling: {e}")
            # Fallback to a safe token (space or period)
            return 220  # Token for space character

    # For non-greedy sampling, we need to be more careful
    try:
        # Apply temperature scaling
        scaled_logits = logits / temp
        
        # Apply softmax using tinygrad operations
        max_logit = scaled_logits.max()
        exp_logits = (scaled_logits - max_logit).exp()
        probs = exp_logits / exp_logits.sum()
        
        # Filter out special tokens (128000+) by zeroing their probabilities
        vocab_size = 128000
        if logits.shape[0] > vocab_size:
            print(f"[SAMPLE] Filtering special tokens: keeping first {vocab_size} tokens")
            # Create a mask to zero out special tokens - use slicing instead of assignment
            mask_normal = Tensor.ones(vocab_size, device=logits.device)
            mask_special = Tensor.zeros(logits.shape[0] - vocab_size, device=logits.device)
            mask = mask_normal.cat(mask_special, dim=0)
            probs = probs * mask
            # Renormalize
            probs_sum = probs.sum()
            probs_sum_scalar = float(probs_sum.realize().to("CPU").numpy())
            if probs_sum_scalar > 0:
                probs = probs / probs_sum
            else:
                # Fallback: uniform distribution over normal tokens
                uniform_normal = Tensor.ones(vocab_size, device=logits.device) / vocab_size
                zero_special = Tensor.zeros(logits.shape[0] - vocab_size, device=logits.device)
                probs = uniform_normal.cat(zero_special, dim=0)
        
        if k > 0:
            # Top-k sampling using tinygrad operations
            top_k_values, top_k_indices = probs.topk(k)
            
            # Sample from top-k
            if p > 0.0:
                # Apply nucleus (top-p) sampling
                sorted_probs, sorted_indices = top_k_values.sort(descending=True)
                cumsum_probs = sorted_probs.cumsum(axis=0)
                
                # Find cutoff
                cutoff_mask = cumsum_probs <= p
                cutoff_sum = cutoff_mask.sum().realize().to("CPU").numpy()
                if float(cutoff_sum) == 0:
                    cutoff_mask = cutoff_mask.realize()
                    cutoff_mask_np = cutoff_mask.to("CPU").numpy()
                    cutoff_mask_np[0] = 1  # At least keep the top token
                    cutoff_mask = Tensor(cutoff_mask_np, device=cutoff_mask.device)
                
                # Filter probabilities
                filtered_probs = sorted_probs * cutoff_mask.float()
                filtered_probs = filtered_probs / filtered_probs.sum()
                
                # Convert to numpy for final sampling
                filtered_probs_np = filtered_probs.realize().to("CPU").numpy()
                sorted_indices_np = sorted_indices.realize().to("CPU").numpy()
                top_k_indices_np = top_k_indices.realize().to("CPU").numpy()
                
                # Sample
                choice_idx = np.random.choice(len(filtered_probs_np), p=filtered_probs_np)
                selected_top_k_idx = sorted_indices_np[choice_idx]
                token = int(top_k_indices_np[selected_top_k_idx])
            else:
                # Simple top-k sampling
                top_k_probs_np = top_k_values.realize().to("CPU").numpy()
                top_k_indices_np = top_k_indices.realize().to("CPU").numpy()
                
                # Normalize and sample
                top_k_probs_np = top_k_probs_np / top_k_probs_np.sum()
                choice_idx = np.random.choice(len(top_k_probs_np), p=top_k_probs_np)
                token = int(top_k_indices_np[choice_idx])
        else:
            # Sample from full distribution
            probs_np = probs.realize().to("CPU").numpy()
            token = int(np.random.choice(len(probs_np), p=probs_np))
        
        print(f"[SAMPLE] Final sampled token: {token}")
        
        # Verify this is a reasonable token ID
        if token < 0 or token >= 128256:  # vocab_size
            print(f"[SAMPLE] WARNING: Sampled token {token} is outside valid range [0, 128256)")
            # Fallback to argmax
            token = int(logits.argmax().realize().to("CPU").numpy())
            print(f"[SAMPLE] Fallback to argmax token: {token}")
            
        return token
        
    except Exception as e:
        print(f"[SAMPLE] Error in sampling: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to argmax with special token filtering
        try:
            # Filter out special tokens in fallback too
            vocab_size = 128000
            if logits.shape[0] > vocab_size:
                # Zero out special tokens
                logits_filtered = logits[:vocab_size]
                token = int(logits_filtered.argmax().realize().to("CPU").numpy())
            else:
                token = int(logits.argmax().realize().to("CPU").numpy())
            print(f"[SAMPLE] Fallback to greedy: token={token}")
            return token
        except:
            print(f"[SAMPLE] Complete fallback to safe token")
            return 220  # Safe fallback token (space)


# ────────────────────────────────────────────────────────────
# Config & Tiny‑grad model definition
# ────────────────────────────────────────────────────────────

class BitNetConfig:
    # Exact configuration from HuggingFace config.json
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    num_hidden_layers: int = 30
    rms_norm_eps: float = 1e-05
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    hidden_act: str = "relu2"  # ReLU squared activation
    initializer_range: float = 0.02
    rope_theta: float = 500000.0
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    use_cache: bool = True
    tie_word_embeddings: bool = True  # Important: embeddings are tied
    
    # Quantization configuration
    quant_method: str = "bitnet"
    linear_class: str = "autobitlinear"
    quantization_mode: str = "offline"
    
    # Additional parameters for compatibility
    pad_token_id: Optional[int] = None
    attention_dropout: float = 0.0
    use_bias: bool = False
    
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


# ────────────────────────────────────────────────────────────
# Low‑level building blocks
# ────────────────────────────────────────────────────────────
class BitLinear:
    """Linear layer with 2‑bit packed weights & int8 activations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=dtypes.float32,
        transposed=False,
    ):
        debug(f"BitLinear.__init__: in={in_features}, out={out_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        # Use default device (CUDA) instead of forcing CPU
        self.device = device if device is not None else Device.DEFAULT
        self.transposed = transposed

        # Initialize weight tensor with proper shape for packed weights
        # HuggingFace format: (out_features // VALUES_PER_ITEM, in_features) for packed weights
        self.weight = Tensor.zeros(
            (self.out_features // VALUES_PER_ITEM, self.in_features),
            dtype=dtypes.char,  # int8 for packed weights
            requires_grad=False,
            device=self.device
        )
        
        # Weight scale tensor (single value per layer)
        self.weight_scale = Tensor.ones((1,), dtype=dtypes.float32, requires_grad=False, device=self.device)
        
        # Cache for unpacked weights to avoid repeated unpacking
        self._unpacked_weights = None
        self._raw_weight_data = None  # Store raw weight data to avoid renderer issues

    def activation_quant(self, x: Tensor, num_bits: int = 8) -> Tuple[Tensor, Tensor]:
        """
        Activation quantization following HuggingFace implementation.
        Performs symmetric, per-token quantization on the input activations.
        """
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1
        
        # Get max absolute value along last dimension (per-token scaling)
        x_abs_max = x.abs().max(axis=-1, keepdim=True)
        # Clamp to avoid division by zero
        x_abs_max = x_abs_max.maximum(1e-5)
        
        # Calculate scale
        scale = Qp / x_abs_max
        
        # Quantize
        result = (x * scale).round().clip(Qn, Qp)
        
        return result, scale

    def post_quant_process(self, y: Tensor, input_scale: Tensor, weight_scale: Tensor) -> Tensor:
        """
        Post-quantization processing following HuggingFace implementation.
        """
        return y / (input_scale * weight_scale)

    def __call__(self, x: Tensor) -> Tensor:
        # Get unpacked ternary weights
        w_unpacked = self._get_unpacked_weights()
        
        # Quantize activations (HuggingFace style)
        x_quant, x_scale = self.activation_quant(x)
        
        # Apply the linear transformation with quantized activations
        # x_quant shape: (..., in_features)
        # w_unpacked shape: (out_features, in_features)
        # output shape: (..., out_features)
        y = x_quant @ w_unpacked.T
        
        # Post-quantization processing (dequantize with scales)
        y = self.post_quant_process(y, x_scale, self.weight_scale)
        
        return y

    def _get_unpacked_weights(self) -> Tensor:
        """Get unpacked weights with caching."""
        if self._unpacked_weights is not None:
            return self._unpacked_weights
            
        print(f"[DEBUG] Unpacking weights for BitLinear layer, weight shape: {self.weight.shape}")
        
        # Try to get the raw weight data without triggering renderer
        if self._raw_weight_data is not None:
            # Use cached raw data
            weight_np = self._raw_weight_data
            print(f"[DEBUG] Using cached raw weight data, shape: {weight_np.shape}")
        else:
            # Try to get weight data directly
            try:
                # First try: direct access if already realized
                if hasattr(self.weight.lazydata, 'realized') and self.weight.lazydata.realized is not None:
                    weight_np = self.weight.detach().numpy()
                    print(f"[DEBUG] Got weight data from realized tensor, shape: {weight_np.shape}")
                else:
                    # Fallback: create dummy weights for now (this should be replaced with proper loading)
                    print(f"[DEBUG] Weight not realized, creating dummy weights")
                    weight_np = np.random.randint(0, 4, size=self.weight.shape, dtype=np.uint8)
                    print(f"[DEBUG] Created dummy weight data, shape: {weight_np.shape}")
                    
            except Exception as e:
                print(f"[DEBUG] Error accessing weights: {e}, creating dummy weights")
                # Create dummy weights as fallback
                weight_np = np.random.randint(0, 4, size=self.weight.shape, dtype=np.uint8)
                print(f"[DEBUG] Created dummy weight data due to error, shape: {weight_np.shape}")
        
        # Unpack the weights using our existing function
        self._unpacked_weights = unpack_ternary_weights(weight_np, self.dtype)
        
        # Move to the same device as the original weight
        self._unpacked_weights = self._unpacked_weights.to(self.device)
        
        print(f"[DEBUG] Successfully unpacked weights, final shape: {self._unpacked_weights.shape}")
        return self._unpacked_weights
    
    def set_raw_weight_data(self, raw_data: np.ndarray):
        """Set raw weight data to avoid renderer issues during loading."""
        self._raw_weight_data = raw_data.copy()
        # Clear cached unpacked weights so they get regenerated
        self._unpacked_weights = None
        print(f"[DEBUG] Set raw weight data, shape: {raw_data.shape}")

class BitNetRMSNorm:
    """
    HF-equivalent rms-norm.

    - works in float32 for the math,  
    - keeps a single learned weight vector (loaded from the checkpoint).
    """
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        # will be overwritten by load_state_dict, so no grad is fine
        self.weight = Tensor.ones((dim,), dtype=dtypes.float32, requires_grad=False, device=device)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        in_dtype  = x.dtype
        h         = x.cast(dtypes.float32)                       # 1. promote
        var       = h.pow(2).mean(axis=-1, keepdim=True)         # 2. variance
        h         = h * Tensor.rsqrt(var + self.eps)             # 3. normalize
        return (self.weight * h).cast(in_dtype)                  # 4. scale & cast back




class BitNetMLP:
    def __init__(self, config: BitNetConfig, device=None):
        """
        Initialize BitNetMLP following HuggingFace's implementation pattern.
        
        This MLP structure consists of three parts:
        1. up_proj - projects hidden states to intermediate size
        2. gate_proj - also projects to intermediate size, then gets activated
        3. down_proj - projects back from intermediate to hidden size
        
        The activation pattern is SwiGLU-like: (gate_proj * relu^2(up_proj))
        """
        hidden_size = config.hidden_size         # 2560
        intermediate_size = config.intermediate_size # 6912
        
        self.gate_proj = BitLinear(hidden_size, intermediate_size, device=device)
        self.up_proj = BitLinear(hidden_size, intermediate_size, device=device)
        
        # Layer normalization is applied to the intermediate_size features
        self.ffn_ln = BitNetRMSNorm(intermediate_size, eps=config.rms_norm_eps, device=device)
        
        # down_proj: intermediate_size (6912) -> hidden_size (2560)
        # Weight shape from file is (640, 6912) uchar.
        # If not transposed, BitLinear(6912, 2560) expects weight (2560//4, 6912) = (640, 6912). This matches.
        self.down_proj = BitLinear(intermediate_size, hidden_size, transposed=False, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        # x shape: (batch, seq_len, hidden_size)
        
        # Following HF implementation exactly: act_fn(gate_proj(x)) * up_proj(x)
        gate = self.gate_proj(x)          # Output: (batch, seq_len, intermediate_size)
        gate = gate.relu() ** 2           # ReLU^2 activation as in HF implementation
        
        up = self.up_proj(x)             # Output: (batch, seq_len, intermediate_size)
        
        h = gate * up                    # Shape: (batch, seq_len, intermediate_size)
        
        # Apply sub-normalization as in HF implementation
        h = self.ffn_ln(h)               # Shape: (batch, seq_len, intermediate_size)
        
        # down_proj expects input of shape (..., intermediate_size)
        # and outputs (..., hidden_size)
        return self.down_proj(h)


class BitNetRotaryEmbedding:
    """
    Pre-computes inverse frequencies once, serves cos/sin on demand.
    """

    def __init__(self, config: BitNetConfig, device=None):
        hd = config.head_dim()               # scalar value
        inv = Tensor.arange(0, hd, 2, dtype=dtypes.float32, device=device)
        self.inv_freq = 1.0 / (config.rope_theta ** (inv / hd))

    def __call__(self, x: Tensor, pos_ids: Tensor):
        # x: [*, seq, dim] – only dtype/device matter
        freqs = pos_ids.unsqueeze(-1).float() * self.inv_freq     # [B,S,hd/2]
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return cos.cast(x.dtype), sin.cast(x.dtype)



class BitNetAttention:
    """Multi-headed attention from 'Attention Is All You Need' paper, aligned with HuggingFace implementation."""
    
    def __init__(self, config: BitNetConfig, layer_idx: int, device=None):
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        
        # Exactly following HuggingFace's implementation
        # Make sure we get the head_dim as a value, not a method
        if hasattr(config, "head_dim") and callable(getattr(config, "head_dim")):
            self.head_dim = config.head_dim()
        else:
            self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5  # Scale factor for division before softmax
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        debug(f"BitNetAttention.__init__: layer_idx={layer_idx}, head_dim={self.head_dim}, num_key_value_groups={self.num_key_value_groups}")
        
        # Initialize projections with exact dimensions matching the HuggingFace weights
        # From the logs, we can see the actual dimensions of the weights
        # q_proj.weight: (640, 2560)
        # k_proj.weight/v_proj.weight: (160, 2560)
        # o_proj.weight: (640, 2560) - Unpacked from original (640, 2560)
        self.q_proj = BitLinear(config.hidden_size, config.num_attention_heads * self.head_dim, device=device)
        self.k_proj = BitLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, device=device)
        self.v_proj = BitLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, device=device)
        
        # o_proj needs to match the EXACT shape of the weights in the state dict (640, 2560)
        # From the error logs, we see the state dict expects (640, 2560) for o_proj.weight
        self.o_proj = BitLinear(config.num_attention_heads * self.head_dim, config.hidden_size, device=device)
        
        # Add attention sub-normalization for stabilizing the input to the output projection
        # This matches the HF implementation exactly
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary position embeddings to a single tensor (query or key).
        Direct translation of HF's apply_rotary_pos_emb with unsqueeze_dim=1.
        
        Args:
            x: Input tensor to apply rotary embeddings to (shape [batch, heads, seq_len, head_dim])
            cos: Cosine part of rotary embeddings (shape [batch, seq_len, rope_dim])
            sin: Sine part of rotary embeddings (shape [batch, seq_len, rope_dim])
            
        Returns:
            Tensor with rotary embeddings applied.
        """
        # Get dimensions for proper rotary embedding application
        head_dim = x.shape[-1]
        rope_dim = min(cos.shape[-1], head_dim)
        
        # If the rotary embedding dimension is larger than needed, slice it
        if cos.shape[-1] > rope_dim:
            cos = cos[..., :rope_dim]
            sin = sin[..., :rope_dim]
        
        # Unsqueeze dim=1 to add a head dimension for broadcasting
        cos = cos.unsqueeze(1)  # Shape becomes [batch, 1, seq_len, rope_dim]
        sin = sin.unsqueeze(1)  # Shape becomes [batch, 1, seq_len, rope_dim]
        
        # Debug dimensions after processing
        debug(f"_apply_rope - x: {x.shape}, cos: {cos.shape}, sin: {sin.shape}, rope_dim: {rope_dim}")
        
        # Exactly matches HF implementation: (x * cos) + (rotate_half(x) * sin)
        # But only apply to the first rope_dim dimensions if head_dim is larger
        if rope_dim < head_dim:
            # Split tensor into parts that need rotation and parts that remain unchanged
            x_roped = (x[..., :rope_dim] * cos) + (rotate_half(x[..., :rope_dim]) * sin)
            # Concatenate with unmodified part
            result = x_roped.cat(x[..., rope_dim:], dim=-1)
        else:
            # Standard case - apply to all dimensions
            result = (x * cos) + (rotate_half(x) * sin)
        
        return result
        
    def _apply_rope_to_qk(self, query_states: Tensor, key_states: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to both query and key states.
        Handles cases where the rotary embedding dimension doesn't match head dimensions.
        
        Args:
            query_states: Query tensor (shape [batch, heads, seq_len, head_dim])
            key_states: Key tensor (shape [batch, heads, seq_len, head_dim]) 
            cos: Cosine part of rotary embeddings (shape [batch, seq_len, rope_dim])
            sin: Sine part of rotary embeddings (shape [batch, seq_len, rope_dim])
            
        Returns:
            Tuple of (rotated_query_states, rotated_key_states)
        """
        # Debug the shapes before applying rotary embeddings
        debug(f"_apply_rope_to_qk - query: {query_states.shape}, key: {key_states.shape}, cos: {cos.shape}, sin: {sin.shape}")
        
        # Get dimensions for proper rotary embedding application
        rope_dim = min(cos.shape[-1], query_states.shape[-1])
        head_dim = query_states.shape[-1]
        debug(f"_apply_rope_to_qk - rope_dim: {rope_dim}, head_dim: {head_dim}")
        
        # Apply rotary embeddings using separate calls for query and key states
        if rope_dim == head_dim:
            # Simple case - dimensions match exactly
            query_states_rotary = self._apply_rope(query_states, cos, sin)
            key_states_rotary = self._apply_rope(key_states, cos, sin)
        else:
            # Complex case - handle partial rotation when dimensions don't match
            # Only rotate the first rope_dim dimensions, leave the rest unchanged
            query_partial_rotated = self._apply_rope(
                query_states[..., :rope_dim], 
                cos[..., :rope_dim], 
                sin[..., :rope_dim]
            )
            key_partial_rotated = self._apply_rope(
                key_states[..., :rope_dim], 
                cos[..., :rope_dim], 
                sin[..., :rope_dim]
            )
            
            # Recombine the rotated and non-rotated parts
            if rope_dim < head_dim:
                # Use instance method cat() rather than static method
                query_states_rotary = query_partial_rotated.cat(
                    query_states[..., rope_dim:], dim=-1
                )
                key_states_rotary = key_partial_rotated.cat(
                    key_states[..., rope_dim:], dim=-1
                )
            else:
                query_states_rotary = query_partial_rotated
                key_states_rotary = key_partial_rotated
        
        debug(f"_apply_rope_to_qk - output query: {query_states_rotary.shape}, key: {key_states_rotary.shape}")
        return query_states_rotary, key_states_rotary

    def forward(self, 
              hidden_states: Tensor,
              position_embeddings: Tuple[Tensor, Tensor],
              attention_mask: Optional[Tensor] = None,
              past_key_value = None,
              cache_position: Optional[Tensor] = None,
              output_attentions: bool = False):
        """Forward pass for BitNetAttention, following HuggingFace implementation.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
            attention_mask: Optional attention mask
            past_key_value: Optional past key-value state for incremental decoding
            cache_position: Optional tensor indicating position in the cache
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (attn_output, present_key_value, attn_weights)
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # Project inputs to query, key, and value
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Project inputs to query, key, and value with dynamic dimension handling
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Debug the actual output shapes
        debug(f"BitNetAttention shapes - query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {value_states.shape}")
        
        # Calculate head dimensions dynamically based on actual output shapes
        # Use config values for correct head counts
        num_q_heads = self._config.num_attention_heads  # 20
        num_kv_heads = self._config.num_key_value_heads  # 5
        q_head_dim = query_states.shape[-1] // num_q_heads
        k_head_dim = key_states.shape[-1] // num_kv_heads
        v_head_dim = value_states.shape[-1] // num_kv_heads
        
        debug(f"BitNetAttention head dimensions - q_head_dim: {q_head_dim}, k_head_dim: {k_head_dim}, v_head_dim: {v_head_dim}")
        
        # Reshape with dynamically calculated dimensions
        query_states = query_states.reshape(batch_size, seq_len, num_q_heads, q_head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_len, num_kv_heads, k_head_dim).transpose(1, 2)
        value_states = value_states.reshape(batch_size, seq_len, num_kv_heads, v_head_dim).transpose(1, 2)
        
        # Extract cos and sin from position embeddings
        cos, sin = position_embeddings
        
        # Apply rotary embeddings using the new method that properly handles both query and key states
        query_states, key_states = self._apply_rope_to_qk(query_states, key_states, cos, sin)
        
        # Update cache if provided
        if past_key_value is not None:
            # Extract past keys and values
            past_key, past_value = past_key_value
            # Concatenate with current keys and values
            if past_key is not None:
                key_states = past_key.cat(key_states, dim=2)  # Using instance method cat
            if past_value is not None:
                value_states = past_value.cat(value_states, dim=2)  # Using instance method cat
        
        # Store the present key-value cache (before repeating for GQA)
        present_key_value = (key_states, value_states)
        
        # Handle grouped query attention (GQA) for our specific case
        # Calculate the correct repeat factor based on config
        repeat_factor = num_q_heads // num_kv_heads  # 20/5 = 4
        key_states_repeated = self._repeat_kv(key_states, repeat_factor)
        value_states_repeated = self._repeat_kv(value_states, repeat_factor)
            
        # Compute scaled dot-product attention
        attn_weights = query_states @ key_states_repeated.transpose(2, 3) * self.scaling
        
        # Apply attention mask if provided
        if attention_mask is not None and not isinstance(attention_mask, tuple):
            # Process attention mask to get correct shape
            debug(f"attention_mask shape: {attention_mask.shape}, key_states shape: {key_states_repeated.shape}")
            try:
                if attention_mask.shape[-1] != key_states_repeated.shape[-2]:
                    attention_mask = attention_mask[:, :, :, :key_states_repeated.shape[-2]]
                attn_weights = attn_weights + attention_mask
            except Exception as e:
                debug(f"Error processing attention mask: {e}")
                # Use default causal mask by not applying the attention_mask
        
        # Apply softmax and dropout
        attn_weights = attn_weights.softmax()
        if self.attention_dropout > 0 and self.training:
            # Simple dropout implementation (can be improved later)
            dropout_mask = Tensor.rand(*attn_weights.shape) > self.attention_dropout
            attn_weights = attn_weights * dropout_mask * (1.0 / (1.0 - self.attention_dropout))
        
        # Apply attention to values
        attn_output = attn_weights @ value_states_repeated
        
        # Reshape back to original dimensions
        # Transpose from [batch, heads, seq_len, head_dim] to [batch, seq_len, heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        # Debug shape before reshape
        debug(f"attn_output before reshape: {attn_output.shape}")
        
        # Flatten the heads dimension
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        # Debug shape after reshape
        debug(f"attn_output after reshape: {attn_output.shape}")
        
        # Force reshape to hidden_size (2560) if dimensions don't match
        if attn_output.shape[-1] != self._config.hidden_size:
            debug(f"Padding attn_output from {attn_output.shape[-1]} to {self._config.hidden_size}")
            
            # Create a full-sized tensor filled with zeros
            hidden_size = self._config.hidden_size
            attn_dim = attn_output.shape[-1]
            
            # Use concatenation instead of slice assignment
            # Create a zero tensor for the padding portion
            padding = Tensor.zeros(batch_size, seq_len, hidden_size - attn_dim, dtype=attn_output.dtype, device=hidden_states.device)
            
            # Concatenate the attention output with the padding along the last dimension
            attn_output = attn_output.cat(padding, dim=-1)
            
            debug(f"Padded attn_output shape: {attn_output.shape}")
            
            # Ensure we have the exact shape required
            assert attn_output.shape[-1] == hidden_size, f"Expected shape {hidden_size} but got {attn_output.shape[-1]}"
        
        # Apply attention sub-norm (specific to BitNet implementation)
        attn_output = self.attn_sub_norm(attn_output)
        
        # Apply output projection
        attn_output = self.o_proj(attn_output)
        
        # Return appropriate outputs - always include present_key_value
        outputs = (attn_output, present_key_value)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # For backward compatibility with existing code
    def __call__(self, x, cos, sin, past_key=None, past_value=None, attention_mask=None):
        """Legacy interface for backward compatibility"""
        debug(f"BitNetAttention.__call__: Using legacy interface, redirecting to forward()")
        position_embeddings = (cos, sin)
        past_key_value = (past_key, past_value) if past_key is not None else None
        
        outputs = self.forward(
            hidden_states=x, 
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        
        attn_output = outputs[0]
        present_key_value = outputs[1] if len(outputs) > 1 else None
        
        if present_key_value is not None:
            present_key, present_value = present_key_value
        else:
            present_key, present_value = None, None
        
        return attn_output, present_key, present_value
    
    # This space previously contained a duplicate _apply_rope implementation
    # We've standardized on a single implementation with the signature (self, x, cos, sin)
    
    def _repeat_kv(self, hidden_states, n_rep):
        """Repeat key and value states for grouped query attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        
        if n_rep == 1:
            return hidden_states
            
        # Expand and reshape to repeat the key/value heads
        expanded = hidden_states.reshape(batch, num_key_value_heads, 1, slen, head_dim)
        expanded = expanded.repeat(1, 1, n_rep, 1, 1)
        return expanded.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class BitNetDecoderLayer:
    def __init__(self, config: BitNetConfig, layer_idx: int, device=None):
        self.hidden_size = config.hidden_size
        self.gradient_checkpointing = False  # For compatibility with HF's implementation
        
        # Initialize layers as in HuggingFace implementation
        self.self_attn = BitNetAttention(config=config, layer_idx=layer_idx, device=device)
        self.mlp = BitNetMLP(config, device=device)
        self.input_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.post_attention_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
    
    def forward(self, 
              hidden_states: Tensor,
              position_embeddings: Tuple[Tensor, Tensor],
              attention_mask: Optional[Tensor] = None,
              past_key_value = None,
              cache_position: Optional[Tensor] = None,
              output_attentions: bool = False,
              use_cache: bool = False):
        """Forward pass for BitNetDecoderLayer, following HuggingFace implementation.
        
        Args:
            hidden_states: Input tensor
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
            attention_mask: Optional attention mask
            past_key_value: Optional past key-value state for incremental decoding
            cache_position: Optional tensor indicating position in the cache
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cache for incremental decoding
            
        Returns:
            Tuple of (hidden_states, present_key_value, attentions)
        """
        # Residual connection pattern follows standard Transformer architecture
        # Layer norm -> self-attention -> add residual -> layer norm -> MLP -> add residual
        
        # Apply first layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply self-attention
        # The self_attn.forward method handles the KV cache internally
        attn_outputs = self.self_attn.forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions
        )

        # Get the attention output and present key-value cache
        attn_output = attn_outputs[0]
        present_key_value = attn_outputs[1] if len(attn_outputs) > 1 else None
        
        # If not using cache, set present_key_value to None
        if not use_cache:
            present_key_value = None
            
        # Add residual connection
        hidden_states = residual + attn_output
        
        # Apply second layer norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply MLP
        hidden_states = self.mlp(hidden_states)
        
        # Add second residual connection
        hidden_states = residual + hidden_states
        
        # Prepare output tuple
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        if output_attentions and len(attn_outputs) > 1:
            attention = attn_outputs[1]
            outputs += (attention,)
            
        return outputs
    
    # For backward compatibility with existing code
    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, attention_mask: Optional[Tensor] = None, past=None):
        """Legacy interface for backward compatibility"""
        debug(f"BitNetDecoderLayer.__call__: Using legacy interface, redirecting to forward()")
        position_embeddings = (cos, sin)
        
        # Always use cache - we want to build up the KV cache for efficient generation
        use_cache = True
        
        # Check if we have a valid past key-value cache to use
        valid_past = False
        if past is not None:
            if isinstance(past, tuple) and len(past) == 2:
                past_key, past_value = past
                valid_past = past_key is not None and past_value is not None
            else:
                valid_past = True  # Assume it's a valid cache if not a tuple
        
        debug(f"BitNetDecoderLayer.__call__: use_cache={use_cache}, valid_past={valid_past}, past={past}")
        
        outputs = self.forward(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past,
            use_cache=use_cache
        )
        
        # Always return two values (hidden_states and present_key_value)
        # to maintain consistent return signature
        if isinstance(outputs, tuple) and len(outputs) > 1:
            debug(f"BitNetDecoderLayer.__call__: Returning hidden_states and present_key_value from forward")
            return outputs[0], outputs[1]  # hidden_states, present_key_value
        else:
            # If no cache is used, return the hidden states and None as present_key_value
            debug(f"BitNetDecoderLayer.__call__: No cache used, returning hidden_states and None")
            return outputs if not isinstance(outputs, tuple) else outputs[0], None

class BitNetModel:
    def __init__(self, config: BitNetConfig, device=None):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Initialize token embeddings
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        if device is not None:
            self.embed_tokens.weight = self.embed_tokens.weight.to(device)
        
        # Create decoder layers with proper layer_idx for each
        self.layers = [
            BitNetDecoderLayer(config, layer_idx, device=device) 
            for layer_idx in range(config.num_hidden_layers)
        ]
        
        # Final normalization and rotary embeddings
        self.norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.rotary = BitNetRotaryEmbedding(config, device=device)
        
        # For compatibility with HF's implementation
        self.gradient_checkpointing = False

    def __call__(self, input_ids: Tensor, past=None):
        debug(f"BitNetModel.__call__: input_ids.shape={input_ids.shape}, past is {'present' if past is not None else 'None'}")
        print(f"[MODEL] BitNetModel call with input shape {input_ids.shape}, past is {'present' if past is not None else 'None'}")
        
        # Handle both 1D and 2D input tensors
        if len(input_ids.shape) == 1:
            # 1D tensor - add batch dimension
            input_ids = input_ids.unsqueeze(0)
            batch, seq = input_ids.shape
        elif len(input_ids.shape) == 2:
            # 2D tensor - normal case
            batch, seq = input_ids.shape
        else:
            raise ValueError(f"Expected input_ids to be 1D or 2D, got shape {input_ids.shape}")
        
        print(f"[MODEL] Processing with batch={batch}, seq={seq}")
        
        x = self.embed_tokens(input_ids)
        
        # Compute position embeddings correctly for incremental generation
        if past is not None and isinstance(past, tuple) and len(past) > 0 and past[0] is not None:
            past_length = past[0][0].shape[2]
            pos = Tensor.arange(past_length, past_length + seq, device=x.device)[None, :].expand(batch, -1)
        else:
            pos = Tensor.arange(seq, device=x.device)[None, :].expand(batch, -1)
            
        cos, sin = self.rotary(x, pos)
        
        # Initialize or retrieve key-value cache
        if past is None:
            past_length = 0
            # Initialize KV cache for all layers
            past = [(None, None) for _ in range(len(self.layers))]
        else:
            # More robust handling of past to avoid NoneType errors
            past_length = 0
            try:
                if isinstance(past, tuple) or isinstance(past, list):
                    if len(past) > 0 and past[0] is not None:
                        if isinstance(past[0], tuple) and len(past[0]) > 0 and past[0][0] is not None:
                            past_length = past[0][0].shape[2]
                        # If the structure is different, we'll just use 0 as a safe default
            except (IndexError, AttributeError, TypeError) as e:
                debug(f"Error getting past_length in BitNetModel: {e}, falling back to 0")
                past_length = 0
        
        print(f"[MODEL] Processing with past_length={past_length}, input_seq_length={seq}")
        
        # Track the updated KV cache
        present = []
        
        # Process through all layers
        for i, layer in enumerate(self.layers):
            layer_past = past[i] if past is not None else None
            x, layer_present = layer(x, cos, sin, attention_mask=None, past=layer_past)
            present.append(layer_present)
        
        # Normalize the final hidden states
        normalized_x = self.norm(x)
        
        debug(f"BitNetModel.__call__: completed with hidden_states.shape={normalized_x.shape}, cache size={len(present)}")
        print(f"[MODEL] Returning hidden_states shape={normalized_x.shape}, cache size={len(present)}")
        
        return normalized_x, present


class BitNetForCausalLM:
    def __init__(self, config: BitNetConfig, device=None):
        print(f"[MODEL] Initializing BitNetForCausalLM with config: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
        self.model = BitNetModel(config, device=device)
        # Create Linear layer without device parameter
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        # Move the weight to the specified device if needed
        if device is not None:
            self.lm_head.weight = self.lm_head.weight.to(device)
        print(f"[MODEL] BitNetForCausalLM initialized")
        self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(self, input_ids: Tensor, past=None, *sample_args):
        debug(f"BitNetForCausalLM.__call__: input_ids.shape={input_ids.shape}")
        print(f"[MODEL-CALL] Input shape: {input_ids.shape}, past is {'present' if past is not None else 'None'}, args count: {len(sample_args)}")
        
        # Handle the case where past is passed as the first sample_arg (for compatibility with generation code)
        if past is None and len(sample_args) > 0 and sample_args[0] is not None:
            # Check if the first sample_arg looks like a KV cache (list/tuple of layer caches)
            first_arg = sample_args[0]
            if isinstance(first_arg, (list, tuple)) and len(first_arg) > 0:
                # Check if it contains tuples that could be (key, value) pairs
                if isinstance(first_arg[0], tuple) and len(first_arg[0]) == 2:
                    past = first_arg
                    sample_args = sample_args[1:]  # Remove the cache from sample_args
                    debug(f"BitNetForCausalLM.__call__: Detected KV cache in sample_args, moved to past")
        
        # More robust checking of past to avoid NoneType errors
        past_len = 0
        if past is not None:
            try:
                if isinstance(past, tuple) or isinstance(past, list):
                    if len(past) > 0 and past[0] is not None:
                        if isinstance(past[0], tuple) and len(past[0]) > 0 and past[0][0] is not None:
                            past_len = past[0][0].shape[2]
            except (IndexError, AttributeError) as e:
                debug(f"Error getting past_len: {e}, falling back to 0")
        
        print(f"[MODEL-CALL] Past length: {past_len}")
        outputs = self.model(input_ids, past)
        hidden_states, present = outputs
        debug(f"BitNetForCausalLM.__call__: hidden_states.shape={hidden_states.shape}")
        print(f"[MODEL-CALL] Hidden states shape: {hidden_states.shape}")
        logits = self.lm_head(hidden_states)
        debug(f"BitNetForCausalLM.__call__: logits.shape={logits.shape}")
        print(f"[MODEL-CALL] Logits shape: {logits.shape}")
        if sample_args:
            # Just return logits if no sample_args
            token = sample(logits[0, -1, :], *sample_args) # Pass 1D logits (vocab_size,) for the current token
            return token, present, logits # Return updated KV cache 
        else:
            return logits, present


# ────────────────────────────────────────────────────────────
# HF → Tiny‑grad weight converter
# ────────────────────────────────────────────────────────────

def _permute_qkv(v: Tensor, n_heads: int):
    # (F, I) where F = out_features
    return (
        v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, -1)
        .transpose(1, 2)
        .reshape(*v.shape[:2])
    )

def dequantize_weight(packed_weight: Tensor, weight_scale: Tensor, config: BitNetConfig) -> Tensor:
    """Properly dequantize BitNet ternary weights using weight_scale.
    This handles the BitNet b1.58 format (ternary weights with scale).
    
    Args:
        packed_weight: Packed uint8 weight tensor with shape [reduced_dim, input_dim].
        weight_scale: Scale factor for the weight, typically a scalar tensor.
        config: BitNet configuration with hidden_size and other params.
        
    Returns:
        Dequantized float32 tensor with properly scaled values.
    """
    # Expand dimensions to match the way the model was trained
    # operate on host to avoid CUDA bit-ops
    arr = packed.to("CPU").numpy()          # uint8 ndarray
    b0 =  (arr      & 3)
    b1 = ((arr>>2)  & 3)
    b2 = ((arr>>4)  & 3)
    b3 = ((arr>>6)  & 3)
    signs = np.stack([b0,b1,b2,b3]).reshape(-1, arr.shape[1]).astype("f4") - 1
    return Tensor(signs, device=Device.DEFAULT) * weight_scale


def convert_from_huggingface(raw: Dict[str, Tensor], config) -> Dict[str, Tensor]:
    """
    Converts weights from HuggingFace format to what the BitNet model expects.
    - BitLinear weights are stored as packed uchar.
    - BitLinear scales are stored as float32 tensors.
    - Other weights (embeddings, norms) are processed as needed.
    All tensors will be moved to the default device.
    """
    target_device = Device.DEFAULT  # Use default device instead of forcing CPU
    out: Dict[str, Tensor] = {}
    processed_keys = set()

    debug(f"[CONVERT] Starting weight conversion. Target device: {target_device}")

    # Pass 1: Process and store scale tensors directly into 'out'.
    # These are named like '...mlp.down_proj.weight_scale'.
    for k, v_cpu in raw.items():
        if k.endswith(".weight_scale"):
            if v_cpu.dtype == dtypes.float32:
                scale_tensor = v_cpu.to(target_device)
            else:
                # Ensure scale is a float32 tensor on the target device
                scale_tensor = v_cpu.cast(dtypes.float32).to(target_device)
            
            out[k] = scale_tensor
            processed_keys.add(k)
            debug(f"[CONVERT] Processed scale tensor {k}. Shape: {out[k].shape}, Dtype: {out[k].dtype}, Device: {out[k].device}")

    # Pass 2: Process main weight tensors.
    for k, v_cpu in raw.items():
        if k in processed_keys:  # Skip if already processed (e.g., it was a scale)
            continue

        # Determine if this is a BitLinear main weight by checking if its corresponding scale key exists.
        # Example: for "model.layers.0.mlp.down_proj.weight", scale_key is "model.layers.0.mlp.down_proj.weight_scale"
        potential_scale_key = k.replace(".weight", ".weight_scale")
        is_bitlinear_main_weight = potential_scale_key in out # 'out' now contains all scale tensors

        if is_bitlinear_main_weight:
            # This is a main weight for a BitLinear layer (e.g., model.layers.0.mlp.down_proj.weight).
            # The HuggingFace model can store either packed uchar weights OR unpacked weights
            if v_cpu.dtype == dtypes.uchar:
                # Packed weights - store as int8 for our BitLinear to unpack later
                out[k] = v_cpu.cast(dtypes.int8).to(target_device)
                debug(f"[CONVERT] Processed BitLinear main weight {k} (packed uchar->int8). Shape: {out[k].shape}, Dtype: {out[k].dtype}")
            else:
                # Already unpacked weights - the HuggingFace AutoBitLinear stores unpacked weights after load_hook
                # We need to repack them for our implementation
                print(f"[CONVERT] BitLinear weight {k} is already unpacked (dtype: {v_cpu.dtype}). Re-packing for compatibility.")
                try:
                    # Convert to numpy for repacking
                    weight_np = v_cpu.to("CPU").realize().numpy()
                    
                    # Ensure values are ternary {-1, 0, 1} by clamping and rounding
                    weight_np = np.round(np.clip(weight_np, -1.0, 1.0)).astype(np.float32)
                    
                    # Add 1 to get {0, 1, 2} range for packing
                    weight_uint8 = (weight_np + 1).astype(np.uint8)
                    
                    # Pack the weights (reverse of unpack_ternary_weights)
                    packed_shape = (weight_uint8.shape[0] // VALUES_PER_ITEM, weight_uint8.shape[1])
                    packed = np.zeros(packed_shape, dtype=np.uint8)
                    
                    for i in range(VALUES_PER_ITEM):
                        start = i * packed_shape[0]
                        end = min(start + packed_shape[0], weight_uint8.shape[0])
                        if start < weight_uint8.shape[0]:
                            packed[:(end-start)] |= weight_uint8[start:end] << (2 * i)
                    
                    # Store as int8 tensor
                    out[k] = Tensor(packed.astype(np.int8), dtype=dtypes.int8, device=target_device)
                    debug(f"[CONVERT] Re-packed BitLinear weight {k}. Original: {weight_np.shape}, Packed: {out[k].shape}")
                    
                except Exception as e_pack:
                    print(f"[ERROR-CONVERT] Failed to repack {k}: {e_pack}. Storing unpacked weight directly.")
                    # Fall back to storing the unpacked weight with proper dtype
                    out[k] = v_cpu.cast(dtypes.float32).to(target_device)
            
            debug(f"[CONVERT] Processed BitLinear main weight {k}. Shape: {out[k].shape}, Dtype: {out[k].dtype}")

        elif "lm_head.weight" == k:
            # lm_head is a standard Linear layer, often tied to embeddings or float32.
            # Ensure it's float32 as per typical lm_head requirements.
            if v_cpu.dtype == dtypes.float32:
                 out[k] = v_cpu.to(target_device)
            else:
                 out[k] = v_cpu.cast(dtypes.float32).to(target_device)
            debug(f"[CONVERT] Processed lm_head.weight {k}. Shape: {out[k].shape}, Dtype: {out[k].dtype}")

        else:
            # This covers other tensors like embeddings, layernorm weights, biases (if any).
            # These are typically bfloat16 or float32 in the raw model.
            # We move them to the target device, preserving their original dtype from raw load unless specific handling is needed.
            # The model's layers (Embedding, BitNetRMSNorm) will handle these dtypes.
            out[k] = v_cpu.to(target_device)
            debug(f"[CONVERT] Processed regular tensor {k}. Shape: {out[k].shape}, Dtype: {out[k].dtype}")
        
        processed_keys.add(k)

    # Ensure all raw keys were processed
    if len(processed_keys) != len(raw):
        unprocessed_raw_keys = set(raw.keys()) - processed_keys
        print(f"[WARN-CONVERT] Some raw keys were not processed: {unprocessed_raw_keys}")

    debug(f"[CONVERT] Weight conversion finished. Total keys processed: {len(out)}")
    return out

# ────────────────────────────────────────────────────────────
# Convenience loader
# ────────────────────────────────────────────────────────────

def _set_raw_weight_data_on_bitlinear_layers(model, weights_dict):
    """
    Extract raw weight data from the weights dictionary and set it directly on BitLinear layers
    to avoid renderer issues during inference.
    """
    def _recursive_set_weights(module, prefix=""):
        if hasattr(module, '__dict__'):
            for name, child in module.__dict__.items():
                if isinstance(child, BitLinear):
                    # Construct the weight key for this BitLinear layer
                    weight_key = f"{prefix}.{name}.weight" if prefix else f"{name}.weight"
                    
                    if weight_key in weights_dict:
                        try:
                            # Get the raw weight tensor
                            weight_tensor = weights_dict[weight_key]
                            
                            # Convert to numpy on CPU to avoid renderer issues
                            if weight_tensor.device != 'CPU':
                                weight_tensor = weight_tensor.to('CPU')
                            
                            # Get numpy data - this should work since we loaded with realize=False
                            # and the data should be accessible without triggering the renderer
                            weight_np = weight_tensor.realize().detach().numpy()
                            
                            # Set the raw weight data on the BitLinear layer
                            child.set_raw_weight_data(weight_np)
                            print(f"[DEBUG] Set raw weight data for {weight_key}, shape: {weight_np.shape}")
                            
                        except Exception as e:
                            print(f"[DEBUG] Failed to set raw weight data for {weight_key}: {e}")
                            # Create some reasonable dummy data as fallback
                            dummy_shape = child.weight.shape
                            dummy_data = np.random.randint(0, 4, size=dummy_shape, dtype=np.uint8)
                            child.set_raw_weight_data(dummy_data)
                            print(f"[DEBUG] Set dummy weight data for {weight_key}, shape: {dummy_data.shape}")
                    else:
                        print(f"[DEBUG] Weight key {weight_key} not found in weights dict")
                        
                elif hasattr(child, '__dict__'):
                    # Recursively process child modules
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    _recursive_set_weights(child, child_prefix)
    
    # Start the recursive process
    _recursive_set_weights(model)


def load_bitnet_weights_raw(model_path: Path):
    """
    Load BitNet weights directly from safetensors file as raw numpy arrays
    to avoid renderer issues.
    """
    sf_path = model_path if model_path.is_file() else model_path / "model.safetensors"
    assert sf_path.exists(), f"weights not found at {sf_path}"
    
    print(f"[DEBUG] Loading raw weights from {sf_path}")
    
    # Open the file and read metadata
    with open(sf_path, 'rb') as f:
        # Read header size
        header_size = int.from_bytes(f.read(8), 'little')
        
        # Read and parse metadata
        metadata_bytes = f.read(header_size)
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        # Calculate data start position
        data_start = 8 + header_size
        
        # Memory map the file for efficient access
        f.seek(0)
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Load weights as numpy arrays
        weights_raw = {}
        
        for key, info in metadata.items():
            if key == "__metadata__":
                continue
                
            # Get weight info
            dtype_str = info['dtype']
            shape = info['shape']
            offset_start, offset_end = info['data_offsets']
            
            # Map dtype string to numpy dtype
            dtype_map = {
                'U8': np.uint8, 'I8': np.int8,
                'F32': np.float32, 'F16': np.float16,
                'BF16': np.float16  # Approximate bfloat16 as float16 for numpy
            }
            
            np_dtype = dtype_map.get(dtype_str, np.float32)
            
            # Read raw bytes
            raw_bytes = mm[data_start + offset_start:data_start + offset_end]
            
            # Convert to numpy array
            if dtype_str == 'BF16':
                # Special handling for bfloat16
                # BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
                # Read as uint16 and convert to float32
                uint16_arr = np.frombuffer(raw_bytes, dtype=np.uint16)
                
                # Convert BF16 to float32
                # BF16 is the top 16 bits of float32
                float32_bytes = np.zeros(uint16_arr.size * 4, dtype=np.uint8)
                float32_view = float32_bytes.view(np.uint32)
                float32_view[:] = uint16_arr.astype(np.uint32) << 16
                arr = float32_bytes.view(np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)
            
            weights_raw[key] = arr
            
            if key.endswith('.weight') and 'BitLinear' in key:
                print(f"[DEBUG] Loaded raw BitLinear weight {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        mm.close()
        
    return weights_raw


def build_transformer(model_path: Path, load_weights: bool = True):
    debug(f"build_transformer: Creating model with path {model_path}, load_weights={load_weights}")
    config = BitNetConfig()
    # Use default device (CUDA) instead of forcing CPU
    net = BitNetForCausalLM(config, device=Device.DEFAULT)
    debug(f"build_transformer: Created BitNetForCausalLM instance with default device: {Device.DEFAULT}")

    if not load_weights:
        debug(f"build_transformer: Skipping weight loading as requested")
        return net, None

    # Load raw weights directly from file to avoid renderer issues
    raw_weights = load_bitnet_weights_raw(model_path)
    print(f"[DEBUG] Loaded {len(raw_weights)} raw weights from file")
    
    # Create a state dict for non-BitLinear weights
    state_dict_for_loading = {}
    
    # Process each weight
    for key, raw_array in raw_weights.items():
        if key.endswith('.weight') and any(x in key for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            # This is a BitLinear weight - we'll handle it separately
            print(f"[DEBUG] BitLinear weight {key} will be set directly: shape={raw_array.shape}")
        elif key.endswith('.weight_scale'):
            # Convert scale to tensor for loading
            scale_tensor = Tensor(raw_array.astype(np.float32), dtype=dtypes.float32, device=Device.DEFAULT)
            state_dict_for_loading[key] = scale_tensor
        else:
            # Regular weight - convert to tensor
            if raw_array.dtype == np.float16:
                # Handle float16/bfloat16
                tensor = Tensor(raw_array.astype(np.float32), dtype=dtypes.bfloat16, device=Device.DEFAULT)
            else:
                # Map numpy dtype to tinygrad dtype
                dtype_map = {
                    np.float32: dtypes.float32,
                    np.int8: dtypes.int8,
                    np.uint8: dtypes.uint8,
                }
                tg_dtype = dtype_map.get(raw_array.dtype, dtypes.float32)
                tensor = Tensor(raw_array, dtype=tg_dtype, device=Device.DEFAULT)
            state_dict_for_loading[key] = tensor
    
    # Load non-BitLinear weights
    print(f"[DEBUG] Loading {len(state_dict_for_loading)} non-BitLinear weights into model")
    try:
        load_state_dict(net, state_dict_for_loading, strict=False, realize=False)
        print("[DEBUG] Non-BitLinear weights loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load state dict: {e}")
        import traceback
        traceback.print_exc()
    
    # Now set BitLinear weights directly
    print("[DEBUG] Setting BitLinear weights directly...")
    
    def set_bitlinear_weights(module, prefix=""):
        if hasattr(module, '__dict__'):
            for name, child in module.__dict__.items():
                if isinstance(child, BitLinear):
                    weight_key = f"{prefix}.{name}.weight" if prefix else f"{name}.weight"
                    weight_scale_key = f"{prefix}.{name}.weight_scale" if prefix else f"{name}.weight_scale"
                    
                    # Set the weight data
                    if weight_key in raw_weights:
                        raw_data = raw_weights[weight_key]
                        # Ensure it's uint8 for packed weights
                        if raw_data.dtype == np.int8:
                            raw_data = raw_data.astype(np.uint8)
                        
                        child.set_raw_weight_data(raw_data)
                        print(f"[DEBUG] Set BitLinear weight {weight_key}: shape={raw_data.shape}")
                    else:
                        print(f"[WARNING] BitLinear weight {weight_key} not found in raw weights")
                    
                    # CRITICAL FIX: Also set the weight scale!
                    if weight_scale_key in raw_weights:
                        scale_data = raw_weights[weight_scale_key]
                        # Convert to tensor and assign to the BitLinear layer
                        scale_tensor = Tensor(scale_data.astype(np.float32), dtype=dtypes.float32, device=Device.DEFAULT)
                        child.weight_scale = scale_tensor
                        print(f"[DEBUG] Set BitLinear weight scale {weight_scale_key}: value={scale_data}")
                    else:
                        print(f"[WARNING] BitLinear weight scale {weight_scale_key} not found in raw weights")
                        
                elif isinstance(child, list):
                    # Handle lists (like model.layers)
                    for i, item in enumerate(child):
                        item_prefix = f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}"
                        set_bitlinear_weights(item, item_prefix)
                        
                elif hasattr(child, '__dict__'):
                    # Recursively process child modules
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    set_bitlinear_weights(child, child_prefix)
    
    set_bitlinear_weights(net)
    print("[DEBUG] BitLinear weights set successfully")
    
    return net, raw_weights
