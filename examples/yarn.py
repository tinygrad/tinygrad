import torch
import math
import numpy as np
from tinygrad import Tensor, Device
from tinygrad import dtypes

# Set device for tinygrad
Device.DEFAULT = "CPU"

# ============ PYTORCH IMPLEMENTATION (from gpt-oss) ============

def _apply_rotary_emb_pytorch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbeddingPyTorch(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self):
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        print(f'pytorch apply_rotary {query.shape=}')
        query = _apply_rotary_emb_pytorch(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb_pytorch(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


# ============ TINYGRAD IMPLEMENTATION ============

def precompute_freqs_cis_tinygrad(dim:int, end:int, theta:float = 10000.0, scale:float=1.0, ntk_alpha:float=1, ntk_beta:float=32, initial_context_length:int=4096) -> Tensor:
  half = dim // 2
  freqs = (theta ** (-(Tensor.arange(half, dtype="float32") / half)))[None, :]

  # rope
  if scale <= 1:
    freqs = Tensor.arange(end, dtype="float32")[:, None] * freqs
    return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, half, 2)

  # yarn https://arxiv.org/pdf/2309.00071
  def _ratio(ntk): return half * math.log(initial_context_length / (ntk * 2 * math.pi)) / math.log(theta)
  low, high = _ratio(ntk_alpha), _ratio(ntk_beta)
  interpolation, extrapolation = freqs, freqs / scale
  ramp = (Tensor.arange(half, dtype=dtypes.float32, device=freqs.device) - low) / (high - low)
  mask = 1 - ramp.clamp(0, 1)
  freqs = interpolation * (1 - mask) + extrapolation * mask
  freqs = Tensor.arange(end, dtype=dtypes.float32)[:, None] * freqs
  mscale = 0.1 * math.log(scale) + 1.0
  return Tensor.stack(freqs.cos() * mscale, freqs.sin() * mscale, dim=-1).reshape(1, end, 1, half, 2)

def complex_mult_tinygrad(A, c, d):
    a, b = A[..., 0:1], A[..., 1:2]
    ro = a*c - b*d
    co = a*d + b*c
    return ro.cat(co, dim=-1)


def apply_rotary_emb_tinygrad(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    xq = xq.reshape(*xq.shape[0:-1], -1, 2)
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)
    assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
    print(f'tinygrad apply_rotary {xq.shape=}')
    c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
    xq_out = complex_mult_tinygrad(xq, c, d)
    xk_out = complex_mult_tinygrad(xk, c, d)
    return xq_out.flatten(3), xk_out.flatten(3)


# ============ COMPARISON TESTS ============

def print_section(title, char="=", width=80):
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}")

def print_subsection(title, char="-", width=80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def compare_implementations():
    print_section("PYTORCH (gpt-oss) vs TINYGRAD IMPLEMENTATION COMPARISON")

    # Test parameters
    head_dim = 64
    num_tokens = 128
    num_heads = 8
    base = 10000.0
    initial_context_length = 4096

    # Test different scaling factors
    test_configs = [
        {"scale": 1.0, "name": "No scaling (baseline RoPE)"},
        {"scale": 2.0, "name": "YaRN scale=2"},
        {"scale": 8.0, "name": "YaRN scale=8"},
        {"scale": 16.0, "name": "YaRN scale=16"},
    ]

    all_passed = True

    for config_idx, config in enumerate(test_configs, 1):
        scale = config["scale"]

        print_section(f"TEST {config_idx}/{len(test_configs)}: {config['name']}", char="=")

        # ============ PYTORCH ============
        rope_pytorch = RotaryEmbeddingPyTorch(
            head_dim=head_dim,
            base=int(base),
            dtype=torch.float32,
            initial_context_length=initial_context_length,
            scaling_factor=scale,
            ntk_alpha=1.0,
            ntk_beta=32.0,
            device=torch.device("cpu")
        )

        cos_pt, sin_pt = rope_pytorch._compute_cos_sin(num_tokens)
        concentration_pt, inv_freq_pt = rope_pytorch._compute_concentration_and_inv_freq()

        # ============ TINYGRAD ============
        freqs_cis_tg = precompute_freqs_cis_tinygrad(
            dim=head_dim,
            end=num_tokens,
            theta=base,
            scale=scale,
            ntk_alpha=1.0,
            ntk_beta=32.0,
            initial_context_length=initial_context_length
        )

        # Extract cos and sin from tinygrad format
        # freqs_cis_tg is [1, num_tokens, 1, head_dim//2, 2]
        cos_tg = freqs_cis_tg[0, :, 0, :, 0]  # [num_tokens, head_dim//2]
        sin_tg = freqs_cis_tg[0, :, 0, :, 1]  # [num_tokens, head_dim//2]

        # ============ TEST 1: FREQUENCY COMPUTATION ============
        print_subsection("Test 1.1: Concentration Factor & Inverse Frequencies")

        print(f"\nPyTorch concentration: {concentration_pt:.6f}")
        print(f"Tinygrad mscale:       {0.1 * math.log(scale) + 1.0 if scale > 1 else 1.0:.6f}")

        if abs(concentration_pt - (0.1 * math.log(scale) + 1.0 if scale > 1 else 1.0)) < 1e-6:
            print("✓ Concentration factors match")
        else:
            print("✗ Concentration factors don't match")
            all_passed = False

        print(f"\nInverse frequency statistics:")
        print(f"  PyTorch  - shape: {inv_freq_pt.shape}, first 3: {inv_freq_pt[:3].numpy()}")
        print(f"  PyTorch  - last 3: {inv_freq_pt[-3:].numpy()}")

        # ============ TEST 2: COS/SIN VALUES ============
        print_subsection("Test 1.2: Cosine and Sine Values")

        cos_pt_np = cos_pt.numpy()
        sin_pt_np = sin_pt.numpy()
        cos_tg_np = cos_tg.numpy()
        sin_tg_np = sin_tg.numpy()

        cos_diff = np.abs(cos_pt_np - cos_tg_np)
        sin_diff = np.abs(sin_pt_np - sin_tg_np)

        print(f"\nShape comparison:")
        print(f"  PyTorch:  cos {cos_pt.shape}, sin {sin_pt.shape}")
        print(f"  Tinygrad: cos {cos_tg.shape}, sin {sin_tg.shape}")

        print(f"\nNumerical differences:")
        print(f"  Cosine  - Max: {cos_diff.max():.2e}, Mean: {cos_diff.mean():.2e}")
        print(f"  Sine    - Max: {sin_diff.max():.2e}, Mean: {sin_diff.mean():.2e}")

        freq_test_passed = cos_diff.max() < 1e-5 and sin_diff.max() < 1e-5

        if freq_test_passed:
            print("✓ PASS: Frequency computation matches!")
        else:
            print("✗ FAIL: Frequency computation doesn't match!")
            print(f"\nSample comparison at token 0, first 5 dimensions:")
            print(f"  PyTorch cos:  {cos_pt_np[0, :5]}")
            print(f"  Tinygrad cos: {cos_tg_np[0, :5]}")
            print(f"  Difference:   {cos_diff[0, :5]}")
            all_passed = False

        # ============ TEST 3: FULL FORWARD PASS ============
        print_subsection("Test 2: Full Rotary Embedding Application")

        # Create dummy query and key tensors
        batch_size = 2
        seq_len = num_tokens

        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # PyTorch format: [seq_len, num_heads, head_dim]
        query_pt = torch.randn(seq_len, num_heads, head_dim)
        key_pt = torch.randn(seq_len, num_heads, head_dim)

        # Apply PyTorch RoPE
        query_rope_pt, key_rope_pt = rope_pytorch(query_pt, key_pt)

        # Tinygrad format: [batch, seq_len, num_heads, head_dim]
        # Convert PyTorch to Tinygrad format
        query_tg = Tensor(query_pt.numpy()).reshape(1, seq_len, num_heads, head_dim)
        key_tg = Tensor(key_pt.numpy()).reshape(1, seq_len, num_heads, head_dim)

        # Apply Tinygrad RoPE
        query_rope_tg, key_rope_tg = apply_rotary_emb_tinygrad(query_tg, key_tg, freqs_cis_tg)

        # Convert back for comparison
        query_rope_tg_np = query_rope_tg.numpy().reshape(seq_len, num_heads, head_dim)
        key_rope_tg_np = key_rope_tg.numpy().reshape(seq_len, num_heads, head_dim)

        query_diff = np.abs(query_rope_pt.numpy() - query_rope_tg_np)
        key_diff = np.abs(key_rope_pt.numpy() - key_rope_tg_np)

        print(f"\nQuery tensor differences:")
        print(f"  Max:  {query_diff.max():.2e}")
        print(f"  Mean: {query_diff.mean():.2e}")
        print(f"  Shape: {query_diff.shape}")

        print(f"\nKey tensor differences:")
        print(f"  Max:  {key_diff.max():.2e}")
        print(f"  Mean: {key_diff.mean():.2e}")
        print(f"  Shape: {key_diff.shape}")

        # Find position of maximum difference for query
        query_max_pos = np.unravel_index(query_diff.argmax(), query_diff.shape)
        key_max_pos = np.unravel_index(key_diff.argmax(), key_diff.shape)

        print(f"\n--- Query: Maximum difference location ---")
        print(f"Position: [seq={query_max_pos[0]}, head={query_max_pos[1]}, dim={query_max_pos[2]}]")
        print(f"  PyTorch value:  {query_rope_pt.numpy()[query_max_pos]:.6f}")
        print(f"  Tinygrad value: {query_rope_tg_np[query_max_pos]:.6f}")
        print(f"  Difference:     {query_diff[query_max_pos]:.6e}")

        # Show a small window around the max difference
        seq, head, dim = query_max_pos
        dim_start = max(0, dim - 2)
        dim_end = min(head_dim, dim + 3)
        print(f"\nContext (dims {dim_start}:{dim_end}):")
        print(f"  PyTorch:  {query_rope_pt.numpy()[seq, head, dim_start:dim_end]}")
        print(f"  Tinygrad: {query_rope_tg_np[seq, head, dim_start:dim_end]}")
        print(f"  Diff:     {query_diff[seq, head, dim_start:dim_end]}")

        print(f"\n--- Key: Maximum difference location ---")
        print(f"Position: [seq={key_max_pos[0]}, head={key_max_pos[1]}, dim={key_max_pos[2]}]")
        print(f"  PyTorch value:  {key_rope_pt.numpy()[key_max_pos]:.6f}")
        print(f"  Tinygrad value: {key_rope_tg_np[key_max_pos]:.6f}")
        print(f"  Difference:     {key_diff[key_max_pos]:.6e}")

        # Show a small window around the max difference
        seq, head, dim = key_max_pos
        dim_start = max(0, dim - 2)
        dim_end = min(head_dim, dim + 3)
        print(f"\nContext (dims {dim_start}:{dim_end}):")
        print(f"  PyTorch:  {key_rope_pt.numpy()[seq, head, dim_start:dim_end]}")
        print(f"  Tinygrad: {key_rope_tg_np[seq, head, dim_start:dim_end]}")
        print(f"  Diff:     {key_diff[seq, head, dim_start:dim_end]}")

        forward_test_passed = query_diff.max() < 1e-5 and key_diff.max() < 1e-5

        if forward_test_passed:
            print("\n✓ PASS: Full forward pass matches!")
        else:
            print("\n✗ FAIL: Full forward pass doesn't match!")
            all_passed = False

        # ============ TEST SUMMARY ============
        print_subsection(f"Summary for {config['name']}")
        print(f"  Frequency computation: {'✓ PASS' if freq_test_passed else '✗ FAIL'}")
        print(f"  Forward pass:          {'✓ PASS' if forward_test_passed else '✗ FAIL'}")

    # ============ FINAL SUMMARY ============
    print_section("FINAL RESULTS", char="=")
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The Tinygrad implementation matches PyTorch (gpt-oss) implementation.")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Check the output above for details on which tests failed.")
    print()


if __name__ == "__main__":
    compare_implementations()
