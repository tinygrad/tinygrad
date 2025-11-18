from dataclasses import dataclass
from transformers import AutoTokenizer
from tinygrad import nn, Tensor, dtypes
from tinygrad.helpers import Timing
from transformers import AutoConfig, AutoModelForCausalLM
import torch

@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256
    hidden_dim: int = 8192
    norm_eps: float = 1e-5
    max_seq_len: int = 64 # small for testing.

class Transformer:
    def __init__(self, args: ModelArgs):
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args.dim, args.n_heads, args.n_kv_heads, args.hidden_dim, args.norm_eps, args.max_seq_len) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, tokens: Tensor) -> Tensor:
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.output(h)

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2, dtype=dtypes.float16) / dim))
    t = Tensor.arange(end)
    freqs = t.unsqueeze(1) @ freqs.unsqueeze(0)
    freqs_cat = freqs.cat(freqs, dim=-1)
    return freqs_cat.cos(), freqs_cat.sin()

# See "Computational efficient realization of rotary matrix multiplication"
# https://arxiv.org/pdf/2104.09864
def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return (-x2).cat(x1, dim=-1)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (bs, seq_len, n_heads, head_dim)
    # cos, sin: (max_seq_len, head_dim)
    cos = cos[:x.shape[1]].reshape((1, x.shape[1], 1, x.shape[3]))
    sin = sin[:x.shape[1]].reshape((1, x.shape[1], 1, x.shape[3]))
    return (x * cos) + (rotate_half(x) * sin)

class MultiHeadAttention:
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len=2048, linear=nn.Linear):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = linear(self.n_heads * self.head_dim, dim, bias=False)

        # Precompute the rotary matrix.
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.freqs_cos.realize()
        self.freqs_sin.realize()

    def __call__(self, x:Tensor):
        bs, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.reshape(bs, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bs, seq_len, self.n_kv_heads, self.head_dim)

        xk = xk.repeat((1, 1, self.n_rep, 1))
        xv = xv.repeat((1, 1, self.n_rep, 1))

        xq = apply_rope(xq, self.freqs_cos, self.freqs_sin)
        xk = apply_rope(xk, self.freqs_cos, self.freqs_sin)

        xq = xq.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)

        scores = xq @ xk.transpose(-2, -1) / self.head_dim**0.5

        mask = Tensor.full(scores.shape[-2:], float("-inf")).triu(1)
        scores = scores + mask

        scores = scores.softmax(-1)

        output = scores @ xv

        output = output.permute(0, 2, 1, 3).reshape(bs, seq_len, -1)

        return self.wo(output)

class FeedForward:
    def __init__(self, dim, hidden_dim, linear=nn.Linear):
        self.w1 = linear(dim, hidden_dim, bias=False)
        self.w3 = linear(dim, hidden_dim, bias=False)
        self.w2 = linear(hidden_dim, dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.w3(x))

class TransformerBlock:
    def __init__(self, dim, n_heads, n_kv_heads, hidden_dim, norm_eps, max_seq_len):
        self.attention = MultiHeadAttention(dim, n_heads, n_kv_heads, max_seq_len)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)

    def __call__(self, x: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

def bench_tiny(sample: int = 10):
    args = ModelArgs() # llama3.2:1b config by default.
    model = Transformer(args)
    input_tokens = Tensor.randint((1, 10), low=0, high=args.vocab_size)

    # warm-up
    (model(input_tokens)).realize()

    with Timing(f"Tinygrad {sample} runs:", on_exit=lambda x: f" {sample*10/(x*1e-9):.2f} tok/s"):
        for _ in range(sample):
            (model(input_tokens)).realize()

def bench_torch(sample: int = 10):
    # from_config creates a model with random weights
    config = AutoConfig.from_pretrained("unsloth/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_config(config)
    input_tokens = torch.randint(0, config.vocab_size, (1, 10))

    # warm-up
    with torch.no_grad(): model(input_tokens)

    with Timing(f"Torch    {sample} runs:", on_exit=lambda x: f" {sample*10/(x*1e-9):.2f} tok/s"):
        with torch.no_grad():
            for _ in range(sample):
                model(input_tokens)

# This does not work... yet
def bench_torch_compile(sample: int = 10):
    config = AutoConfig.from_pretrained("unsloth/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_config(config)

    # Compile the core model, not the top-level wrapper.
    model.model = torch.compile(model.model, mode="max-autotune")

    input_tokens = torch.randint(0, config.vocab_size, (1, 10))
    attention_mask = torch.ones_like(input_tokens)

    # warm-up
    with torch.no_grad(): model(input_tokens, attention_mask=attention_mask)

    with Timing(f"Compiled {sample} runs:", on_exit=lambda x: f" {sample*10/(x*1e-9):.2f} tok/s"):
        with torch.no_grad():
            for _ in range(sample):
                model(input_tokens, attention_mask=attention_mask)

if __name__ == "__main__":
    bench_tiny(sample=10)
    bench_torch(sample=10)

    # HuggingFace transformer's library implementation prevents compilation.
    # bench_torch_compile(sample=10)
