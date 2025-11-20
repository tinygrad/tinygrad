# Test Tiny vs Torch Llama3.2:1b inference (no weights/autoregression).
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = '1'
os.environ["CPU_COUNT"] = '1'
from examples.llama3 import MODEL_PARAMS
from extra.models.llama import Transformer as TinyTransformer
from tinygrad import Tensor
import torch, math, unittest, time
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

# --- Begin PyTorch Llama3.2 implementation ---
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.weight = eps, torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

class Attention(torch.nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, max_context, rope_theta):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = torch.nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_context * 2, rope_theta))

    def forward(self, x: torch.Tensor, start_pos: int, mask: torch.Tensor = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1,2)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1,2)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1,2)

        def apply_rotary_emb(x, freqs_cis):
            x_ = x.float().reshape(*x.shape[:-1], -1, 2)
            x_out = torch.view_as_complex(x_)
            freqs_cis = freqs_cis.view(1, 1, *x_out.shape[2:])
            x_out = x_out * freqs_cis
            x_out = torch.view_as_real(x_out)
            return x_out.flatten(3).type_as(x)

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # n_kv_heads != n_heads implies GQA/MQA [arxiv/2307.09288, A.2.1]
        def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
            bs, n_kv_heads, slen, head_dim = x.shape
            return x if n_rep == 1 else (
                x[:, :, None, :, :]
                .expand(bs, n_kv_heads, n_rep, slen, head_dim)
                .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
            )

        n_rep = self.n_heads // self.n_kv_heads
        xk = repeat_kv(xk, n_rep)
        xv = repeat_kv(xv, n_rep)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None: scores = scores + mask

        scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, hidden_dim, norm_eps, max_context, rope_theta):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, max_context, rope_theta)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, mask: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), start_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class TorchTransformer(torch.nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, n_layers, norm_eps, rope_theta, vocab_size, hidden_dim, max_context=1024):
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, norm_eps, max_context, rope_theta)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = torch.nn.Linear(dim, vocab_size, bias=False)
        self.max_context = max_context

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, start_pos, mask)

        h = self.norm(h)
        return self.output(h)
# --- end of PyTorch Llama3.2 implementation ---

def benchmark(fxn, count):
    times = []
    for _ in range(count):
        st = time.perf_counter_ns()
        fxn()
        times.append(time.perf_counter_ns() - st)
    return sum(times) / len(times)  # mean time (ns)

class TestTinyVsTorchLlama1b(unittest.TestCase):
    def test_tiny_vs_torch_llama1b(self):
        count = int(os.getenv("COUNT", 5))
        tg_tokens = Tensor([[1,2]], dtype="int32", device="cpu").realize() # 2 input tokens enables masking.
        tg_model = TinyTransformer(**MODEL_PARAMS["1B"]["args"], disable_kv_cache=False, max_context=5, jit=True)

        # Warmup tinygrad; temperature=math.nan bypasses sampling, returns logits.
        for _ in range(2): tg_model.forward_jit(tg_tokens,2,temperature=math.nan,top_k=0,top_p=0,alpha_f=0,alpha_p=0).realize()
        tg_bench = benchmark(lambda:tg_model.forward_jit(tg_tokens,2,temperature=math.nan,top_k=0,top_p=0,alpha_f=0,alpha_p=0).realize(), count)

        tr_tokens = torch.tensor([[1,2]], device='cpu', dtype=torch.int32)
        tr_model = TorchTransformer(**MODEL_PARAMS["1B"]["args"], max_context=5).eval().to('cpu')
        for _ in range(2): tr_model(tr_tokens, 2) # Warmup Torch.
        tr_bench = benchmark(lambda: tr_model(tr_tokens, 2), count)

        seq_len = 2 # 2 input tokens.
        print(f"tinygrad count: {count}, avg: {seq_len/(tg_bench*1e-9):.3f} tok/s (total: {tg_bench*count} ns)")
        print(f"torch count: {count}, avg: {seq_len/(tr_bench*1e-9):.3f} tok/s (total: {tr_bench*count} ns)")

        assert tg_bench < tr_bench

if __name__ == "__main__":
    unittest.main()
