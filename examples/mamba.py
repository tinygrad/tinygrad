import os, sys, math, argparse
sys.path.append(os.getcwd())
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import fetch
from extra.models.llama import RMSNorm
from tinygrad.nn.state import load_state_dict, torch_load

MODELS = {
    "130m": {
        "dim": 768,
        "n_layers": 24,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
    "370m": {
        "dim": 1024,
        "n_layers": 48,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
    "790m": {
        "dim": 1536,
        "n_layers": 48,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
    "1.4b": {
        "dim": 2048,
        "n_layer": 48,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
    "2.8b": {
        "dim": 2560,
        "n_layer": 64,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
}

def fetch_weights(model_name: str):
    if model_name not in MODELS.keys(): raise Exception(f"Requested unknown mamba model: {model_name}")
    downloaded = fetch(f"https://huggingface.co/state-spaces/mamba-{model_name}/resolve/main/pytorch_model.bin?download=true")
    weights = torch_load(downloaded)
    return weights


class MambaMixer:
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
    ):
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.dim)
        self.dt_rank = math.ceil(self.dim / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.dim, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            self.dt_proj.weight = Tensor.full(*self.dt_proj.weight.shape, dt_init_std)
        elif dt_init == "random":
            self.dt_proj.weight = Tensor.uniform(*self.dt_proj.weight.shape, low=-dt_init_std, high=dt_init_std)
        else:
            raise NotImplementedError

        dt = (
            Tensor.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).exp().maximum(dt_init_floor)
        inv_dt = dt + (-((-dt).exp() - Tensor.ones(*dt.shape))).log() # TODO: implement torch.expm1?
       
        self.dt_proj.bias.assign(inv_dt)

        # S4D real initialization
        self.A_log = Tensor.arange(1, self.d_state + 1, dtype=dtypes.float32).repeat([self.d_inner, 1]).contiguous().log()

        # D "skip" parameter
        self.D = Tensor.ones(self.d_inner)  # Keep in fp32

        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=bias)

class MambaBlock:
    def __init__(self, dim: int, norm_eps: float = 1e-5, rms_norm: bool = True, layer_idx: int = None):
        self.mixer = MambaMixer(dim, layer_idx=layer_idx)
        if rms_norm: self.norm = RMSNorm(dim, norm_eps)
        else: raise NotImplementedError

class MambaBackbone:
    def __init__(self, dim: int, n_layers: int, vocab_size: int):
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = [MambaBlock(dim, layer_idx=i) for i in range(n_layers)]

class Mamba:
    def __init__(self, dim: int, n_layers: int, vocab_size: int, pad_vocab_size_multiple: int = 1):
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.backbone = MambaBackbone(dim, n_layers, vocab_size)

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mamba in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for LLM completion")
    parser.add_argument("--size", type=str, default="130m", help=f"Size of model to use [{', '.join([k for k in MODELS.keys()])}]")
    args = parser.parse_args()
    weights = fetch_weights(args.size)
    model = Mamba(**MODELS[args.size])
    load_state_dict(model, weights)
    