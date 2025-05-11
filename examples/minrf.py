# much taken from https://github.com/cloneofsimo/minRF
import math
from tinygrad import Tensor, nn, GlobalCounters, TinyJit
from tinygrad.helpers import getenv, trange
from extra.models.llama import Attention, FeedForward, precompute_freqs_cis

DUMB = bool(getenv("DUMB", 0))

def modulate(x:Tensor, shift:Tensor, scale:Tensor) -> Tensor: return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def timestep_embedding(t:Tensor, dim:int, max_period=10000):
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half)/half).exp()
  args = t[:, None] * freqs[None]
  return Tensor.cat(args.cos(), args.sin(), dim=-1)

class TimestepEmbedder:
  def __init__(self, hidden_size, frequency_embedding_size=256):
    self.mlp = [nn.Linear(frequency_embedding_size, hidden_size), Tensor.silu, nn.Linear(hidden_size, hidden_size)]
    self.frequency_embedding_size = frequency_embedding_size
  def __call__(self, t:Tensor): return timestep_embedding(t, self.frequency_embedding_size).sequential(self.mlp)

class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps=1e-5):
    self.attention = Attention(dim, n_heads)
    self.feed_forward = FeedForward(dim, 4*dim)
    self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
    self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)
    self.adaLN_modulation = nn.Linear(dim, 6 * dim, bias=True)

  def __call__(self, x:Tensor, freqs_cis:Tensor, adaln_input:Tensor):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input.silu()).chunk(6, dim=1)
    x = x + gate_msa.unsqueeze(1) * self.attention(modulate(self.attention_norm(x), shift_msa, scale_msa), 0, freqs_cis)
    x = x + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(x), shift_mlp, scale_mlp))
    return x

class FinalLayer:
  def __init__(self, dim, patch_size, out_channels):
    self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(dim, patch_size*patch_size*out_channels, bias=True)
    self.adaLN_modulation = nn.Linear(dim, 2 * dim, bias=True)

    # TODO: how does real model do this?
    #self.linear.weight.replace(self.linear.weight.zeros_like())
    #self.linear.bias.replace(self.linear.bias.zeros_like())

  def __call__(self, x:Tensor, c:Tensor):
    shift, scale = self.adaLN_modulation(c.silu()).chunk(2, dim=1)
    x = modulate(self.norm_final(x), shift, scale)
    return self.linear(x)

# channels=1, input_size=32, dim=64, n_layers=6, n_heads=4, num_classes=10
class DiT_Llama:
  def __init__(self, in_channels=1, dim=64, n_layers=6, n_heads=4, num_classes=10, patch_size=2):
    self.patch_size = patch_size
    self.out_channels = in_channels

    self.init_conv_seq = [
      nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
      Tensor.silu,
      nn.GroupNorm(32, dim//2),
      nn.Conv2d(dim //2, dim // 2, kernel_size=5, padding=2, stride=1),
      Tensor.silu,
      nn.GroupNorm(32, dim//2),
    ]

    if not DUMB:
      self.freqs_cis = precompute_freqs_cis(dim // n_heads, 4096)
      self.x_embedder = nn.Linear(self.patch_size * self.patch_size * dim // 2, dim, bias=True)
      self.t_embedder = TimestepEmbedder(dim)
      self.y_embedder = nn.Embedding(num_classes+1, dim)
      self.layers = [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
      self.final_layer = FinalLayer(dim, self.patch_size, self.out_channels)
    else:
      N = self.patch_size * self.patch_size * dim // 2
      self.dumb_model = [
        nn.Conv2d(N+11+1, N, kernel_size=3, padding='same'), Tensor.silu,
        nn.Conv2d(N, N, kernel_size=3, padding='same'), Tensor.silu,
        nn.Conv2d(N, N, kernel_size=3, padding='same'), Tensor.silu,
        nn.Conv2d(N, N, kernel_size=3, padding='same'), Tensor.silu,
        nn.Conv2d(N, self.patch_size*self.patch_size, kernel_size=3, padding='same'),
      ]

  def unpatchify(self, x:Tensor):
    c, p = self.out_channels, self.patch_size
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = x.rearrange("n h w p q c -> n c h p w q")
    return x.reshape(shape=(x.shape[0], c, h * p, h * p))

  def patchify(self, x:Tensor):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
    return x  # B <H*W ish> <C*patch_size*patch_size>

  def __call__(self, x:Tensor, t:Tensor, y:Tensor, dropout_prob=None) -> Tensor:
    if dropout_prob is not None: y = (Tensor.rand(y.shape[0]) < dropout_prob).where(y.full_like(-1), y)
    x = x.sequential(self.init_conv_seq)
    x = self.patchify(x)
    if not DUMB:
      x = self.x_embedder(x)
      adaln_input = self.t_embedder(t) + self.y_embedder(y)
      for layer in self.layers: x = layer(x, self.freqs_cis[:, :x.size(1)], adaln_input=adaln_input)
      x = self.final_layer(x, adaln_input)
    else:
      b = x.size(0)
      d = x.size(1)
      dumb = Tensor.cat(x, t.reshape(b,1,1).expand(b,d,1), y.one_hot(11).reshape(b,1,11).expand(b,d,11), dim=2)
      dumb = dumb.permute(0,2,1).reshape(b,-1,32//self.patch_size,32//self.patch_size)
      x = dumb.sequential(self.dumb_model)
      x = x.reshape(b,-1,(32//self.patch_size)*(32//self.patch_size)).permute(0,2,1)
    return self.unpatchify(x)

  def rf(self, x:Tensor, cond:Tensor):
    b = x.shape[0]
    # self.ln is True
    t = Tensor.randn((b,)).sigmoid()
    texp = t.view([b, *([1] * len(x.shape[1:]))])

    # TODO: Tensor.randn_like
    # this is rectified flow
    z1 = Tensor.randn(x.shape)
    zt = (1 - texp) * x + texp * z1
    vtheta = self(zt, t, cond, dropout_prob=0.1)

    # MSE loss
    return ((z1 - x) - vtheta).square().mean()

  def sample(self, z, cond, null_cond, sample_steps=50, cfg=2.0):
    b = z.size(0)
    dt = Tensor.full((b,)+(1,)*len(z.shape[1:]), fill_value=1.0/sample_steps)
    images = [z]
    for i in range(sample_steps, 0, -1):
      t = Tensor.full((b,), fill_value=i/sample_steps).realize()
      vc = self(z, t, cond)
      vu = self(z, t, null_cond)
      vc = vu + cfg * (vc - vu)
      z = z - dt * vc
      images.append(z)
    return images

def mviz(t:Tensor):
  ft = t.permute(1,2,0,3).reshape(32, -1)
  assert ft.shape[-1]%32 == 0
  print("")
  for y in ((ft+1)/2).clamp(0,1).tolist():
    ln = [f"\033[38;5;{232+int(x*23)}m██" for x in y]
    print(''.join(ln) + "\033[0m")

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = nn.datasets.mnist()
  X_train = X_train.pad((2,2,2,2))
  X_train = ((X_train.float()/255)-0.5)/0.5
  Y_train = Y_train.int()

  #mviz(X_train[0:3])

  Tensor.training = True

  model = DiT_Llama(patch_size=4)
  for r in nn.state.get_parameters(model): r.realize()
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=5e-4)

  @TinyJit
  def train_step():
    samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
    optimizer.zero_grad()
    X, Y = X_train[samples], Y_train[samples]
    loss = model.rf(X, Y)
    loss.backward()
    optimizer.step()
    return loss

  @TinyJit
  def sample(z:Tensor, cond:Tensor) -> Tensor: return model.sample(z, cond, Tensor.full_like(cond, 10), sample_steps=20)[-1]

  for steps in (t:=trange(getenv("STEPS", 1000))):
    if steps%10 == 0: mviz(sample(Tensor.randn(3, 1, 32, 32), Tensor([5,0,4])))
    GlobalCounters.reset()
    loss = train_step()
    t.set_description(f"loss: {loss.item():9.2f}")
