import math
from typing import Union

from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import prod, argfix



# rejection sampling truncated randn
def rand_truncn(*shape, dtype=None, truncstds=2, **kwargs) -> Tensor:
  CNT=8
  x = Tensor.randn(*(*shape, CNT), dtype=dtype, **kwargs)
  ctr = Tensor.arange(CNT).reshape((1,) * len(x.shape[:-1]) + (CNT,)).expand(x.shape)
  take = (x.abs() <= truncstds).where(ctr, CNT).min(axis=-1, keepdim=True)  # set to 0 if no good samples
  return (ctr == take).where(x, 0).sum(axis=-1)

# https://github.com/keras-team/keras/blob/v2.15.0/keras/initializers/initializers.py#L1026-L1065
def he_normal(*shape, a: float = 0.00, **kwargs) -> Tensor:
  std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:])) / 0.87962566103423978
  return std * rand_truncn(*shape, **kwargs)

class Conv2dHeNormal(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.in_channels, self.out_channels = in_channels, out_channels  # for testing
    self.weight = he_normal(out_channels, in_channels//groups, *self.kernel_size, a=0.0, dtype=dtypes.float32)
    if bias: self.bias = self.bias.cast(dtypes.float32)
  def __call__(self, x: Tensor):
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

class Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias=bias)
    self.weight = Tensor.normal((out_features, in_features), mean=0.0, std=0.01, dtype=dtypes.float32)
    if bias: self.bias = Tensor.zeros(out_features, dtype=dtypes.float32)
  def __call__(self, x:Tensor):
    return x.linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)

class LinearBert(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, std=0.02):
    self.weight = std * rand_truncn(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.zeros(out_features, dtype=dtypes.float32) if bias else None
  
  def __call__(self, x:Tensor):
    return x.cast(dtypes.default_float).linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)

class LinearBitNet(nn.Linear):
    """
    BitNet Linear layer with 2-bit packed weights and int8 activations.
    Matches HuggingFace implementation but uses tinygrad operations
    https://github.com/huggingface/transformers/blob/096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4/src/transformers/integrations/bitnet.py#L126
    """

    VALUES_PER_ITEM = 4  # 4 ternary values packed per uint8 byte
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        
        # Packed weight tensor: (out_features//4, in_features)
        self.weight = Tensor.zeros(
            (out_features // VALUES_PER_ITEM, in_features),
            dtype=dtypes.uint8,
        )
        
        # Weight scale (single scalar)
        self.weight_scale = Tensor.ones(
            (1,),
            dtype=dtypes.float32,
        )
        
        # Optional bias
        if bias:
            self.bias = Tensor.zeros(
                (out_features,),
                dtype=dtypes.float32,
            )
        else:
            self.bias = None
        
        # Cache for unpacked weights
        self._unpacked_weights = None
        self._weight_hash = None


  def unpack_weights(packed: Tensor, dtype) -> Tensor:
      """
      Unpacks a tensor of quantized weights that were stored in a packed format using 2 bits per value.

      Parameters:
      -----------
      packed : Tensor
          A tensor containing packed weights where each element represents 4 quantized values (using 2 bits per value).
      dtype : dtypes
          The dtype of the returned Tensor
      Returns:
      --------
      Tensor
          A tensor of unpacked weights, where each value is converted from its packed 2-bit representation.

      Example:
      --------
      packed = Tensor([[0b10100001, 0b00011000],
                      [0b10010000, 0b00001010]], dtype=dtypes.uint8)

      # Unpack the values
      unpacked = unpack_weights(packed, dtypes.float32)

      # Resulting unpacked tensor
      print(unpacked)
      # Output: tensor([[ 0, -1],
                        [-1,  1],
                        [-1,  1],
                        [-1,  1],
                        [ 1,  0],
                        [ 0, -1],
                        [ 1, -1],
                        [ 1, -1]])

      Explanation of the example:
      ---------------------------
      Let's take the first value for example 0b10100001, we we will only focus on the first column,
      because every element is unpacked across the first dimension
      - First 2 bits: `01` → 0 at [0][0]
      - Second 2 bits: `00` → -1 at [0][2]
      - Third 2 bits: `10` → 1 at [0][4]
      - Fourth 2 bits: `10` → 1 at [0][6]
      the second value of the same row (0b10010000) will give the values for [0][1], [0][3], [0][5], [0][7]

      We subtract 1 because during the packing process, it's easier to work with values like 0, 1, and 2. To make this possible,
      we add 1 to the original ternary weights (which are typically -1, 0, and 1) when packing them. When unpacking, we reverse
      this by subtracting 1 to restore the original ternary values.
      """
      packed_shape = packed.shape

      if len(packed_shape) == 1:
          original_row_dim = packed_shape[0] * VALUES_PER_ITEM
          unpacked_shape = (original_row_dim,)
      else:
          original_row_dim = packed_shape[0] * VALUES_PER_ITEM
          unpacked_shape = (original_row_dim, *packed_shape[1:])

      unpacked = Tensor.zeros(unpacked_shape, device=packed.device, dtype=dtypes.uint8)

      for i in range(VALUES_PER_ITEM):
          start = i * packed_shape[0]
          end = start + packed_shape[0]
          mask = 3 << (2 * i)
          unpacked[start:end] = (packed & mask) >> (2 * i)

      return unpacked.cast(dtype) - 1

    
    def _get_unpacked_weights(self) -> Tensor:
        """Get unpacked weights with caching."""
        # Simple hash for cache invalidation
        current_hash = hash(str(self.weight.lazydata))
        
        if self._unpacked_weights is None or self._weight_hash != current_hash:
            self._unpacked_weights = unpack_weights(self.weight, dtypes.float32)
            self._weight_hash = current_hash
        
        return self._unpacked_weights
    
    def activation_quant(self, input: Tensor, num_bits: int = 8) -> Tuple[Tensor, Tensor]:
        """
        Activation function : Performs symmetric, per-token quantization on the input activations.
        Parameters:
        -----------
        input : Tensor
            Input activations to be quantized.
        num_bits : int, optional (default=8)
            Number of bits to use for quantization, determining the quantization range.

        Returns:
        --------
        result : Tensor
            Quantized activation tensor, with values mapped to an `int8` range.
        scale : Tensor
            The per-channel scaling factors used to quantize the tensor.
        """
        Qn = -(2 ** (num_bits - 1))  # -128
        Qp = 2 ** (num_bits - 1) - 1  # 127
        
        # Per-token scaling
        scale = Qp / input.abs().max(axis=-1, keepdim=True).clamp(min_=1e-5)
        
        # Quantize and clamp
        result = (input * scale).round().clip(Qn, Qp)
        
        return result.cast(dtypes.int8), scale
    
    def post_quant_process(self, input: Tensor, input_scale: Tensor, weight_scale: Tensor) -> Tensor:
        """Apply post-quantization scaling."""
        out = input / (input_scale * weight_scale)
        return out

    def __call__(self, input: Tensor) -> Tensor:
        # Get unpacked ternary weights
        w_quant = self._get_unpacked_weights()
        
        # Apply activation quantization
        input_quant, input_scale = self.activation_quant(input)
        
        # Cast to computation dtype for matrix multiplication
        input_float = input_quant.cast(dtypes.float32)
        
        # Matrix multiplication: input_float @ w_quant.T
        if len(w_quant.shape) == 1:
            y = input_float * w_quant
        else:
            y = input_float @ w_quant.T
        
        # Apply post-quantization processing
        y = self.post_quant_process(y, input_scale, self.weight_scale)
        
        # Add bias if present
        if self.bias is not None:
            y = y + self.bias.reshape(1, -1).expand(*y.shape)
        
        return y

class EmbeddingBert(nn.Embedding):
  def __init__(self, vocab_size:int, embed_size:int, std=0.02):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight = std * rand_truncn(vocab_size, embed_size, dtype=dtypes.float32)

  def __call__(self, idx:Tensor) -> Tensor:
    if idx.numel() == 0: return Tensor.empty(idx.shape+(self.embed_sz,), dtype=self.weight.dtype, device=self.weight.device)
    arange_shp, weight_shp, big_shp = (1, 1, self.vocab_sz, 1), (1, 1, self.vocab_sz, self.embed_sz), idx.shape+(self.vocab_sz, self.embed_sz,)
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).reshape(arange_shp)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1,)).expand(big_shp), self.weight.cast(dtypes.default_float).reshape(weight_shp).expand(big_shp)
    # TODO: contiguous() here because the embedding dropout creates different asts on each device, and search becomes very slow.
    # Should fix with fixing random ast on multi device, and fuse arange to make embedding fast.
    return (arange == idx).mul(vals).sum(2, dtype=vals.dtype).contiguous()

class LayerNormBert:
  def __init__(self, normalized_shape:Union[int, tuple[int, ...]], eps:float=1e-12, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight, self.bias = (Tensor.ones(*self.normalized_shape, dtype=dtypes.float32), Tensor.zeros(*self.normalized_shape, dtype=dtypes.float32)) if elementwise_affine else (None, None)

  def __call__(self, x:Tensor):
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    xn = x.cast(dtypes.float32).layernorm(eps=self.eps, axis=self.axis).cast(x.dtype)
    if not self.elementwise_affine: return xn
    return (xn * self.weight.cast(dtypes.default_float) + self.bias.cast(dtypes.default_float))

class FrozenBatchNorm2dRetinaNet(nn.BatchNorm2d):
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight = Tensor.ones(sz, dtype=dtypes.float32, requires_grad=False) if affine else None
    self.bias = Tensor.zeros(sz, dtype=dtypes.float32, requires_grad=False) if affine else None

    if track_running_stats: self.running_mean, self.running_var = Tensor.zeros(sz, dtype=dtypes.float32, requires_grad=False), Tensor.ones(sz, dtype=dtypes.float32, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, dtype=dtypes.long, requires_grad=False)

  def __call__(self, x:Tensor) -> Tensor:
    batch_mean, batch_var = super().calc_stats(x.cast(dtypes.float32))
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach().cast(self.running_mean.dtype))
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * x.numel()/(x.numel()-x.shape[1]) * batch_var.detach().cast(self.running_var.dtype))
      self.num_batches_tracked += 1
    return x.cast(dtypes.float32).batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt()).cast(x.dtype)

class Conv2dNormalRetinaNet(nn.Conv2d):
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...],
               stride:int=1, padding:int|tuple[int, ...]|str=0, dilation:int=1, groups:int=1,
               bias:bool=True, prior_prob:float|None=None):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.weight = Tensor.normal(*self.weight.shape, std=0.01, dtype=dtypes.float32)
    if bias:
      if prior_prob:
        prior_prob = Tensor(prior_prob, device=self.bias.device, dtype=dtypes.float32).expand(*self.bias.shape)
        self.bias = -(((1 - prior_prob) / prior_prob).log())
      else: self.bias = Tensor.zeros_like(self.bias, dtype=dtypes.float32)

  def __call__(self, x:Tensor) -> Tensor:
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    groups=self.groups, stride=self.stride, padding=self.padding)

class Conv2dKaimingUniformRetinaNet(nn.Conv2d):
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...],
               stride:int=1, padding:int|tuple[int, ...]|str=0, dilation:int=1, groups:int=1,
               bias:bool=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.weight = Tensor.kaiming_uniform(*self.weight.shape, a=1, dtype=dtypes.float32)
    if bias: self.bias = Tensor.zeros_like(self.bias, dtype=dtypes.float32)

  def __call__(self, x:Tensor) -> Tensor:
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    groups=self.groups, stride=self.stride, padding=self.padding)

class Conv2dRetinaNet(nn.Conv2d):
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...],
               stride:int=1, padding:int|tuple[int, ...]|str=0, dilation:int=1, groups:int=1,
               bias:bool=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale, dtype=dtypes.float32)
    self.bias: Tensor|None = Tensor.uniform(out_channels, low=-scale, high=scale, dtype=dtypes.float32) if bias else None

  def __call__(self, x:Tensor) -> Tensor:
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    groups=self.groups, stride=self.stride, dilation=self.dilation, padding=self.padding)
