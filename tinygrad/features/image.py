from typing import Tuple
from tinygrad.helpers import prod, IMAGE, getenv, DEBUG
from tinygrad.dtype import dtypes

# *** image Tensor function replacements ***

def image_dot(self, w, acc_dtype=None):
  # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
  n1, n2 = len(self.shape), len(w.shape)
  assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
  assert self.shape[-1] == w.shape[-min(n2, 2)], f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"  # noqa: E501
  bs, groups, cin, cout = prod(self.shape[0:-2]), prod(w.shape[0:-2]), w.shape[-2], w.shape[-1]
  out_shape_t = self.shape[0:-2] + (cout,-1) if len(self.shape) > 1 else (cout, )

  # NOTE: with NHWC we can remove the transposes
  # bs x groups*cin x H x W
  cx = self.transpose(self.ndim-1, self.ndim-2).reshape((bs//groups, groups*cin, -1, 1))
  # groups*cout x cin x H, W
  cw = w.transpose(w.ndim-1, w.ndim-2).reshape((groups*cout, cin, 1, 1))
  return image_conv2d(cx, cw, groups=groups, acc_dtype=acc_dtype).reshape(out_shape_t).transpose(self.ndim-1, self.ndim-2)

def image_conv2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, acc_dtype=None):
  base_image_type = dtypes.imageh if getenv("FLOAT16", 0) else dtypes.imagef

  (bs,_,iy,ix), (cout,cin,H,W) = self.shape, weight.shape
  x, w = self, weight.reshape(groups, (rcout := cout//groups), cin, H, W)

  # hack for non multiples of 4 on cin
  if cin % 4 != 0 and not (cin == 1 and groups%4 == 0):
    x = x.reshape(bs, groups, cin, iy, ix)   # do this always?
    added_input_channels = 4 - (cin % 4)
    w = w.pad(tuple((0, added_input_channels) if i == 2 else None for i in range(w.ndim)))
    x = x.pad(tuple((0, added_input_channels) if i == 2 else None for i in range(x.ndim)))
    cin = cin + added_input_channels
    x = x.reshape(bs, groups*cin, iy, ix)

  # hack for non multiples of 4 on rcout
  added_output_channels = 0
  if rcout % 4 != 0 and not (rcout == 1 and groups%4 == 0):
    added_output_channels = 4 - (rcout % 4)
    rcout += added_output_channels
    cout = groups * rcout
    w = w.pad(tuple((0, added_output_channels) if i == 1 else None for i in range(w.ndim)))

  # packed (note: flipping bs and iy would make the auto-padding work)
  x = x.permute(0,2,3,1)
  cin_last = iy == 1 and ix == 1
  if cin == 1: w = w.reshape(cout//4,4,H,W).permute(0,2,3,1)
  elif cin_last: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,1,3)
  else: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,3,1)

  # contiguous creates the image, and early realize static weights (TODO: test for the static weight)
  if IMAGE >= 2: x,w = x.cast(base_image_type((bs*iy, ix*groups*cin//4, 4))), w.cast(base_image_type((cout//4, H*W*cin, 4)))
  x, w = x.contiguous(), w.contiguous()

  # expand out
  rcin_hi, rcin_lo = cin//4 if cin >= 4 else 1, 4 if cin >= 4 else 1
  cout_expand = [groups//4 if cin == 1 else groups, 4 if cin == 1 else 1, rcout//4 if rcout >= 4 else 1, 4 if rcout >= 4 else 1]
  x = x.reshape(bs, iy, ix, groups, rcin_hi, rcin_lo)
  if cin_last: w = w.reshape(cout//4, H, rcin_hi, W, 4, rcin_lo)
  else: w = w.reshape(cout//4, H, rcin_hi, W, rcin_lo, 4).permute(0,1,2,3,5,4)

  # padding
  padding_ = [padding]*4 if isinstance(padding, int) else (padding if len(padding) == 4 else [padding[1], padding[1], padding[0], padding[0]])
  x = x.slice((None, (-padding_[2], x.shape[1]+padding_[3]), (-padding_[0], x.shape[2]+padding_[1]), None, None, None))

  # prepare input
  x = x.permute(0,3,4,5,1,2)._pool((H, W), stride, dilation) # -> (bs, groups, rcin_hi, rcin_lo, oy, ox, H, W)
  x = x.permute(0,4,5,1,2,3,6,7).reshape(bs, (oy := x.shape[4]), (ox := x.shape[5]), *cout_expand[0:2], 1, 1, rcin_hi, rcin_lo, H, W)

  # prepare weights
  w = w.permute(0,4,2,5,1,3).reshape((1, 1, 1, *cout_expand, rcin_hi, rcin_lo, H, W))

  # the conv!
  ret = (x*w).cast(base_image_type((bs*oy, ox*cout//4, 4)) if IMAGE >= 2 else dtypes.float32).sum((-4, -3, -2, -1), acc_dtype=acc_dtype)

  # undo hack for non multiples of 4 on C.rcout
  if added_output_channels != 0:
    ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
    cout = groups * (rcout - added_output_channels)

  # NCHW output
  ret = ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)
  return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))

# *** images have weird indexing requirements ***

from tinygrad.shape.symbolic import Node
def to_image_idx(base_shape:Tuple[int, ...], idxy:Node, valid:Node) -> Tuple[Tuple[Node, Node], Node]:
  idx, idy = (idxy // 4) % base_shape[1], (idxy // (4 * base_shape[1]))
  # TODO: bring back the valid removal logic (correct!)
  if DEBUG>=5: print("to_image_idx", base_shape, idx.min, idx.max, idy.min, idy.max, idx, idy, valid)
  return (idx, idy), valid
