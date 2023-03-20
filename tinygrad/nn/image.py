import numpy as np
from tinygrad.helpers import prod, IMAGE, ImageDType, getenv, dtypes
from tinygrad.lazy import get_single_root

FLOAT16 = getenv("FLOAT16", 0)
base_image_type = (100, 2, "imageh", np.float16) if FLOAT16 else (100, 4, "imagef", np.float32)

def image_dot(self, w):
  # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
  bs, groups = prod(self.shape[0:-2]), prod(w.shape[0:-2])
  cin, cout = w.shape[-2], w.shape[-1]
  out_shape_t = self.shape[0:-2] + (cout,-1)
  if len(self.shape) > 1:
    order = tuple(range(len(self.shape)-2)) + (len(self.shape)-1, len(self.shape)-2)
  else:
    order, out_shape_t = (0,), (cout, )
  worder = tuple(range(len(w.shape)-2)) + (len(w.shape)-1, len(w.shape)-2)

  # NOTE: with NHWC we can remove the transposes
  # bs x groups*cin x H x W
  cx = self.permute(order=order).reshape(shape=(bs//groups, groups*cin, -1, 1))
  # groups*cout x cin x H, W
  cw = w.permute(order=worder).reshape(shape=(groups*cout, cin, 1, 1))
  return cx.conv2d(cw, groups=groups).reshape(shape=out_shape_t).permute(order=order)

def image_conv2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0):
  (bs,_,iy,ix), (cout,cin,H,W) = self.shape, weight.shape
  rcout = cout//groups
  x, w = self, weight.reshape(groups, rcout, cin, H, W)

  # hack for non multiples of 4 on cin
  if cin % 4 != 0 and not (cin == 1 and groups%4 == 0):
    x = x.reshape(bs, groups, cin, iy, ix)   # do this always?
    added_input_channels = 4 - (cin % 4)
    w = w.pad(tuple((0, added_input_channels) if i == 2 else (0, 0) for i in range(len(w.shape))))
    x = x.pad(tuple((0, added_input_channels) if i == 2 else (0, 0) for i in range(len(x.shape))))
    cin = cin + added_input_channels
    x = x.reshape(bs, groups*cin, iy, ix)

  # hack for non multiples of 4 on rcout
  added_output_channels = 0
  if rcout % 4 != 0 and not (rcout == 1 and groups%4 == 0):
    added_output_channels = 4 - (rcout % 4)
    rcout += added_output_channels
    cout = groups * rcout
    w = w.slice(tuple((0, rcout) if i == 1 else (0, w.shape[i]) for i in range(len(w.shape))))

  # packed (note: flipping bs and iy would make the auto-padding work)
  x = x.permute(0,2,3,1).reshape(bs * iy, ix * groups * cin//4, 4)
  cin_last = iy == 1 and ix == 1
  if cin == 1: w = w.reshape(cout//4,4,H*W).permute(0,2,1)
  elif cin_last: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,1,3).reshape(cout//4, H*cin//4*W*4, 4)
  else: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,3,1).reshape(cout//4, H*cin//4*W*4, 4)

  # contiguous creates the image, and early realize static weights (TODO: test for the static weight)
  if IMAGE >= 2: x,w = x.cast(ImageDType(*base_image_type, shape=x.shape)), w.cast(ImageDType(*base_image_type, shape=w.shape))
  x, w = x.contiguous(), w.contiguous()
  if get_single_root(w.lazydata).realized: w.realize()

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
  oy, ox = x.shape[4:6]
  x = x.permute(0,4,5,1,2,3,6,7).reshape(bs, oy, ox, *cout_expand[0:2], 1, 1, rcin_hi, rcin_lo, H, W)
  x = x.expand(bs, oy, ox, *cout_expand, rcin_hi, rcin_lo, H, W)

  # prepare weights
  w = w.permute(0,4,2,5,1,3)
  w = w.reshape((1, 1, 1, *cout_expand, rcin_hi, rcin_lo, H, W)).expand(x.shape)

  # the conv! (+ the bias)
  ret = (x*w).cast(dtypes.float32).sum((-4, -3, -2, -1))

  # reshape to image and cast back to image
  ret = ret.reshape(bs*oy, ox*cout//4, 4)
  if IMAGE >= 2: ret = ret.cast(ImageDType(*base_image_type, shape=ret.shape))
  if IMAGE >= 3: ret = ret.contiguous()

  # undo hack for non multiples of 4 on C.rcout
  if added_output_channels != 0:
    ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
    rcout -= added_output_channels
    cout = groups * rcout

  # NCHW output
  ret = ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)
  return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))
