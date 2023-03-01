from tinygrad.tensor import HLOP
from tinygrad.helpers import getenv

IMAGE = getenv("IMAGE", 0)

def image_conv2d_decorator(normal_conv):
  if not HLOP or IMAGE == 0: return normal_conv

  def image_conv2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0):
    (bs,_,iy,ix), (cout,cin,H,W) = self.shape, weight.shape
    rcout = cout//groups
    x, w = self, weight.reshape(groups, rcout, cin, H, W)

    # hack for non multiples of 4 on cin
    if cin % 4 != 0 and not (cin == 1 and groups%4 == 0):
      x = x.reshape(bs, groups, cin, iy, ix)   # do this always?
      added_input_channels = 4 - (cin % 4)
      cin = cin + added_input_channels
      w = w.slice(tuple((0, cin) if i == 2 else (0, w.shape[i]) for i in range(len(w.shape))))
      x = x.slice(tuple((0, cin) if i == 2 else (0, x.shape[i]) for i in range(len(x.shape))))
      x = x.reshape(bs, groups*cin, iy, ix)

    # hack for non multiples of 4 on rcout
    added_output_channels = 0
    if rcout % 4 != 0 and not (rcout == 1 and groups%4 == 0):
      added_output_channels = 4 - (rcout % 4)
      rcout += added_output_channels
      cout = groups * rcout
      w = w.slice(tuple((0, rcout) if i == 1 else (0, w.shape[i]) for i in range(len(w.shape))))

    # packed
    x = x.permute(0,2,3,1).reshape(bs * iy, ix * groups * cin//4, 4)
    if cin == 1: w = w.reshape(cout//4,4,H*W).permute(0,2,1)
    else: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,1,3).reshape(cout//4, H*cin*W, 4)

    # contiguous creates the image, and early realize static weights (TODO: don't always realize)
    x, w = x.contiguous(), w.contiguous().realize()

    # expand out
    rcin_hi, rcin_lo = cin//4 if cin >= 4 else 1, 4 if cin >= 4 else 1
    cout_expand = [groups//4 if cin == 1 else groups, 4 if cin == 1 else 1, rcout//4 if rcout >= 4 else 1, 4 if rcout >= 4 else 1]
    x = x.reshape(bs, iy, ix, groups, rcin_hi, rcin_lo)
    w = w.reshape(cout//4, H, rcin_hi, W, 4, rcin_lo)

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
    w = w.reshape((1, 1, 1, *cout_expand, rcin_hi, rcin_lo, H, W))

    # the conv!
    ret = (x*w).sum((-4, -3, -2, -1)).reshape(bs*oy, ox*cout//4, 4)
    if IMAGE >= 3: ret = ret.contiguous()

    # undo hack for non multiples of 4 on C.rcout
    if added_output_channels != 0:
      ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
      rcout -= added_output_channels
      cout = groups * rcout

    # NCHW output
    ret = ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)
    return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))
  return image_conv2d
