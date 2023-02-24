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

    # put it back for normal conv
    x = x.reshape(bs, iy, ix, groups*cin).permute(0,3,1,2)
    w = w.reshape(cout//4, H, cin//4 if cin >= 4 else 1, W, 4, 4 if cin >= 4 else 1).permute(0,4,2,5,1,3).reshape(cout, cin, H, W)

    # run normal conv
    ret = normal_conv(x, w, None, groups, stride, dilation, padding)

    # make image sized
    oy, ox = ret.shape[2:]
    ret = ret.permute(0,2,3,1).reshape(bs*oy, ox*cout//4, 4).contiguous()

    # undo hack for non multiples of 4 on C.rcout
    if added_output_channels != 0:
      ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
      rcout -= added_output_channels
      cout = groups * rcout

    # NCHW output
    ret = ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)
    return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))
  return image_conv2d
