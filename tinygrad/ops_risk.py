import numpy as np
from .tensor import Function

class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    if type(ctx.stride) == int:
      ctx.stride = (ctx.stride, ctx.stride)
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_ = x.shape[0], x.shape[1]
    oy,ox = (x.shape[2]-(H-ys))//ys, (x.shape[3]-(W-xs))//xs
    assert cin*ctx.groups == cin_
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups
    
    if H == 1 and W == 1 and ctx.groups == 1 and ctx.stride == (1,1):
      d1 = x.shape[0] * x.shape[2] * x.shape[3]
      assert x.shape[1] == w.shape[1]
      d2 = x.shape[1]
      d3 = w.shape[0]
      print("1x1 CONV FAST == %d x %d x %d matmul" % (d1, d2, d3))

      x11 = x.reshape(x.shape[1], x.shape[2]*x.shape[3]).T
      w11 = w.reshape(w.shape[0], w.shape[1]).T
      print(x11.shape, w11.shape)

      ret = x11 @ w11

      return ret.T.reshape(1, w.shape[0], x.shape[2], x.shape[3])

    gx = x.reshape(bs,ctx.groups,cin,x.shape[2],x.shape[3])
    tx = np.lib.stride_tricks.as_strided(gx,
      shape=(bs, ctx.groups, cin, oy, ox, H, W),
      strides=(*gx.strides[0:3], gx.strides[3]*ys, gx.strides[4]*xs, *gx.strides[3:5]),
      writeable=False,
    )
    tw = w.reshape(ctx.groups, rcout, cin, H, W)
    ctx.save_for_backward(tx, tw, x.shape)

    ret = np.zeros((bs,ctx.groups,oy,ox,rcout),dtype=x.dtype)
    for g in range(ctx.groups):
      #ijYXyx,kjyx -> iYXk ->ikYX
      ret[:,g] += np.tensordot(tx[:,g], tw[g], ((1,4,5),(1,2,3)))
    ret = np.moveaxis(ret,4,2).reshape(bs, cout, oy, ox)

    print(x.shape, w.shape, "->", ret.shape)
    return ret


  def backward(ctx, grad_output):
    pass

