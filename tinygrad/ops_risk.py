import numpy as np
from .tensor import Function
from extra.risk import risk_matmul

class Matmul(Function):
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return risk_matmul(input, weight)

  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = risk_matmul(grad_output, weight, transpose_w=True)
    grad_weight = risk_matmul(input, grad_output, transpose_x=True)
    return grad_input, grad_weight


"""
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
    bs,_,oy,ox = grad_output.shape
    tx, tw, x_shape = ctx.saved_tensors
    _,rcout,cin,H,W = tw.shape
    ys,xs = ctx.stride
    OY,OX = x_shape[2:4]

    ggg = grad_output.reshape(bs,ctx.groups,rcout,oy,ox)

    gdw = np.zeros((ctx.groups,rcout,cin,H,W), dtype=tx.dtype)
    for g in range(ctx.groups):
      #'ikYX,ijYXyx -> kjyx'
      gdw[g] += np.tensordot(ggg[:,g], tx[:,g], ((0,2,3),(0,2,3)))

    # needs to be optimized
    gdx = np.zeros((bs,ctx.groups,cin,OY,OX), dtype=tx.dtype)
    for k in range(oy*ox):
      Y, X = k//ox, k%ox
      iY,iX = Y*ys, X*xs
      #gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
      for g in range(ctx.groups):
        tg = np.dot(ggg[:,g,:,Y,X].reshape(bs, -1), tw[g].reshape(rcout, -1))
        gdx[:, g, :, iY:iY+H, iX:iX+W] += tg.reshape((bs, cin, H, W))

    return gdx.reshape((bs, ctx.groups*cin, OY, OX)), gdw.reshape((ctx.groups*rcout, cin, H, W))
"""

