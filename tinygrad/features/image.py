from typing import Tuple, Dict, Any
from tinygrad.helpers import prod, IMAGE, getenv, dtypes, DEBUG

# *** image Tensor function replacements ***

def image_dot(self, w):
  # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
  n1, n2 = len(self.shape), len(w.shape)
  assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
  assert self.shape[-1] == w.shape[-min(n2, 2)], f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"
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
  return image_conv2d(cx, cw, groups=groups).reshape(shape=out_shape_t).permute(order=order)

def image_conv2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0):
  base_image_type = dtypes.imageh if getenv("FLOAT16", 0) else dtypes.imagef

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
    w = w.slice(tuple((0, rcout) if i == 1 else (0, s) for i,s in enumerate(w.shape)))

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
  oy, ox = x.shape[4:6]
  x = x.permute(0,4,5,1,2,3,6,7).reshape(bs, oy, ox, *cout_expand[0:2], 1, 1, rcin_hi, rcin_lo, H, W)
  x = x.expand(bs, oy, ox, *cout_expand, rcin_hi, rcin_lo, H, W)

  # prepare weights
  w = w.permute(0,4,2,5,1,3)
  w = w.reshape((1, 1, 1, *cout_expand, rcin_hi, rcin_lo, H, W)).expand(x.shape)

  # the conv! (+ the bias)
  ret = x*w
  if IMAGE >= 2: ret = ret.cast(base_image_type((bs*oy, ox*cout//4, 4)))
  ret = ret.sum((-4, -3, -2, -1))

  # undo hack for non multiples of 4 on C.rcout
  if added_output_channels != 0:
    ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
    rcout -= added_output_channels
    cout = groups * rcout

  # NCHW output
  ret = ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)
  return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))

# *** images have weird indexing requirements ***

from tinygrad.shape.symbolic import Node, AndNode, Variable, NumNode, SumNode, LtNode

def to_image_idx(base_shape:Tuple[int, ...], idxy:Node, valid:Node) -> Tuple[Tuple[Node, Node], Node]:
  idx = (idxy // 4) % base_shape[1]
  idy = (idxy // (4 * base_shape[1]))

  if valid.min == 0 and isinstance(idxy, SumNode):
    nodes = valid.nodes if isinstance(valid, AndNode) else [valid]
    val_dict: Dict[Node, Any] = {}
    # TODO: is this correct? should it check there's only one variable from each component?
    idxy_flat_var = [(i, list(i.vars())[0]) for i in idxy.flat_components if not isinstance(i, NumNode)]

    for node in nodes:
      assert isinstance(node, LtNode)
      node_flat, node_vars = node.a.flat_components if isinstance(node.a, SumNode) else [node.a], node.vars()
      same_sym = [i for (i, var) in idxy_flat_var if var in node_vars]
      if len(same_sym) == 0: continue
      first, second = sorted(same_sym)[0], sorted(node_flat)[0]
      f_b = 1 if isinstance(first, Variable) else first.b
      s_b = 1 if isinstance(second, Variable) else second.b
      sig = -1 if s_b < 0 else 1
      key_node = sig*node.a
      if key_node not in val_dict: val_dict[key_node] = [key_node.min, key_node.max, abs(f_b//s_b)]
      val_dict[key_node][(sig + 1)//2] = sig*(node.b - 1)

    fakes = {}
    for cnt, (key_node, (mnn, mxn, multip)) in enumerate(val_dict.items()):
      if mnn > mxn: return (idx, idy), valid  # TODO: why is this happening?
      fake_var = Variable("fake_" + str(cnt), mnn, mxn)
      fakes[fake_var] = key_node
      idxy += multip*(fake_var - key_node)

    idx = (idxy // 4) % base_shape[1]
    idy = (idxy // (4 * base_shape[1]))

    fake_rep = {fake: node for fake, node in fakes.items()}

    idx = idx.substitute(fake_rep)
    idy = idy.substitute(fake_rep)

    idy_vars, idx_vars, ones = set(idy.vars()), set(idx.vars()), []
    for node in nodes:
      node_vars = set(node.vars())
      if not node_vars & (idx_vars | idy_vars): continue #There is simplified NumNode which can not go outside the bounds
      # NOTE: Why does only idy is problematic? and not the idx
      if idy_vars == node_vars or idy_vars & node_vars == set(): ones.append(node)
    valid = Variable.ands([i for i in nodes if i not in ones])

  if DEBUG>=5: print("to_image_idx", base_shape, idx.min, idx.max, idy.min, idy.max, idx, idy, valid)
  return (idx, idy), valid
