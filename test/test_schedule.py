# this will be the new test_ops for the next level
# schedule confirms the right things are capable of fusing
# NOTE: this has overlap with external_test_opt.py

import unittest
import numpy as np
from typing import List, Optional, Union
from tinygrad.engine.realize import run_schedule
from tinygrad.tensor import Tensor
from tinygrad.ops import BinaryOps, LoadOps, ReduceOps
from tinygrad.helpers import DEBUG, flatten
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.engine.graph import print_tree
from tinygrad.engine.schedule import create_schedule
from tinygrad import nn, dtypes
from test.helpers import is_dtype_supported

def check_schedule(t:Union[Tensor, List[Tensor]], allowed:int, to_prerealize:Optional[List[Tensor]]=None, filter_loadops=True):
  if isinstance(t, Tensor): t = [t]
  seen = set()
  if to_prerealize:
    for pre in to_prerealize:
      for s in pre.schedule(seen=seen.copy()):
        for i,out in enumerate(s.outputs):
          seen.add(out)
  sched = create_schedule(flatten([r.lazydata.lbs for r in t]), seen)
  if filter_loadops: sched = [s for s in sched if s.ast[0].op not in LoadOps]
  if len(sched) != allowed: print(f"SCHEDULE ISSUE, expecting {allowed} got {len(sched)}")
  if len(sched) != allowed or DEBUG >= 3:
    for i, s in enumerate(sched):
      print("kernel", i+1)
      for op in s.ast: print_tree(op)
  assert len(sched) == allowed, f"{len(sched)} != {allowed}"
  # test the (non loadops) ops linearize
  for s in sched:
    if s.ast[0].op in LoadOps: continue
    l = Linearizer(*s.ast)
    l.hand_coded_optimizations()
    l.linearize()
  return sched

class TestSchedule(unittest.TestCase):
  def test_basic_binop_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = a+b+c
    check_schedule(d, 1)

  def test_basic_binop_fusion_deep(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = Tensor.empty(10)
    e = a+b+c+d
    check_schedule(e, 1)

  def test_mulacc_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = (a*b).sum()
    check_schedule(c, 1)

  def test_mulacc_relu_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = (a*b).sum().relu()
    check_schedule(c, 1)

  def test_binop_reshape_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(5,2)
    d = (a+b).reshape(5,2)+c
    check_schedule(d, 1)

  def test_binop_permute_fusion(self):
    a = Tensor.empty(2,5)
    b = Tensor.empty(2,5)
    c = Tensor.empty(5,2)
    d = (a+b).permute(1,0)+c
    check_schedule(d, 1)

  def test_constants_are_embedded(self):
    a = Tensor.empty(3,3) * 2
    check_schedule(a, 2, filter_loadops=False)

  def test_binop_elu_fusion(self):
    a = Tensor.empty(10)
    b = a.elu()
    check_schedule(b, 1)

  def test_binop_reshape_reduce_fusion(self):
    a = Tensor.empty(100)
    b = Tensor.empty(100)
    c = (a+b).reshape(10, 10).sum(axis=0, keepdim=True)
    check_schedule(c, 1)

  def test_reduce_reshape_binop_fusion(self):
    a = Tensor.empty(10,10)
    b = Tensor.empty(10)
    c = a.sum(axis=0) + b
    check_schedule(c, 1)

  @unittest.skip("not pushing permutes through reduces")
  def test_reduce_permute_binop_fusion(self):
    a = Tensor.empty(10,10,10)
    b = Tensor.empty(10,10,1)
    c = a.sum(axis=0, keepdim=True).permute(2,1,0) + b
    check_schedule(c, 1)

  def test_binop_early_reshape_reduce_fusion(self):
    a = Tensor.empty(100)
    b = Tensor.empty(100)
    c = Tensor.empty(10,10)
    d = ((a+b).reshape(10,10) + c).sum(axis=0)
    check_schedule(d, 1)

  def test_diamond_folded(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = Tensor.empty(10)
    ab = a+b
    e = (ab+c) + (ab+d)
    check_schedule(e, 1)

  def test_cache_binaryop(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a+b
    d = a+b
    check_schedule(d, 0, [c])

  @unittest.skip("failing in old lazy")
  def test_cache_binaryop_reshaped(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a+b
    d = a.reshape(10,1)+b.reshape(10,1)
    check_schedule(d, 0, [c])

  @unittest.skip("failing in new lazy")
  def test_cache_binaryop_transpose(self):
    a = Tensor.empty(10,10)
    b = Tensor.empty(10,10)
    c = (a.T*b.T).T #.contiguous()
    d = a*b
    check_schedule(d, 0, [c])

  def test_cache_two_reduceops(self):
    a = Tensor.empty(10)
    b = a.sum()
    c = a.sum()
    bc = b+c
    check_schedule(bc, 1)

  def test_cache_reduce_parent(self):
    x = Tensor.empty(32)
    r0 = x.mean(axis=0, keepdim=True)
    r1 = (x - r0).sum(axis=0).div(2)
    out = r0 + r1
    schedule = check_schedule(out, 2)
    reduceops = [x for si in schedule for out in si.ast for x in out.lazyops if x.op in ReduceOps]
    assert len(reduceops) == 2

  def test_cache_reduce_multiple_children(self):
    x = Tensor.empty(32)
    y = Tensor.empty(4, 4)
    r0 = x.mean(axis=0, keepdim=True)
    r1 = (x - r0).sum(axis=0).div(2)
    out0 = r0 + y
    out1 = r1 + y
    schedule = check_schedule([out0, out1], 4)
    reduceops = [x for si in schedule for out in si.ast for x in out.lazyops if x.op in ReduceOps]
    assert len(reduceops) == 2

  def test_fold_double_unary(self):
    y = Tensor.empty(2)
    out = y.sum(keepdim=True).sqrt().__neg__()
    check_schedule(out, 1)

  #@unittest.skip("may want to reconsider this")
  def test_fold_batchnorm(self):
    with Tensor.train():
      img = Tensor.empty(1,32,4,4)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(img)
      check_schedule(out, 3)

  def test_fold_conv_batchnorm_notrain(self):
    with Tensor.train(False):
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(c1(img)).relu()
      check_schedule(out, 1, [c1.weight, c1.bias])

  def test_fold_conv_batchnorm(self):
    with Tensor.train():
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(c1(img)).relu()
      check_schedule(out, 4, [c1.weight, c1.bias])

  def test_fold_conv_batchnorm_optim(self):
    # this is too high
    for optim, cnt in [(nn.optim.Adam, 19), (nn.optim.SGD, 17)]:
      with self.subTest(optim=optim.__name__):
        with Tensor.train():
          img = Tensor.ones(1,3,4,4)
          c1 = nn.Conv2d(3,32,3)
          bn = nn.BatchNorm2d(32, track_running_stats=False)
          opt = optim(nn.state.get_parameters([c1, bn]))
          img_bn = bn(c1(img)).elu().sum()
          opt.zero_grad()
          img_bn.backward()
          check_schedule(opt.schedule_step(), cnt)

  def test_fold_conv_relu_backward(self):
    c1 = nn.Conv2d(3,16,3, bias=False)
    c1.weight.requires_grad = True

    # run
    img = Tensor.rand(2,3,64,64, requires_grad=True)
    c1(img).relu().mean().backward()
    # TODO: this should be 4, not 5
    # img.grad is requiring two reduces
    check_schedule([img.grad, c1.weight.grad], 5)

  def test_fold_batchnorm_backward(self):
    with Tensor.train():
      x = Tensor.empty((2, 16, 8, 8)).contiguous()
      bn = nn.BatchNorm2d(16)
      bn.weight.requires_grad = bn.bias.requires_grad = x.requires_grad = True
      fw = bn(x).contiguous_backward().relu().contiguous()
      fw.sum().backward()
      # TODO: this is too many
      check_schedule([x.grad, bn.weight.grad, bn.bias.grad, fw], 10)

  def test_fold_conv_relu(self):
    c1 = nn.Conv2d(3,16,3)

    # run
    img = Tensor.ones(2,3,64,64)
    out = c1(img).relu()
    check_schedule(out, 1, [c1.weight, c1.bias])

  def test_fold_conv_relu_nobias(self):
    img = Tensor.ones(1,4,8,8)
    c1 = nn.Conv2d(4, 4, kernel_size=3, bias=False)
    c2 = nn.Conv2d(4, 4, kernel_size=3, bias=False)
    out = img.sequential([c1, Tensor.relu, c2, Tensor.relu])
    check_schedule(out, 2, [c1.weight, c2.weight, img])

  def test_fold_conv_elu(self):
    c1 = nn.Conv2d(3,16,3)

    # run
    img = Tensor.rand(2,3,64,64)
    out = c1(img).elu()
    check_schedule(out, 1, [c1.weight, c1.bias, img])

  def test_two_sum(self):
    img = Tensor.empty(64,64)
    x = (img.sum(0) + img.sum(1))
    out = x.relu()
    del x    # is 3 without this
    check_schedule(out, 2)

  #@unittest.skip("failing in old lazy")
  def test_push_permute_through_reshape(self):
    a = Tensor.empty(16,16)
    b = Tensor.empty(16,16)
    c = (a+b).reshape(4,4,4,4).permute(2,3,0,1).contiguous()
    check_schedule(c, 1)

  #@unittest.skip("failing in old lazy")
  def test_push_permute_through_reshape_alt(self):
    a = Tensor.empty(4,4,4,4)
    b = Tensor.empty(4,4,4,4)
    c = (a+b).reshape(16,16).permute(1,0).contiguous()
    check_schedule(c, 1)

  def test_no_binop_rerun(self):
    a = Tensor.empty(16)
    b = Tensor.empty(16)
    c = a+b
    d = (a+b).reshape(16,1)
    check_schedule(d, 0, [c])

  def test_multi_permute_should_collapse(self):
    a = Tensor.empty(4,4,4,4)
    b = Tensor.empty(16)
    c = a.sum((0,1)).cast(dtypes.float16).permute(1,0).reshape(4,4,1).permute(1,0,2).reshape(16) + b
    check_schedule(c, 1)

  @unittest.skip("failing in old lazy")
  def test_fancy_reshape_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a+b
    d = a.reshape(10,1)+b.reshape(10,1)
    out = c.sum() + d.sum()
    check_schedule(out, 1)

  # NOTE: for this to pass, LazyViews must be children of LazyBuffers so the (a+b) runs first
  @unittest.skip("not real world")
  def test_children_dont_push(self):
    a = Tensor.empty(10, 10, 1)
    b = Tensor.empty(10, 10, 1)
    d = (a+b).expand(10, 10, 10)
    e = (a+b).permute(2,1,0)
    f = d+e
    check_schedule(f, 2)

  @unittest.skip("failing in new lazy")
  def test_dont_fuse_binops_with_children(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    keep_me = a+b
    e = keep_me.sum() # noqa: F841 give keep_me a child (NOTE: BinaryOps won't be a child since it will instant fuse)
    d = keep_me+c
    check_schedule(d, 2)
    check_schedule(keep_me, 0, [d])

  #@unittest.skip("failing in old lazy")
  def test_permute_breaks_fusion(self):
    a = Tensor.empty(10, 10, 10)
    b = Tensor.empty(10, 10)
    c = (a.sum(axis=2) + b).permute(1,0)
    d = c.permute(1,0)
    check_schedule(d, 1)

  def test_some_permute_fusion(self):
    a = Tensor.empty(8192, 16)
    b = Tensor.empty(1, 16)
    d = (a.T + b.expand(8192, 16).T)
    c = a + b.expand(8192, 16)
    e = d.T
    check_schedule(c, 1)
    check_schedule(e, 1)

  def test_shrink_fuse(self):
    a = Tensor.empty(8192, 16)
    b = Tensor.empty(8192, 16)
    c = a * b
    d = Tensor.empty(1, 16)
    e = c[0] * d
    check_schedule(e, 1)

  def test_expand_nofuse(self):
    a = Tensor.empty(1, 16)
    b = Tensor.empty(1, 16)
    c = a * b
    d = Tensor.empty(8192, 16)
    e = c * d
    check_schedule(e, 2)

  # this is the failing case in openpilot...it's very simple like this
  @unittest.skip("failing in old lazy")
  def test_image_conv_fusion(self):
    from tinygrad.features.image import image_conv2d
    w1 = Tensor.empty(16, 16, 1, 1)
    b1 = Tensor.empty(16)
    w2 = Tensor.empty(16, 16, 1, 1)
    b2 = Tensor.empty(16)
    w3 = Tensor.empty(16, 16, 1, 1)
    b3 = Tensor.empty(16)

    x = Tensor.empty(1, 16, 32, 32)
    x = base = image_conv2d(x, w1, b1)
    x = image_conv2d(x, w2, b2) + base
    x = image_conv2d(x, w3, b3)

    # NOOP, 3 convs, contiguous
    check_schedule(x, 5)

  def test_image_conv_fusion_minimal(self):
    b1 = Tensor.empty(16)
    b2 = Tensor.empty(16)
    def p(x): return x.permute(1,0).contiguous().reshape(32,16,1).expand(32,16,16).sum(axis=2).permute(1,0)

    x = Tensor.empty(16, 32)
    x = base = p(x) + b1.reshape(16,1)
    x = p(x)
    x = x + b2.reshape(16,1)
    x = x + base
    del base
    x = p(x)
    check_schedule(x, 4)

  def test_image_conv_fusion_more_minimal(self):
    b1 = Tensor.empty(16)
    def p(x): return x.permute(1,0).contiguous().reshape(32,16,1).expand(32,16,16).sum(axis=2).permute(1,0)

    x = Tensor.empty(16, 32)
    x = base = p(x) + b1.reshape(16,1)
    x = p(x)
    del base
    check_schedule(x, 3)

  def test_resnet_block(self):
    Tensor.training = False

    in_planes, planes = 64, 64
    conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(planes)
    conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    bn2 = nn.BatchNorm2d(planes)

    x = Tensor.empty(1, 64, 32, 32)
    out = bn1(conv1(x)).relu()
    out = bn2(conv2(out))
    out = (out + x).relu()
    check_schedule(out, 2, [conv1.weight, conv2.weight])

  def test_contiguous_while_contiguous(self):
    x = Tensor.empty(1, 64, 32, 32)
    out = x.contiguous()
    check_schedule(out, 1, filter_loadops=False)

  def test_contiguous_while_not_contiguous(self):
    x = Tensor.empty(1, 64, 32, 32)
    out = x.permute(0,2,3,1).contiguous()
    check_schedule(out, 2, filter_loadops=False)

  def test_double_from(self):
    x = Tensor([1,2,3,4])
    out = x.to('npy')
    check_schedule(out, 0, filter_loadops=False)

  def test_pow_const_tensor_simplified(self):
    x = Tensor([1,2,3,4])
    # NOTE: this does not test ** Tensor(2) is simpler in ast than ** Tensor(2.5)
    out = x ** Tensor(2)
    check_schedule(out, 1)

  def test_pow_const_tensor_to_zero(self):
    x = Tensor([1,2,3,4])
    out = x ** Tensor(0)
    # NOTE: this is ConstBuffer 0 + ConstBuffer 1
    check_schedule(out, 0)

  def test_zero_size(self):
    x = Tensor.empty(2, 3, 0)
    out = x + 1
    check_schedule(out, 0, filter_loadops=False)

  def test_reduce_permute_nofuse(self):
    x = Tensor.empty(32, 32, 32)
    y = Tensor.empty(32, 32)
    out = x.sum(axis=2).T+y
    check_schedule(out, 2)

  def test_two_elus_sum(self):
    x = Tensor.empty(32, 32)
    y = Tensor.empty(32, 32)
    out = x.sum(1).relu().elu() + y.sum(1).relu().elu()
    check_schedule(out, 2)

  def test_multistage_reduce(self):
    x = Tensor.empty(32, 32, 32)
    out = x.sum(2).relu().sum(1)
    check_schedule(out, 2)

  def test_multistage_reduce_fork(self):
    x = Tensor.empty(32, 32, 32)
    x = x.sum(2)
    out2 = x + 1
    out = x.relu().sum(1) + out2[0]
    check_schedule(out, 2)

  def test_example_matmul(self):
    x = Tensor.eye(64, requires_grad=True)
    y = Tensor.eye(64, requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    check_schedule(out, 2)

  def test_contiguous_add(self):
    x = Tensor.empty(32)
    y = Tensor.empty(32)
    z = Tensor.empty(32)
    out = (x+y).contiguous()+z
    check_schedule(out, 2)

  def test_double_sum_ref(self):
    x = Tensor.empty(32, 32, 32)
    x = x.sum(2)
    out = x + x[:, 4]
    check_schedule(out, 2)

  def test_reduce_shrink(self):
    x = Tensor.empty(32, 32)
    y = Tensor.empty(16)
    x = x.sum(1)
    x = x[:16]
    out = x + y
    check_schedule(out, 2)  # TODO: this should be 1

  @unittest.skip("broken due to const folding and two contiguous are different kernels")
  def test_const_no_recompute(self):
    x = Tensor(2) + Tensor(2)
    y = Tensor(2) + Tensor(2)
    out = x.contiguous() + y.contiguous()
    check_schedule(out, 2)

  def test_reduce_same_size(self):
    a = Tensor.empty(4, 4)
    out0 = a.sum() + 2
    out1 = a.sum() + 4
    out2 = out0 * out1
    check_schedule([out0, out1, out2], 1)

  def test_reduce_multiple_paths(self):
    a = Tensor.empty(4, 4)
    out0 = a.sum().exp2()
    # out1 has two paths to a.sum()
    out1 = a.sum() + out0
    check_schedule([out0, out1], 1)

  def test_reduce_ext_reduce_child(self):
    a = Tensor.empty((4, 4))
    b = Tensor.empty((4, 4))
    # b.sum() is not a descendant of the fused nodes
    out0 = a.sum() + b.sum() + 2
    out1 = a.sum() + b.sum() + 4
    check_schedule([out0, out1], 4)

  def test_reduce_multiple_paths_midreduce(self):
    a = Tensor.empty(4, 4)
    r = a.sum()
    out0 = r.exp2()
    # reduce node in the indirect path from r to out2
    out1 = (a - out0).max()
    out2 = r + out1
    check_schedule([r, out0, out1, out2], 4)

  def test_reduce_multiple_paths_midreduce_fused(self):
    a = Tensor.empty(4, 4)
    b = Tensor.empty(4, 4)
    out0 = a.sum() + 4
    out1 = b.max() + out0*2
    out2 = a.sum() + out1
    check_schedule([out0, out1, out2], 4)

  def test_reduce_multiple_paths_midexpand(self):
    a = Tensor.empty(4, 4)
    b = Tensor.empty(4, 4, 4)
    r = a.sum()
    out0 = r.exp2()
    # e1 is in the indirect path from a.sum() to out1
    e = b + out0
    out1 = r + e[0][0][0]
    check_schedule([r, out0, out1, e], 4)

  def test_reduce_expand_child(self):
    a = Tensor.empty((32, 32, 32))
    b = Tensor.empty((1, 16))
    out0 = a.sum() + 2
    out1 = a.sum() + b
    check_schedule([out0, out1], 4)

  def test_reduce_shrink_child(self):
    a = Tensor.empty(100, 100)
    b = Tensor.empty(10,)
    c = a.sum() + b[0]
    d = a.sum() + 2
    check_schedule([c, d], 1)

  def test_reduce_multiple_paths_midshrink(self):
    a = Tensor.empty(4, 4)
    r = a.sum(axis=1)
    out0 = r.exp2()
    out1 = out0[0] + out0
    check_schedule([r, out0, out1], 3)

  def test_reduce_shrink_output(self):
    a = Tensor.empty(4, 4)
    r = a.sum(keepdim=True)
    out0 = r.exp2()
    out1 = out0[0] + Tensor.empty(1, )
    check_schedule([r, out0, out1], 3)

  def test_softmax_fusion(self):
    out = Tensor.empty(4, 12, 64, 64).softmax()
    check_schedule(out, 3)

  def test_layernorm_onelayer_fusion(self):
    layer = nn.LayerNorm([10, 10])
    x = Tensor.empty(20, 5, 10, 10)
    check_schedule(layer(x), 3)

  def test_scaled_dot_product_attention_fusion(self):
    x, y, z, m = (Tensor.empty(32, 8, 16, 16) for _ in range(4))
    out = Tensor.scaled_dot_product_attention(x, y, z, attn_mask=m)
    check_schedule(out, 5)

  def test_scaled_dot_product_attention_causal_fusion(self):
    x, y, z, m = (Tensor.empty(32, 8, 16, 16) for _ in range(4))
    out = Tensor.scaled_dot_product_attention(x, y, z, attn_mask=m, is_causal=True)
    check_schedule(out, 7)

  def test_adam_step_fusion(self):
    with Tensor.train():
      x = Tensor.empty(4, 64, 768)
      layer = nn.Linear(768, 768*4)
      opt = nn.optim.Adam(nn.state.get_parameters(layer), lr=1e-4)
      layer(x).relu().sum().backward()
      check_schedule(opt.schedule_step(), 11)

  def test_adam_conv_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      opt = nn.optim.Adam(nn.state.get_parameters(c1), lr=1e-4)
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 11)

  def test_adam_2convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,3,bias=False)
      opt = nn.optim.Adam(nn.state.get_parameters([c1, c2]), lr=1e-4)
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 13)

  def test_sgd_conv_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      opt = nn.optim.SGD(nn.state.get_parameters(c1))
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 7)

  def test_sgd_2convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,3,bias=False)
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]))
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 7)

  def test_fold_2convs_sgd_nesterov_momentum_wd(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,3,bias=False)
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]), nesterov=True, momentum=0.9, weight_decay=0.1)
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 9)

  def test_sgd_4convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,64,64)
      c1 = nn.Conv2d(3,4,3,bias=False)
      c2 = nn.Conv2d(4,8,3,bias=False)
      c3 = nn.Conv2d(8,16,3,bias=False)
      c4 = nn.Conv2d(16,32,3,bias=False)
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 22)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_prefer_half_buffer(self):
    x = Tensor.ones(4).contiguous().realize()
    # y = Tensor.ones(4).contiguous().realize()
    z = Tensor.ones(4, 4).contiguous().realize()

    # should not create extra kernel if output will be realized anyways
    dummy = x.sum().half().float()
    check_schedule(dummy, 1)
    dummy = x.sum().half().float().contiguous() + 1
    check_schedule(dummy, 2)

    # shared between two outputs
    shared = x.sum().half().float()
    a = shared * 2
    b = shared * 3
    sched = check_schedule([a, b], 1)
    for si in sched[:-2]: assert all(out.dtype is dtypes.half for out in si.outputs)

    # reduce
    a = z.sum(axis=0).half().float().sum(axis=0)
    sched = check_schedule(a, 2)
    for si in sched[:-1]: assert all(out.dtype is dtypes.half for out in si.outputs)

    # expand
    # expand will realize just after the .float(), so requires change to realize-before-expand
    # normal = (x.sum().half().float().reshape(1) * y).sum()
    # sched = check_schedule(normal, 2)
    # for si in sched[:-1]: assert all(out.dtype == dtypes.half for out in si.outputs[:-1])

    # parallel reduce
    # a = x.sum().half().float() * y.sum().half().float()
    # b = a + 1
    # c = a + 2
    # sched = check_schedule([b, c], 4)
    # doesn't store either in half because it doesn't chase

  def test_reduce_simple_chase(self):
    a = Tensor.empty(4, 4, 4)
    r = a.sum(0) + 6
    b = r.sum(0) * 4
    c = r.sum(1) * 2
    schedule = check_schedule([b, c], 3)
    assert schedule[0].ast[0].src[0].op is BinaryOps.ADD

  def test_push_permute_chase(self):
    a = Tensor.empty(4, 4, 4)
    b = Tensor.empty(4, 4)
    r = a.sum(2) + b
    d = r.T * 4
    e = r * d
    schedule = check_schedule([d, e], 3)
    assert schedule[0].ast[0].src[0].op is BinaryOps.ADD

  def test_push_shrink_chase(self):
    a = Tensor.empty(16, 16)
    b = Tensor.empty(4)
    c = Tensor.empty(16, )
    r = a.sum(1) + c
    d = r[:4] * b
    schedule = check_schedule(d, 2)
    assert schedule[0].ast[0].src[0].op is BinaryOps.ADD

  def test_midreduce_nochase(self):
    a = Tensor.empty(16, 16)
    b = (a.sum(0) + a.max(1)) + 2
    schedule = check_schedule(b, 2)
    assert schedule[0].ast[0].src[0].op is ReduceOps.MAX

  # pattern in test_transformer
  def test_partial_fuse1(self):
    a = Tensor.empty(16, 16)
    b = Tensor.empty(16, 16)
    c = a.sum() + 2
    d = (a.sum() - b.sum()) * 4
    check_schedule([c, d], 3)

  # pattern in conv
  def test_partial_fuse2(self):
    a = Tensor.empty(16, 16)
    b = Tensor.empty(16, 16)
    c = a.sum() + 2
    d = b.sum() - c
    check_schedule([c, d], 2)

  # pattern in adam
  def test_partial_fuse3(self):
    a = Tensor.empty(16, 16)
    b = Tensor.empty(16, 16)
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = b.sum() - e
    check_schedule([c, d, e, f], 2)

  def test_partial_fuse4(self):
    a = Tensor.empty(16, 16)
    b = Tensor.empty(16, 16)
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = (b - d).sum() - e
    check_schedule([c, d, e, f], 3)

  def test_pad_reduce_safe(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(3, 4, 5).realize()
    b = Tensor.rand(3, 4, 5).realize()
    out = (a + b).pad(((0, 1), (0, 1), (0, 1)), 1.0).sum().contiguous()
    run_schedule(check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), np.pad(a.numpy()+b.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum())

  def test_pad_reduce_usafe(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(3, 4, 5).realize()
    out = a.log2().pad(((0, 1), (0, 1), (0, 1)), 1.0).sum().contiguous()
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.pad(np.log2(a.numpy()), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), rtol=1e-6)

  def test_shrink_pad_safe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    b = Tensor.ones((3, )).contiguous().realize()
    out = (a + b).shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    run_schedule(check_schedule(out, 1))
    np.testing.assert_equal(out.numpy(), [2, 0])

  # TODO: should not shuffle unsafe pad ops through any pads, even if buffer is shrunk overall (#3437)
  def test_shrink_pad_unsafe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    out = a.exp2().shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    run_schedule(check_schedule(out, 1))
    with self.assertRaises(AssertionError):
      np.testing.assert_equal(out.numpy(), [2, 0])

  def test_base_change_shrink_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:-1, :-1]
    d = c.pad(((0, 1), (0, 1))) * 2
    run_schedule(check_schedule(d, 1))
    with self.assertRaises(AssertionError): # TODO unsafe pads
      np.testing.assert_equal(d.numpy(), np.pad(np.exp2(a.numpy())[:-1, :-1], ((0, 1), (0, 1)))*2)

  def test_base_change_expand_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:, None, :]
    d = c.pad(((0, 0), (1, 1), (0, 0))) * 2
    run_schedule(check_schedule(d, 2))
    np.testing.assert_equal(d.numpy(), np.pad(np.exp2(a.numpy())[:, None, :], ((0, 0), (1, 1), (0, 0)))*2)

  # TODO like openpilot with imagef
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_base_change_expand_expand(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.cast(dtypes.half).expand(2, 4, 4)
    c = b.cast(dtypes.int).expand(2, 2, 4, 4)
    run_schedule(check_schedule(c, 2))
    np.testing.assert_equal(c.numpy(), np.ones(((2, 2, 4, 4)), dtype=np.int32))

  def test_base_change_pad_expand(self):
    a = Tensor.full((4, 4), 1.).contiguous().realize()
    b = Tensor.full((4, 4), 2.).contiguous().realize()
    c = (a + b).pad(((1, 1), (1, 1)))
    d = c.cast(dtypes.int).expand((2, 6, 6)) * 4
    run_schedule(check_schedule(d, 2))
    c_np = np.pad((np.full((4, 4), 2., dtype=np.float32) + np.full((4, 4), 1., dtype=np.float32)), ((1, 1), (1, 1)), constant_values=0.0)
    np.testing.assert_equal(d.numpy(), np.broadcast_to(c_np.astype(np.half), (2, *c_np.shape)) * 4)

if __name__ == '__main__':
  unittest.main(verbosity=2)
