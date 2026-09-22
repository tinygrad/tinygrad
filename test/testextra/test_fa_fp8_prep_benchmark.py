"""Exact per-GPU Llama prep benchmark: DEV=AMD HCQ2=0 python -m unittest test.testextra.test_fa_fp8_prep_benchmark -v."""
import functools, statistics, unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import compile_linear, time_call
from tinygrad.helpers import Context
from extra.thunder.amd.fa_fp8_bwd import custom_fp8_backward_init, custom_fp8_backward_prep, custom_fp8_backward_scales

@unittest.skipUnless(Device.DEFAULT.split(":")[0] == "AMD" and Device[Device.DEFAULT].renderer.target.arch == "gfx950",
                    "full Llama gfx950 benchmark")
class TestFP8PrepLlama(unittest.TestCase):
  def test_correctness_and_speed(self):
    # kernel_graph: 67,108,864 BF16 inputs, 524,288 FP32 deltas, 1,024 partial maxima per rank.
    shape = (2,8192,32,128)
    Tensor.manual_seed(28200)
    do,out = [(Tensor.randn(*shape)*0.2).bfloat16().realize() for _ in range(2)]
    dq,partial = Tensor.custom_kernel(Tensor.full(shape,17.,dtype=dtypes.bfloat16).realize(),Tensor.empty(2,512),do,
                                     fxn=custom_fp8_backward_init)[:2]
    Tensor.realize(dq,partial)
    self.assertEqual(dq.float().abs().max().item(),0.)
    self.assertEqual((partial-do.float().abs().reshape(2,512,-1).max(-1)).abs().max().item(),0.)
    scale = ((partial.max()+1e-8)/57344.).realize()
    rounded = (do.float()/scale).clamp(-57344,57344).cast(dtypes.fp8e5m2).contiguous().realize()
    expected = (out.float()*(rounded.float()*scale)).sum(-1).transpose(1,2).contiguous().realize()
    # Independent reference for the original gfx950 kernel's accumulation and scaling order.
    products = (out.float()*rounded.float()).numpy()
    accum = np.zeros((*shape[:-1],8),dtype=np.float32)
    for i in range(16): accum += products[...,i*8:(i+1)*8]
    exact = np.zeros(shape[:-1],dtype=np.float32)
    for i in range(8): exact += accum[...,i]
    numerator = np.float32(partial.max().item())+np.float32(1e-8)
    exact = (exact*numerator*np.float32(1/57344.)).transpose(0,2,1)
    del products,accum
    for ds in (0.,0.3):
      state,vs = Tensor([1.,ds]).realize(),Tensor([0.37]).realize()
      scales,next_amax = Tensor.custom_kernel(Tensor.empty(5),Tensor([17.,29.]).realize(),partial,state,vs,
        fxn=functools.partial(custom_fp8_backward_scales,D=128))[:2]
      ret = Tensor.custom_kernel(Tensor.empty(shape,dtype=dtypes.fp8e5m2),Tensor.empty(2,32,8192),do,out,
        scales,fxn=custom_fp8_backward_prep)+[next_amax]
      with Context(BEAM=3):
        linear,values = Tensor.linear_with_vars(ret[0],ret[1],ret[4],ret[5])
        compiled = compile_linear(linear)
      # Device timestamps exclude Python dispatch and compilation. Include every kernel in the prep pipeline.
      timers = [time_call(call,values) for call in compiled.src]
      samples = [sum(next(timer) for timer in timers) for _ in range(25)]
      print(f"FA prep shape={shape} ds={ds}: median {statistics.median(samples[5:])*1e6:.2f} us, kernels={len(timers)}",flush=True)
      Tensor.realize(ret[0],ret[1],ret[4],ret[5])
      self.assertEqual((ret[0].bitcast(dtypes.uint8)!=rounded.bitcast(dtypes.uint8)).sum().item(),0)
      np.testing.assert_allclose(ret[1].numpy(),expected.numpy(),rtol=2e-5,atol=2e-6)
      np.testing.assert_array_equal(ret[1].numpy(),exact)
      effective_ds = ds/57344. if ds else 4*128*scale.item()*0.37*448.
      np.testing.assert_allclose(ret[4][:4].numpy(),[0.37,scale.item(),(1.+1e-8)/448.,effective_ds],rtol=2e-6)
      np.testing.assert_array_equal(ret[4][4].numpy(),numerator)
      np.testing.assert_array_equal(ret[5].numpy(),[0.,0.])

if __name__ == '__main__': unittest.main()
