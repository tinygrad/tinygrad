import functools, math, unittest
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes

def round_fp8(x:Tensor, descale:Tensor, dtype) -> Tensor:
  limit = 448 if dtype == dtypes.fp8e4m3 else 57344
  return (x.float()/descale).clamp(-limit, limit).cast(dtype).float()*descale

def fp8_backward_reference(q8, k8, v8, v_descale, do, out, lse, do_descale, p_descale, ds_descale, *, quantize=True):
  """Small, materialized reference. Saved LSE is natural-log; Q/K scores use base 2.

  p_descale/ds_descale are immutable snapshots of delayed state, not amaxes from this invocation.
  Return new amaxes separately so capture/replay cannot overwrite scales needed by an earlier microbatch.
  """
  B, N, H, D = q8.shape
  assert N <= 512, "quadratic reference is only for unit tests"
  Hkv = k8.shape[2]
  def heads(x): return x.transpose(1, 2).float()
  def expand(x): return heads(x).reshape(B,Hkv,1,N,D).expand(B,Hkv,H//Hkv,N,D).reshape(B,H,N,D)
  q, k, v = heads(q8), expand(k8), expand(v8)*v_descale
  dh = heads(round_fp8(do, do_descale, dtypes.fp8e5m2) if quantize else do)
  mask = Tensor.arange(N).to(q8.device).reshape(N,1) >= Tensor.arange(N).to(q8.device).reshape(1,N)
  p = mask.where((q @ k.transpose(-1,-2) - lse.reshape(B,H,N,1)*math.log2(math.e)).exp2(), 0)
  delta = (heads(out)*dh).sum(-1, keepdim=True)
  if quantize:
    d8 = heads((do.float()/do_descale).clamp(-57344,57344).cast(dtypes.fp8e5m2))
    # Native FP8 GEMMs accumulate first, then apply operand descales. Moving
    # descales into the operands changes FP32 rounding before dS quantization.
    dp = (d8 @ expand(v8).transpose(-1,-2))*do_descale*v_descale
  else: dp = dh @ v.transpose(-1,-2)
  ds = p*(dp - delta)
  pq = round_fp8(p, p_descale, dtypes.fp8e4m3) if quantize else p
  dsq = round_fp8(ds, ds_descale, dtypes.fp8e5m2) if quantize else ds
  # d/dQ of ln(2) * Q8 K8.T, including our producer's sqrt(a*log2(e)) scaling.
  grad_scale = D**-0.5 / math.sqrt(D**-0.5 * math.log2(math.e))
  dq = (dsq @ k)*grad_scale
  dk = (dsq.transpose(-1,-2) @ q)*grad_scale
  dv = pq.transpose(-1,-2) @ dh
  def reduce_gqa(x): return x.reshape(B,Hkv,H//Hkv,N,D).sum(2).transpose(1,2)
  return dq.transpose(1,2), reduce_gqa(dk), reduce_gqa(dv), p.abs().max(), ds.abs().max()

class TestFP8BackwardReference(unittest.TestCase):
  def test_fused_state_update(self):
    from examples.mlperf.models.flat_llama import FlatTransformer
    model = FlatTransformer.__new__(FlatTransformer)
    model._fp8_amax,model._fp8_next_amax = {},{}
    model._fp8_grad_amax = {"fa":[Tensor([1.,0.]).realize() for _ in range(2)]}
    model._fp8_next_grad_amax = {"fa":[Tensor([3.,4.]).realize(),Tensor([5.,6.]).realize()]}
    loss = Tensor([17.]).realize()
    @TinyJit
    def step():
      snapshot = loss.clone()
      reset = model.update_amax(reset=loss)
      Tensor.realize(snapshot,reset,*model._fp8_grad_amax["fa"])
      return snapshot
    for i in range(3):
      loss.assign(17.+i).realize()
      model._fp8_next_grad_amax["fa"][0].assign(Tensor([3.+i,4.+i])).realize()
      np.testing.assert_array_equal(step().numpy(),[17.+i])
      np.testing.assert_array_equal(loss.numpy(),[0.])
      np.testing.assert_array_equal(model._fp8_grad_amax["fa"][0].numpy(),[3.+i,4.+i])
      np.testing.assert_array_equal(model._fp8_grad_amax["fa"][1].numpy(),[5.,6.])

  def test_backward_scale_finalization(self):
    from extra.thunder.amd.fa_fp8_bwd import custom_fp8_backward_init, custom_fp8_backward_prep
    rng = np.random.default_rng(17)
    shape = (2,64,4,128)
    do,out = [Tensor(rng.standard_normal(shape).astype(np.float32)*0.2).bfloat16().realize() for _ in range(2)]
    _,partial = Tensor.custom_kernel(Tensor.empty_like(do),Tensor.empty(2,512),do,fxn=custom_fp8_backward_init)[:2]
    for ds in (0.,0.3):
      state,vs = Tensor([1.,ds]).realize(),Tensor([0.37]).realize()
      ret = Tensor.custom_kernel(Tensor.empty_like(do,dtype=dtypes.fp8e5m2),Tensor.empty(2,4,64),do,out,
        Tensor.empty(4),Tensor([17.,29.]).realize(),partial,state,vs,fxn=functools.partial(custom_fp8_backward_prep,finalize=True))
      scale = (do.float().abs().max()+1e-8)/57344.
      rounded = (do.float()/scale).clamp(-57344,57344).cast(dtypes.fp8e5m2).contiguous().realize()
      expected = (out.float()*(rounded.float()*scale)).sum(-1).transpose(1,2).contiguous().realize()
      np.testing.assert_array_equal(ret[0].float().numpy(),rounded.float().numpy())
      np.testing.assert_allclose(ret[1].numpy(),expected.numpy(),rtol=2e-5,atol=2e-6)
      effective_ds = ds/57344. if ds else 4*128*scale.item()*0.37*448.
      np.testing.assert_allclose(ret[4].numpy(),[0.37,scale.item(),(1.+1e-8)/448.,effective_ds],rtol=2e-6)
      np.testing.assert_array_equal(ret[5].numpy(),[0.,0.])

  def test_backward_init(self):
    from extra.thunder.amd.fa_fp8_bwd import custom_fp8_backward_init
    rng = np.random.default_rng(28200)
    do = Tensor(rng.standard_normal((2,64,4,128)).astype(np.float32)).bfloat16().realize()
    for dtype in (dtypes.bfloat16,dtypes.float32):
      dq,partial = Tensor.custom_kernel(Tensor.full(do.shape,17.,dtype=dtype).realize(),Tensor.empty(2,512),do,
                                        fxn=custom_fp8_backward_init)[:2]
      np.testing.assert_array_equal(dq.float().numpy(),np.zeros(do.shape))
      np.testing.assert_array_equal(partial.numpy(),np.abs(do.float().numpy()).reshape(2,512,-1).max(-1))

  def test_fused_backward_prep(self):
    from extra.thunder.amd.fa_fp8_bwd import custom_fp8_backward_prep
    rng = np.random.default_rng(28200)
    shape = (2,64,4,128)
    for magnitude,descale in ((0.,1e-12), (1e-8,1e-10), (1.,0.01), (1e4,0.01)):
      do = Tensor(rng.standard_normal(shape).astype(np.float32)*magnitude).bfloat16().realize()
      out = Tensor(rng.standard_normal(shape).astype(np.float32)).bfloat16().realize()
      scale = Tensor([descale]).realize()
      scales = Tensor.cat(Tensor([1.]),scale,Tensor([1.,1.])).contiguous().realize()
      expected_do8 = (do.float()/scale).clamp(-57344,57344).cast(dtypes.fp8e5m2).contiguous().realize()
      expected_delta = (out.float()*(expected_do8.float()*scale)).sum(-1).transpose(1,2).contiguous().realize()
      next_amax = Tensor([17.,29.]).realize()
      prepared = Tensor.custom_kernel(Tensor.empty(*shape,dtype=dtypes.fp8e5m2),
        Tensor.empty(shape[0],shape[2],shape[1]),do,out,scales,next_amax,fxn=custom_fp8_backward_prep)
      actual_do8,actual_delta = prepared[:2]
      np.testing.assert_array_equal(actual_do8.float().numpy(),expected_do8.float().numpy())
      np.testing.assert_allclose(actual_delta.numpy(),expected_delta.numpy(),rtol=2e-5,atol=max(magnitude*1e-5,1e-15))
      np.testing.assert_array_equal(prepared[5].numpy(),[0.,0.])

  def test_prescaled_derivative_and_gqa(self):
    rng = np.random.default_rng(3)
    B,N,H,Hkv,D = 1,32,4,2,128
    c = math.sqrt(D**-0.5*math.log2(math.e))
    q = Tensor(rng.standard_normal((B,N,H,D)).astype(np.float32)*c).cast(dtypes.fp8e4m3).realize()
    k = Tensor(rng.standard_normal((B,N,Hkv,D)).astype(np.float32)*c).cast(dtypes.fp8e4m3).realize()
    v = Tensor(rng.standard_normal((B,N,Hkv,D)).astype(np.float32)).cast(dtypes.fp8e4m3).realize()
    do = Tensor(rng.standard_normal((B,N,H,D)).astype(np.float32))
    qn,kn,vn = [x.float().numpy().transpose(0,2,1,3) for x in (q,k,v)]
    kn,vn = (np.repeat(x,H//Hkv,axis=1) for x in (kn,vn))
    scores = qn @ kn.swapaxes(-1,-2)
    scores = np.where(np.arange(N)[:,None]>=np.arange(N),scores,-np.inf)
    mx = scores.max(-1,keepdims=True)
    p = np.exp2(scores-mx)
    lse = mx+np.log2(p.sum(-1,keepdims=True))
    p /= p.sum(-1,keepdims=True)
    out = (p@vn).transpose(0,2,1,3)
    one = Tensor([1.])
    actual = fp8_backward_reference(q,k,v,one,do,Tensor(out),Tensor(lse[...,0]*math.log(2)),one,one,one,quantize=False)
    dn = do.numpy().transpose(0,2,1,3)
    ds = p*((dn@vn.swapaxes(-1,-2))-(out.transpose(0,2,1,3)*dn).sum(-1,keepdims=True))
    def reduce(x): return x.reshape(B,Hkv,H//Hkv,N,D).sum(2).transpose(0,2,1,3)
    expected = ((ds@kn*math.log(2)*c).transpose(0,2,1,3),reduce(ds.swapaxes(-1,-2)@qn*math.log(2)*c),reduce(p.swapaxes(-1,-2)@dn))
    for a,b in zip(actual,expected): np.testing.assert_allclose(a.numpy(),b,atol=2e-5,rtol=2e-4)

  def test_fp8_formats_and_clipping(self):
    x = Tensor([-1e6,-1.,0.,1.,1e6])
    for dtype,limit in [(dtypes.fp8e4m3,448),(dtypes.fp8e5m2,57344)]:
      np.testing.assert_array_equal(round_fp8(x,Tensor([2.]),dtype).numpy(),[-2*limit,-1,0,1,2*limit])

class TestHipKittensFP8Backward(unittest.TestCase):
  def assert_gradients(self,actual,expected,peak_tolerance=0.002):
    for a,b in zip(actual,expected):
      aa,bb = a.numpy(),b.numpy()
      # exp2/reduction rounding can cross FP8 boundaries. Check both peak
      # and aggregate error; the latter remains below 0.2% in every case.
      self.assertTrue(np.isfinite(aa).all())
      self.assertLess(np.max(np.abs(aa-bb))/max(np.max(np.abs(bb)),1e-20),peak_tolerance)
      self.assertLess(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-20),0.002)
  def test_matches_quantized_reference(self):
    for N,H,Hkv in [(64,2,1),(128,4,2),(256,4,1)]:
      with self.subTest(N=N,H=H,Hkv=Hkv): self.run_case(N,H,Hkv)

  def test_jit_scale_inputs(self): self.run_case(128,4,2,jit=True)

  def test_delayed_scale_updates(self): self.run_case(128,4,2,delayed=True)

  def test_data_parallel_local_scales(self): self.run_case(64,4,2,dp=True)

  def test_small_gradient_bootstrap(self): self.run_case(128,4,2,bootstrap=True)

  def test_native_bf16_accumulation(self):
    for N in (64,128,256):
      with self.subTest(N=N): self.run_case(N,4,2,native=True)
    self.run_case(256,4,2,native=True,bootstrap=True)

  def test_multiple_kv_tiles(self):
    self.run_case(512,4,2)
    self.run_case(512,4,2,native=True)

  def test_saturating_conversions(self):
    self.run_case(256,4,2,clipping=True)
    self.run_case(256,4,2,clipping=True,native=True)

  def run_case(self,N,H,Hkv,jit=False,delayed=False,dp=False,bootstrap=False,native=False,clipping=False):
    if Device[Device.DEFAULT].renderer.target.arch != "gfx950": self.skipTest("requires gfx950")
    from extra.thunder.amd.fa_fp8_bwd import fp8_backward
    rng = np.random.default_rng(17)
    B,D = (2 if dp else 1),128
    q,k,v = [Tensor(rng.standard_normal(s).astype(np.float32)*0.3).cast(dtypes.fp8e4m3).contiguous().realize()
             for s in [(B,N,H,D),(B,N,Hkv,D),(B,N,Hkv,D)]]
    vs,ps,dss = [Tensor([x]).realize() for x in ([0.37,1e-5,1e-9] if clipping else [0.37,1/448,1e-5])]
    do = Tensor(rng.standard_normal((B,N,H,D)).astype(np.float32)*0.2).bfloat16().realize()
    if bootstrap:
      do = (do.float()*1e-8).bfloat16().realize()
      dss = Tensor([0.]).realize()
    if dp: do = (do.float()*Tensor([1.,37.]).reshape(B,1,1,1)).bfloat16().realize()
    qn,kn,vn = [x.float().numpy().transpose(0,2,1,3) for x in (q,k,v)]
    kn,vn = (np.repeat(x,H//Hkv,axis=1) for x in (kn,vn))
    scores = qn@kn.swapaxes(-1,-2)
    scores = np.where(np.arange(N)[:,None]>=np.arange(N),scores,-np.inf)
    mx = scores.max(-1,keepdims=True)
    pn = np.exp2(scores-mx)
    ln = mx+np.log2(pn.sum(-1,keepdims=True))
    pn /= pn.sum(-1,keepdims=True)
    out = Tensor((pn@(vn*0.37)).transpose(0,2,1,3).copy()).bfloat16().realize()
    lse = Tensor((ln[...,0]*math.log(2)).copy()).realize()
    inputs = [q,k,v,vs,do,out,lse,ps,dss]
    if dp:
      devices = (Device.DEFAULT,f"{Device.DEFAULT.split(':')[0]}:1")
      inputs = [x.shard(devices,axis=0 if i in (0,1,2,4,5,6) else None).contiguous().realize() for i,x in enumerate(inputs)]
    actual = fp8_backward(*inputs,native=native)
    Tensor.realize(*actual)
    if dp:
      # Compare each shard with its single-device invocation as well as the
      # mathematical reference, so a global amax cannot hide in tolerances.
      for shard in range(B):
        local = [x[shard:shard+1].contiguous() if i in (0,1,2,4,5,6) else x for i,x in enumerate([q,k,v,vs,do,out,lse,ps,dss])]
        expected_local = fp8_backward(*local)
        Tensor.realize(*expected_local)
        for a,b in zip(actual,expected_local):
          aa = a.numpy()[shard:shard+1]
          np.testing.assert_allclose(aa,b.numpy(),rtol=1e-5,atol=1e-6)
    dos = (do.float().abs().max(axis=(1,2,3),keepdim=True)+1e-8)/57344. if dp else ((do.float().abs().max()+1e-8)/57344.).reshape(1)
    effective_dss = 4*D*dos*vs*448. if bootstrap else dss
    reference = fp8_backward_reference(q,k,v,vs,do,out,lse,dos,ps,effective_dss)
    # Stress cases (dS clipping and tiny gradients) produce sparse rounding outliers.
    if native:
      # Packed BF16 atomics round each dQ contribution, as in the ASM BF16 backward.
      for a,b in zip(actual[:3],reference[:3]):
        aa,bb = a.float().numpy(),b.numpy()
        self.assertTrue(np.isfinite(aa).all())
        self.assertLess(np.linalg.norm(aa-bb)/np.linalg.norm(bb),0.01)
        self.assertLess(np.max(np.abs(aa-bb))/np.max(np.abs(bb)),0.02)
    else: self.assert_gradients(actual[:3],reference[:3],peak_tolerance=0.02 if dp or bootstrap or clipping else 0.002)
    np.testing.assert_allclose(actual[3].numpy().reshape(-1,2).max(0),[r.item() for r in reference[3:]],rtol=2e-4,atol=1e-6)
    if bootstrap:
      unquantized = fp8_backward_reference(q,k,v,vs,do,out,lse,dos,ps,effective_dss,quantize=False)
      for a,b in zip(actual[:3],unquantized[:3]):
        self.assertLess(np.linalg.norm(a.numpy()-b.numpy())/np.linalg.norm(b.numpy()),0.1)
    if delayed:
      current = Tensor([1.,0.],dtype=dtypes.float32).realize()
      nxt = Tensor.zeros(2,dtype=dtypes.float32).contiguous().realize()
      @TinyJit
      def train_backward(q,k,v,vs,do,out,lse,current,nxt):
        nxt.assign(0)
        result = fp8_backward(q,k,v,vs,do,out,lse,(current[0]+1e-8)/448.,current[1]/57344.,nxt,native=True)
        Tensor.realize(*result,nxt)
        return result
      for magnitude in [1.,1.,0.5,2.]:
        old = current.numpy().copy()
        incoming = (do.float()*magnitude).bfloat16().contiguous().realize()
        result = train_backward(q,k,v,vs,incoming,out,lse,current,nxt)
        dos = ((incoming.float().abs().max()+1e-8)/57344.).reshape(1)
        ds_expected = Tensor([float(old[1]/57344.)]) if old[1] > 0 else 4*D*dos*vs*448.
        expected = fp8_backward_reference(q,k,v,vs,incoming,out,lse,dos,Tensor([float((old[0]+1e-8)/448.)]),ds_expected)
        for a,b in zip(result[:3],expected[:3]):
          aa,bb = a.float().numpy(),b.numpy()
          self.assertTrue(np.isfinite(aa).all())
          self.assertLess(np.linalg.norm(aa-bb)/np.linalg.norm(bb),0.01)
        np.testing.assert_allclose(nxt.numpy(),[x.item() for x in expected[3:]],rtol=2e-4,atol=1e-6)
        np.testing.assert_array_equal(current.numpy(),old)
        current.assign(nxt).realize()
    if jit:
      @TinyJit
      def run(q,k,v,vs,do,out,lse,ps,dss):
        ret = fp8_backward(q,k,v,vs,do,out,lse,ps,dss)
        Tensor.realize(*ret)
        return ret
      for scale in [1e-5,1e-5,2e-5,5e-6]:
        ds_input = Tensor([scale]).realize()
        result = run(q,k,v,vs,do,out,lse,ps,ds_input)
        expected = fp8_backward_reference(q,k,v,vs,do,out,lse,dos,ps,ds_input)
        self.assert_gradients(result[:3],expected[:3])

if __name__ == '__main__': unittest.main()
