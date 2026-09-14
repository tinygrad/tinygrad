import functools, pathlib, unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops
from extra.llama_kernels.rmsnorm import rmsnorm_mul_mxfp4, rmsnorm_add_mul_mxfp4

def names(linear):
  return [p.src[0].arg.name for c in linear.src if (p:=c.without_after.src[0]).op is Ops.PROGRAM]

@functools.cache
def original_row_backward(*args):
  from extra.llama_kernels import compile_hip
  from extra.llama_kernels.rmsnorm import _rmsnorm_add_mul_bwd_mxfp4_row
  program = _rmsnorm_add_mul_bwd_mxfp4_row(*args)
  root = pathlib.Path(__file__).parents[2]/"extra/llama_kernels"
  source = (root/"rmsnorm/rmsnorm_add_mul_bwd.hip").read_text()
  defines = ["-DROWS=16384", "-DHIDDEN=4096", "-DNUM_WG=1024", "-DTHREADS=128", "-DWRITE_MXFP4_ROW=1",
             f"-I{root/'quantize_mxfp4'}"]
  return UOp(Ops.PROGRAM,src=(*program.src[:2],UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=compile_hip(source,defines))))

@functools.cache
def original_quantize(*args, row, shuffle_row, shuffle_col):
  from extra.llama_kernels import compile_hip
  from extra.llama_kernels.quantize_mxfp4 import _custom_quantize_mxfp4
  program = _custom_quantize_mxfp4(*args,write_row=row,write_col=True,shuffle_row=shuffle_row,shuffle_col=shuffle_col)
  root = pathlib.Path(__file__).parents[2]/"extra/llama_kernels/quantize_mxfp4"
  source = (root/"quantize_mxfp4.cpp").read_text()
  defines = ["-DKERNEL_NAME=original_quantize", "-DM_DIM=16384", "-DN_DIM=4096", f"-DWRITE_ROWWISE_VALUE={int(row)}",
             "-DWRITE_COLWISE_VALUE=1", f"-DSHUFFLE_ROWWISE_FP4_VALUE={int(shuffle_row)}",
             f"-DSHUFFLE_COLWISE_FP4_VALUE={int(shuffle_col)}", f"-I{root}"]
  return UOp(Ops.PROGRAM,src=(*program.src[:2],UOp(Ops.SOURCE,arg=source),UOp(Ops.BINARY,arg=compile_hip(source,defines))))

@unittest.skipUnless(Device.DEFAULT.startswith("NULL"), "NULL graph tests")
class TestMXFP4QuantizationGraph(unittest.TestCase):
  def test_norm_gradient_keeps_production_backward(self):
    from extra.gemm.cdna_asm_gemm import asm_gemm
    x = Tensor.empty(16384,4096,dtype=dtypes.bfloat16)
    w = Tensor.empty(4096,4096,dtype=dtypes.bfloat16)
    z = asm_gemm(x,w.T,mxfp4=True)
    y,h,_,quant = rmsnorm_add_mul_mxfp4(z,Tensor.empty_like(z),Tensor.empty(4096,dtype=dtypes.bfloat16),1e-5,quantized_only=True)
    out = asm_gemm(y,Tensor.empty(256,4096,dtype=dtypes.bfloat16).T,mxfp4=True,mxfp4_x=quant)
    dw = (out.sum()+h.sum()).gradient(w)[0]
    kernels = names(dw.schedule_linear())
    self.assertIn("rmsnorm_add_mul_bwd_mxfp4_row_16384_4096",kernels)
    self.assertIn("quantize_mxfp4_col_16384_4096",kernels)

  def test_norm_backward_uses_produced_column_input(self):
    from extra.gemm.cdna_asm_gemm import asm_gemm
    for residual in (False,True):
      x = Tensor.empty(16384,4096,dtype=dtypes.bfloat16)
      norm = Tensor.empty(4096,dtype=dtypes.bfloat16)
      if residual: y,_,_,quant = rmsnorm_add_mul_mxfp4(x,Tensor.empty_like(x),norm,1e-5,quantized_only=True)
      else: y,_,quant = rmsnorm_mul_mxfp4(x,norm,1e-5,quantized_only=True)
      self.assertEqual([t.shape for t in quant],[(16384,2048),(16384,128),(4096,8192),(4096,512)])
      w = Tensor.empty(256,4096,dtype=dtypes.bfloat16)
      out = asm_gemm(y,w.T,mxfp4=True,mxfp4_x=quant)
      dw = out.gradient(w,gradient=Tensor.empty_like(out))[0]
      kernels = names(dw.schedule_linear())
      self.assertIn(f"rmsnorm_{'add_' if residual else ''}mul_fwd_mxfp4_dual_16384_4096",kernels)
      self.assertNotIn("quantize_mxfp4_col_16384_4096",kernels)
      self.assertNotIn("quantize_mxfp4_dual_16384_4096",kernels)

@unittest.skipUnless(Device.DEFAULT.split(":")[0] == "AMD", "requires free AMD GPU for numerical validation")
class TestRMSNormDualNumerics(unittest.TestCase):
  def test_exact_backward(self):
    from extra.llama_kernels.rmsnorm import _rmsnorm_add_mul_bwd_mxfp4_row
    from extra.llama_kernels.quantize_mxfp4 import alloc_mxfp4_row_outputs
    Tensor.manual_seed(28200)
    g,d,h = [Tensor.randn(16384,4096,dtype=dtypes.bfloat16).realize() for _ in range(3)]
    w = Tensor.randn(4096,dtype=dtypes.bfloat16).realize()
    rrms = (h.float().square().mean(-1)+1e-5).rsqrt().realize()
    outputs = [Tensor.custom_kernel(Tensor.empty_like(h),Tensor.empty(1024,4096,dtype=dtypes.float32),
      *alloc_mxfp4_row_outputs(h),g,d,h,rrms,w,fxn=fxn)[:4] for fxn in (original_row_backward,_rmsnorm_add_mul_bwd_mxfp4_row)]
    for a,b in zip(*outputs): self.assertEqual((a!=b).sum().item(),0)

  def test_exact_quantize_layouts(self):
    from extra.llama_kernels.quantize_mxfp4 import alloc_mxfp4_outputs, quantize_mxfp4
    Tensor.manual_seed(28200)
    x = Tensor.randn(16384,4096,dtype=dtypes.bfloat16).realize()
    for row,shuffle_row,shuffle_col in ((False,False,False),(False,False,True),(True,False,False),(True,False,True),(True,True,True)):
      reference = Tensor.custom_kernel(*alloc_mxfp4_outputs(x),x,
        fxn=functools.partial(original_quantize,row=row,shuffle_row=shuffle_row,shuffle_col=shuffle_col))[:4]
      actual = quantize_mxfp4(x,row=row,shuffle_row=shuffle_row,shuffle_col=shuffle_col)
      for a,b in zip(reference[0 if row else 2:],actual[0 if row else 2:]): self.assertEqual((a!=b).sum().item(),0)

  def test_exact_dual_outputs(self):
    from extra.llama_kernels.quantize_mxfp4 import quantize_mxfp4
    Tensor.manual_seed(28200)
    x,residual = [Tensor.randn(16384,4096,dtype=dtypes.bfloat16).realize() for _ in range(2)]
    weight = Tensor.randn(4096,dtype=dtypes.bfloat16).realize()
    for add in (False,True):
      if add:
        old,h_old,rrms_old,row_old = rmsnorm_add_mul_mxfp4(x,residual,weight,1e-5)
        _,h_new,rrms_new,quant = rmsnorm_add_mul_mxfp4(x,residual,weight,1e-5,quantized_only=True)
      else:
        old,rrms_old,row_old = rmsnorm_mul_mxfp4(x,weight,1e-5)
        _,rrms_new,quant = rmsnorm_mul_mxfp4(x,weight,1e-5,quantized_only=True)
      reference = (*row_old[:2],*quantize_mxfp4(old,row=False,shuffle_col=True)[2:])
      for a,b in zip(quant,reference): self.assertEqual((a!=b).sum().item(),0)
      self.assertEqual((rrms_old!=rrms_new).sum().item(),0)
      if add: self.assertEqual((h_old!=h_new).sum().item(),0)

if __name__ == '__main__': unittest.main()
