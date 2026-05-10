from __future__ import annotations
import os, numpy as np
from typing import Any
from fastmcp import FastMCP

from tinygrad import Tensor, Device
from extra.agentic_opt.models import CandidateEvaluation, CorrectnessResult, HardwareDescriptor, Kernel, KernelDescriptor, KernelRuntimeProfile

class AgenticOpt:
  def __init__(self, device:str, reference_kernel:KernelDescriptor):
    self.device = Device.canonicalize(device)
    self.hardware = HardwareDescriptor.from_dev(device)
    self.reference_kernel = reference_kernel

    # NOTE: assumming all buffers are purely in or out, not both
    self.test_bufs = [Tensor.rand(*arg.shape, dtype=arg.dtype, device=self.device).realize() for arg in self.reference_kernel.buffer_args]
    
    self.correct_bufs = [buf.clone().realize() for buf in self.test_bufs]
    t = self._run_kernel(self.reference_kernel.kernel, self.correct_bufs)
    ref_eval = CandidateEvaluation(candidate=self.reference_kernel.kernel, correctness=CorrectnessResult(passed=True), profile=KernelRuntimeProfile(runtime_ms=t))

    self.best_evaluation: CandidateEvaluation = ref_eval
    self.history: list[CandidateEvaluation] = [self.best_evaluation]

  def _run_kernel(self, kernel:Kernel, args: list[Tensor]) -> float:
    dev = Device[self.device]
    lib = dev.compiler.compile(kernel.source)
    rt = dev.runtime(self.reference_kernel.entry_point, lib)
    bufs = [arg._buffer() for arg in args]
    t = rt(*[buf._buf for buf in bufs], global_size=self._dim3(kernel.global_size), local_size=self._dim3(kernel.local_size or (1,)), vals=(), wait=True)
    assert isinstance(t, float)
    dev.synchronize()
    return t * 1000

  @staticmethod
  def _dim3(dims: tuple[int, ...]) -> tuple[int, int, int]:
    if len(dims) > 3: raise ValueError(f"expected at most 3 launch dimensions, got {dims}")
    return (dims + (1, 1, 1))[:3]

  def evaluate_kernel(self, candidate:Kernel) -> CandidateEvaluation:
    bufs = [buf.clone().realize() for buf in self.test_bufs]
    try:
      t = self._run_kernel(candidate, bufs)
      for ref_buf, buf in zip(self.correct_bufs, bufs):
        np.testing.assert_allclose(ref_buf.numpy(), buf.numpy())
      eval = CandidateEvaluation(candidate=candidate, profile=KernelRuntimeProfile(runtime_ms=t), correctness=CorrectnessResult(passed=True))
      if eval.profile.runtime_ms < self.best_evaluation.profile.runtime_ms:
        self.best_evaluation = eval
    except Exception as exc:
      eval = CandidateEvaluation(candidate=candidate, correctness=CorrectnessResult(passed=False, message=str(exc)))
    self.history.append(eval)
    return eval

  def get_history(self) -> tuple[CandidateEvaluation, ...]:
    return tuple(self.history)

  def create_server(self, name:str="agentic-opt") -> Any:
    mcp = FastMCP(name)

    @mcp.tool()
    def get_hardware_descriptor() -> HardwareDescriptor: return self.hardware

    @mcp.tool()
    def get_reference_kernel() -> KernelDescriptor: return self.reference_kernel

    @mcp.tool()
    def evaluate_kernel(candidate:Kernel) -> CandidateEvaluation: return self.evaluate_kernel(candidate)

    @mcp.tool()
    def get_history() -> list[CandidateEvaluation]: return self.history

    return mcp

def llama2_70b_lora_dummy_step(
  bs:int=1,
  seqlen:int=128,
  layers:int|None=1,
  device:str|None="NULL",
  quantize:bool|None=False,
  lr:float=1e-4,
  seed:int=1337,
  realize_model:bool=False,
) -> dict[str, Any]:
  """
  Build a flat Llama2-70B-LoRA model with dummy token data, create the MLPerf
  LoRA optimizer, and run one dummy training/optimizer step.

  Defaults are intentionally small/safe for scratch work: the model uses the
  70B hidden/head dimensions, but only one layer unless `layers=None` or a
  different value is passed. Set `device=None` to use the existing tinygrad
  default device.
  """
  os.environ.setdefault("LORA", "1")
  os.environ.setdefault("LOAD_MODEL", "0")
  os.environ.setdefault("AGENTIC_OPT", "1")
  os.environ.setdefault("FUSED_INPUT_QUANTIZE", "0")
  os.environ.setdefault("FUSED_ADD_NORM_MUL_QUANTIZE", "0")
  os.environ.setdefault("FUSED_SILU_W13", "0")
  os.environ.setdefault("FUSED_PAD_GRAD_ACCUM", "0")
  if device is not None: os.environ.setdefault("DEV", device)
  if quantize is not None: os.environ["QUANTIZE"] = "1" if quantize else "0"

  from tinygrad import Tensor, dtypes
  from tinygrad.nn.state import get_parameters, get_state_dict
  from examples.mlperf.model_train import LLAMA2_70B_ARGS
  from examples.mlperf.optim import GradAccClipAdamW
  from examples.mlperf.models import flat_llama

  if not flat_llama.LORA:
    raise RuntimeError("flat_llama was imported with LORA=0. Run this in a fresh process with LORA=1.")

  Tensor.manual_seed(seed)
  model_params = dict(LLAMA2_70B_ARGS)
  if layers is not None: model_params["n_layers"] = layers

  model = flat_llama.FlatTransformer(**model_params, max_context=seqlen)

  params = get_parameters(model)
  for p in params:
    if not p.requires_grad: p.requires_grad_(False)

  optim = GradAccClipAdamW(params, lr=lr, b1=0.9, b2=0.95, eps=1e-5, weight_decay=0.1, grad_acc=1, clip_norm=1.0)
  for p in optim.params:
    grad_dtype = dtypes.bfloat16 if p.dtype == flat_llama.FP8_DTYPE else p.dtype
    p.grad = p.zeros_like(dtype=grad_dtype).contiguous()
  grads = [p.grad for p in optim.params]

  state = get_state_dict(model)
  if realize_model:
    Tensor.realize(*state.values(), *optim.params, *optim.buffers)

  tokens = Tensor.randint(bs, seqlen, low=0, high=model.vocab_size, dtype=dtypes.int)

  with Tensor.train():
    logits = model(tokens)[:, :-1]
    loss = logits.sparse_categorical_crossentropy(tokens[:, 1:])
    for grad_buf, new_grad in zip(grads, loss.gradient(*optim.params)):
      flat_llama.apply_grad(grad_buf, new_grad.uop)

  loss_cpu = loss.flatten().float().to("CPU")
  Tensor.realize(loss_cpu, *grads)

  with Tensor.train():
    grad_norm = optim.fstep(grads)
  grad_norm_cpu = grad_norm.flatten().float().to("CPU")
  Tensor.realize(grad_norm_cpu)

  return {
    "model": model,
    "optimizer": optim,
    "tokens": tokens,
    "loss": loss_cpu,
    "grad_norm": grad_norm_cpu,
    "model_params": model_params,
    "trainable_params": optim.params,
    "state": state,
  }


if __name__ == "__main__":
  out = llama2_70b_lora_dummy_step()
  print({"loss": out["loss"].item(), "grad_norm": out["grad_norm"].item(), "trainable_params": len(out["trainable_params"])})
