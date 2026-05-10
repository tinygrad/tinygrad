from __future__ import annotations
import os
from typing import Any
from fastmcp import FastMCP
from extra.agentic_opt.models import CandidateEvaluation, HardwareDescriptor, Kernel, KernelDescriptor
class AgenticOpt:
  def __init__(self, device:str, reference_kernel:KernelDescriptor):
    self.hardware = HardwareDescriptor.from_dev(device)
    self.reference_kernel = reference_kernel
    self.history: list[CandidateEvaluation] = []
    self.best_evaluation: CandidateEvaluation|None = None

  def evaluate_kernel(self, candidate:Kernel) -> CandidateEvaluation:
    ...
    # TODO: run, time, and verify kernel, add to history, save best kernel
    # evaluation = self.evaluator(candidate)
    # self.history.append(evaluation)
    # if self._is_better(evaluation, self.best_evaluation): self.best_evaluation = evaluation
    # return evaluation

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

  @staticmethod
  def _is_better(candidate:CandidateEvaluation, best:CandidateEvaluation|None) -> bool:
    if not candidate.correctness.passed or candidate.runtime is None: return False
    if best is None or best.runtime is None: return True
    return candidate.runtime.runtime_ms < best.runtime.runtime_ms

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
