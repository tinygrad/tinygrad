from __future__ import annotations
import os, numpy as np
from typing import Any
from fastmcp import FastMCP

from tinygrad import Tensor, Device
from extra.agentic_opt.models import AgenticOptResult, CandidateEvaluation, CorrectnessResult, HardwareDescriptor, Kernel, KernelDescriptor, KernelRuntimeProfile

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

  def create_agent(self, model:Any|None=None) -> Any:
    from pydantic_ai import Agent

    agent = Agent(model, output_type=Kernel, instructions=self._agent_instructions(), name="agentic-kernel-optimizer")

    @agent.tool_plain
    def get_hardware_descriptor() -> HardwareDescriptor:
      """Return the target hardware and compiler descriptor."""
      return self.hardware

    @agent.tool_plain
    def get_reference_kernel() -> KernelDescriptor:
      """Return the reference kernel source, launch geometry, type family, and buffer argument metadata."""
      return self.reference_kernel

    @agent.tool_plain
    def get_history() -> list[CandidateEvaluation]:
      """Return all candidate evaluations so far, including failures."""
      return self.history

    return agent

  def optimize_kernel(self, model:Any|None=None, max_iterations:int=10, patience:int=3, target_speedup:float|None=None,
                      max_failures:int|None=None, prompt:str|None=None) -> AgenticOptResult:
    agent = self.create_agent(model)
    messages, stale_iters, failures = None, 0, 0
    history_start = len(self.history)
    stop_reason = "max_iterations"

    for iteration in range(max_iterations):
      best_before = self.best_evaluation.profile.runtime_ms
      result = agent.run_sync(self._iteration_prompt(iteration, target_speedup, prompt), message_history=messages)
      messages = result.all_messages()

      evaluation = self.evaluate_kernel(result.output)
      if evaluation.correctness.passed:
        stale_iters = 0 if self.best_evaluation.profile.runtime_ms < best_before else stale_iters + 1
      else:
        failures += 1
        stale_iters += 1

      if target_speedup is not None and self._speedup(self.best_evaluation) >= target_speedup:
        stop_reason = "target_speedup"
        break
      if max_failures is not None and failures >= max_failures:
        stop_reason = "max_failures"
        break
      if patience > 0 and stale_iters >= patience:
        stop_reason = "patience"
        break

    return AgenticOptResult(best=self.best_evaluation, history=tuple(self.history), iterations=len(self.history)-history_start, stop_reason=stop_reason)

  def _speedup(self, evaluation:CandidateEvaluation) -> float:
    assert self.history[0].profile is not None and evaluation.profile is not None
    return self.history[0].profile.runtime_ms / evaluation.profile.runtime_ms

  def _iteration_prompt(self, iteration:int, target_speedup:float|None, prompt:str|None) -> str:
    best_ms = self.best_evaluation.profile.runtime_ms
    best_speedup = self._speedup(self.best_evaluation)
    extra = f"\nAdditional caller guidance:\n{prompt}\n" if prompt is not None else ""
    target = f" Target speedup is {target_speedup:.4g}x versus the reference." if target_speedup is not None else ""
    return (
      f"Optimization iteration {iteration + 1}.{target}\n"
      f"Current best runtime is {best_ms:.6f} ms ({best_speedup:.4g}x versus reference).\n"
      "Use the available tools to inspect the target, reference kernel, and candidate history. "
      "Return exactly one Kernel candidate. The runner will compile, execute, verify, profile, and record it after your response."
      f"{extra}"
    )

  @staticmethod
  def _agent_instructions() -> str:
    return (
      "You optimize a single tinygrad-rendered kernel by proposing replacement kernel source code.\n"
      "Your output must be a Kernel object with source, global_size, and optional local_size.\n"
      "Preserve the reference kernel entry point name, function signature, argument order, argument count, and externally visible semantics.\n"
      "You may change the implementation and launch geometry.\n"
      "Do not add new global buffers or require new arguments.\n"
      "Use get_hardware_descriptor, get_reference_kernel, and get_history before proposing candidates.\n"
      "Use candidate history to avoid repeating failed compiles, incorrect kernels, or slower variants.\n"
      "Prefer small, justified changes early; after correctness passes, optimize for lower runtime_ms.\n"
    )

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
