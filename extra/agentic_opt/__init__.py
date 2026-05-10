from __future__ import annotations
import os, time, numpy as np
from typing import Any

from tinygrad import Tensor, Device
from extra.agentic_opt.models import AgenticOptResult, BufferArg, CandidateEvaluation, CorrectnessResult, HardwareDescriptor, Kernel, KernelDescriptor, KernelRuntimeProfile

class AgenticOpt:
  def __init__(self, device:str, reference_kernel:KernelDescriptor, test_bufs:list[Tensor]|None=None, correct_bufs:list[Tensor]|None=None,
               reference_runtime_ms:float|None=None):
    self.device = Device.canonicalize(device)
    self.hardware = HardwareDescriptor.from_dev(device)
    self.reference_kernel = reference_kernel

    # NOTE: assumming all buffers are purely in or out, not both
    self.test_bufs = test_bufs if test_bufs is not None else \
      [Tensor.rand(*arg.shape, dtype=arg.dtype, device=self.device).realize() for arg in self.reference_kernel.buffer_args]
    
    self.correct_bufs = correct_bufs if correct_bufs is not None else [buf.clone().realize() for buf in self.test_bufs]
    if correct_bufs is None:
      reference_runtime_ms = self._run_kernel(self.reference_kernel.kernel, self.correct_bufs)
      correctness = CorrectnessResult(passed=True)
    else:
      correctness = CorrectnessResult(passed=True, message="external correctness outputs")
    profile = KernelRuntimeProfile(runtime_ms=reference_runtime_ms if reference_runtime_ms is not None else float("inf"))
    ref_eval = CandidateEvaluation(candidate=self.reference_kernel.kernel, correctness=correctness, profile=profile)

    self.best_evaluation: CandidateEvaluation = ref_eval
    self.history: list[CandidateEvaluation] = [self.best_evaluation]

  def _run_kernel(self, kernel:Kernel, args: list[Tensor]) -> float:
    if self.reference_kernel.entry_point not in kernel.source:
      raise RuntimeError(f"candidate source does not define required entry point {self.reference_kernel.entry_point!r}")
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
        np.testing.assert_allclose(ref_buf.numpy(), buf.numpy(), rtol=float(os.getenv("AGENTIC_RTOL", "1e-3")),
                                   atol=float(os.getenv("AGENTIC_ATOL", "1e-2")))
      eval = CandidateEvaluation(candidate=candidate, profile=KernelRuntimeProfile(runtime_ms=t), correctness=CorrectnessResult(passed=True))
      if eval.profile.runtime_ms < self.best_evaluation.profile.runtime_ms:
        self.best_evaluation = eval
    except Exception as exc:
      eval = CandidateEvaluation(candidate=candidate, correctness=CorrectnessResult(passed=False, message=str(exc)))
    self.history.append(eval)
    return eval

  def create_agent(self, model:Any|None=None) -> Any:
    from pydantic_ai import Agent
    from pydantic_ai.builtin_tools import WebSearchTool

    builtin_tools = [WebSearchTool()] if os.getenv("AGENTIC_WEB_SEARCH", "1") != "0" else []
    if builtin_tools and isinstance(model, str) and model.startswith("openai:"):
      from pydantic_ai.models.openai import OpenAIResponsesModel
      model = OpenAIResponsesModel(model.removeprefix("openai:"))

    agent = Agent(model, output_type=Kernel, instructions=self._agent_instructions(), name="agentic-kernel-optimizer", builtin_tools=builtin_tools)

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
      self._print_evaluation(iteration, evaluation)
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

  def _print_evaluation(self, iteration:int, evaluation:CandidateEvaluation):
    if evaluation.profile is not None:
      speedup = self._speedup(evaluation)
      print(f"candidate {iteration+1}: passed={evaluation.correctness.passed} runtime={evaluation.profile.runtime_ms:.6f} ms speedup={speedup:.4f}x")
    else:
      msg = "" if evaluation.correctness.message is None else f" error={evaluation.correctness.message[:500]}"
      print(f"candidate {iteration+1}: passed={evaluation.correctness.passed}{msg}")

  def _speedup(self, evaluation:CandidateEvaluation) -> float:
    assert self.history[0].profile is not None and evaluation.profile is not None
    return self.history[0].profile.runtime_ms / evaluation.profile.runtime_ms

  def _iteration_prompt(self, iteration:int, target_speedup:float|None, prompt:str|None) -> str:
    best_ms = self.best_evaluation.profile.runtime_ms
    best_speedup = self._speedup(self.best_evaluation)
    extra = f"\nAdditional caller guidance:\n{prompt}\n" if prompt is not None else ""
    target = f" Target speedup is {target_speedup:.4g}x versus the reference." if target_speedup is not None else ""
    history = "\n".join(self._history_lines(8))
    return (
      f"Optimization iteration {iteration + 1}.{target}\n"
      f"Current best runtime is {best_ms:.6f} ms ({best_speedup:.4g}x versus reference).\n"
      f"Hardware descriptor:\n{self.hardware.model_dump_json()}\n"
      f"Reference kernel descriptor:\n{self.reference_kernel.model_dump_json()}\n"
      f"Recent candidate history:\n{history}\n"
      "Return exactly one Kernel candidate. The runner will compile, execute, verify, profile, and record it after your response."
      f"{extra}"
    )

  def _history_lines(self, limit:int) -> list[str]:
    lines = []
    for i, ev in enumerate(self.history[-limit:]):
      if ev.profile is not None:
        lines.append(f"{i}: passed={ev.correctness.passed} runtime_ms={ev.profile.runtime_ms:.6f} speedup={self._speedup(ev):.4f}")
      else:
        msg = "" if ev.correctness.message is None else ev.correctness.message[:300].replace("\n", "\\n")
        lines.append(f"{i}: passed={ev.correctness.passed} runtime_ms=None error={msg}")
    return lines

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
      "Search the internet for reference kernels and documentation"
    )

def render_gemm_kernel_descriptor(n:int=2048, device:str="METAL", dtype:str="bfloat16") -> KernelDescriptor:
  from tinygrad.engine.realize import compile_linear
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import Ops
  from extra.agentic_opt.gemm import gemm

  device = Device.canonicalize(device)
  with Context(BEAM=0, CACHELEVEL=0):
    a = Tensor.empty(n, n, dtype=dtype, device=device)
    b = Tensor.empty(n, n, dtype=dtype, device=device)
    linear = compile_linear(gemm(a, b).schedule_linear(), beam=0)

  if len(linear.src) != 1: raise RuntimeError(f"expected one GEMM kernel, got {len(linear.src)}")
  prg = linear.src[0].src[0]
  source = next(x.arg for x in prg.src if x.op is Ops.SOURCE)

  return KernelDescriptor(
    entry_point=prg.arg.function_name,
    type_family="gemm",
    kernel=Kernel(source=source, global_size=tuple(int(x) for x in prg.arg.global_size),
                  local_size=tuple(int(x) for x in prg.arg.local_size) if prg.arg.local_size is not None else None),
    buffer_args=[
      # Tensor.custom_kernel sees the unsqueezed A/output buffers for 2D GEMM.
      BufferArg(shape=(1, n, n), dtype=dtype),
      BufferArg(shape=(1, n, n), dtype=dtype),
      BufferArg(shape=(n, n), dtype=dtype),
    ],
    description=f"Square GEMM C = A @ B for N={n}. Reference mode=naive. Buffer ABI is C[1,N,N], A[1,N,N], B[N,N].",
  )

def _extract_tinyjit_gemm_descriptor(matmul:Any, n:int, dtype:str, reference:str) -> KernelDescriptor:
  from tinygrad.uop.ops import Ops

  if matmul.captured is None: raise RuntimeError("TinyJit did not capture a GEMM kernel")
  calls = [call for call in matmul.captured.linear.src if call.src[0].op is Ops.PROGRAM]
  if len(calls) != 1: raise RuntimeError(f"expected one captured GEMM program, got {len(calls)}")
  prg = calls[0].src[0]
  source = next((x.arg for x in prg.toposort() if x.op is Ops.SOURCE), None)
  if source is None: raise RuntimeError("captured GEMM program has no rendered source")
  global_size, local_size = prg.arg.launch_dims({})

  return KernelDescriptor(
    entry_point=prg.arg.function_name,
    type_family="gemm",
    kernel=Kernel(source=source, global_size=tuple(int(x) for x in global_size),
                  local_size=tuple(int(x) for x in local_size) if local_size is not None else None),
    buffer_args=[
      BufferArg(shape=(1, n, n), dtype=dtype),
      BufferArg(shape=(1, n, n), dtype=dtype),
      BufferArg(shape=(n, n), dtype=dtype),
    ],
    description=f"Square GEMM C = A @ B for N={n}. Reference mode={reference}; source is the settled TinyJit/JITBEAM matmul kernel. "
                "Buffer ABI is C[1,N,N], A[1,N,N], B[N,N].",
  )

def _time_tinyjit_gemm_reference(n:int, device:str, dtype:str, jitbeam:int, warmup:int, repeats:int) -> tuple[list[Tensor], list[Tensor], float, KernelDescriptor]:
  from tinygrad import TinyJit
  from tinygrad.helpers import Context

  device = Device.canonicalize(device)
  out = Tensor.empty(1, n, n, dtype=dtype, device=device).realize()
  a = Tensor.rand(1, n, n, dtype=dtype, device=device).realize()
  b = Tensor.rand(n, n, dtype=dtype, device=device).realize()

  @TinyJit
  def matmul(x:Tensor, y:Tensor) -> Tensor:
    return (x.squeeze(0) @ y).reshape(1, n, n).realize()

  old_jitbeam = os.environ.get("JITBEAM")
  os.environ["JITBEAM"] = str(jitbeam)
  times: list[float] = []
  try:
    with Context(BEAM=jitbeam, CACHELEVEL=0):
      correct = None
      for _ in range(max(warmup, 2)):
        correct = matmul(a, b)
        Device[device].synchronize()
      reference_kernel = _extract_tinyjit_gemm_descriptor(matmul, n, dtype, "beam")
      for _ in range(repeats):
        st = time.perf_counter()
        correct = matmul(a, b)
        Device[device].synchronize()
        times.append((time.perf_counter() - st) * 1000)
  finally:
    if old_jitbeam is None: os.environ.pop("JITBEAM", None)
    else: os.environ["JITBEAM"] = old_jitbeam

  assert correct is not None
  return [out, a, b], [correct.realize(), a.clone().realize(), b.clone().realize()], sorted(times)[len(times)//2], reference_kernel

def _main_gemm_optimization():
  device = os.getenv("AGENTIC_DEVICE", os.getenv("DEV", "METAL"))
  n = int(os.getenv("AGENTIC_N", "2048"))
  dtype = os.getenv("AGENTIC_DTYPE", "bfloat16")
  jitbeam = int(os.getenv("AGENTIC_JITBEAM", os.getenv("JITBEAM", "3")))
  warmup = int(os.getenv("AGENTIC_WARMUP", "1"))
  repeats = int(os.getenv("AGENTIC_REPEATS", "3"))
  max_iterations = int(os.getenv("AGENTIC_MAX_ITERATIONS", "10"))
  patience = int(os.getenv("AGENTIC_PATIENCE", "3"))
  max_failures = int(os.getenv("AGENTIC_MAX_FAILURES", "5"))
  target_speedup = float(x) if (x:=os.getenv("AGENTIC_TARGET_SPEEDUP")) else None
  model = os.getenv("AGENTIC_MODEL")
  if model is None and (vendor:=os.getenv("AGENTIC_VENDOR")) and (model_name:=os.getenv("AGENTIC_MODEL_NAME")):
    model = f"{vendor}:{model_name}"
  reference = os.getenv("AGENTIC_REFERENCE", "naive")
  if reference not in ("naive", "beam"): raise ValueError(f"unknown AGENTIC_REFERENCE={reference!r}, expected 'naive' or 'beam'")

  if model is None:
    raise SystemExit("set AGENTIC_MODEL, for example AGENTIC_MODEL=openai:gpt-5.4 or AGENTIC_MODEL=anthropic:claude-sonnet-4-5")

  print(f"timing tinygrad baseline with TinyJit/JITBEAM={jitbeam}")
  test_bufs, correct_bufs, baseline_ms, beam_kernel = _time_tinyjit_gemm_reference(n, device, dtype, jitbeam, warmup, repeats)
  print(f"tinygrad baseline median: {baseline_ms:.6f} ms")

  if reference == "beam":
    reference_kernel = beam_kernel
  else:
    print(f"rendering generic UOp GEMM for {device=} {n=} {dtype=} {reference=}")
    reference_kernel = render_gemm_kernel_descriptor(n=n, device=device, dtype=dtype)
  print(f"reference entry={reference_kernel.entry_point} global={reference_kernel.kernel.global_size} local={reference_kernel.kernel.local_size}")

  opt = AgenticOpt(device, reference_kernel, test_bufs=test_bufs, correct_bufs=correct_bufs, reference_runtime_ms=baseline_ms)
  source_note = "the settled TinyJit/JITBEAM matmul kernel" if reference == "beam" else "the generic UOp GEMM scaffold"
  result = opt.optimize_kernel(
    model=model, max_iterations=max_iterations, patience=patience, target_speedup=target_speedup, max_failures=max_failures,
    prompt=(
      f"Optimize this square GEMM kernel for the provided target. The reference source is {source_note}. "
      "The baseline runtime in history is the tinygrad TinyJit/JITBEAM matmul baseline, not the naive source runtime."
    ),
  )
  best_ms = result.best.profile.runtime_ms if result.best.profile is not None else float("inf")
  print({"stop_reason": result.stop_reason, "iterations": result.iterations, "best_ms": best_ms,
         "speedup_vs_tinygrad": baseline_ms / best_ms if best_ms != 0 else float("inf")})

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
  _main_gemm_optimization()
