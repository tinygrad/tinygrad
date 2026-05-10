from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HardwareDescriptor(BaseModel):
  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  hardware_family: str = Field(description="Hardware vendor/family, for example AMD, NVIDIA, Apple, Intel.", examples=["AMD"])
  hardware_architecture: str = Field(description="Concrete hardware architecture/target, for example gfx942, sm_90, or apple7.", examples=["gfx942"])
  source_language_dialect: str = Field(description="Candidate kernel source language or dialect.", examples=["HIP C++"])
  compiler: str = Field(description="Compiler/toolchain used for candidate source.", examples=["hipcc"])
  compiler_version: str = Field(description="Compiler/toolchain version string.", examples=["ROCm 6.3.0"])

  @field_validator("*")
  @classmethod
  def _nonempty(cls, value:str) -> str:
    if value == "": raise ValueError("must be non-empty")
    return value

  @classmethod
  def from_dev(cls, device:str|None=None) -> "HardwareDescriptor":
    """
    Build the descriptor from tinygrad's active DEV target.

    This intentionally avoids shelling out to compiler binaries. It reports the
    concrete tinygrad device, selected renderer, compiler object, and arch.
    """
    from tinygrad.device import Device

    dev = Device[Device.DEFAULT if device is None else device]
    renderer = dev.renderer
    compiler = renderer.compiler
    source_language_dialect, compiler_name = cls.candidate_source_and_compiler(dev)

    base_device = renderer.target.device or dev.device.split(":", 1)[0].upper()
    arch = cls._first_nonempty(
      renderer.target.arch,
      getattr(dev, "arch", None),
      "unknown",
    )

    return cls(
      hardware_family=base_device,
      hardware_architecture=arch,
      source_language_dialect=source_language_dialect,
      compiler=compiler_name,
      compiler_version=cls._first_nonempty(getattr(compiler, "version", None), "unknown"),
    )

  @staticmethod
  def _first_nonempty(*values:object) -> str:
    return next((str(x) for x in values if x is not None and str(x).strip()), "unknown")

  @staticmethod
  def candidate_source_and_compiler(device:Any|None=None) -> tuple[str, str]:
    from tinygrad.device import Device
    from tinygrad.helpers import getenv

    dev = Device[Device.DEFAULT] if device is None else Device[device] if isinstance(device, str) else device
    renderer = dev.renderer
    compiler = renderer.compiler
    renderer_name, compiler_name = type(renderer).__name__, type(compiler).__name__

    match renderer_name:
      case "HIPCCRenderer": return "HIP C++", "hipcc"
      case "HIPRenderer": return "HIP C++", "COMGR"
      case "AMDLLVMRenderer": return "LLVM IR", "tinygrad AMDLLVM"
      case "CUDARenderer": return "CUDA C++", "nvrtc"
      case "NVCCRenderer": return "CUDA C++", "nvcc"
      case "PTXRenderer": return "PTX", "ptxas"
      case "MetalRenderer": return "Metal Shading Language", "Metal runtime compiler"
      case "OpenCLRenderer" | "IntelRenderer" | "QCOMCLRenderer": return "OpenCL C", "OpenCL runtime compiler"
      case "ClangRenderer" | "ClangJITRenderer": return "C", getenv("CC", "clang")
      case "CPULLVMRenderer": return "LLVM IR", "LLVM"

    return renderer_name, compiler_name


class KernelRuntimeProfile(BaseModel):
  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  runtime_ms: float = Field(description="Elapsed GPU/runtime time in milliseconds.")


class KernelDescriptor(BaseModel):
  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  entry_point: str = Field(description="Kernel function name. Candidates must preserve this name.")
  type_family: str = Field(description="High-level kernel family, for example gemm, gemm_bwd, fa_fwd, fa_bwd_pre, fa_bwd, or fa_bwd_post.")
  source: str = Field(description="Current candidate kernel source code.")
  global_size: tuple[int, ...] = Field(description="Launch grid/global dimensions.")
  buffer_shapes: tuple[tuple[int, ...]] = Field(description="Shapes for buffer arguments in kernel ABI order. Buffers are assumed dense row-major contiguous.")
  local_size: tuple[int, ...]|None = Field(default=None, description="Launch workgroup/local dimensions, if the backend uses them.")
  description: str|None = Field(default=None, description="Optional concise semantic note supplied by the MCP or caller.")

  @field_validator("name", "type_family", "source")
  @classmethod
  def _nonempty(cls, value:str) -> str:
    if value == "": raise ValueError("must be non-empty")
    return value


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
