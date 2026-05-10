from typing import Any
from tinygrad.dtype import DType

from pydantic import BaseModel, ConfigDict, Field, field_validator

class HardwareDescriptor(BaseModel):
  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  hardware_family: str = Field(description="Hardware vendor/family, for example AMD, NVIDIA, Apple, Intel.", examples=["AMD"])
  hardware_architecture: str = Field(description="Concrete hardware architecture/target, for example gfx942, sm_90, or apple7.", examples=["gfx942"])
  source_language_dialect: str = Field(description="Candidate kernel source language or dialect.", examples=["HIP C++"])
  compiler: str = Field(description="Compiler/toolchain used for candidate source.", examples=["hipcc"])
  compiler_version: str = Field(description="Compiler/toolchain version string.", examples=["ROCm 6.3.0"])

  # @field_validator("*")
  # @classmethod
  # def _nonempty(cls, value:str) -> str:
  #   if value == "": raise ValueError("must be non-empty")
  #   return value

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

class Kernel(BaseModel):
  source: str = Field(description="kernel source code.")
  global_size: tuple[int, ...] = Field(description="Launch grid/global dimensions.")
  local_size: tuple[int, ...]|None = Field(default=None, description="Launch workgroup/local dimensions, if the backend uses them.")

class CorrectnessResult(BaseModel):
  passed: bool
  max_abs_error: float | None = None
  max_rel_error: float | None = None
  message: str | None = None

class CandidateEvaluation(BaseModel):
  candidate: Kernel
  correctness: CorrectnessResult
  runtime: KernelRuntimeProfile | None = None
  compiler_log: str | None = None

class BufferArg(BaseModel):
  shape: tuple[int, ...] = Field(description="Shapes for buffer arg. Buffers are assumed dense row-major contiguous.")
  dtype: DType = Field(description="Datatype for buffer arg. Buffers are assumed dense row-major contiguous.")

class KernelDescriptor(BaseModel):
  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  entry_point: str = Field(description="Kernel function name. Candidates must preserve this name.")
  type_family: str = Field(description="High-level kernel family, for example gemm, gemm_bwd, fa_fwd, fa_bwd_pre, fa_bwd, or fa_bwd_post.")
  kernel: Kernel = Field(description="the kernel.")
  buffer_args: list[BufferArg] = Field(description="buffer arguments in kernel ABI order")
  description: str|None = Field(default=None, description="Optional concise semantic note supplied by the MCP or caller.")

  # @field_validator("entry_point", "type_family"
  # @classmethod
  # def _nonempty(cls, value:str) -> str:
  #   if value == "": raise ValueError("must be non-empty")
  #   return value
