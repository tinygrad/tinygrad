import json, os, unittest
from dataclasses import dataclass
from typing import Any, cast

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.late.reduce import bind_coupled_reduce_descriptors, lower_coupled_reduce_plan, CoupledReduceLowered
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import prod
from tinygrad.uop.ops import KernelInfo, Ops, UOp

RUN_SDPA_METRICS = os.getenv("RUN_SDPA_METRICS") == "1"
SHAPE = {"B": 1, "H": 4, "N": 128, "D": 32}

@dataclass(frozen=True)
class SDPACase:
  name: str
  dtype: Any
  parity_target: float

CASES = (
  SDPACase("fp32", dtypes.float32, 1e-4),
  SDPACase("fp16", dtypes.float16, 1e-2),
  SDPACase("bf16", dtypes.bfloat16, 1e-2),
)

def _err(e:Exception) -> str:
  return f"{type(e).__name__}: {e}"

def _shape_numel(u:UOp) -> int|None:
  try: return prod(cast(Any, u.shape))
  except Exception: return None

def _kernel_count(linear:UOp) -> int:
  return sum((len(call.device) if isinstance(call.device, tuple) else 1) for call in linear.src if call.src[0].op is Ops.SINK)

def _scheduled_buffers(linear:UOp) -> list[UOp]:
  ret: list[UOp] = []
  seen: set[UOp] = set()
  def visit(u:UOp):
    if u in seen: return
    seen.add(u)
    if u.op in {Ops.BUFFER, Ops.BUFFER_VIEW}: ret.append(u)
    elif u.op in {Ops.MSELECT, Ops.MSTACK}:
      for s in u.src: visit(s)
  for call in linear.src:
    for src in call.src[1:]: visit(src)
  return ret

def _full_attention_allocation(linear:UOp) -> bool:
  full_attention_numel = SHAPE["B"] * SHAPE["H"] * SHAPE["N"] * SHAPE["N"]
  return any(_shape_numel(buf) == full_attention_numel for buf in _scheduled_buffers(linear))

def _coupled_reduce_report(linear:UOp) -> dict[str, Any]:
  descriptors: list[dict[str, Any]] = []
  descriptor_field_widths: list[list[int]] = []
  binding_state = "none"

  for kernel_index, call in enumerate(linear.src):
    sink = call.src[0]
    if not isinstance(sink.arg, KernelInfo): continue
    coupled_reduce = sink.arg.coupled_reduce or ()
    if not coupled_reduce: continue

    binding_state = "bound"
    try:
      bound = bind_coupled_reduce_descriptors(sink, coupled_reduce)
    except Exception as e:
      binding_state = "error"
      for descriptor_index, descriptor in enumerate(coupled_reduce):
        widths = [field.dtype.count for field in descriptor.plan.fields]
        descriptor_field_widths.append(widths)
        descriptors.append({
          "kernel_index": kernel_index, "descriptor_index": descriptor_index, "bound": False,
          "field_count": len(descriptor.plan.fields), "field_widths": widths, "binding_error": _err(e),
        })
      continue

    for descriptor_index, (target, plan) in enumerate(bound.items()):
      lowered = lower_coupled_reduce_plan(plan, target=target)
      widths = [field.dtype.count for field in plan.fields]
      descriptor_field_widths.append(widths)
      descriptors.append({
        "kernel_index": kernel_index, "descriptor_index": descriptor_index, "bound": True,
        "target_dtype": str(target.dtype), "field_count": len(plan.fields), "field_widths": widths,
        "reduce_range_count": len(plan.reduce_ranges),
        "accumulator_slots": lowered.accumulator_slots if isinstance(lowered, CoupledReduceLowered) else None,
        "lowering_state": "lowerable" if isinstance(lowered, CoupledReduceLowered) else f"rejected:{lowered.reason.name}",
      })

  return {
    "descriptor_count": len(descriptor_field_widths),
    "descriptor_field_widths": descriptor_field_widths,
    "descriptor_binding_state": binding_state,
    "descriptors": descriptors,
  }

def _base_report(case:SDPACase, torch_available:bool, numpy_available:bool) -> dict[str, Any]:
  return {
    "shape": SHAPE, "dtype": case.name, "device": Device.DEFAULT,
    "kernel_count": None, "full_attention_allocation": None,
    "descriptor_count": None, "descriptor_field_widths": [], "descriptor_binding_state": "unknown",
    "duplicate_qk_sweeps": None, "duplicate_qk_sweeps_status": "not_measured",
    "max_abs_error": None, "max_rel_error": None,
    "parity_target": case.parity_target, "torch_available": torch_available, "numpy_available": numpy_available,
    "schedule_status": "not_run", "parity_status": "not_run", "status": "not_run",
  }

def _schedule_report(report:dict[str, Any], case:SDPACase) -> UOp|None:
  if not is_dtype_supported(case.dtype):
    report["schedule_status"] = report["status"] = "skipped_dtype_unavailable"
    return None
  try:
    B, H, N, D = SHAPE["B"], SHAPE["H"], SHAPE["N"], SHAPE["D"]
    q, k, v = (Tensor.empty(B, H, N, D, dtype=case.dtype, device=Device.DEFAULT) for _ in range(3))
    linear = q.scaled_dot_product_attention(k, v).schedule_linear()
  except Exception as e:
    report["schedule_status"] = report["status"] = "schedule_error"
    report["error"] = _err(e)
    return None
  report["kernel_count"] = _kernel_count(linear)
  report["full_attention_allocation"] = _full_attention_allocation(linear)
  report.update(_coupled_reduce_report(linear))
  report["schedule_status"] = report["status"] = "scheduled"
  return linear

def _parity_report(report:dict[str, Any], case:SDPACase, np:Any, torch:Any) -> None:
  if np is None:
    report["parity_status"] = report["status"] = "skipped_numpy_unavailable"
    return
  if torch is None:
    report["parity_status"] = report["status"] = "skipped_torch_unavailable"
    return
  try:
    B, H, N, D = SHAPE["B"], SHAPE["H"], SHAPE["N"], SHAPE["D"]
    rng = np.random.default_rng(1337)
    inputs = [rng.standard_normal((B, H, N, D), dtype=np.float32) * 0.125 for _ in range(3)]
    tq, tk, tv = (Tensor(x, dtype=case.dtype, device=Device.DEFAULT).realize() for x in inputs)
    tiny = tq.scaled_dot_product_attention(tk, tv).realize().cast(dtypes.float32).numpy().astype(np.float32)

    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[case.name]
    with torch.no_grad():
      ref = torch.nn.functional.scaled_dot_product_attention(*(torch.from_numpy(x).to(dtype=torch_dtype) for x in inputs))
    ref_np = ref.detach().cpu().to(dtype=torch.float32).numpy().astype(np.float32)
    abs_err = np.abs(tiny - ref_np)
    report["max_abs_error"] = float(abs_err.max())
    report["max_rel_error"] = float((abs_err / np.maximum(np.abs(ref_np), 1e-12)).max())
    report["parity_status"] = report["status"] = "ok" if bool(np.allclose(tiny, ref_np, atol=case.parity_target,
                                                                          rtol=case.parity_target)) else "parity_mismatch"
  except Exception as e:
    report["parity_status"] = report["status"] = "skipped_parity_unavailable"
    report["error"] = _err(e)

@unittest.skipUnless(RUN_SDPA_METRICS, "set RUN_SDPA_METRICS=1 to run SDPA metrics harness")
class TestSDPAMetrics(unittest.TestCase):
  def test_sdpa_metrics_report(self):
    np: Any
    try:
      import numpy as _np
      np = _np
    except Exception:
      np = None
    torch: Any
    try:
      import torch as _torch
      torch = _torch
    except Exception:
      torch = None

    reports = []
    for case in CASES:
      report = _base_report(case, torch is not None, np is not None)
      if _schedule_report(report, case) is not None:
        _parity_report(report, case, np, torch)
      reports.append(report)

    print(json.dumps({"schema_version": 1, "reports": reports}, sort_keys=True))

if __name__ == "__main__":
  unittest.main()
