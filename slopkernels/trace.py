"""Offline-only template tracing utilities for slopkernels.

WARNING: Offline generation/verification tooling ONLY. Runtime code (`tinygrad/*`, `slopkernels/__init__.py`,
`slopkernels/precompiled.py`, `slopkernels/src/*`) must NEVER import or use this module.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tinygrad import UOp, dtypes
from tinygrad.dtype import DType, PyConst
from tinygrad.uop.spec import spec_tensor, type_verify
from tinygrad.uop.uir import parse_uir, render_uir

_DTYPES: dict[str, DType] = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}


@dataclass(frozen=True)
class TemplateParam:
  """Specification for tracing a scalar int/float kernel parameter as a named `AddrSpace.TEMPLATE` UOp."""
  name: str
  dtype: DType = dtypes.int32
  vmin_vmax: tuple[PyConst, PyConst] | None = None

  def to_uop(self) -> UOp: return UOp.template(self.name, self.vmin_vmax, self.dtype)

  def to_dict(self) -> dict[str, Any]:
    d: dict[str, Any] = {"dtype": self.dtype.name}
    if self.vmin_vmax is not None: d["vmin_vmax"] = list(self.vmin_vmax)
    return d

  @classmethod
  def from_spec(cls, name:str, raw:TemplateParam|DType|dict[str, Any]) -> TemplateParam:
    if isinstance(raw, TemplateParam): return raw
    if isinstance(raw, DType): return cls(name, dtype=raw)
    if isinstance(raw, dict):
      dt_raw = raw.get("dtype", "i32")
      dt = dt_raw if isinstance(dt_raw, DType) else _DTYPES[dt_raw]
      vv_raw = raw.get("vmin_vmax")
      vv = (vv_raw[0], vv_raw[1]) if vv_raw is not None else None
      return cls(name, dtype=dt, vmin_vmax=vv)
    raise TypeError(f"unsupported template spec for {name!r}: {raw!r}")


def verify_template_uir(uir:str, samples:Sequence[dict[str, Any]],
                        *, ref_fn:Callable[[dict[str, Any]], str|UOp|None]|None=None) -> None:
  """Verify a traced templated UIR round-trips losslessly and specializes cleanly for every sample."""
  u = parse_uir(uir)
  if parse_uir(render_uir(u)) is not u: raise ValueError("templated UIR failed lossless round-trip")
  for sample in samples:
    spec_u = u.specialize(**sample)
    if spec_u.has_templates: raise ValueError(f"specialization left templates for sample {sample}")
    type_verify(spec_u, spec_tensor, enter_calls=True)
    if ref_fn is not None:
      ref = ref_fn(sample)
      if ref is None: raise ValueError(f"reference builder returned None for sample {sample}")
      ref_u = parse_uir(ref) if isinstance(ref, str) else ref
      if spec_u.op is not ref_u.op or spec_u._shape != ref_u._shape:
        raise ValueError(f"specialized UOp mismatch for {sample}: got ({spec_u.op}, {spec_u._shape}), expected ({ref_u.op}, {ref_u._shape})")


def trace_builder(builder:Callable[..., str|UOp|None], spec:dict[str, Any],
                  templates:Mapping[str, TemplateParam|DType|dict[str, Any]],
                  *, verify_samples:Sequence[dict[str, Any]]=()) -> str:
  """Offline-trace `builder` with named `UOp.template` parameters substituted for `templates` keys."""
  if not templates: raise ValueError("trace_builder requires at least one template parameter")
  traced_spec = dict(spec)
  tparams: dict[str, TemplateParam] = {}
  for key, raw in templates.items():
    tp = TemplateParam.from_spec(key, raw)
    tparams[key] = tp
    traced_spec[key] = tp.to_uop()
  res = builder(**traced_spec)
  if res is None: raise ValueError(f"builder {builder} returned None for traced spec")
  from dataclasses import replace as _replace
  from tinygrad.uop.ops import Ops as _Ops
  u = parse_uir(res) if isinstance(res, str) else res
  if not isinstance(res, str):
    allocs = [x for x in u.toposort(enter_calls=True) if x.op is _Ops.ALLOC]
    if allocs: u = u.substitute({b: b.replace(arg=_replace(b.arg, slot=i)) for i, b in enumerate(allocs)}, enter_calls=True)
  uir = res if isinstance(res, str) else render_uir(u, color=False)
  found = {t.arg.name for t in u.templates()}
  expected = {tp.name for tp in tparams.values()}
  if missing := sorted(expected - found):
    raise ValueError(f"traced UIR did not retain template parameter(s): {', '.join(missing)}")
  if parse_uir(render_uir(u)) is not u:
    raise ValueError("traced templated UIR failed lossless round-trip")
  if verify_samples:
    verify_template_uir(uir, verify_samples, ref_fn=lambda s: builder(**{**spec, **s}))
  return uir


def trace_kernel(name:str, spec:dict[str, Any], templates:Mapping[str, TemplateParam|DType|dict[str, Any]],
                 *, verify_samples:Sequence[dict[str, Any]]=()) -> str:
  """Offline-trace a registered slopkernel by name into a templated UIR string."""
  import slopkernels
  if name not in slopkernels._REGISTRY:
    raise KeyError(f"unknown slopkernel {name!r}; available: {slopkernels.list_kernels()}")
  fn = slopkernels._REGISTRY[name][0]
  return trace_builder(fn, spec, templates, verify_samples=verify_samples)


def discover_traceable_params(name:str, spec:dict[str, Any]) -> tuple[str, ...]:
  """Offline probe: return sorted scalar int/float spec keys of `name` that trace cleanly as `UOp.template`."""
  import slopkernels
  if name not in slopkernels._REGISTRY: raise KeyError(f"unknown slopkernel {name!r}")
  fn = slopkernels._REGISTRY[name][0]
  traceable: list[str] = []
  for k, v in sorted(spec.items()):
    if isinstance(v, bool) or not isinstance(v, (int, float)): continue
    tp = TemplateParam(k, dtype=dtypes.int32 if isinstance(v, int) else dtypes.float32)
    try:
      trace_builder(fn, spec, {k: tp}, verify_samples=({k: v},))
    except Exception:
      continue
    traceable.append(k)
  return tuple(traceable)
