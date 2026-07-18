#!/usr/bin/env python3
"""Time representative compiled openpilot kernels outside the graph."""
import argparse, hashlib, pickle, statistics

from tinygrad import Tensor
from tinygrad.device import Buffer
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import get_runtime, resolve_params
from tinygrad.uop.ops import Ops
from extra.gemm.qcom_ir3_matmul_patch import plain_name


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("reference", nargs="?")
  parser.add_argument("--runs", type=int, default=10)
  parser.add_argument("--unique", action="store_true", help="separate programs with the same display name")
  parser.add_argument("--name", help="only profile programs with this display name")
  parser.add_argument("--details", action="store_true", help="show launch geometry and register metadata")
  parser.add_argument("--list-only", action="store_true", help="list program families without running the model")
  parser.add_argument("--no-warm", action="store_true", help="profile captured calls without first executing the full graph")
  parser.add_argument("--output-dtype", help="only profile calls with an output dtype whose name contains this text")
  parser.add_argument("--dry-bufs", action="store_true", help="print resolved buffer/runtime metadata without executing")
  parser.add_argument("--program-hash", help="only profile this eight-character library SHA1 prefix")
  parser.add_argument("--fresh-bufs", action="store_true", help="execute with fresh dedicated buffers instead of captured buffers")
  parser.add_argument("--local-size", help="override launch local size, for example 8,4,4")
  parser.add_argument("--global-size", help="override launch work-group counts, for example 1,8,16")
  parser.add_argument("--zero-bufs", action="store_true", help="zero all execution buffers before profiling")
  parser.add_argument("--zero-buffer-indices", help="comma-separated execution buffer indices to zero")
  parser.add_argument("--all", action="store_true", help="profile every program family instead of only the top entries")
  args = parser.parse_args()
  local_size_override = tuple(int(x) for x in args.local_size.split(",")) if args.local_size else None
  global_size_override = tuple(float(x) for x in args.global_size.split(",")) if args.global_size else None
  with open(args.model, "rb") as f: model = pickle.load(f)
  batch = model.captured.linear.src[0].src[0].src[0].src
  if args.list_only:
    families: dict[tuple[str, tuple, tuple], int] = {}
    for call in batch:
      if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM:
        program = call.src[0]
        key = (plain_name(program.arg.name), tuple(program.arg.global_size), tuple(program.arg.local_size))
        families[key] = families.get(key, 0) + 1
    for (name, global_size, local_size), count in sorted(families.items()):
      print(f"{count:3d} {name} global={global_size} local={local_size}")
    return
  inputs = {}
  for name, (view, _vars, dtype, device) in zip(model.captured.expected_names, model.captured.expected_input_info):
    inputs[name] = Tensor.zeros(*view.shape, dtype=dtype, device=device).contiguous().realize()
  input_uops, var_vals, _names, _info = _prepare_jit_inputs((), inputs)
  if not args.no_warm: model(**inputs).numpy()
  families: dict[object, list] = {}
  for call in batch:
    if call.op is Ops.CALL and call.src[0].op is Ops.PROGRAM:
      name = plain_name(call.src[0].arg.name)
      if args.name is not None and name != args.name: continue
      if args.output_dtype is not None and not any(args.output_dtype in str(call.src[i+1].dtype) for i in call.src[0].arg.outs): continue
      if args.program_hash is not None and hashlib.sha1(call.src[0].src[3].arg).hexdigest()[:8] != args.program_hash: continue
      key = (name, hashlib.sha1(call.src[0].src[3].arg).hexdigest()[:8]) if args.unique else name
      families.setdefault(key, []).append(call)
  ranked = sorted(families.items(), key=lambda x: sum(int(c.src[0].src[0].arg.estimates.ops) for c in x[1]), reverse=True)
  for key, calls in (ranked if args.all else ranked[:80 if args.unique else 30]):
    name = f"{key[0]}#{key[1]}" if isinstance(key, tuple) else key
    call, program = calls[0], calls[0].src[0]
    resolved = resolve_params(call, tuple(input_uops))
    bufs = ([Buffer(resolved[0].device, u.buffer.size, u.dtype).allocate()._buf for u in resolved] if args.fresh_bufs else
            [u.buffer.ensure_allocated()._buf for u in resolved])
    runtime = get_runtime(resolved[0].device, program)
    if args.dry_bufs:
      print(name, "bufs", [(x.dtype, x.buffer.size, hex(int(b.va_addr)), int(b.va_addr)%4096, b.size) for x,b in zip(resolved, bufs)],
            "runtime_offs", runtime.buf_offs, "tex", runtime.tex_cnt, "ibo", runtime.ibo_cnt)
      continue
    zero_indices = set(range(len(bufs))) if args.zero_bufs else {int(x) for x in (args.zero_buffer_indices or "").split(",") if x}
    for i in zero_indices: bufs[i].cpu_view().mv[:] = bytes(bufs[i].size)
    launch_local = local_size_override or program.arg.local_size
    launch_global = global_size_override or (tuple(program.arg.global_size[i]*program.arg.local_size[i]/launch_local[i] for i in range(3))
                                                if local_size_override else program.arg.global_size)
    for _ in range(2): runtime(*bufs, global_size=launch_global, local_size=launch_local, vals=(), wait=True)
    times = [runtime(*bufs, global_size=launch_global, local_size=launch_local,
                     vals=(), wait=True)*1e3 for _ in range(args.runs)]
    ops = int(program.src[0].arg.estimates.ops)
    med = statistics.median(times)
    detail = ""
    if args.details:
      detail = f" global={program.arg.global_size} local={program.arg.local_size}"
    print(f"{med:8.4f} ms {ops/med/1e6:8.1f} GFLOP/s x{len(calls):2d} total={med*len(calls):8.3f} ms {name}{detail}")


if __name__ == "__main__": main()
