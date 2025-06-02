from __future__ import annotations
import itertools, operator, colorsys, tabulate
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, UPat, PatternMatcher, sint_to_uop
from tinygrad.helpers import getenv, prod, Context
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, unravel, canonicalize_strides
from tinygrad.codegen.kernel import Kernel


def _viz(ctx: Kernel, uop: UOp):
  st, buf = uop.st_arg, uop.src[0].src[0]
  print(f"\nBuf [{buf.arg}] (op: {'st' if uop.op is Ops.STORE else 'ld'} {'global' if buf.op is Ops.DEFINE_GLOBAL else 'shared'})")
  # shrink global, reduce and broadcasted upcast dims and expand local dims
  st = st.shrink(tuple((0, 1) if i < ctx.global_dims or (ctx.first_reduce <= i < ctx.first_upcast) else (0, s) for i, s in enumerate(st.shape)))
  st = st.shrink(tuple((0, 1) if (ctx.first_upcast <= i and s == 0) else (0, st.shape[i]) for i, s in enumerate(st.real_strides(True))))
  st = st.expand(tuple(ctx.full_shape[i] if ctx.global_dims <= i < ctx.first_reduce else s for i, s in enumerate(st.shape)))

  # thread and upcast indices increment in col-major order
  colmajor_strides = canonicalize_strides(st.shape, tuple(itertools.accumulate(st.shape, operator.mul, initial=1)))
  tile_st = ShapeTracker((View.create(shape=st.shape, strides=colmajor_strides),))

  layout: dict = {}
  # print(f"{uop.st_arg}{uop.st_arg.size} {uop.st_arg.real_size()}\n{st}{st.size} {st.real_size()}\n{tile_st}{tile_st.size} {tile_st.real_size()}")
  with Context(TRACK_MATCH_STATS=0):
    for i in range(0, tile_st.real_size()):
      logical_coords: tuple[UOp, ...] = tuple(sint_to_uop(c) for c in unravel(tile_st.shape, i))
      idx, idx_valid = st.to_indexed_uops(logical_coords)
      tile_idx, tile_idx_valid = tile_st.to_indexed_uops(logical_coords)
      if idx_valid.arg and tile_idx_valid.arg:
        layout.setdefault(idx.arg, []).append(tile_idx.arg)

  matrix, elems, width, tidx = None, [], 1, getenv("VIZ_TILE_TIDX", -1)
  local_size = prod(s for s in tile_st.shape[ctx.global_dims : ctx.first_reduce])
  upcast_size = prod(s for s in tile_st.shape[ctx.first_upcast :])
  local_w, upcast_w = len(str(local_size - 1)), len(str(upcast_size - 1))

  def ansi(t: int) -> str:
    _R, _G, _B = (int(x * 5 + 0.5) for x in colorsys.hsv_to_rgb(t / 32, 0.65, 0.80))
    return f"\x1b[38;5;{17 + 36 * _R + 6 * _G + _B}m{t:0{local_w}d}\x1b[0m" if tidx == -1 or tidx == t else f"{t:0{local_w}d}"

  for i, coords in sorted(layout.items()):
    thread_idxs = tuple(sorted(set(cs % local_size for cs in coords)))
    upcast_idx = tuple(set(cs // local_size for cs in coords))[0]
    elems += [f"T({','.join((f'{chr(10)}  ' if i > 0 and i % 4 == 0 else '') + ansi(thread_idx) for i, thread_idx in enumerate(thread_idxs))})\n"
            + f"V[{upcast_idx:0{upcast_w}d}]"]

  for stride, shape in sorted((stride, shape) for stride, shape in zip(st.real_strides(True), st.shape) if stride != 0):
    if width == stride and width * shape <= getenv("VIZ_TILE_MAX_WIDTH", 32): width *= shape
    else: break

  if buf.op is Ops.DEFINE_LOCAL: width = 32 * 4 // buf.dtype.itemsize  # override width to visualize smem banks
  elif len(elems) % width != 0: width = 1  # fallback to width 1

  matrix = [elems[i : i + width] for i in range(0, len(elems), width)]
  if matrix:
    print(tabulate.tabulate(matrix, tablefmt="simple_grid", showindex=True, headers=tuple(str(i) for i in range(width))))
  else:
    print("<< failed to viz tile >>")

  return None


def viz_tile(kernel: Kernel, ast: UOp) -> None:
  graph_rewrite(ast, PatternMatcher([(UPat((Ops.LOAD, Ops.STORE), name="uop"), _viz)]), ctx=kernel)
