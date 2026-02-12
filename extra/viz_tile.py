from __future__ import annotations
import itertools, operator, colorsys, tabulate
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, UPat, PatternMatcher, sint_to_uop
from tinygrad.helpers import getenv, prod, Context
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, unravel, canonicalize_strides
from tinygrad.codegen.kernel import Kernel

# TODO: address masking/padding
# TODO: generic shapetracker layout viz
# TODO: remove tabulate dependency

VIZ_TILE_DEBUG = getenv("VIZ_TILE_DEBUG", 0)
VIZ_TILE_MAX_WIDTH = getenv("VIZ_TILE_MAX_WIDTH", 32)
VIZ_TILE_LIDX = getenv("VIZ_TILE_LIDX", -1)

def _viz(ctx: Kernel, uop: UOp):
  st, buf = uop.st_arg, uop.src[0].base
  print(f"\nBuf [{buf.arg}] (op: {'st' if uop.op is Ops.STORE else 'ld'} {'global' if buf.op is Ops.DEFINE_GLOBAL else 'shared'})")

  # shrink global, reduce and broadcasted upcast dims
  # expand local dims
  st = st.shrink(tuple((0, 1) if i < ctx.global_dims or (ctx.first_reduce <= i < ctx.first_upcast) else (0, s) for i, s in enumerate(st.shape)))
  st = st.shrink(tuple((0, 1) if (ctx.first_upcast <= i and s == 0) else (0, st.shape[i]) for i, s in enumerate(st.real_strides(True))))
  st = st.expand(tuple(ctx.full_shape[i] if ctx.global_dims <= i < ctx.first_reduce else s for i, s in enumerate(st.shape)))

  # thread and upcast indices increment in col-major order
  colmajor_strides = canonicalize_strides(st.shape, tuple(itertools.accumulate(st.shape, operator.mul, initial=1)))
  threads_st = ShapeTracker((View.create(shape=st.shape, strides=colmajor_strides),))

  if VIZ_TILE_DEBUG:
    print(f"{uop.st_arg=} {uop.st_arg.size=} {uop.st_arg.real_size()=}")
    print(f"{st=} {st.size=} {st.real_size()=}")
    print(f"{threads_st=} {threads_st.size=} {threads_st.real_size()=}")

  layout: dict = {}
  with Context(TRACK_MATCH_STATS=0):
    for i in range(0, threads_st.real_size()): # match logial coord to thread/upcast index
      logical_coords: tuple[UOp, ...] = tuple(sint_to_uop(c) for c in unravel(threads_st.shape, i))
      idx, idx_valid = st.to_indexed_uops(logical_coords)
      threads_idx, threads_idx_valid = threads_st.to_indexed_uops(logical_coords)
      if idx_valid.arg and threads_idx_valid.arg:
        layout.setdefault(idx.arg, []).append(threads_idx.arg)

  local_size = prod(s for s in threads_st.shape[ctx.global_dims : ctx.first_reduce])
  upcast_size = prod(s for s in threads_st.shape[ctx.first_upcast :])
  local_w, upcast_w = len(str(local_size - 1)), len(str(upcast_size - 1))

  def ansi(t: int) -> str:
    _R, _G, _B = (int(x * 5 + 0.5) for x in colorsys.hsv_to_rgb(t / 32, 0.65, 0.80))
    return f"\x1b[38;5;{17 + 36 * _R + 6 * _G + _B}m{t:0{local_w}d}\x1b[0m" if VIZ_TILE_LIDX in (-1, t) else f"{t:0{local_w}d}"

  elems = []
  for i, coords in sorted(layout.items()):
    thread_idxs = tuple(sorted(set(cs % local_size for cs in coords)))
    upcast_idx = tuple(set(cs // local_size for cs in coords))[0]
    elems += [f"T({','.join((f'{chr(10)}  ' if i > 0 and i % 4 == 0 else '') + ansi(thread_idx) for i, thread_idx in enumerate(thread_idxs))})\n"
            + f"V[{upcast_idx:0{upcast_w}d}]"]

  if buf.op is Ops.DEFINE_LOCAL: # set width to 128 bytes (32 floats) to simulate smem layout and visualize bank conflicts
    width = 32 * 4 // buf.dtype.itemsize
  else: # set witdh based on contiguous strides
    width = 1
    for stride, shape in sorted((stride, shape) for stride, shape in zip(st.real_strides(True), st.shape) if stride != 0):
      if width == stride and width * shape <= VIZ_TILE_MAX_WIDTH:
        width *= shape
      else:
        break
    if len(elems) % width != 0: # fallback to width 1
      width = 1

  if tile := [elems[i : i + width] for i in range(0, len(elems), width)]:
    print(tabulate.tabulate(tile, tablefmt="simple_grid", showindex=True, headers=tuple(str(i) for i in range(width))))
  else:
    print("<< failed to viz tile >>")

  return None


def viz_tile(kernel: Kernel, ast: UOp) -> None:
  """
  Visualize LOAD/STORE tiles.

  The function walks the `ast` with `graph_rewrite`, dispatching `_viz`
  on each matching UOp to emit an ANSI-coloured table of the buffer
  region accessed.  It is completely side-effectful (prints to stdout)
  and returns `None`.

  Typical integration (add inside `Kernel.linearize` **after** you've
  produced the final AST you want to run):

  ```
  modified_ast = self.get_optimized_ast(name_override)
  if ast_transform is not None:modified_ast = ast_transform(self, modified_ast)
  from extra import viz_tile              # 1 import helper
  viz_tile.viz_tile(self, modified_ast)   # 2 show tiles
  ```
  """
  graph_rewrite(ast, PatternMatcher([(UPat((Ops.LOAD, Ops.STORE), name="uop"), _viz)]), ctx=kernel)
