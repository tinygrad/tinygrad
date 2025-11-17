#!/usr/bin/env python3
"""
Phase-by-phase investigation of winograd performance.
We'll manually modify schedule/rangeify.py::winowrite to return early at each phase.

Instructions:
1. Backup current winowrite
2. Replace it with each phase version below
3. Run: python3 wino_phase_test.py
4. Observe compile time changes
5. Move to next phase

IMPORTANT: Ensure all created ranges are consumed in bufferize operations!
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time

B, Cin, Cout, H, W = 1, 16, 16, 32, 32

def test_phase(phase_name):
    print(f"\n{'='*80}")
    print(f"Testing: {phase_name}")
    print(f"{'='*80}")

    np.random.seed(42)
    x = Tensor.randn(B, Cin, H, W, dtype=dtypes.float32).realize()
    w = Tensor.randn(Cout, Cin, 3, 3, dtype=dtypes.float32).realize()

    # Baseline
    with Context(WINO=0, WINO_OLD=0):
        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()
        c_base = time.time() - t0

    # NEW (modified)
    with Context(WINO=1, WINO_OLD=0):
        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()
        c_new = time.time() - t0

    # OLD (full)
    with Context(WINO=0, WINO_OLD=1):
        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()
        c_old = time.time() - t0

    print(f"BASE compile: {c_base*1000:>7.1f}ms")
    print(f"NEW compile:  {c_new*1000:>7.1f}ms ({c_new/c_base:.1f}× overhead)")
    print(f"OLD compile:  {c_old*1000:>7.1f}ms ({c_old/c_base:.1f}× overhead)")

    if c_new < c_old:
        print(f"✓ NEW is {((c_old-c_new)/c_old*100):.1f}% FASTER than OLD")
    else:
        print(f"❌ NEW is {((c_new-c_old)/c_old*100):.1f}% SLOWER than OLD")

print("""
================================================================================
PHASE 1: XHAT Only (minimal winograd)
================================================================================

Replace winowrite (lines 165-194) with:

def winowrite(ctx: IndexingContext, lhs: UOp, rhs: UOp, redu: UOp):
  if not (g := winoguard(lhs, rhs, redu)): return None
  act_like, w_like, k_axes, o_axes, o_adds = g
  reduce_ranges = list(redu.src[1:]); device = redu.device
  other_reduces = [ax for ax in act_like.ranges if ax not in k_axes and ax in reduce_ranges]
  other_loops_x = [ax for ax in act_like.ranges if ax not in reduce_ranges+o_axes]

  # Create only the ranges we need for XHAT
  tile_ranges = [ctx.new_range((int(b.vmax+1)+3)//4, AxisType.LOOP) for b in o_axes]
  inner6 = [ctx.new_range(6, AxisType.LOOP) for _ in o_axes]

  # PHASE 1: Just XHAT, return immediately
  X_tiled = act_like.substitute({add: tr*4 + u for add, tr, u in zip(o_adds, tile_ranges, inner6)})
  XHAT = kron(X_tiled, winograd_Bt, other_reduces+other_loops_x+tile_ranges, inner6, device, ctx)
  return XHAT  # STOP HERE - no GHAT, no MHAT, no output transform

After modifying, run:
""")

test_phase("PHASE 1: Press Enter when ready to test")
input()

print("""
================================================================================
PHASE 2: XHAT + GHAT (both transforms, no multiply)
================================================================================

Replace winowrite with:

def winowrite(ctx: IndexingContext, lhs: UOp, rhs: UOp, redu: UOp):
  if not (g := winoguard(lhs, rhs, redu)): return None
  act_like, w_like, k_axes, o_axes, o_adds = g
  reduce_ranges = list(redu.src[1:]); device = redu.device
  other_reduces = [ax for ax in act_like.ranges if ax not in k_axes and ax in reduce_ranges]
  other_loops_x = [ax for ax in act_like.ranges if ax not in reduce_ranges+o_axes]
  other_loops_w = [ax for ax in w_like.ranges if ax not in reduce_ranges]

  tile_ranges = [ctx.new_range((int(b.vmax+1)+3)//4, AxisType.LOOP) for b in o_axes]
  inner6 = [ctx.new_range(6, AxisType.LOOP) for _ in o_axes]
  kranges = [ctx.new_range(3, AxisType.LOOP) for _ in o_axes]

  # PHASE 2: Create both XHAT and GHAT
  X_tiled = act_like.substitute({add: tr*4 + u for add, tr, u in zip(o_adds, tile_ranges, inner6)})
  XHAT = kron(X_tiled, winograd_Bt, other_reduces+other_loops_x+tile_ranges, inner6, device, ctx)

  w_sub = w_like.substitute({k: r for k, r in zip(k_axes, kranges)})
  GHAT = kron(w_sub, winograd_G, other_reduces+other_loops_w, kranges, device, ctx)

  return GHAT  # Return GHAT (both are created, but return one)

After modifying, run:
""")

test_phase("PHASE 2: Press Enter when ready to test")
input()

print("""
================================================================================
PHASE 3: XHAT * GHAT (multiply, no reduce)
================================================================================

Replace winowrite with:

def winowrite(ctx: IndexingContext, lhs: UOp, rhs: UOp, redu: UOp):
  if not (g := winoguard(lhs, rhs, redu)): return None
  act_like, w_like, k_axes, o_axes, o_adds = g
  reduce_ranges = list(redu.src[1:]); device = redu.device
  other_reduces = [ax for ax in act_like.ranges if ax not in k_axes and ax in reduce_ranges]
  other_loops_x = [ax for ax in act_like.ranges if ax not in reduce_ranges+o_axes]
  other_loops_w = [ax for ax in w_like.ranges if ax not in reduce_ranges]

  tile_ranges, tile_ranges1 = [[ctx.new_range((int(b.vmax+1)+3)//4, AxisType.LOOP) for b in o_axes] for _ in range(2)]
  inner6, inner6_1 = [[ctx.new_range(6, AxisType.LOOP) for _ in o_axes] for _ in range(2)]
  other_loop_ranges_xhat, other_loop_ranges_ghat = [[ctx.new_range(r.vmax+1, AxisType.LOOP) for r in rs] for rs in (other_loops_x, other_loops_w)]
  kranges = [ctx.new_range(3, AxisType.LOOP) for _ in o_axes]

  X_tiled = act_like.substitute({add: tr*4 + u for add, tr, u in zip(o_adds, tile_ranges, inner6)})
  XHAT = kron(X_tiled, winograd_Bt, other_reduces+other_loops_x+tile_ranges, inner6, device, ctx)

  w_sub = w_like.substitute({k: r for k, r in zip(k_axes, kranges)})
  GHAT = kron(w_sub, winograd_G, other_reduces+other_loops_w, kranges, device, ctx)

  # PHASE 3: Multiply XHAT and GHAT (no reduce yet)
  # Note: Commented out reduce, just multiply and bufferize
  product = (XHAT.index(*other_reduces, *other_loop_ranges_xhat, *tile_ranges1, *inner6_1) *
             GHAT.index(*other_reduces, *other_loop_ranges_ghat, *inner6_1))
  result = product.bufferize(*other_reduces, *other_loop_ranges_xhat, *other_loop_ranges_ghat, *tile_ranges1, *inner6_1,
                             arg=BufferizeOpts(device=device, addrspace=AddrSpace.GLOBAL))
  return result

After modifying, run:
""")

test_phase("PHASE 3: Press Enter when ready to test")
input()

print("""
Done with phased investigation!
Check which phase caused the compile time to jump significantly.
""")
