import unittest
import textwrap

from tinygrad import Device, Tensor
from tinygrad.uop.ops import UOp, Ops, track_rewrites
from tinygrad.renderer import ProgramSpec
from tinygrad.helpers import TracingKey
from tinygrad.engine.realize import ExecItem, CompiledRunner

from extra.sqtt.active_sqtt_parse import template

@track_rewrites(name=lambda *args,ret,**kwargs: TracingKey(ret.name, ret=ret))
def run_asm(name:str, src:str) -> ProgramSpec:
  prg = ProgramSpec(name, template.replace("INSTRUCTION", textwrap.dedent(src)), Device.DEFAULT, UOp(Ops.SINK))
  ei = ExecItem(UOp(Ops.SINK), [Tensor.empty(1).uop.buffer.ensure_allocated()], prg=CompiledRunner(prg))
  ei.run()
  return prg

@unittest.skipUnless(Device.DEFAULT == "AMD", "only on AMD")
class TestCfg(unittest.TestCase):
  def setUp(self):
    arch = Device["AMD"].arch
    if not any(arch.startswith(a) for a in {"gfx11", "gfx12"}):
      self.skipTest(f"tests written for RDNA, got arch {arch}")

  def test_simple(self):
    run_asm("simple", """
      entry:
        s_branch bb1
      bb1:
        s_endpgm
    """)

  def test_diamond(self):
    run_asm("diamond", """
      entry:
        s_cmp_eq_i32 s0, 0
        s_cbranch_scc1 if
        s_branch else
      if:
        s_nop 1
        s_branch end
      else:
        s_nop 0
      end:
        s_endpgm
    """)

  def test_loop(self):
    run_asm("loop", """
      entry:
        s_mov_b32 s1, 4
      loop:
        s_add_u32 s1, s1, -1
        s_cmp_eq_i32 s1, 0
        s_cbranch_scc0 loop
        s_endpgm
    """)

  def test_loop_branch(self):
    run_asm("loop_if", """
      entry:
        s_mov_b32 s1, 4
      loop:
        s_add_u32 s1, s1, -1
        s_cmp_eq_i32 s1, 2
        s_cbranch_scc1 cond
        s_branch cont
      cond:
        s_add_u32 s1, s1, -2
      cont:
        s_cmp_eq_i32 s1, 0
        s_cbranch_scc0 loop
        s_endpgm
    """)

  def test_loop_break(self):
    run_asm("loop_break", """
      entry:
        s_mov_b32 s1, 8
      loop:
        s_add_u32 s1, s1, -1
        s_cmp_eq_i32 s1, 5
        s_cbranch_scc1 break
        s_cmp_eq_i32 s1, 0
        s_cbranch_scc0 loop
      break:
        s_endpgm
    """)

  def test_switch(self):
    run_asm("switch_case", """
      entry:
        s_cmp_eq_i32 s0, 0
        s_cbranch_scc1 case0
        s_cmp_eq_i32 s0, 1
        s_cbranch_scc1 case1
        s_branch case2
      case0:
        s_nop 0
        s_branch join
      case1:
        s_nop 1
        s_branch join
      case2:
        s_nop 2
        s_branch join
      join:
        s_endpgm
    """)

  def test_ping_pong(self):
    run_asm("ping_pong", """
      entry:
        s_cmp_eq_i32 s0, 0
        s_cbranch_scc1 ping
        s_branch pong
      ping:
        s_cmp_eq_i32 s1, 0
        s_cbranch_scc1 pong
        s_branch end
      pong:
        s_cmp_eq_i32 s2, 0
        s_cbranch_scc1 ping
      end:
        s_endpgm
    """)

if __name__ == "__main__":
  unittest.main()
