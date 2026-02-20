# ruff: noqa: F405, F403
# allow define from star imports

import unittest

from tinygrad import Device, Tensor
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.viz.serve import amdgpu_cfg

from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s

# TODO: this belongs to the dsl infrastructure
from extra.gemm.amd_asm_matmul import Kernel

def run_asm(name:str, k:Kernel):
  insts = k.finalize()
  def fxn(out:UOp) -> UOp:
    lidx = UOp.special(1, "lidx0")
    gidx = UOp.special(1, "gidx0")
    sink = UOp.sink(out.base, lidx, gidx, arg=KernelInfo(name=name))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
  out = Tensor.custom_kernel(Tensor.empty(1), fxn=fxn)[0]
  ei = out.schedule()[-1].lower()
  ei.run()
  return ei

@unittest.skipUnless(Device.DEFAULT == "AMD", "only on AMD")
class TestCfg(unittest.TestCase):
  def setUp(self):
    self.arch = Device["AMD"].arch
    if not any(self.arch.startswith(a) for a in {"gfx11", "gfx12"}):
      self.skipTest(f"tests written for RDNA, got arch {arch}")

  def test_simple(self):
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_branch(), target="bb1")
    k.label("bb1")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    run_asm("simple", k)

  def test_diamond(self):
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_mov_b32(s[0], 0))
    k.emit(s_mov_b32(s[1], 0))
    k.emit(s_cmp_eq_u64(s[0:1], 0))
    k.emit(s_cbranch_scc1(), target="if")
    k.emit(s_branch(), target="else")
    k.label("if")
    k.emit(s_nop(1))
    k.emit(s_branch(), target="end")
    k.label("else")
    k.emit(s_nop(0))
    k.label("end")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    ei = run_asm("diamond", k)
    cfg = amdgpu_cfg(ei.prg.p.lib, self.arch)["data"]
    self.assertEqual(len(cfg["blocks"]), 5)
    edge_count = sum(len(v) for v in cfg["paths"].values())
    self.assertEqual(edge_count, 5)
    references:dict[str, list[str]] = {}
    for pc, tokens in cfg["pc_tokens"].items():
      for t in tokens:
        for key in t["keys"]: references.setdefault(key, []).append(pc)
    self.assertEqual(len(references["r0"]), 2)
    insts = [cfg["pc_tokens"][pc][0]["st"] for pc in references["r0"]]
    self.assertEqual(insts, ['s_mov_b32', 's_cmp_eq_u64'])

  def test_loop(self):
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_mov_b32(s[1], 4))
    k.label("loop")
    k.emit(s_add_u32(s[1], s[1], -1))
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_cbranch_scc0(), target="loop")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    run_asm("simple_loop", k)

  def test_loop_branch(self):
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_mov_b32(s[1], 4))
    k.label("loop")
    k.emit(s_add_u32(s[1], s[1], -1))
    k.emit(s_cmp_eq_i32(s[1], 2))
    k.emit(s_cbranch_scc1(), target="cond")
    k.emit(s_branch(), target="cont")
    k.label("cond")
    k.emit(s_add_u32(s[1], s[1], -2))
    k.label("cont")
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_cbranch_scc0(), target="loop")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    run_asm("loop_if", k)

  def test_loop_break(self):
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_mov_b32(s[1], 8))
    k.label("loop")
    k.emit(s_add_u32(s[1], s[1], -1))
    k.emit(s_cmp_eq_i32(s[1], 5))
    k.emit(s_cbranch_scc1(), target="break")
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_cbranch_scc0(), target="loop")
    k.label("break")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    run_asm("loop_break", k)

  def test_switch(self):
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_cmp_eq_i32(s[0], 0))
    k.emit(s_cbranch_scc1(), target="case0")
    k.emit(s_cmp_eq_i32(s[0], 1))
    k.emit(s_cbranch_scc1(), target="case1")
    k.emit(s_branch(), target="case2")
    k.label("case0")
    k.emit(s_nop(0))
    k.emit(s_branch(), target="join")
    k.label("case1")
    k.emit(s_nop(1))
    k.emit(s_branch(), target="join")
    k.label("case2")
    k.emit(s_nop(2))
    k.emit(s_branch(), target="join")
    k.label("join")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    run_asm("switch_case", k)

  def test_ping_pong(self):
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_cmp_eq_i32(s[0], 0))
    k.emit(s_cbranch_scc1(), target="ping")
    k.emit(s_branch(), target="pong")
    k.label("ping")
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_cbranch_scc1(), target="pong")
    k.emit(s_branch(), target="end")
    k.label("pong")
    k.emit(s_cmp_eq_i32(s[2], 0))
    k.emit(s_cbranch_scc1(), target="ping")
    k.label("end")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    run_asm("ping_pong", k)

  def test_colored_blocks(self):
    N = 10
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_branch(), target="init0")
    for i in range(N):
      loop = f"loop{i}"
      k.label(f"init{i}")
      k.emit(s_mov_b32(s[1], i + 1))
      k.emit(s_branch(), target=loop)
      k.label(loop)
      k.emit(s_nop(i & 7))
      k.emit(s_add_u32(s[1], s[1], -1))
      k.emit(s_cmp_eq_i32(s[1], 0))
      k.emit(s_cbranch_scc0(), target=loop)
      k.emit(s_branch(), target=f"init{i+1}" if i + 1 < N else "end")
    k.label("end")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    run_asm("test_colored_blocks", k)

  def test_jump_back_to_end(self):
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_mov_b32(s[1], 2))
    k.emit(s_cbranch_execz(), target="loop")
    k.label("end")
    k.emit(s_endpgm())
    k.label("loop")
    k.emit(s_add_u32(s[1], s[1], -1))
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_branch(), target="end")
    k.emit(s_code_end())
    run_asm("jump_back_to_end", k)

  def test_hit_count(self):
    k = Kernel(arch=Device["AMD"].arch)
    k.label("entry")
    k.emit(s_mov_b32(s[1], 1))
    k.emit(s_branch(), target="alt")
    k.label("continue")
    k.emit(s_mov_b32(s[2], 2))
    k.emit(s_add_u32(s[1], s[1], s[2]))
    k.label("alt")
    k.emit(s_add_u32(s[1], s[1], -1))
    k.emit(s_endpgm())
    k.emit(s_code_end())
    run_asm("test_hit_count", k)

if __name__ == "__main__":
  unittest.main()
