"""Shared test helpers for RDNA3 tests."""
import shutil
from dataclasses import dataclass

@dataclass
class KernelInfo:
  code: bytes
  global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  buf_idxs: list[int]  # indices into shared buffer pool
  buf_sizes: list[int]  # sizes for each buffer index

# LLVM tool detection (shared across test files)
def get_llvm_mc():
  """Find llvm-mc executable, preferring newer versions."""
  for p in ['llvm-mc', 'llvm-mc-21', 'llvm-mc-20']:
    if shutil.which(p): return p
  raise FileNotFoundError("llvm-mc not found")

def get_llvm_objdump():
  """Find llvm-objdump executable, preferring newer versions."""
  for p in ['llvm-objdump', 'llvm-objdump-21', 'llvm-objdump-20']:
    if shutil.which(p): return p
  raise FileNotFoundError("llvm-objdump not found")

ARCH_TO_TARGET:dict[str, list[str]] = {
  "rdna3":["gfx1100"],
  "rdna4":["gfx1200"],
  "cdna":["gfx950", "gfx942"],
}

TARGET_TO_ARCH:dict[str, str] = {t:arch for arch,targets in ARCH_TO_TARGET.items() for t in targets}

def get_target(arch:str) -> str: return ARCH_TO_TARGET[arch][0]

def get_mattr(arch:str) -> str:
  return {"rdna3":"+real-true16,+wavefrontsize32", "rdna4":"+real-true16,+wavefrontsize32", "cdna":"+wavefrontsize64"}[arch]

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION CONTEXT (for testing compiled pseudocode)
# ═══════════════════════════════════════════════════════════════════════════════

class ExecContext:
  """Context for running compiled pseudocode in tests."""
  def __init__(self, s0=0, s1=0, s2=0, d0=0, scc=0, vcc=0, lane=0, exec_mask=0xffffffff, literal=0, vgprs=None, src0_idx=0, vdst_idx=0):
    from extra.assembly.amd.pcode import Reg, MASK32, MASK64, TypedView
    self._Reg, self._MASK64, self._TypedView = Reg, MASK64, TypedView
    self.S0, self.S1, self.S2 = Reg(s0), Reg(s1), Reg(s2)
    self.D0, self.D1 = Reg(d0), Reg(0)
    self.SCC, self.VCC, self.EXEC = Reg(scc), Reg(vcc), Reg(exec_mask)
    self.tmp, self.saveexec = Reg(0), Reg(exec_mask)
    self.lane, self.laneId, self.literal = lane, lane, literal
    self.SIMM16, self.SIMM32 = Reg(literal), Reg(literal)
    self.VGPR = vgprs if vgprs is not None else {}
    self.SRC0, self.VDST = Reg(src0_idx), Reg(vdst_idx)

  def run(self, code: str):
    """Execute compiled code."""
    import extra.assembly.amd.pcode as pcode
    ns = {k: getattr(pcode, k) for k in dir(pcode) if not k.startswith('_')}
    # Also include underscore-prefixed helpers that compiled pseudocode uses
    for k in ['_pack', '_pack32']:
      if hasattr(pcode, k): ns[k] = getattr(pcode, k)
    ns.update({
      'S0': self.S0, 'S1': self.S1, 'S2': self.S2, 'D0': self.D0, 'D1': self.D1,
      'SCC': self.SCC, 'VCC': self.VCC, 'EXEC': self.EXEC,
      'EXEC_LO': self._TypedView(self.EXEC, 31, 0), 'EXEC_HI': self._TypedView(self.EXEC, 63, 32),
      'tmp': self.tmp, 'saveexec': self.saveexec,
      'lane': self.lane, 'laneId': self.laneId, 'literal': self.literal,
      'SIMM16': self.SIMM16, 'SIMM32': self.SIMM32, 'VGPR': self.VGPR, 'SRC0': self.SRC0, 'VDST': self.VDST,
    })
    exec(code, ns)
    def _sync(ctx_reg, ns_val):
      if isinstance(ns_val, self._Reg): ctx_reg._val = ns_val._val
      else: ctx_reg._val = int(ns_val) & self._MASK64
    for name in ('SCC', 'VCC', 'EXEC', 'D0', 'D1', 'tmp', 'saveexec'):
      if ns.get(name) is not getattr(self, name): _sync(getattr(self, name), ns[name])

  def result(self) -> dict: return {"d0": self.D0._val, "scc": self.SCC._val & 1}
