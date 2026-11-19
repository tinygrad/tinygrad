from __future__ import annotations
import dataclasses, struct

# Rogue PDS instructions used by compute launches. These are the same encodings
# as pvr_pds_compute_shader(), restricted to features exposed by MTGPU metadata.
PDS_UNUSED = 0xffffffff

@dataclasses.dataclass(frozen=True)
class MTPDSProgram:
  code: bytes
  data: bytes
  temps_used: int = 3

  @classmethod
  def from_words(cls, code:list[int], data:list[int]) -> MTPDSProgram:
    return cls(struct.pack(f'<{len(code)}I',*code),struct.pack(f'<{len(data)}I',*data))

def _bra(addr:int) -> int: return 0xc0000000 | (2<<24) | (1<<23) | (8<<19) | addr
def _halt() -> int: return 0xd0000000 | (6<<23)
def _doutu(src0:int=0) -> int: return 0xf0000000 | (1<<26) | (src0<<8) | 2
def _doutw(src1:int, src0:int, end:bool=False) -> int:
  return 0xf0000000 | (int(end)<<26) | (src1<<16) | (src0<<8) | 1

def _control(dest:int, part:str, store:str, last:bool) -> int:
  masks = {'lower32':0,'upper32':1,'lower64':2}
  stores = {'unified':0,'common':1}
  return (dest<<13) | masks[part] | (stores[store]<<28) | (int(last)<<31)

def _reg_list(regs:tuple[int,int,int]) -> list[int]: return [i for i,reg in enumerate(regs) if reg != PDS_UNUSED]

def compute_pds(execution_address:int, local_regs:tuple[int,int,int]=(PDS_UNUSED,)*3,
                workgroup_regs:tuple[int,int,int]=(PDS_UNUSED,)*3, barrier_coefficient:int=PDS_UNUSED,
                coefficient_update:bool=False, prefix:tuple[int,...]=()) -> MTPDSProgram:
  """Build compute PDS code/data for one MTGPU kernel.

  Constant zero is the 64-bit USC task-control word. ID control words follow
  it in the same order as their DOUTW instructions. Workgroup/local IDs are
  sourced from CDM temp registers 64 and 65, as in pvr_pds_compute_shader().
  """
  code, data = [_bra(2)], list(prefix) or [execution_address & 0xffffffff,execution_address >> 32]
  local, workgroup = _reg_list(local_regs), _reg_list(workgroup_regs)
  if coefficient_update: data += [execution_address & 0xffffffff,execution_address >> 32]

  if barrier_coefficient != PDS_UNUSED:
    code.append(_doutw(4,1))  # barrier control at dword 4, zero64 at dword 2
    data += [0,0,_control(barrier_coefficient,'lower64','common',not workgroup)]

  if 0 in workgroup and 1 in workgroup:
    code.append(_doutw(len(data),64))
    data.append(_control(workgroup_regs[0],'lower64','common',2 not in workgroup))
  else:
    for i in (0,1):
      if i in workgroup:
        code.append(_doutw(len(data),64))
        data.append(_control(workgroup_regs[i],'lower32' if i == 0 else 'upper32','common',2 not in workgroup))
  if 2 in workgroup:
    code.append(_doutw(len(data),65))
    data.append(_control(workgroup_regs[2],'upper32','common',True))

  if coefficient_update: code.append(_doutu(1))
  code.append(_halt())
  branch_target = len(code)
  if local:
    if 0 in local and (1 in local or 2 in local): dest,part = local_regs[0],'lower64'
    elif 0 in local: dest,part = local_regs[0],'lower32'
    else: dest,part = local_regs[1] if 1 in local else local_regs[2],'upper32'
    code.append(_doutw(len(data),64))
    data.append(_control(dest,part,'unified',True))

  code[0] = _bra(branch_target)
  code += [_doutu(),_halt()]
  return MTPDSProgram.from_words(code,data)
