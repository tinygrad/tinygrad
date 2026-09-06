# ruff: noqa: F403,F405
from extra.gemm.gemm_mxfp4 import Kernel, build_kernel as build_generic
from tinygrad.runtime.autogen.amd.cdna.ins import *


def build_kernel(tiles_per_wg:int=8, prefetch:bool=True):
  """Persistent 16384x4096x4096 GEMM, reusing the generic 256x256 matrix pipeline.

  s80 is the persistent M-tile index; s84:87 retain the previous C descriptor.
  The input prologue uses v0:7 and v136:234, leaving the epilogue's v8:19
  scratch and v235:250 output addresses intact. Input LDS is reused only
  after the matrix body's final wait/barrier. The next LDS reads are delayed
  until after the current epilogue, when its scratch registers are free.
  """
  insts = build_generic(16384, 4096, 4096, 256, 256, target_optimization=False)
  positions = {x._pos:i for i,x in enumerate(insts)}
  labels = {}
  for x in insts:
    if x._target is not None:
      delta = x.simm16 if x.simm16 < 32768 else x.simm16-65536
      labels[x._target] = positions[x._pos+x.size()+4*delta]
  labels_at = {i:[n for n,j in labels.items() if j == i] for i in labels.values()}
  setup_end = next(i for i,x in enumerate(insts) if x.op_name == 'S_WAITCNT')+1
  lane_decode = next(i for i,x in enumerate(insts) if x.op_name == 'V_LSHRREV_B32_E32')
  group_x = next(i for i,x in enumerate(insts) if str(x) == str(s_mov_b32(s[49], s[2])))
  wave_id = next(i for i,x in enumerate(insts) if x.op_name == 'V_READFIRSTLANE_B32_E32')
  k = Kernel()
  def copy(start, end):
    for i in range(start, end):
      for name in labels_at.get(i, []): k.label(name)
      x = insts[i]
      k.emit(x, target=x._target)
  assert tiles_per_wg in (1, 2, 4, 8)
  if prefetch:
    assert tiles_per_wg > 1
    first_lds = next(i for i,x in enumerate(insts) if x.op_name == 'DS_READ_B128')
    epilogue = labels['L2_3B10']
    assert insts[first_lds-2].op_name == 'S_WAITCNT' and insts[first_lds-1].op_name == 'S_BARRIER'
    assert insts[epilogue].op_name == 'S_WAITCNT' and insts[epilogue+1].op_name == 'S_BARRIER'
    copy(0, setup_end)
    k.emit(s_mov_b32(s[80], s[3]))
    copy(labels['L2_0194']+1, first_lds)
    k.label('compute_tile')
    copy(first_lds, epilogue+2)
    k.emit(s_add_u32(s[80], s[80], 64//tiles_per_wg))
    k.emit(s_cmp_lt_u32(s[80], 64))
    k.emit(s_cbranch_scc0(), target='final_epilogue')
    # Keep the current output descriptor while rebuilding the next input descriptors.
    for j in range(4): k.emit(s_mov_b32(s[84+j], s[4+j]))
    copy(1, lane_decode)
    k.emit(s_mov_b32(s[49], s[2]))
    k.emit(s_mov_b32(s[47], s[80]))
    k.emit(s_waitcnt(49279))
    for x in insts[labels['L2_0194']+1:first_lds-2]:
      if x.op_name != 'V_ACCVGPR_WRITE': k.emit(type(x).from_bytes(x.to_bytes()))
    # Input transfers now overlap output conversion/stores. Clear each accumulator only after reading it.
    for x in insts[epilogue+2:-2]:
      x = type(x).from_bytes(x.to_bytes())
      if x.op_name == 'BUFFER_STORE_DWORDX4': x.srsrc = s[84:87]
      k.emit(x)
      if x.op_name == 'V_ACCVGPR_READ': k.emit(v_accvgpr_write(x.src0, 0))
    k.emit(s_waitcnt(20345))
    k.emit(s_barrier())
    k.emit(s_branch(), target='compute_tile')
    k.label('final_epilogue')
    copy(epilogue+2, len(insts))
  elif tiles_per_wg == 1:
    copy(0, setup_end)
    copy(labels['L2_0194']+1, len(insts))
  else:
    # Decode lane/wave IDs once. s80 survives the original matrix body.
    copy(0, 1)
    copy(lane_decode, group_x)
    copy(wave_id, wave_id+1)
    k.emit(s_mov_b32(s[80], s[3]))
    k.label('tile_loop')
    copy(1, lane_decode)
    k.emit(s_mov_b32(s[49], s[2]))
    k.emit(s_mov_b32(s[47], s[80]))
    k.emit(s_waitcnt(49279))
    copy(labels['L2_0194']+1, len(insts)-1)
    k.emit(s_add_u32(s[80], s[80], 64//tiles_per_wg))
    k.emit(s_cmp_lt_u32(s[80], 64))
    k.emit(s_cbranch_scc1(), target='tile_loop')
    k.emit(s_endpgm())
  return k.finalize()
