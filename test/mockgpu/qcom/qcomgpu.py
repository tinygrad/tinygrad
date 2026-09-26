import ctypes
from tinygrad.helpers import to_mv
from tinygrad.runtime.autogen import mesa
from test.mockgpu.gpu import VirtGPU
from test.mockgpu.qcom.ir3 import launch

def _fld(val, field): return (val & getattr(mesa, f"{field}__MASK")) >> getattr(mesa, f"{field}__SHIFT")

class QCOMGPU(VirtGPU):
  def __init__(self):
    super().__init__(0)
    self.timestamp, self.mapped_ranges, self.shader_sz = 0, [], 0

  def map_range(self, vaddr, size): self.mapped_ranges.append((vaddr, size))
  def unmap_range(self, vaddr, size): self.mapped_ranges.remove((vaddr, size))

  def _clip(self, addr, want):
    for base, sz in self.mapped_ranges:
      if base <= addr < base + sz: return min(want, base + sz - addr)
    return min(want, 4096)

  def submit(self, addr, size):
    self.timestamp, self.shader_sz = (self.timestamp + 1) & 0xffffffff, 0
    words, i, n = to_mv(addr, size).cast('I'), 0, size // 4
    regs, const = {}, bytearray()
    while i < n and (hdr:=int(words[i])):
      if (hdr & mesa.CP_TYPE7_PKT) == mesa.CP_TYPE4_PKT:
        cnt, reg = hdr & 0x7f, (hdr >> 8) & 0x3ffff
        for k in range(cnt): regs[reg + k] = int(words[i + 1 + k])
        i += 1 + cnt
      elif (hdr & mesa.CP_TYPE7_PKT) == mesa.CP_TYPE7_PKT:
        cnt, opc = hdr & 0x3fff, (hdr >> 16) & 0x7f
        body = [int(words[i + 1 + k]) for k in range(cnt)]
        i += 1 + cnt
        if opc in (mesa.CP_EXEC_CS, mesa.CP_RUN_OPENCL): self._launch(regs, const, body if opc == mesa.CP_EXEC_CS else None)
        elif opc in (mesa.CP_LOAD_STATE6, mesa.CP_LOAD_STATE6_FRAG): const = self._load(const, body)
        elif opc == mesa.CP_EVENT_WRITE: self._event(body)
        elif opc == mesa.CP_REG_TO_MEM: self._reg_to_mem(body)
        elif opc == mesa.CP_WAIT_REG_MEM: self._wait(body)
        elif opc in (mesa.CP_NOP, mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES, mesa.CP_SET_MARKER): pass
        else: raise RuntimeError(f'unhandled pm4 opcode {opc}')
      else: raise RuntimeError(f'bad pm4 header {hdr:#x}')
    return self.timestamp

  def _load(self, const, body):
    state = body[0]
    dst, typ, src = _fld(state, 'CP_LOAD_STATE6_0_DST_OFF'), _fld(state, 'CP_LOAD_STATE6_0_STATE_TYPE'), _fld(state, 'CP_LOAD_STATE6_0_STATE_SRC')
    block, nunit = _fld(state, 'CP_LOAD_STATE6_0_STATE_BLOCK'), _fld(state, 'CP_LOAD_STATE6_0_NUM_UNIT')
    if src == mesa.SS6_INDIRECT and len(body) >= 3 and typ == mesa.ST_SHADER and block == mesa.SB6_CS_SHADER: self.shader_sz = nunit * 128
    if typ == mesa.ST_CONSTANTS and block == mesa.SB6_CS_SHADER and src == mesa.SS6_INDIRECT and len(body) >= 3 and nunit:
      addr, nbytes, off = body[1] | (body[2] << 32), nunit * 16, dst * 16
      buf = bytearray(const)
      if len(buf) < off + nbytes: buf.extend(bytes(off + nbytes - len(buf)))
      buf[off:off + nbytes] = to_mv(addr, nbytes).tobytes()
      return buf
    return const

  def _launch(self, regs, const, pkt):
    nd = regs.get(mesa.REG_A6XX_SP_CS_NDRANGE_0, 0)
    lx = _fld(nd, 'A6XX_SP_CS_NDRANGE_0_LOCALSIZEX') + 1
    ly = _fld(nd, 'A6XX_SP_CS_NDRANGE_0_LOCALSIZEY') + 1
    lz = _fld(nd, 'A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ') + 1
    if pkt is not None and len(pkt) >= 4: gx, gy, gz = pkt[1], pkt[2], pkt[3]
    else: gx, gy, gz = (regs.get(mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X, 1), regs.get(mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Y, 1),
                        regs.get(mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Z, 1))
    cfg = regs.get(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 0xfcfcfcfc)
    wgid, lid = _fld(cfg, 'A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID'), _fld(cfg, 'A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID')
    lib = regs.get(mesa.REG_A6XX_SP_CS_BASE, 0) | (regs.get(mesa.REG_A6XX_SP_CS_BASE + 1, 0) << 32)
    prg = regs.get(mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET, 0)
    image = to_mv(lib + prg, self._clip(lib + prg, self.shader_sz or 4096)).tobytes()
    tex = regs.get(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, 0) | (regs.get(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE + 1, 0) << 32)
    uav = regs.get(mesa.REG_A6XX_SP_CS_UAV_BASE, 0) | (regs.get(mesa.REG_A6XX_SP_CS_UAV_BASE + 1, 0) << 32)
    psz = _fld(regs.get(mesa.REG_A6XX_SP_CS_PVT_MEM_PARAM, 0), 'A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM') << 9
    if not lib: return
    launch(image, const, gx, gy, gz, lx, ly, lz, lid, wgid, 1 << 16, psz, tex, uav, self.mapped_ranges)

  def _event(self, body):
    if len(body) < 4 or _fld(body[0], 'CP_EVENT_WRITE_0_EVENT') not in (mesa.CACHE_FLUSH_TS, mesa.CACHE_FLUSH_AND_INV_TS_EVENT): return
    ctypes.c_uint64.from_address(body[1] | (body[2] << 32)).value = body[3]
  def _reg_to_mem(self, body):
    if len(body) < 3: return
    ctypes.c_uint64.from_address(body[1] | (body[2] << 32)).value = self.timestamp
  def _wait(self, body):
    if len(body) < 4: return
    if (got:=ctypes.c_uint32.from_address(body[1] | (body[2] << 32)).value) < body[3]: raise RuntimeError(f'pm4 wait {got} < {body[3]}')
