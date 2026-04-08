from __future__ import annotations
import functools, math, struct
from typing import Any
from tinygrad.runtime.autogen import mesa
from tinygrad.helpers import data64_le, round_up, next_power2, to_mv

# PM4 packet helpers
def _qreg_exec(__reg, __val=0, **kwargs):
  for k, v in kwargs.items():
    reg_name = f"{__reg[4:]}_{k.removeprefix('_').upper()}"
    __val |= (getattr(mesa, reg_name) if v else 0) if type(v) is bool else (v << getattr(mesa, f'{reg_name}__SHIFT'))
  return __val
qreg: Any = type("QREG", (object,), {name[4:].lower(): functools.partial(_qreg_exec, name) for name in mesa.__dict__.keys() if name[:4] == 'REG_'})

def ctz(v): return (v & -v).bit_length() - 1

def parity(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return mesa.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return mesa.CP_TYPE4_PKT | cnt & 0x7F | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

def parse_ir3_shader(prg, lib):
  """Parse IR3 shader binary and set program metadata attributes on prg."""
  from tinygrad.runtime.support.compiler_mesa import IR3Compiler
  v, cs, imm_vals, prg.image = IR3Compiler.unpack_lib(lib)
  prg.prg_offset, prg.brnchstck, prg.image_size, prg.pvtmem, prg.shmem = 0, v.branchstack, v.info.size, v.pvtmem_size, v.shared_size
  prg.wgsz = alloc.offset_vec4 * 4 + 8 if (alloc:=cs.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]).size_vec4 else 0xfc
  prg.wgid, prg.lid = v.cs.work_group_id, v.cs.local_invocation_id # register ids
  prg.buf_off, imm_off = cs.ubo_state.range[0].offset, cs.allocs.max_const_offset_vec4 * 16
  prg.consts_info = [(struct.unpack_from("<I", imm_vals, i)[0], imm_off + i, 4) for i in range(0, len(imm_vals), 4)]
  # see https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_shader.h#L525
  # and https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L5389
  prg.samp_cnt, prg.tex_cnt, prg.ibo_cnt = (nt:=v.image_mapping.num_tex), nt, v.num_uavs - nt
  prg.tex_to_image = v.image_mapping.tex_to_image[:]
  # IR3 outputs a sampler for every texture (https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L1714)
  prg.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                   qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0] * prg.samp_cnt
  prg.tex_off, prg.ibo_off, prg.samp_off = 2048, 2048 + 0x40 * prg.tex_cnt, 2048 + 0x40 * (prg.tex_cnt + prg.ibo_cnt)
  prg.fregs, prg.hregs = v.info.max_reg + 1, v.info.max_half_reg + 1

def compute_program_sizes(prg):
  """Compute derived sizes from shader metadata. Sets attributes on prg."""
  prg.pvtmem_size_per_item: int = round_up(prg.pvtmem, 512) >> 9
  prg.pvtmem_size_total: int = prg.pvtmem_size_per_item * 128 * 2
  prg.hw_stack_offset: int = round_up(next_power2(round_up(prg.pvtmem, 512)) * 128 * 16, 0x1000)
  prg.shared_size: int = max(1, (prg.shmem - 1) // 1024)
  prg.max_threads = min(1024, ((384 * 32) // (max(1, (prg.fregs + round_up(prg.hregs, 2) // 2)) * 128)) * 128)
  prg.kernargs_alloc_size = round_up(2048 + (prg.tex_cnt + prg.ibo_cnt) * 0x40 + len(prg.samplers) * 4, 0x100)

def build_a6xx_tex_descriptor(imgdt, va_addr, ibo=False):
  """Build a6xx texture/IBO descriptor words."""
  fmt = mesa.FMT6_32_32_32_32_FLOAT if imgdt.itemsize == 4 else mesa.FMT6_16_16_16_16_FLOAT
  return [qreg.a6xx_tex_const_0(fmt=fmt) if ibo else qreg.a6xx_tex_const_0(0x8, swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3, fmt=fmt),
          qreg.a6xx_tex_const_1(width=imgdt.shape[1], height=imgdt.shape[0]),
          qreg.a6xx_tex_const_2(type=mesa.A6XX_TEX_2D, pitch=imgdt.pitch, pitchalign=ctz(imgdt.pitch)-6), 0, *data64_le(va_addr),
          qreg.a6xx_tex_const_6(plane_pitch=0x400000), qreg.a6xx_tex_const_7(13), 0, 0, 0, 0, 0, 0, 0, 0]

def build_a6xx_compute_pm4(cmd, reg, prg, args_va, lib_va, stack_va, border_color_va, global_size, local_size, *, nir=True):
  """Emit a6xx compute dispatch PM4 register sequence.
  cmd(opcode, *vals) and reg(register, *vals) are PM4 append primitives."""
  def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
  global_size_mp = [cast_int(g*l) for g,l in zip(global_size, local_size)]
  isammode = mesa.ISAMMODE_GL if nir else mesa.ISAMMODE_CL

  cmd(mesa.CP_SET_MARKER, qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE))
  reg(mesa.REG_A6XX_SP_UPDATE_CNTL, qreg.a6xx_sp_update_cntl(cs_state=True, cs_uav=True))
  reg(mesa.REG_A6XX_SP_UPDATE_CNTL, 0x0)
  reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(0x80)) # is this right? mesa uses 1
  reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(0x40)) # mesa also uses 1
  reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=isammode))
  reg(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, qreg.a6xx_sp_perfctr_shader_mask(cs=True))
  reg(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=isammode))
  reg(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
  cmd(mesa.CP_WAIT_FOR_IDLE)

  reg(mesa.REG_A6XX_SP_CS_NDRANGE_0,
      qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
      global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf, 0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=mesa.THREAD64),
      cast_int(global_size[0], ceil=True), cast_int(global_size[1], ceil=True), cast_int(global_size[2], ceil=True))

  reg(mesa.REG_A6XX_SP_CS_CNTL_0,
      qreg.a6xx_sp_cs_cntl_0(threadsize=mesa.THREAD64, halfregfootprint=prg.hregs, fullregfootprint=prg.fregs, branchstack=prg.brnchstck),
      qreg.a6xx_sp_cs_cntl_1(constantrammode=mesa.CONSTLEN_256, shared_size=prg.shared_size), # should this be CONSTLEN_512?
      0, prg.prg_offset, *data64_le(lib_va),
      qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=prg.pvtmem_size_per_item), *data64_le(stack_va),
      qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=prg.pvtmem_size_total))

  if nir and prg.wgsz != 0xfc: to_mv(args_va + prg.wgsz * 4, 12)[:] = struct.pack("III", *local_size)
  cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                       state_block=mesa.SB6_CS_SHADER, num_unit=1024 // 4),
      *data64_le(args_va))
  cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                       state_block=mesa.SB6_CS_SHADER, num_unit=round_up(prg.image_size, 128) // 128),
      *data64_le(lib_va))

  reg(mesa.REG_A6XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc, qreg.a6xx_sp_cs_const_config(constlen=1024 // 4, enabled=True))

  reg(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_stack_offset(prg.hw_stack_offset))
  reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE, qreg.a6xx_sp_cs_instr_size(prg.image_size // 4))

  if prg.samp_cnt > 0:
    cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                         state_block=mesa.SB6_CS_TEX, num_unit=prg.samp_cnt),
        *data64_le(args_va + prg.samp_off))
    reg(mesa.REG_A6XX_SP_CS_SAMPLER_BASE, *data64_le(args_va + prg.samp_off))
    reg(mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, *data64_le(border_color_va))

  if prg.tex_cnt > 0:
    cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                         state_block=mesa.SB6_CS_TEX, num_unit=min(16, prg.tex_cnt)),
        *data64_le(args_va + prg.tex_off))
    reg(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, *data64_le(args_va + prg.tex_off))

  if prg.ibo_cnt > 0:
    cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST6_UAV, state_src=mesa.SS6_INDIRECT,
                                                         state_block=mesa.SB6_CS_SHADER, num_unit=prg.ibo_cnt),
        *data64_le(args_va + prg.ibo_off))
    reg(mesa.REG_A6XX_SP_CS_UAV_BASE, *data64_le(args_va + prg.ibo_off))

  reg(mesa.REG_A6XX_SP_CS_CONFIG,
      qreg.a6xx_sp_cs_config(enabled=True, nsamp=prg.samp_cnt, ntex=prg.tex_cnt, nuav=prg.ibo_cnt))

  if nir:
    reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
        qreg.a6xx_sp_cs_const_config_0(wgidconstid=prg.wgid, wgsizeconstid=prg.wgsz, wgoffsetconstid=0xfc, localidregid=prg.lid),
        qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=mesa.THREAD64))
    cmd(mesa.CP_EXEC_CS, 0,
        qreg.cp_exec_cs_1(ngroups_x=global_size[0]), qreg.cp_exec_cs_2(ngroups_y=global_size[1]), qreg.cp_exec_cs_3(_ngroups_z=global_size[2]))
  else: cmd(mesa.CP_RUN_OPENCL, 0)
