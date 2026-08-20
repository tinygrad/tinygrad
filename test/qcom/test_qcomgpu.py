import ctypes
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import data64_le, pkt4_hdr, pkt7_hdr, qreg
from test.mockgpu.qcom.qcomgpu import QCOMGPU, decode_pm4

def test_wait_for_idle_packet():
  packets = decode_pm4([pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0)])
  assert packets == [(7, mesa.CP_WAIT_FOR_IDLE, ())]
  
def test_register_packet():
  packets = decode_pm4([pkt4_hdr(mesa.REG_A6XX_SP_UPDATE_CNTL, 2), 1, 2])
  assert packets == [(4, mesa.REG_A6XX_SP_UPDATE_CNTL, (1, 2))]
  
def test_register_write():
  gpu = QCOMGPU()
  reg = mesa.REG_A6XX_SP_UPDATE_CNTL
  gpu.execute([pkt4_hdr(reg, 2), 10, 20])
  assert gpu.regs == {reg: 10, reg + 1: 20}
  
def test_wait_reg_mem_ready():
  value = ctypes.c_uint32(5)
  payload = (qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
    *data64_le(ctypes.addressof(value)), qreg.cp_wait_reg_mem_3(ref=4),
    qreg.cp_wait_reg_mem_4(mask=0xffffffff), qreg.cp_wait_reg_mem_5(delay_loop_cycles=32))
  QCOMGPU().execute([pkt7_hdr(mesa.CP_WAIT_REG_MEM, len(payload)), *payload])
  
def test_event_write():
  value = ctypes.c_uint32()
  payload = (qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS),
    *data64_le(ctypes.addressof(value)), 7)
  QCOMGPU().execute([pkt7_hdr(mesa.CP_EVENT_WRITE, len(payload)), *payload])
  assert value.value == 7

def test_wait_mem_writes():
  QCOMGPU().execute([pkt7_hdr(mesa.CP_WAIT_MEM_WRITES, 0)])
  
def test_set_compute_marker():
  payload = (qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE),)
  QCOMGPU().execute([pkt7_hdr(mesa.CP_SET_MARKER, len(payload)), *payload])
  
def test_load_state6():
  address = ctypes.addressof(ctypes.c_uint32())
  control = qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS,
    state_src=mesa.SS6_INDIRECT, state_block=mesa.SB6_CS_SHADER, num_unit=1)
  gpu = QCOMGPU()
  gpu.execute([pkt7_hdr(mesa.CP_LOAD_STATE6_FRAG, 3), control, *data64_le(address)])
  assert gpu.state[(mesa.SB6_CS_SHADER, mesa.ST_CONSTANTS)] == address
  
def test_exec_cs():
  image = (ctypes.c_ubyte * 128)(*range(128))
  calls = []
  gpu = QCOMGPU(lambda code, grid, local, regs, state: calls.append((code, grid, local, regs, state)))
  gpu.state[(mesa.SB6_CS_SHADER, mesa.ST_SHADER)] = ctypes.addressof(image)
  gpu.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0] = qreg.a6xx_sp_cs_ndrange_0(
    kerneldim=3, localsizex=5, localsizey=6, localsizez=7)
  gpu.regs[mesa.REG_A6XX_SP_CS_INSTR_SIZE] = qreg.a6xx_sp_cs_instr_size(1)
  payload = (0, qreg.cp_exec_cs_1(ngroups_x=2),
    qreg.cp_exec_cs_2(ngroups_y=3), qreg.cp_exec_cs_3(_ngroups_z=4))
  gpu.execute([pkt7_hdr(mesa.CP_EXEC_CS, len(payload)), *payload])
  assert calls == [(bytes(image), (2, 3, 4), (5, 6, 7), gpu.regs, gpu.state)]

def test_shader_image():
  image = (ctypes.c_ubyte * 128)(*range(128))
  gpu = QCOMGPU()
  gpu.state[(mesa.SB6_CS_SHADER, mesa.ST_SHADER)] = ctypes.addressof(image)
  gpu.regs[mesa.REG_A6XX_SP_CS_INSTR_SIZE] = qreg.a6xx_sp_cs_instr_size(1)
  assert gpu.shader_image() == bytes(image)
  
def test_local_size():
  gpu = QCOMGPU()
  gpu.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0] = qreg.a6xx_sp_cs_ndrange_0(
    kerneldim=3, localsizex=2, localsizey=3, localsizez=4)
  assert gpu.local_size() == (2, 3, 4)
