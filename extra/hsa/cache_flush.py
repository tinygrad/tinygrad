import ctypes
import gpuctypes.hsa as hsa
from tinygrad.runtime.driver.hsa import *
from tinygrad.runtime.ops_hsa import HSADevice, HSAProgram
from tinygrad.helpers import Timing, from_mv

class AmdAqlPm4Ib(ctypes.Structure):
    _fields_ = [
        ("header", ctypes.c_uint16),
        ("amd_format", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint8),
        ("ib_jump_cmd", ctypes.c_uint32 * 4),
        ("dw_cnt_remain", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32 * 8),
        ("completion_signal", hsa.hsa_signal_t)
    ]
    _pack_ = 1

if __name__ == "__main__":
  device = HSADevice("HSA:0")
  device_1 = HSADevice("HSA:1")
  
  # https://lxr.missinglinkelectronics.com/linux/drivers/gpu/drm/amd/amdgpu/nvd.h
  pm4_cache_inv_cmd_len = 8
  pm4_cache_inv_cmd = [
    PM4_HDR(PM4_HDR_IT_OPCODE_ACQUIRE_MEM, pm4_cache_inv_cmd_len, 11),
    0,
    PM4_ACQUIRE_MEM_DW2_COHER_SIZE(0xFFFFFFFF),
    PM4_ACQUIRE_MEM_DW3_COHER_SIZE_HI(0xFF),
    0,
    0,
    0,
    PM4_ACQUIRE_MEM_DW7_GCR_CNTL(
      PM4_ACQUIRE_MEM_GCR_CNTL_GLI_INV(1) |
      # PM4_ACQUIRE_MEM_GCR_CNTL_GLK_INV |
      # PM4_ACQUIRE_MEM_GCR_CNTL_GLV_INV |
      PM4_ACQUIRE_MEM_GCR_CNTL_GL1_INV |
      PM4_ACQUIRE_MEM_GCR_CNTL_GL2_INV |
      0)
  ]

  assert len(pm4_cache_inv_cmd) == pm4_cache_inv_cmd_len
  device.hw_queue.submit_pm4(pm4_cache_inv_cmd, pm4_cache_inv_cmd_len)

  l2_cache_size = 3 << 20
  buf0 = device.allocator.alloc(l2_cache_size)
  buf1 = device.allocator.alloc(l2_cache_size)
  for i in range(4):
    check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(copy_signal := hsa.hsa_signal_t())))
    with Timing("copy     l2 ", lambda x: f" {l2_cache_size/x:.2f} GB/s"):
      check(hsa.hsa_amd_memory_async_copy(buf0, device.agent, buf1, device.agent, l2_cache_size, 0, None, copy_signal))
      hsa.hsa_signal_wait_acquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)

  device.hw_queue.submit_pm4(pm4_cache_inv_cmd, pm4_cache_inv_cmd_len)

  check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(copy_signal := hsa.hsa_signal_t())))
  with Timing("copy inv l2 ", lambda x: f" {l2_cache_size/x:.2f} GB/s"):
    check(hsa.hsa_amd_memory_async_copy(buf0, device.agent, buf1, device.agent, l2_cache_size, 0, None, copy_signal))
    hsa.hsa_signal_wait_acquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)

  for i in range(2):
    check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(copy_signal := hsa.hsa_signal_t())))
    with Timing("copy     l2 ", lambda x: f" {l2_cache_size/x:.2f} GB/s"):
      check(hsa.hsa_amd_memory_async_copy(buf0, device.agent, buf1, device.agent, l2_cache_size, 0, None, copy_signal))
      hsa.hsa_signal_wait_acquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)

  device.hw_queue.submit_pm4(pm4_cache_inv_cmd, pm4_cache_inv_cmd_len)

  check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(copy_signal := hsa.hsa_signal_t())))
  with Timing("copy inv l2 ", lambda x: f" {l2_cache_size/x:.2f} GB/s"):
    check(hsa.hsa_amd_memory_async_copy(buf0, device.agent, buf1, device.agent, l2_cache_size, 0, None, copy_signal))
    hsa.hsa_signal_wait_acquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)
