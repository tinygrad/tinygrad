# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
MTFWIF_CONNECTION_FW_STATE: dict[int, str] = {(MTFW_CONNECTION_FW_OFFLINE:=0): 'MTFW_CONNECTION_FW_OFFLINE', (MTFW_CONNECTION_FW_READY:=1): 'MTFW_CONNECTION_FW_READY', (MTFW_CONNECTION_FW_ACTIVE:=2): 'MTFW_CONNECTION_FW_ACTIVE', (MTFW_CONNECTION_FW_OFFLOADING:=3): 'MTFW_CONNECTION_FW_OFFLOADING', (MTFW_CONNECTION_FW_DECLINE:=4): 'MTFW_CONNECTION_FW_DECLINE', (MTFW_CONNECTION_FW_STATE_COUNT:=5): 'MTFW_CONNECTION_FW_STATE_COUNT'}
MTFWIF_CONNECTION_OS_STATE: dict[int, str] = {(MTFW_CONNECTION_OS_OFFLINE:=0): 'MTFW_CONNECTION_OS_OFFLINE', (MTFW_CONNECTION_OS_READY:=1): 'MTFW_CONNECTION_OS_READY', (MTFW_CONNECTION_OS_ACTIVE:=2): 'MTFW_CONNECTION_OS_ACTIVE', (MTFW_CONNECTION_OS_STATE_COUNT:=3): 'MTFW_CONNECTION_OS_STATE_COUNT'}
@c.record
class MTFW_FWIF_STARTUP_INFO(c.Struct):
  SIZE = 4
  field: MTFW_FWIF_STARTUP_INFO_field
  value: int
@c.record
class MTFW_FWIF_STARTUP_INFO_field(c.Struct):
  SIZE = 4
  magicValue: int
  threadNum: int
uint32_t: TypeAlias = ctypes.c_uint32
MTFW_FWIF_STARTUP_INFO_field.register_fields([('magicValue', uint32_t, 0, 8, 0), ('threadNum', uint32_t, 1, 1, 0)])
MTFW_FWIF_STARTUP_INFO.register_fields([('field', MTFW_FWIF_STARTUP_INFO_field, 0), ('value', uint32_t, 0)])
@c.record
class MTFW_FWIF_WATCHDOG(c.Struct):
  SIZE = 16
  t0WatchDogCounter: int
  t1WatchDogCounter: int
  t0State: int
  t1State: int
MTFW_FWIF_WATCHDOG.register_fields([('t0WatchDogCounter', uint32_t, 0), ('t1WatchDogCounter', uint32_t, 4), ('t0State', uint32_t, 8), ('t1State', uint32_t, 12)])
@c.record
class MTFW_FWIF_VGPU(c.Struct):
  SIZE = 48
  vgpuFWVersion: int
  vgpuFlag: int
  osIdStart: int
  osIdCount: int
  vgpuSchedulingPolicy: int
  vgpuTimeSlicedValue: int
  vgpuUtil: int
  vgpuQosEnable: int
  vgpuSharedInfoAddr: int
  vgpuReleaseVersion: int
uint64_t: TypeAlias = ctypes.c_uint64
fw_va: TypeAlias = ctypes.c_uint64
MTFW_FWIF_VGPU.register_fields([('vgpuFWVersion', uint64_t, 0), ('vgpuFlag', uint64_t, 0), ('osIdStart', uint32_t, 8), ('osIdCount', uint32_t, 12), ('vgpuSchedulingPolicy', uint32_t, 16), ('vgpuTimeSlicedValue', uint32_t, 20), ('vgpuUtil', uint32_t, 24), ('vgpuQosEnable', uint32_t, 28), ('vgpuSharedInfoAddr', fw_va, 32), ('vgpuReleaseVersion', uint64_t, 40)])
@c.record
class struct_MTFW_FWIF_TAG(c.Struct):
  SIZE = 265952
  flags: struct_MTFW_FWIF_TAG_flags
  fwStarted: int
  fwBuildVer: c.Array[ctypes.c_ubyte, Literal[8]]
  watchDog: int
  ccbQueueArrayAddr: int
  ccbFastQueueArrayAddr: int
  levelQueueArrayFecAddr: int
  levelQueueArrayMetaAddr: int
  nodeQueueArrayAddr: int
  ccbQueueArrayNum: int
  ccbFastQueueArrayNum: int
  nodeQueueArrayNum: int
  levelQueueArrayNum: int
  startUpCounter: int
  ccbQueueArray: c.Array[MTFW_CCB_QUEUE, Literal[5]]
  nodeQueueArray: c.Array[MTFW_NODE_QUEUE, Literal[6]]
  traceBufAddr: int
  traceBufAddrFEC: int
  metaPerfEventBufCtrlArray: int
  fecPerfEventBufCtrlArray: int
  metaPerfEventBufCtrlNum: int
  fecPerfEventBufCtrlNum: int
  faultDumpBuffer: int
  gpuUtilBufAddr: int
  vgpuFWVersion: int
  vgpuFlag: int
  osIdStart: int
  osIdCount: int
  vgpuSchedulingPolicy: int
  vgpuTimeSlicedValue: int
  vgpuUtil: int
  vgpuQosEnable: int
  vgpuSharedInfoAddr: int
  vgpuReleaseVersion: int
  socTimerFreq: int
  gpuFrequence: int
  multicoreClusterMask: int
  fwCfgs: struct_MTFW_FWIF_TAG_fwCfgs
  firmwareCoreMemory: c.Array[ctypes.c_ubyte, Literal[98304]]
@c.record
class struct_MTFW_FWIF_TAG_flags(c.Struct):
  SIZE = 4
  field: struct_MTFW_FWIF_TAG_flags_field
  value: int
@c.record
class struct_MTFW_FWIF_TAG_flags_field(c.Struct):
  SIZE = 4
  version: int
struct_MTFW_FWIF_TAG_flags_field.register_fields([('version', uint32_t, 0, 4, 0)])
struct_MTFW_FWIF_TAG_flags.register_fields([('field', struct_MTFW_FWIF_TAG_flags_field, 0), ('value', uint32_t, 0)])
uint8_t: TypeAlias = ctypes.c_ubyte
@c.record
class MTFW_CCB_QUEUE(c.Struct):
  SIZE = 16432
  reqCcbQ: c.Array[c.Array[MTFW_CCB_Q_CELL, Literal[256]], Literal[2]]
  respCcbQ: c.Array[MTFW_CCB_RESP, Literal[256]]
  reqCcbCtrl: c.Array[MTFW_RING_CTRL, Literal[2]]
  respCcbCtrl: MTFW_RING_CTRL
@c.record
class MTFW_CCB_Q_CELL(c.Struct):
  SIZE = 8
  ccbItemAddr: int
MTFW_CCB_Q_CELL.register_fields([('ccbItemAddr', fw_va, 0)])
@c.record
class MTFW_CCB_RESP(c.Struct):
  SIZE = 48
  flags: MTFW_FWIF_HEAD
  respValue: int
  ccbItemToken: int
  hostPrivateData: int
  startTimestamp: int
  runTime: int
  extInfo: MTFW_CCB_RESP_EXT_INFO
@c.record
class MTFW_FWIF_HEAD(c.Struct):
  SIZE = 4
  field: MTFW_FWIF_HEAD_field
  value: int
@c.record
class MTFW_FWIF_HEAD_field(c.Struct):
  SIZE = 4
  version: int
  crc: int
MTFW_FWIF_HEAD_field.register_fields([('version', uint32_t, 0, 12, 0), ('crc', uint32_t, 1, 16, 4)])
MTFW_FWIF_HEAD.register_fields([('field', MTFW_FWIF_HEAD_field, 0), ('value', uint32_t, 0)])
MTFW_CCB_RESP_VALUE: dict[int, str] = {(MTFW_CCB_RESP_DONE:=0): 'MTFW_CCB_RESP_DONE', (MTFW_CCB_RESP_CANCEL:=1): 'MTFW_CCB_RESP_CANCEL', (MTFW_CCB_RESP_BREAK:=2): 'MTFW_CCB_RESP_BREAK', (MTFW_CCB_RESP_PARTIAL:=3): 'MTFW_CCB_RESP_PARTIAL', (MTFW_CCB_RESP_QSEMP_TIMEOUT:=7): 'MTFW_CCB_RESP_QSEMP_TIMEOUT', (MTFW_CCB_RESP_STALL_TIMEOUT:=8): 'MTFW_CCB_RESP_STALL_TIMEOUT', (MTFW_CCB_RESP_ENGINE_SYNC_TIMEOUT:=9): 'MTFW_CCB_RESP_ENGINE_SYNC_TIMEOUT', (MTFW_CCB_EVENT_START:=80): 'MTFW_CCB_EVENT_START', (MTFW_CCB_EVENT_PERF:=80): 'MTFW_CCB_EVENT_PERF', (MTFW_CCB_RESP_ERROR_HANDLE_START:=256): 'MTFW_CCB_RESP_ERROR_HANDLE_START', (MTFW_CCB_RESP_INVALID:=256): 'MTFW_CCB_RESP_INVALID', (MTFW_CCB_RESP_FAULT:=257): 'MTFW_CCB_RESP_FAULT', (MTFW_CCB_RESP_ERROR:=258): 'MTFW_CCB_RESP_ERROR', (MTFW_CCB_RESP_VALUE_INVALID:=259): 'MTFW_CCB_RESP_VALUE_INVALID'}
@c.record
class MTFW_CCB_RESP_EXT_INFO(c.Struct):
  SIZE = 8
  extInfo: int
MTFW_CCB_RESP_EXT_INFO.register_fields([('extInfo', uint64_t, 0)])
MTFW_CCB_RESP.register_fields([('flags', MTFW_FWIF_HEAD, 0), ('respValue', ctypes.c_uint32, 4), ('ccbItemToken', uint64_t, 8), ('hostPrivateData', uint64_t, 16), ('startTimestamp', uint64_t, 24), ('runTime', uint32_t, 32), ('extInfo', MTFW_CCB_RESP_EXT_INFO, 40)])
@c.record
class MTFW_RING_CTRL(c.Struct):
  SIZE = 16
  wrIdx: int
  wrIdxAddr: int
  rdIdx: int
  reserve: int
MTFW_RING_CTRL.register_fields([('wrIdx', uint32_t, 0), ('wrIdxAddr', fw_va, 0), ('rdIdx', uint32_t, 8), ('reserve', uint32_t, 12)])
MTFW_CCB_QUEUE.register_fields([('reqCcbQ', c.Array[c.Array[MTFW_CCB_Q_CELL, Literal[256]], Literal[2]], 0), ('respCcbQ', c.Array[MTFW_CCB_RESP, Literal[256]], 4096), ('reqCcbCtrl', c.Array[MTFW_RING_CTRL, Literal[2]], 16384), ('respCcbCtrl', MTFW_RING_CTRL, 16416)])
@c.record
class MTFW_NODE_QUEUE(c.Struct):
  SIZE = 12848
  reqNodeQ: c.Array[c.Array[MTFW_NODE_ITEM, Literal[64]], Literal[2]]
  respNodeQ: c.Array[MTFW_NODE_RESP, Literal[64]]
  reqNodeCtrl: c.Array[MTFW_RING_CTRL, Literal[2]]
  respNodeCtrl: MTFW_RING_CTRL
@c.record
class MTFW_NODE_ITEM(c.Struct):
  SIZE = 88
  flags: MTFW_FWIF_HEAD
  priority: int
  attribute: MTFW_SUBMISSION_ATTRIBUTE
  submissionCmd: int
  ccbItemToken: int
  hostPrivateData: int
  itemPara: union__MTFW_NODE_ITEM_PARA_
  sequenceNums: int
  processId: int
@c.record
class MTFW_SUBMISSION_ATTRIBUTE(c.Struct):
  SIZE = 4
  field: MTFW_SUBMISSION_ATTRIBUTE_field
  value: int
@c.record
class MTFW_SUBMISSION_ATTRIBUTE_field(c.Struct):
  SIZE = 4
  errorPause: int
  userPause: int
  reSubmit: int
MTFW_SUBMISSION_ATTRIBUTE_field.register_fields([('errorPause', uint32_t, 0, 1, 0), ('userPause', uint32_t, 0, 1, 1), ('reSubmit', uint32_t, 0, 1, 2)])
MTFW_SUBMISSION_ATTRIBUTE.register_fields([('field', MTFW_SUBMISSION_ATTRIBUTE_field, 0), ('value', uint32_t, 0)])
MTFW_SUBMISSION_CMD_TYPE: dict[int, str] = {(MTFW_SUBMISSION_CMD_NOP:=0): 'MTFW_SUBMISSION_CMD_NOP', (MTFW_SUBMISSION_CMD_CONTINUE:=1): 'MTFW_SUBMISSION_CMD_CONTINUE', (MTFW_SUBMISSION_CMD_CANCEL:=2): 'MTFW_SUBMISSION_CMD_CANCEL', (MTFW_SUBMISSION_CMD_TERMINATED:=3): 'MTFW_SUBMISSION_CMD_TERMINATED', (MTFW_SUBMISSION_CMD_CANCEL_SUBM_Q:=4): 'MTFW_SUBMISSION_CMD_CANCEL_SUBM_Q', (MTFW_SUBMISSION_CMD_TERMINATED_SUBM_Q:=5): 'MTFW_SUBMISSION_CMD_TERMINATED_SUBM_Q', (MTFW_SUBMISSION_CMD_INIT_Q_SEMAPHORE:=6): 'MTFW_SUBMISSION_CMD_INIT_Q_SEMAPHORE', (MTFW_SUBMISSION_CMD_WRITE_Q_SEMAPHORE:=7): 'MTFW_SUBMISSION_CMD_WRITE_Q_SEMAPHORE', (MTFW_SUBMISSION_CMD_SCH_RESUME:=8): 'MTFW_SUBMISSION_CMD_SCH_RESUME', (MTFW_SUBMISSION_CMD_SCH_SKIP:=9): 'MTFW_SUBMISSION_CMD_SCH_SKIP', (MTFW_SUBMISSION_CMD_SET_LOG_GROUP:=10): 'MTFW_SUBMISSION_CMD_SET_LOG_GROUP', (MTFW_SUBMISSION_CMD_SET_LOG_GROUP_FEC:=11): 'MTFW_SUBMISSION_CMD_SET_LOG_GROUP_FEC', (MTFW_SUBMISSION_CMD_SET_PERF_REQUESET:=12): 'MTFW_SUBMISSION_CMD_SET_PERF_REQUESET', (MTFW_SUBMISSION_CMD_FLUSH_MMUCACHE:=13): 'MTFW_SUBMISSION_CMD_FLUSH_MMUCACHE', (MTFW_SUBMISSION_CMD_FLUSH_ALL_MMUCACHE:=14): 'MTFW_SUBMISSION_CMD_FLUSH_ALL_MMUCACHE', (MTFW_SUBMISSION_CMD_SET_GPU_UTIL:=15): 'MTFW_SUBMISSION_CMD_SET_GPU_UTIL', (MTFW_SUBMISSION_CMD_SET_HALT:=16): 'MTFW_SUBMISSION_CMD_SET_HALT', (MTFW_SUBMISSION_CMD_SCH_ABORT:=17): 'MTFW_SUBMISSION_CMD_SCH_ABORT', (MTFW_SUBMISSION_CMD_TRIG_PC_SAMPLER:=18): 'MTFW_SUBMISSION_CMD_TRIG_PC_SAMPLER', (MTFW_SUBMISSION_CMD_VM_INIT_CHECK:=70): 'MTFW_SUBMISSION_CMD_VM_INIT_CHECK', (MTFW_SUBMISSION_CMD_VM_OFFLINE:=71): 'MTFW_SUBMISSION_CMD_VM_OFFLINE', (MTFW_SUBMISSION_CMD_VM_CREATE:=72): 'MTFW_SUBMISSION_CMD_VM_CREATE', (MTFW_SUBMISSION_CMD_VM_SCHEDULE_UPDATE:=73): 'MTFW_SUBMISSION_CMD_VM_SCHEDULE_UPDATE', (MTFW_SUBMISSION_CMD_VM_QOS_UPDATE:=74): 'MTFW_SUBMISSION_CMD_VM_QOS_UPDATE', (MTFW_SUBMISSION_CMD_VM_REONLINE:=75): 'MTFW_SUBMISSION_CMD_VM_REONLINE', (MTFW_SUBMISSION_CMD_GPU_START:=100): 'MTFW_SUBMISSION_CMD_GPU_START', (MTFW_SUBMISSION_CMD_GPU_NOP:=100): 'MTFW_SUBMISSION_CMD_GPU_NOP', (MTFW_SUBMISSION_CMD_GPU_PREEMPTION:=101): 'MTFW_SUBMISSION_CMD_GPU_PREEMPTION', (MTFW_SUBMISSION_CMD_GPU_UQ:=102): 'MTFW_SUBMISSION_CMD_GPU_UQ', (MTFW_SUBMISSION_CMD_GPU_TDM:=103): 'MTFW_SUBMISSION_CMD_GPU_TDM', (MTFW_SUBMISSION_CMD_GPU_CDM:=104): 'MTFW_SUBMISSION_CMD_GPU_CDM', (MTFW_SUBMISSION_CMD_GPU_CE:=105): 'MTFW_SUBMISSION_CMD_GPU_CE', (MTFW_SUBMISSION_CMD_GPU_CDM_STREAM:=106): 'MTFW_SUBMISSION_CMD_GPU_CDM_STREAM', (MTFW_SUBMISSION_CMD_GPU_GFX:=107): 'MTFW_SUBMISSION_CMD_GPU_GFX', (MTFW_SUBMISSION_CMD_MAX_NUMS:=108): 'MTFW_SUBMISSION_CMD_MAX_NUMS'}
@c.record
class union__MTFW_NODE_ITEM_PARA_(c.Struct):
  SIZE = 48
  submissionData: MTFW_NODE_CMD_SUBMISSION
  preemptionData: MTFW_NODE_CTRL_PREEMPTION
  logCfg: MTFW_FWIF_TRACE_LOG_CONFIG
  perfRequest: MTFW_FWIF_PERF_REQUEST
  gpuUtilCfg: MTFW_FWIF_GPU_UTIL_CONFIG
  vmInfo: MTFW_NODE_CMD_VM_SCH_INFO
  haltCfg: MTFW_FWIF_FIRMWARE_HALT
  pcSamplerCfg: MTFW_FWIF_PC_SAMPLER_CFG
MTFW_NODE_ITEM_PARA: TypeAlias = union__MTFW_NODE_ITEM_PARA_
@c.record
class MTFW_NODE_CMD_SUBMISSION(c.Struct):
  SIZE = 32
  pageTableRootAddr: int
  pageTableGeneration: int
  submission: int
  submissionSize: int
  submissionQueueId: int
gpu_pa: TypeAlias = ctypes.c_uint64
gpu_va: TypeAlias = ctypes.c_uint64
MTFW_NODE_CMD_SUBMISSION.register_fields([('pageTableRootAddr', gpu_pa, 0), ('pageTableGeneration', uint64_t, 8), ('submission', gpu_va, 16), ('submissionSize', uint32_t, 24), ('submissionQueueId', uint32_t, 28)])
@c.record
class MTFW_NODE_CTRL_PREEMPTION(c.Struct):
  SIZE = 8
  wrOffset: int
MTFW_NODE_CTRL_PREEMPTION.register_fields([('wrOffset', uint32_t, 0)])
@c.record
class MTFW_FWIF_TRACE_LOG_CONFIG(c.Struct):
  SIZE = 8
  logGroup: int
  fwIndex: int
MTFW_FWIF_TRACE_LOG_CONFIG.register_fields([('logGroup', uint32_t, 0), ('fwIndex', uint8_t, 4)])
@c.record
class MTFW_FWIF_PERF_REQUEST(c.Struct):
  SIZE = 48
  requestType: int
  u: MTFW_FWIF_PERF_REQUEST_u
MTFW_PERF_REQUEST_TYPE: dict[int, str] = {(PERF_REQUEST_PING:=0): 'PERF_REQUEST_PING', (PERF_REQUEST_ENABLE_EVENTS:=1): 'PERF_REQUEST_ENABLE_EVENTS', (PERF_REQUEST_DISABLE_EVENTS:=2): 'PERF_REQUEST_DISABLE_EVENTS', (PERF_REQUEST_CONFIG_COUNTERS:=3): 'PERF_REQUEST_CONFIG_COUNTERS', (PERF_REQUEST_GET_INFO:=4): 'PERF_REQUEST_GET_INFO', (PERF_REQUEST_GET_DATA:=5): 'PERF_REQUEST_GET_DATA', (PERF_REQUEST_FORCE_DWORD:=2147483647): 'PERF_REQUEST_FORCE_DWORD'}
@c.record
class MTFW_FWIF_PERF_REQUEST_u(c.Struct):
  SIZE = 40
  select: MTFW_FWIF_PERF_SELECTOR
  param: int
  reg: MTFW_FWIF_PERF_REQUEST_u_reg
@c.record
class MTFW_FWIF_PERF_SELECTOR(c.Struct):
  SIZE = 40
  ordinal: int
  order: int
  count: int
  selectors: MTFW_FWIF_PERF_SELECTOR_RECORD
@c.record
class MTFW_FWIF_PERF_SELECTOR_RECORD(c.Struct):
  SIZE = 24
  baseAddr: int
  selectAddr: int
  selectValue: int
  indirectAddr: int
  indirectUnitNum: int
  counterSelect: int
  counterAddr: int
uint16_t: TypeAlias = ctypes.c_uint16
MTFW_FWIF_PERF_SELECTOR_RECORD.register_fields([('baseAddr', uint16_t, 0), ('selectAddr', uint16_t, 2), ('selectValue', uint64_t, 8), ('indirectAddr', uint16_t, 16), ('indirectUnitNum', uint8_t, 18), ('counterSelect', uint8_t, 19), ('counterAddr', uint16_t, 20)])
MTFW_FWIF_PERF_SELECTOR.register_fields([('ordinal', uint32_t, 0), ('order', uint32_t, 4), ('count', uint32_t, 8), ('selectors', MTFW_FWIF_PERF_SELECTOR_RECORD, 16)])
@c.record
class MTFW_FWIF_PERF_REQUEST_u_reg(c.Struct):
  SIZE = 16
  addr: int
  val: int
MTFW_FWIF_PERF_REQUEST_u_reg.register_fields([('addr', uint32_t, 0), ('val', uint64_t, 8)])
MTFW_FWIF_PERF_REQUEST_u.register_fields([('select', MTFW_FWIF_PERF_SELECTOR, 0), ('param', uint32_t, 0), ('reg', MTFW_FWIF_PERF_REQUEST_u_reg, 0)])
MTFW_FWIF_PERF_REQUEST.register_fields([('requestType', ctypes.c_uint32, 0), ('u', MTFW_FWIF_PERF_REQUEST_u, 8)])
@c.record
class MTFW_FWIF_GPU_UTIL_CONFIG(c.Struct):
  SIZE = 8
  gpuUtilCfg: int
MTFW_FWIF_GPU_UTIL_CFG_VALUE: dict[int, str] = {(CFG_GPU_UTIL_ALL_DISABLE:=0): 'CFG_GPU_UTIL_ALL_DISABLE', (CFG_GPU_UTIL_OVERALL_ENABLE:=1): 'CFG_GPU_UTIL_OVERALL_ENABLE', (CFG_GPU_UTIL_ALL_ENABLE:=2): 'CFG_GPU_UTIL_ALL_ENABLE'}
MTFW_FWIF_GPU_UTIL_CONFIG.register_fields([('gpuUtilCfg', ctypes.c_uint32, 0)])
@c.record
class MTFW_NODE_CMD_VM_SCH_INFO(c.Struct):
  SIZE = 24
  osid: int
  qosLimit: int
  scheInfo: MTFW_NODE_CMD_VM_SCH_INFO_scheInfo
@c.record
class MTFW_NODE_CMD_VM_SCH_INFO_scheInfo(c.Struct):
  SIZE = 8
  scheTime: int
  sysMemSize: int
MTFW_NODE_CMD_VM_SCH_INFO_scheInfo.register_fields([('scheTime', uint32_t, 0), ('sysMemSize', uint64_t, 0)])
MTFW_NODE_CMD_VM_SCH_INFO.register_fields([('osid', uint32_t, 0), ('qosLimit', uint64_t, 8), ('scheInfo', MTFW_NODE_CMD_VM_SCH_INFO_scheInfo, 16)])
@c.record
class MTFW_FWIF_FIRMWARE_HALT(c.Struct):
  SIZE = 8
  haltReason: int
MTFW_FWIF_FIRMWARE_HALT.register_fields([('haltReason', uint64_t, 0)])
@c.record
class MTFW_FWIF_PC_SAMPLER_CFG(c.Struct):
  SIZE = 32
  cfg: MTFW_FWIF_PC_SAMPLER_CFG_cfg
  dropCntBufVa: int
  dropCntCdmBufVa: int
  dropCntGfxBufVa: int
@c.record
class MTFW_FWIF_PC_SAMPLER_CFG_cfg(c.Struct):
  SIZE = 4
  bits: MTFW_FWIF_PC_SAMPLER_CFG_cfg_bits
  value: int
@c.record
class MTFW_FWIF_PC_SAMPLER_CFG_cfg_bits(c.Struct):
  SIZE = 4
  enable: int
  counterSize: int
  random2: int
  stallReason: int
  periodLevel: int
  rsrv: int
MTFW_FWIF_PC_SAMPLER_CFG_cfg_bits.register_fields([('enable', uint32_t, 0, 1, 0), ('counterSize', uint32_t, 0, 1, 1), ('random2', uint32_t, 0, 1, 2), ('stallReason', uint32_t, 0, 19, 3), ('periodLevel', uint32_t, 2, 5, 6), ('rsrv', uint32_t, 3, 5, 3)])
MTFW_FWIF_PC_SAMPLER_CFG_cfg.register_fields([('bits', MTFW_FWIF_PC_SAMPLER_CFG_cfg_bits, 0), ('value', uint32_t, 0)])
MTFW_FWIF_PC_SAMPLER_CFG.register_fields([('cfg', MTFW_FWIF_PC_SAMPLER_CFG_cfg, 0), ('dropCntBufVa', fw_va, 8), ('dropCntCdmBufVa', fw_va, 16), ('dropCntGfxBufVa', fw_va, 24)])
union__MTFW_NODE_ITEM_PARA_.register_fields([('submissionData', MTFW_NODE_CMD_SUBMISSION, 0), ('preemptionData', MTFW_NODE_CTRL_PREEMPTION, 0), ('logCfg', MTFW_FWIF_TRACE_LOG_CONFIG, 0), ('perfRequest', MTFW_FWIF_PERF_REQUEST, 0), ('gpuUtilCfg', MTFW_FWIF_GPU_UTIL_CONFIG, 0), ('vmInfo', MTFW_NODE_CMD_VM_SCH_INFO, 0), ('haltCfg', MTFW_FWIF_FIRMWARE_HALT, 0), ('pcSamplerCfg', MTFW_FWIF_PC_SAMPLER_CFG, 0)])
MTFW_NODE_ITEM.register_fields([('flags', MTFW_FWIF_HEAD, 0), ('priority', uint32_t, 4), ('attribute', MTFW_SUBMISSION_ATTRIBUTE, 8), ('submissionCmd', ctypes.c_uint32, 12), ('ccbItemToken', uint64_t, 16), ('hostPrivateData', uint64_t, 24), ('itemPara', MTFW_NODE_ITEM_PARA, 32), ('sequenceNums', uint32_t, 80), ('processId', uint32_t, 84)])
@c.record
class MTFW_NODE_RESP(c.Struct):
  SIZE = 24
  flags: MTFW_FWIF_HEAD
  respValue: int
  sequenceNums: int
  runTime: int
  startTimestamp: int
MTFW_NODE_RESP_VALUE: dict[int, str] = {(MTFW_NODE_RESP_DONE:=0): 'MTFW_NODE_RESP_DONE', (MTFW_NODE_RESP_CANCEL:=1): 'MTFW_NODE_RESP_CANCEL', (MTFW_NODE_RESP_BREAK:=2): 'MTFW_NODE_RESP_BREAK', (MTFW_NODE_RESP_PARTIAL:=3): 'MTFW_NODE_RESP_PARTIAL', (MTFW_NODE_RESP_STALL:=4): 'MTFW_NODE_RESP_STALL', (MTFW_NODE_RESP_ENGINE_SYNC:=5): 'MTFW_NODE_RESP_ENGINE_SYNC', (MTFW_NODE_RESP_PREEMPTION:=6): 'MTFW_NODE_RESP_PREEMPTION', (MTFW_NODE_RESP_QSEMP_TIMEOUT:=7): 'MTFW_NODE_RESP_QSEMP_TIMEOUT', (MTFW_NODE_RESP_STALL_TIMEOUT:=8): 'MTFW_NODE_RESP_STALL_TIMEOUT', (MTFW_NODE_RESP_ENGINE_SYNC_TIMEOUT:=9): 'MTFW_NODE_RESP_ENGINE_SYNC_TIMEOUT', (MTFW_NODE_EVENT_START:=80): 'MTFW_NODE_EVENT_START', (MTFW_NODE_EVENT_PERF:=80): 'MTFW_NODE_EVENT_PERF', (MTFW_NODE_RESP_ERROR_HANDLE_START:=256): 'MTFW_NODE_RESP_ERROR_HANDLE_START', (MTFW_NODE_RESP_INVALID:=256): 'MTFW_NODE_RESP_INVALID', (MTFW_NODE_RESP_FAULT:=257): 'MTFW_NODE_RESP_FAULT', (MTFW_NODE_RESP_ERROR:=258): 'MTFW_NODE_RESP_ERROR', (MTFW_NODE_RESP_VALUE_INVALID:=259): 'MTFW_NODE_RESP_VALUE_INVALID'}
MTFW_NODE_RESP.register_fields([('flags', MTFW_FWIF_HEAD, 0), ('respValue', ctypes.c_uint32, 4), ('sequenceNums', uint32_t, 8), ('runTime', uint32_t, 12), ('startTimestamp', uint64_t, 16)])
MTFW_NODE_QUEUE.register_fields([('reqNodeQ', c.Array[c.Array[MTFW_NODE_ITEM, Literal[64]], Literal[2]], 0), ('respNodeQ', c.Array[MTFW_NODE_RESP, Literal[64]], 11264), ('reqNodeCtrl', c.Array[MTFW_RING_CTRL, Literal[2]], 12800), ('respNodeCtrl', MTFW_RING_CTRL, 12832)])
@c.record
class struct_MTFW_FWIF_TAG_fwCfgs(c.Struct):
  SIZE = 8192
  cfg: MTFW_FWIF_CONFIG_GEN1
  value: c.Array[ctypes.c_uint32, Literal[2048]]
@c.record
class MTFW_FWIF_CONFIG_GEN1(c.Struct):
  SIZE = 648
  version: int
  flags: MTFW_FWIF_CONFIG_GEN1_flags
  flStateAddrsSCG: c.Array[ctypes.c_uint64, Literal[2]]
  flStateAddrsMCG2: c.Array[ctypes.c_uint64, Literal[4]]
  flStateAddrsMCG3: c.Array[ctypes.c_uint64, Literal[6]]
  flStateAddrsMCG4: c.Array[ctypes.c_uint64, Literal[8]]
  flStateAddrsMCG5: c.Array[ctypes.c_uint64, Literal[10]]
  flStateAddrsMCG6: c.Array[ctypes.c_uint64, Literal[12]]
  flStateAddrsMCG7: c.Array[ctypes.c_uint64, Literal[14]]
  flStateAddrsMCG8: c.Array[ctypes.c_uint64, Literal[16]]
  yuvCoefficientAddr: int
  pdsHeapBaseAddr: int
  uscHeapBaseAddr: int
  fbcdcHeapBaseAddr: int
  fbcdcLargeHeapBaseAddr: int
  texStateHeapBaseAddr: int
  componentCtrlHeapBaseAddr: int
  dmKillShaderOffset: int
@c.record
class MTFW_FWIF_CONFIG_GEN1_flags(c.Struct):
  SIZE = 4
  field: MTFW_FWIF_CONFIG_GEN1_flags_field
  value: int
@c.record
class MTFW_FWIF_CONFIG_GEN1_flags_field(c.Struct):
  SIZE = 4
  computeOnly: int
  disableHwr: int
MTFW_FWIF_CONFIG_GEN1_flags_field.register_fields([('computeOnly', uint32_t, 0, 1, 0), ('disableHwr', uint32_t, 0, 1, 1)])
MTFW_FWIF_CONFIG_GEN1_flags.register_fields([('field', MTFW_FWIF_CONFIG_GEN1_flags_field, 0), ('value', uint32_t, 0)])
MTFW_FWIF_CONFIG_GEN1.register_fields([('version', uint32_t, 0), ('flags', MTFW_FWIF_CONFIG_GEN1_flags, 4), ('flStateAddrsSCG', c.Array[gpu_va, Literal[2]], 8), ('flStateAddrsMCG2', c.Array[gpu_va, Literal[4]], 24), ('flStateAddrsMCG3', c.Array[gpu_va, Literal[6]], 56), ('flStateAddrsMCG4', c.Array[gpu_va, Literal[8]], 104), ('flStateAddrsMCG5', c.Array[gpu_va, Literal[10]], 168), ('flStateAddrsMCG6', c.Array[gpu_va, Literal[12]], 248), ('flStateAddrsMCG7', c.Array[gpu_va, Literal[14]], 344), ('flStateAddrsMCG8', c.Array[gpu_va, Literal[16]], 456), ('yuvCoefficientAddr', uint64_t, 584), ('pdsHeapBaseAddr', uint64_t, 592), ('uscHeapBaseAddr', uint64_t, 600), ('fbcdcHeapBaseAddr', uint64_t, 608), ('fbcdcLargeHeapBaseAddr', uint64_t, 616), ('texStateHeapBaseAddr', uint64_t, 624), ('componentCtrlHeapBaseAddr', uint64_t, 632), ('dmKillShaderOffset', uint32_t, 640)])
struct_MTFW_FWIF_TAG_fwCfgs.register_fields([('cfg', MTFW_FWIF_CONFIG_GEN1, 0), ('value', c.Array[uint32_t, Literal[2048]], 0)])
struct_MTFW_FWIF_TAG.register_fields([('flags', struct_MTFW_FWIF_TAG_flags, 0), ('fwStarted', uint32_t, 4), ('fwBuildVer', c.Array[uint8_t, Literal[8]], 8), ('watchDog', fw_va, 16), ('ccbQueueArrayAddr', fw_va, 24), ('ccbFastQueueArrayAddr', fw_va, 32), ('levelQueueArrayFecAddr', fw_va, 40), ('levelQueueArrayMetaAddr', fw_va, 48), ('nodeQueueArrayAddr', fw_va, 56), ('ccbQueueArrayNum', uint32_t, 64), ('ccbFastQueueArrayNum', uint32_t, 68), ('nodeQueueArrayNum', uint32_t, 72), ('levelQueueArrayNum', uint32_t, 76), ('startUpCounter', uint32_t, 80), ('ccbQueueArray', c.Array[MTFW_CCB_QUEUE, Literal[5]], 88), ('nodeQueueArray', c.Array[MTFW_NODE_QUEUE, Literal[6]], 82248), ('traceBufAddr', fw_va, 159336), ('traceBufAddrFEC', fw_va, 159344), ('metaPerfEventBufCtrlArray', fw_va, 159352), ('fecPerfEventBufCtrlArray', fw_va, 159360), ('metaPerfEventBufCtrlNum', uint32_t, 159368), ('fecPerfEventBufCtrlNum', uint32_t, 159372), ('faultDumpBuffer', fw_va, 159376), ('gpuUtilBufAddr', fw_va, 159384), ('vgpuFWVersion', uint64_t, 159392), ('vgpuFlag', uint64_t, 159392), ('osIdStart', uint32_t, 159400), ('osIdCount', uint32_t, 159404), ('vgpuSchedulingPolicy', uint32_t, 159408), ('vgpuTimeSlicedValue', uint32_t, 159412), ('vgpuUtil', uint32_t, 159416), ('vgpuQosEnable', uint32_t, 159420), ('vgpuSharedInfoAddr', fw_va, 159424), ('vgpuReleaseVersion', uint64_t, 159432), ('socTimerFreq', uint32_t, 159440), ('gpuFrequence', uint32_t, 159444), ('multicoreClusterMask', uint64_t, 159448), ('fwCfgs', struct_MTFW_FWIF_TAG_fwCfgs, 159456), ('firmwareCoreMemory', c.Array[uint8_t, Literal[98304]], 167648)])
MTFW_FWIF: TypeAlias = struct_MTFW_FWIF_TAG
@c.record
class MTFW_VGPU_SCHEDULE_INFO(c.Struct):
  SIZE = 16
  cmdExecutedTime: int
  maxCmdTime: int
  completeCmdCnt: int
  preemptedCmdCnt: int
  swContextSwitchCnt: int
  hwContextSwitchCnt: int
  preemptionReqCnt: int
  blockedCmdCnt: int
  scheduleWeight: int
  heavyCmdBlockCount: int
MTFW_VGPU_SCHEDULE_INFO.register_fields([('cmdExecutedTime', uint32_t, 0), ('maxCmdTime', uint32_t, 4), ('completeCmdCnt', uint8_t, 8), ('preemptedCmdCnt', uint8_t, 9), ('swContextSwitchCnt', uint8_t, 10), ('hwContextSwitchCnt', uint8_t, 11), ('preemptionReqCnt', uint8_t, 12), ('blockedCmdCnt', uint8_t, 13), ('scheduleWeight', uint8_t, 14), ('heavyCmdBlockCount', uint8_t, 15)])
@c.record
class MTFW_VGPU_SHARED_INFO(c.Struct):
  SIZE = 304
  magic: int
  cswTime: int
  SysMemSize: c.Array[ctypes.c_uint64, Literal[1]]
  ActiveOSid: int
  RunningOSid: c.Array[ctypes.c_uint32, Literal[8]]
  NeedHWROSid: c.Array[ctypes.c_uint32, Literal[8]]
  costTime: int
  quotaOsidMask: c.Array[ctypes.c_uint32, Literal[8]]
  sche_info: c.Array[c.Array[MTFW_VGPU_SCHEDULE_INFO, Literal[1]], Literal[7]]
  qosCredit: c.Array[ctypes.c_uint32, Literal[8]]
  qosLimit: int
  qosAvr: int
  qosStandard: int
  reOnlineOsid: int
MTFW_VGPU_SHARED_INFO.register_fields([('magic', uint32_t, 0), ('cswTime', uint64_t, 8), ('SysMemSize', c.Array[uint64_t, Literal[1]], 16), ('ActiveOSid', uint32_t, 24), ('RunningOSid', c.Array[uint32_t, Literal[8]], 28), ('NeedHWROSid', c.Array[uint32_t, Literal[8]], 60), ('costTime', uint32_t, 92), ('quotaOsidMask', c.Array[uint32_t, Literal[8]], 96), ('sche_info', c.Array[c.Array[MTFW_VGPU_SCHEDULE_INFO, Literal[1]], Literal[7]], 128), ('qosCredit', c.Array[uint32_t, Literal[8]], 240), ('qosLimit', uint64_t, 272), ('qosAvr', uint64_t, 280), ('qosStandard', uint64_t, 288), ('reOnlineOsid', uint32_t, 296)])
@c.record
class struct__MTFW_FWIF_FEC_BOOT_CFG_(c.Struct):
  SIZE = 32
  fwifFwVaLow: int
  fwifFwVaHi: int
  fwifHeapSize: int
  fwvaPageTableRootLow: int
  fwvaPageTableRootHi: int
  reservePPageAddrLow: int
  reservePPageAddrHi: int
  reservePPageNum: int
struct__MTFW_FWIF_FEC_BOOT_CFG_.register_fields([('fwifFwVaLow', uint32_t, 0), ('fwifFwVaHi', uint32_t, 4), ('fwifHeapSize', uint32_t, 8), ('fwvaPageTableRootLow', uint32_t, 12), ('fwvaPageTableRootHi', uint32_t, 16), ('reservePPageAddrLow', uint32_t, 20), ('reservePPageAddrHi', uint32_t, 24), ('reservePPageNum', uint32_t, 28)])
MTFW_FWIF_FEC_BOOT_CFG: TypeAlias = struct__MTFW_FWIF_FEC_BOOT_CFG_
MTFW_FWIF_CCB_TYPE: dict[int, str] = {(MTFW_CCB_TYPE_GP:=0): 'MTFW_CCB_TYPE_GP', (MTFW_CCB_TYPE_UQ:=1): 'MTFW_CCB_TYPE_UQ', (MTFW_CCB_TYPE_TDM:=2): 'MTFW_CCB_TYPE_TDM', (MTFW_CCB_TYPE_CDM:=3): 'MTFW_CCB_TYPE_CDM', (MTFW_CCB_TYPE_CE:=4): 'MTFW_CCB_TYPE_CE', (MTFW_CCB_TYPE_MAX_NUM:=5): 'MTFW_CCB_TYPE_MAX_NUM'}
@c.record
class MTFW_SUBMISSION_PARA(c.Struct):
  SIZE = 1336
  pageTableRootAddr: int
  pageTableGeneration: int
  submissionQueueId: int
  flags: MTFW_SUBMISSION_PARA_flags
  dependQSemValidMask: int
  finishQSemValidMask: int
  dependQSem: c.Array[MTFW_Q_SEMAPHORE, Literal[32]]
  finishQSem: c.Array[MTFW_Q_SEMAPHORE, Literal[8]]
  submissionAddr: int
  submissionSize: int
  doorbellId: int
@c.record
class MTFW_SUBMISSION_PARA_flags(c.Struct):
  SIZE = 4
  field: MTFW_SUBMISSION_PARA_flags_field
  value: int
@c.record
class MTFW_SUBMISSION_PARA_flags_field(c.Struct):
  SIZE = 4
  qSempOutofOrder: int
  withDoorbell: int
MTFW_SUBMISSION_PARA_flags_field.register_fields([('qSempOutofOrder', uint32_t, 0, 1, 0), ('withDoorbell', uint32_t, 0, 1, 1)])
MTFW_SUBMISSION_PARA_flags.register_fields([('field', MTFW_SUBMISSION_PARA_flags_field, 0), ('value', uint32_t, 0)])
@c.record
class MTFW_Q_SEMAPHORE(c.Struct):
  SIZE = 32
  addr: int
  addr32: int
  shadowAddr: int
  shadowAddr32: int
  value: int
  value32: int
  attribute: MTFW_Q_SEMAPHORE_attribute
@c.record
class MTFW_Q_SEMAPHORE_attribute(c.Struct):
  SIZE = 8
  revert: int
MTFW_Q_SEMAPHORE_attribute.register_fields([('revert', uint64_t, 0)])
MTFW_Q_SEMAPHORE.register_fields([('addr', fw_va, 0), ('addr32', uint32_t, 0), ('shadowAddr', fw_va, 8), ('shadowAddr32', uint32_t, 8), ('value', uint64_t, 16), ('value32', uint32_t, 16), ('attribute', MTFW_Q_SEMAPHORE_attribute, 24)])
MTFW_SUBMISSION_PARA.register_fields([('pageTableRootAddr', gpu_pa, 0), ('pageTableGeneration', uint64_t, 8), ('submissionQueueId', uint32_t, 16), ('flags', MTFW_SUBMISSION_PARA_flags, 20), ('dependQSemValidMask', uint64_t, 24), ('finishQSemValidMask', uint64_t, 32), ('dependQSem', c.Array[MTFW_Q_SEMAPHORE, Literal[32]], 40), ('finishQSem', c.Array[MTFW_Q_SEMAPHORE, Literal[8]], 1064), ('submissionAddr', gpu_va, 1320), ('submissionSize', uint32_t, 1328), ('doorbellId', uint32_t, 1332)])
@c.record
class MTFW_SUBMISSION_SCHEDULE_PARA(c.Struct):
  SIZE = 32
  pageTableRootAddr: int
  pageTableGeneration: int
  ccbItemToken: int
  ccbType: int
  submissionQueueId: int
MTFW_SUBMISSION_SCHEDULE_PARA.register_fields([('pageTableRootAddr', gpu_pa, 0), ('pageTableGeneration', uint64_t, 8), ('ccbItemToken', uint64_t, 16), ('ccbType', ctypes.c_uint32, 24), ('submissionQueueId', uint32_t, 28)])
@c.record
class MTFW_SUBMISSION_Q_SEM_PARA(c.Struct):
  SIZE = 48
  pageTableRootAddr: int
  pageTableGeneration: int
  writeQSem: MTFW_Q_SEMAPHORE
MTFW_SUBMISSION_Q_SEM_PARA.register_fields([('pageTableRootAddr', gpu_pa, 0), ('pageTableGeneration', uint64_t, 8), ('writeQSem', MTFW_Q_SEMAPHORE, 16)])
@c.record
class MTFW_SUBMISSION_INVALID_TLB(c.Struct):
  SIZE = 1328
  pageTableRootAddr: int
  pageTableGeneration: int
  dependQSemValidMask: int
  finishQSemValidMask: int
  dependQSem: c.Array[MTFW_Q_SEMAPHORE, Literal[32]]
  finishQSem: c.Array[MTFW_Q_SEMAPHORE, Literal[8]]
  invalidAddr: int
  invalidSize: int
MTFW_SUBMISSION_INVALID_TLB.register_fields([('pageTableRootAddr', gpu_pa, 0), ('pageTableGeneration', uint64_t, 8), ('dependQSemValidMask', uint64_t, 16), ('finishQSemValidMask', uint64_t, 24), ('dependQSem', c.Array[MTFW_Q_SEMAPHORE, Literal[32]], 32), ('finishQSem', c.Array[MTFW_Q_SEMAPHORE, Literal[8]], 1056), ('invalidAddr', gpu_va, 1312), ('invalidSize', uint64_t, 1320)])
@c.record
class MTFW_FWIF_VM_SCH_INFO(c.Struct):
  SIZE = 24
  qosLimit: int
  scheInfo: MTFW_FWIF_VM_SCH_INFO_scheInfo
  osid: int
@c.record
class MTFW_FWIF_VM_SCH_INFO_scheInfo(c.Struct):
  SIZE = 8
  sysMemSize: int
  scheTime: int
MTFW_FWIF_VM_SCH_INFO_scheInfo.register_fields([('sysMemSize', uint64_t, 0), ('scheTime', uint32_t, 0)])
MTFW_FWIF_VM_SCH_INFO.register_fields([('qosLimit', uint64_t, 0), ('scheInfo', MTFW_FWIF_VM_SCH_INFO_scheInfo, 8), ('osid', uint32_t, 16)])
@c.record
class union__MTFW_CCB_ITEM_PARA_(c.Struct):
  SIZE = 1336
  submissionPara: MTFW_SUBMISSION_PARA
  qSempPara: MTFW_SUBMISSION_Q_SEM_PARA
  schPara: MTFW_SUBMISSION_SCHEDULE_PARA
  logCfg: MTFW_FWIF_TRACE_LOG_CONFIG
  perfRequest: MTFW_FWIF_PERF_REQUEST
  gpuUtilCfg: MTFW_FWIF_GPU_UTIL_CONFIG
  vmInfo: MTFW_FWIF_VM_SCH_INFO
  haltCfg: MTFW_FWIF_FIRMWARE_HALT
  invalidTlbCfg: MTFW_SUBMISSION_INVALID_TLB
  pcSamplerCfg: MTFW_FWIF_PC_SAMPLER_CFG
union__MTFW_CCB_ITEM_PARA_.register_fields([('submissionPara', MTFW_SUBMISSION_PARA, 0), ('qSempPara', MTFW_SUBMISSION_Q_SEM_PARA, 0), ('schPara', MTFW_SUBMISSION_SCHEDULE_PARA, 0), ('logCfg', MTFW_FWIF_TRACE_LOG_CONFIG, 0), ('perfRequest', MTFW_FWIF_PERF_REQUEST, 0), ('gpuUtilCfg', MTFW_FWIF_GPU_UTIL_CONFIG, 0), ('vmInfo', MTFW_FWIF_VM_SCH_INFO, 0), ('haltCfg', MTFW_FWIF_FIRMWARE_HALT, 0), ('invalidTlbCfg', MTFW_SUBMISSION_INVALID_TLB, 0), ('pcSamplerCfg', MTFW_FWIF_PC_SAMPLER_CFG, 0)])
MTFW_CCB_ITEM_PARA: TypeAlias = union__MTFW_CCB_ITEM_PARA_
@c.record
class MTFW_FW_CONTEXT(c.Struct):
  SIZE = 64
  field: MTFW_FW_CONTEXT_field
  buffer: c.Array[ctypes.c_uint64, Literal[8]]
@c.record
class MTFW_FW_CONTEXT_field(c.Struct):
  SIZE = 4
  submState: int
MTFW_SUBMISSION_STATE: dict[int, str] = {(MTFW_SUBMISSION_STATE_INIT:=0): 'MTFW_SUBMISSION_STATE_INIT', (MTFW_SUBMISSION_STATE_BLOCK:=1): 'MTFW_SUBMISSION_STATE_BLOCK', (MTFW_SUBMISSION_STATE_READY:=2): 'MTFW_SUBMISSION_STATE_READY', (MTFW_SUBMISSION_STATE_RUNNING:=3): 'MTFW_SUBMISSION_STATE_RUNNING', (MTFW_SUBMISSION_STATE_SUSPEND:=4): 'MTFW_SUBMISSION_STATE_SUSPEND', (MTFW_SUBMISSION_STATE_FINISHED:=5): 'MTFW_SUBMISSION_STATE_FINISHED', (MTFW_SUBMISSION_STATE_NUM:=6): 'MTFW_SUBMISSION_STATE_NUM', (MTFW_SUBMISSION_STATE_INVALID:=6): 'MTFW_SUBMISSION_STATE_INVALID'}
MTFW_FW_CONTEXT_field.register_fields([('submState', ctypes.c_uint32, 0)])
MTFW_FW_CONTEXT.register_fields([('field', MTFW_FW_CONTEXT_field, 0), ('buffer', c.Array[uint64_t, Literal[8]], 0)])
@c.record
class MTFW_CCB_ITEM(c.Struct):
  SIZE = 1456
  flags: MTFW_FWIF_HEAD
  attribute: MTFW_SUBMISSION_ATTRIBUTE
  submissionCmd: int
  priority: int
  ccbItemToken: int
  hostPrivateData: int
  frameId: int
  cbId: int
  processId: int
  ccbItemPara: union__MTFW_CCB_ITEM_PARA_
  fwBuffer: MTFW_FW_CONTEXT
MTFW_CCB_ITEM.register_fields([('flags', MTFW_FWIF_HEAD, 0), ('attribute', MTFW_SUBMISSION_ATTRIBUTE, 4), ('submissionCmd', ctypes.c_uint32, 8), ('priority', uint32_t, 12), ('ccbItemToken', uint64_t, 16), ('hostPrivateData', uint64_t, 24), ('frameId', uint64_t, 32), ('cbId', uint64_t, 40), ('processId', uint32_t, 48), ('ccbItemPara', MTFW_CCB_ITEM_PARA, 56), ('fwBuffer', MTFW_FW_CONTEXT, 1392)])
MTFW_FWIF_GPU_DM_TYPE: dict[int, str] = {(MTFW_FWIF_DM_GP:=0): 'MTFW_FWIF_DM_GP', (MTFW_FWIF_DM_TDM:=1): 'MTFW_FWIF_DM_TDM', (MTFW_FWIF_DM_GEOM:=2): 'MTFW_FWIF_DM_GEOM', (MTFW_FWIF_DM_3D:=3): 'MTFW_FWIF_DM_3D', (MTFW_FWIF_DM_CDM:=4): 'MTFW_FWIF_DM_CDM', (MTFW_FWIF_DM_CE:=5): 'MTFW_FWIF_DM_CE'}
MTFW_FWIF_DUMP_REASON: dict[int, str] = {(MTFW_FWIF_DUMP_REASON_HOST:=0): 'MTFW_FWIF_DUMP_REASON_HOST', (MTFW_FWIF_DUMP_REASON_HALT:=1): 'MTFW_FWIF_DUMP_REASON_HALT', (MTFW_FWIF_DUMP_REASON_FW_HANG:=2): 'MTFW_FWIF_DUMP_REASON_FW_HANG', (MTFW_FWIF_DUMP_REASON_RUN_TIMEOUT:=3): 'MTFW_FWIF_DUMP_REASON_RUN_TIMEOUT', (MTFW_FWIF_DUMP_REASON_K_SEM_TIMEOUT:=4): 'MTFW_FWIF_DUMP_REASON_K_SEM_TIMEOUT', (MTFW_FWIF_DUMP_REASON_Q_SEM_TIMEOUT:=5): 'MTFW_FWIF_DUMP_REASON_Q_SEM_TIMEOUT', (MTFW_FWIF_DUMP_REASON_CONTINUE_TIMEOUT:=6): 'MTFW_FWIF_DUMP_REASON_CONTINUE_TIMEOUT', (MTFW_FWIF_DUMP_REASON_PAGE_FAULT:=7): 'MTFW_FWIF_DUMP_REASON_PAGE_FAULT', (MTFW_FWIF_DUMP_REASON_CSW_TIMEOUT:=8): 'MTFW_FWIF_DUMP_REASON_CSW_TIMEOUT', (MTFW_FWIF_DUMP_REASON_INVALID:=9): 'MTFW_FWIF_DUMP_REASON_INVALID', (MTFW_FWIF_DUMP_REASON_MP_EXCEPTION:=10): 'MTFW_FWIF_DUMP_REASON_MP_EXCEPTION', (MTFW_FWIF_DUMP_REASON_MSS_ERROR:=11): 'MTFW_FWIF_DUMP_REASON_MSS_ERROR', (MTFW_FWIF_DUMP_REASON_MTLINk_ERROR:=12): 'MTFW_FWIF_DUMP_REASON_MTLINk_ERROR', (MTFW_FWIF_DUMP_REASON_VMEM_ECC_ERROR:=13): 'MTFW_FWIF_DUMP_REASON_VMEM_ECC_ERROR', (MTFW_FWIF_DUMP_REASON_STALL_TIMEOUT:=14): 'MTFW_FWIF_DUMP_REASON_STALL_TIMEOUT', (MTFW_FWIF_DUMP_REASON_ENGINE_SYNC_TIMEOUT:=15): 'MTFW_FWIF_DUMP_REASON_ENGINE_SYNC_TIMEOUT', (MTFW_FWIF_DUMP_REASON_DM_RUN_TIMEOUT:=16): 'MTFW_FWIF_DUMP_REASON_DM_RUN_TIMEOUT', (MTFW_FWIF_DUMP_REASON_KILL_TIMEOUT:=17): 'MTFW_FWIF_DUMP_REASON_KILL_TIMEOUT', (MTFW_FWIF_DUMP_REASON_K_EVENT_TIMEOUT:=18): 'MTFW_FWIF_DUMP_REASON_K_EVENT_TIMEOUT', (MTFW_FWIF_DUMP_REASON_MAX_CNT:=19): 'MTFW_FWIF_DUMP_REASON_MAX_CNT'}
@c.record
class MTFW_FWIF_DUMP_PAGE_FAULT_INFO(c.Struct):
  SIZE = 48
  ccbItemToken: int
  hostPrivateData: int
  nodeIdx: int
  sequenceNums: int
  processId: int
  queueId: int
  submissionCmd: int
  cmdOffset: int
  kickPara: int
  kickResult: int
MTFW_FWIF_DUMP_PAGE_FAULT_INFO.register_fields([('ccbItemToken', uint64_t, 0), ('hostPrivateData', uint64_t, 8), ('nodeIdx', uint32_t, 16), ('sequenceNums', uint32_t, 20), ('processId', uint32_t, 24), ('queueId', uint32_t, 28), ('submissionCmd', ctypes.c_uint32, 32), ('cmdOffset', uint32_t, 36), ('kickPara', uint32_t, 40), ('kickResult', uint32_t, 44)])
MTFW_FWIF_DUMP_MP_EXCEPTION_INFO: TypeAlias = MTFW_FWIF_DUMP_PAGE_FAULT_INFO
MTFW_FWIF_DUMP_MSS_ERROR_INFO: TypeAlias = MTFW_FWIF_DUMP_PAGE_FAULT_INFO
MTFW_FWIF_DUMP_DM_TIMEOUT_INFO: TypeAlias = MTFW_FWIF_DUMP_PAGE_FAULT_INFO
@c.record
class MTFW_FWIF_DUMP_INFO_DESC(c.Struct):
  SIZE = 88
  magicWord: int
  crc32: int
  codeTag: MTFW_FWIF_DUMP_INFO_DESC_codeTag
  timeStamp: int
  hostReason: int
  reason: int
  dumpTag: MTFW_FWIF_DUMP_INFO_DESC_dumpTag
  info: MTFW_FWIF_DUMP_INFO_DESC_info
@c.record
class MTFW_FWIF_DUMP_INFO_DESC_codeTag(c.Struct):
  SIZE = 8
  tagValue: int
  tagString: c.Array[ctypes.c_ubyte, Literal[8]]
MTFW_FWIF_DUMP_INFO_DESC_codeTag.register_fields([('tagValue', uint64_t, 0), ('tagString', c.Array[uint8_t, Literal[8]], 0)])
@c.record
class MTFW_FWIF_DUMP_INFO_DESC_dumpTag(c.Struct):
  SIZE = 4
  field: MTFW_FWIF_DUMP_INFO_DESC_dumpTag_field
  value: int
@c.record
class MTFW_FWIF_DUMP_INFO_DESC_dumpTag_field(c.Struct):
  SIZE = 4
  overflow: int
  mapFail: int
MTFW_FWIF_DUMP_INFO_DESC_dumpTag_field.register_fields([('overflow', uint32_t, 0, 1, 0), ('mapFail', uint32_t, 0, 1, 1)])
MTFW_FWIF_DUMP_INFO_DESC_dumpTag.register_fields([('field', MTFW_FWIF_DUMP_INFO_DESC_dumpTag_field, 0), ('value', uint32_t, 0)])
@c.record
class MTFW_FWIF_DUMP_INFO_DESC_info(c.Struct):
  SIZE = 48
  pageFaultInfo: MTFW_FWIF_DUMP_PAGE_FAULT_INFO
  mpExceptionInfo: MTFW_FWIF_DUMP_PAGE_FAULT_INFO
  mssErrorInfo: MTFW_FWIF_DUMP_PAGE_FAULT_INFO
  dmTimeoutInfo: MTFW_FWIF_DUMP_PAGE_FAULT_INFO
MTFW_FWIF_DUMP_INFO_DESC_info.register_fields([('pageFaultInfo', MTFW_FWIF_DUMP_PAGE_FAULT_INFO, 0), ('mpExceptionInfo', MTFW_FWIF_DUMP_MP_EXCEPTION_INFO, 0), ('mssErrorInfo', MTFW_FWIF_DUMP_MSS_ERROR_INFO, 0), ('dmTimeoutInfo', MTFW_FWIF_DUMP_DM_TIMEOUT_INFO, 0)])
MTFW_FWIF_DUMP_INFO_DESC.register_fields([('magicWord', uint32_t, 0), ('crc32', uint32_t, 4), ('codeTag', MTFW_FWIF_DUMP_INFO_DESC_codeTag, 8), ('timeStamp', uint64_t, 16), ('hostReason', uint64_t, 24), ('reason', ctypes.c_uint32, 32), ('dumpTag', MTFW_FWIF_DUMP_INFO_DESC_dumpTag, 36), ('info', MTFW_FWIF_DUMP_INFO_DESC_info, 40)])
MTFW_FWIF_REG_INDIRECT_TYPE: dict[int, str] = {(MTFW_REG_INDIRECT_TYPE_NONE:=0): 'MTFW_REG_INDIRECT_TYPE_NONE', (MTFW_REG_INDIRECT_TYPE_PBE:=1): 'MTFW_REG_INDIRECT_TYPE_PBE', (MTFW_REG_INDIRECT_TYPE_USC:=2): 'MTFW_REG_INDIRECT_TYPE_USC', (MTFW_REG_INDIRECT_TYPE_TPU:=3): 'MTFW_REG_INDIRECT_TYPE_TPU', (MTFW_REG_INDIRECT_TYPE_TEXAS:=4): 'MTFW_REG_INDIRECT_TYPE_TEXAS', (MTFW_REG_INDIRECT_TYPE_MERCER:=5): 'MTFW_REG_INDIRECT_TYPE_MERCER', (MTFW_REG_INDIRECT_TYPE_SHARED:=6): 'MTFW_REG_INDIRECT_TYPE_SHARED', (MTFW_REG_INDIRECT_TYPE_ISP:=7): 'MTFW_REG_INDIRECT_TYPE_ISP', (MTFW_REG_INDIRECT_TYPE_SWIFT:=8): 'MTFW_REG_INDIRECT_TYPE_SWIFT'}
MTFW_FWIF_REG_TYPE: dict[int, str] = {(MTFW_REG_TYPE_NORMAL:=0): 'MTFW_REG_TYPE_NORMAL', (MTFW_REG_TYPE_CE:=1): 'MTFW_REG_TYPE_CE'}
@c.record
class MTFW_FWIF_REG_DUMP_CFG(c.Struct):
  SIZE = 16
  regOffset: int
  regStride: int
  regIdxNum: int
  regType: int
  indirectType: int
  indirectRegOffset: int
  indirectNum: int
  dumpBufNums: int
MTFW_FWIF_REG_DUMP_CFG.register_fields([('regOffset', uint16_t, 0), ('regStride', uint8_t, 2), ('regIdxNum', uint8_t, 3), ('regType', uint8_t, 4), ('indirectType', uint8_t, 5), ('indirectRegOffset', uint16_t, 6), ('indirectNum', uint8_t, 8), ('dumpBufNums', uint16_t, 10)])
@c.record
class MTFW_FWIF_REG_DUMP_DESC(c.Struct):
  SIZE = 8
  desc: MTFW_FWIF_REG_DUMP_DESC_desc
  value: int
@c.record
class MTFW_FWIF_REG_DUMP_DESC_desc(c.Struct):
  SIZE = 8
  magicKey: int
  regDumpNum: int
  regDumpCfgIdx: int
  regIdxCount: int
  indirectCount: int
MTFW_FWIF_REG_DUMP_DESC_desc.register_fields([('magicKey', uint16_t, 0), ('regDumpNum', uint16_t, 2), ('regDumpCfgIdx', uint16_t, 4), ('regIdxCount', uint8_t, 6), ('indirectCount', uint8_t, 7)])
MTFW_FWIF_REG_DUMP_DESC.register_fields([('desc', MTFW_FWIF_REG_DUMP_DESC_desc, 0), ('value', uint64_t, 0)])
@c.record
class MTFW_FWIF_DUMP_CFG(c.Struct):
  SIZE = 16
  regDumpCfgArray: int
  regDumpCfgNum: int
MTFW_FWIF_DUMP_CFG.register_fields([('regDumpCfgArray', gpu_va, 0), ('regDumpCfgNum', uint32_t, 8)])
@c.record
class MTFW_FWIF_ERROR_INFO(c.Struct):
  SIZE = 536
  dumpInfo: MTFW_FWIF_DUMP_INFO_DESC
  schInfo: MTFW_FWIF_SCH_CTX
  regDumpCfgArray: int
  regDumpCfgNums: int
  regDumpBuffAddr: int
  regDumpBuffNum: int
  regDumpCfgUsed: int
  regDumpedNum: int
@c.record
class MTFW_FWIF_SCH_CTX(c.Struct):
  SIZE = 408
  ccbItemToken: int
  pageTableRootAddr: int
  processId: int
  submissionCmd: int
  schCounter: c.Array[MTFW_FW_SCH_KICK_BRIEF, Literal[8]]
@c.record
class MTFW_FW_SCH_KICK_BRIEF(c.Struct):
  SIZE = 48
  flags: MTFW_FW_SCH_KICK_BRIEF_flags
  running: c.Array[ctypes.c_uint32, Literal[8]]
  kicked: int
  finished: int
@c.record
class MTFW_FW_SCH_KICK_BRIEF_flags(c.Struct):
  SIZE = 4
  field: MTFW_FW_SCH_KICK_BRIEF_flags_field
  value: int
@c.record
class MTFW_FW_SCH_KICK_BRIEF_flags_field(c.Struct):
  SIZE = 4
  validFlag: int
  validRunningMask: int
MTFW_FW_SCH_KICK_BRIEF_flags_field.register_fields([('validFlag', uint32_t, 0, 1, 0), ('validRunningMask', uint32_t, 0, 8, 1)])
MTFW_FW_SCH_KICK_BRIEF_flags.register_fields([('field', MTFW_FW_SCH_KICK_BRIEF_flags_field, 0), ('value', uint32_t, 0)])
MTFW_FW_SCH_KICK_BRIEF.register_fields([('flags', MTFW_FW_SCH_KICK_BRIEF_flags, 0), ('running', c.Array[uint32_t, Literal[8]], 4), ('kicked', uint32_t, 36), ('finished', uint32_t, 40)])
MTFW_FWIF_SCH_CTX.register_fields([('ccbItemToken', uint64_t, 0), ('pageTableRootAddr', gpu_pa, 8), ('processId', uint32_t, 16), ('submissionCmd', ctypes.c_uint32, 20), ('schCounter', c.Array[MTFW_FW_SCH_KICK_BRIEF, Literal[8]], 24)])
MTFW_FWIF_ERROR_INFO.register_fields([('dumpInfo', MTFW_FWIF_DUMP_INFO_DESC, 0), ('schInfo', MTFW_FWIF_SCH_CTX, 88), ('regDumpCfgArray', gpu_va, 496), ('regDumpCfgNums', uint32_t, 504), ('regDumpBuffAddr', gpu_va, 512), ('regDumpBuffNum', uint32_t, 520), ('regDumpCfgUsed', uint32_t, 524), ('regDumpedNum', uint32_t, 528)])
@c.record
class MTFW_FWIF_FAULT_INFO(c.Struct):
  SIZE = 520
  dumpInfo: MTFW_FWIF_DUMP_INFO_DESC
  schInfo: MTFW_FWIF_SCH_CTX
  regDumpBuffAddr: int
  regDumpBuffNum: int
  regDumpCfgUsed: int
  regDumpedNum: int
MTFW_FWIF_FAULT_INFO.register_fields([('dumpInfo', MTFW_FWIF_DUMP_INFO_DESC, 0), ('schInfo', MTFW_FWIF_SCH_CTX, 88), ('regDumpBuffAddr', fw_va, 496), ('regDumpBuffNum', uint32_t, 504), ('regDumpCfgUsed', uint32_t, 508), ('regDumpedNum', uint32_t, 512)])
@c.record
class MTFW_FWIF_FAULT_INFO_CTRL(c.Struct):
  SIZE = 33304
  faultInfo: c.Array[MTFW_FWIF_FAULT_INFO, Literal[64]]
  faultInfoWrIdx: int
  faultInfoRdIdx: int
  regDumpCfgArray: int
  regDumpCfgNums: int
MTFW_FWIF_FAULT_INFO_CTRL.register_fields([('faultInfo', c.Array[MTFW_FWIF_FAULT_INFO, Literal[64]], 0), ('faultInfoWrIdx', uint32_t, 33280), ('faultInfoRdIdx', uint32_t, 33284), ('regDumpCfgArray', fw_va, 33288), ('regDumpCfgNums', uint32_t, 33296)])
@c.record
class MTFW_GEOM_CTX_STATE(c.Struct):
  SIZE = 56
  taRegDceCmd0: int
  taRegDceCmd1: int
  taRegDceWrite: int
  taRegDceDraw0: int
  taRegDceDraw1: int
  taRegGtaSoPrim: c.Array[ctypes.c_uint32, Literal[4]]
  taCurrentIdx: int
MTFW_GEOM_CTX_STATE.register_fields([('taRegDceCmd0', uint64_t, 0), ('taRegDceCmd1', uint32_t, 8), ('taRegDceWrite', uint32_t, 12), ('taRegDceDraw0', uint64_t, 16), ('taRegDceDraw1', uint64_t, 24), ('taRegGtaSoPrim', c.Array[uint32_t, Literal[4]], 32), ('taCurrentIdx', uint16_t, 48)])
@c.record
class MTFW_PIXEL_CTX_STATE(c.Struct):
  SIZE = 136
  ispSTORE: c.Array[ctypes.c_uint32, Literal[32]]
  pmDeallocMaskStatus: int
  pmPdsMtileFreeStatus: int
MTFW_PIXEL_CTX_STATE.register_fields([('ispSTORE', c.Array[uint32_t, Literal[32]], 0), ('pmDeallocMaskStatus', uint32_t, 128), ('pmPdsMtileFreeStatus', uint32_t, 132)])
@c.record
class MTFW_GEOM_CTX_REGS(c.Struct):
  SIZE = 128
  vdmContextStateBaseAddr: int
  geContextStateBaseAddr: int
  dceContextStoreTaskVDM0: int
  dceContextStoreTaskVDM1: int
  dceContextStoreTaskVDM2: int
  dceContextResumeTaskVDM0: int
  dceContextResumeTaskVDM1: int
  dceContextResumeTaskVDM2: int
  dceContextStoreTaskDDM0: int
  dceContextStoreTaskDDM1: int
  dceContextStoreTaskDDM2: int
  dceContextStoreTaskXFB: int
  dceContextResumeTaskDDM0: int
  dceContextResumeTaskDDM1: int
  dceContextResumeTaskDDM2: int
  dceContextResumeTaskXFB: int
MTFW_GEOM_CTX_REGS.register_fields([('vdmContextStateBaseAddr', uint64_t, 0), ('geContextStateBaseAddr', uint64_t, 8), ('dceContextStoreTaskVDM0', uint64_t, 16), ('dceContextStoreTaskVDM1', uint64_t, 24), ('dceContextStoreTaskVDM2', uint64_t, 32), ('dceContextResumeTaskVDM0', uint64_t, 40), ('dceContextResumeTaskVDM1', uint64_t, 48), ('dceContextResumeTaskVDM2', uint64_t, 56), ('dceContextStoreTaskDDM0', uint64_t, 64), ('dceContextStoreTaskDDM1', uint64_t, 72), ('dceContextStoreTaskDDM2', uint64_t, 80), ('dceContextStoreTaskXFB', uint64_t, 88), ('dceContextResumeTaskDDM0', uint64_t, 96), ('dceContextResumeTaskDDM1', uint64_t, 104), ('dceContextResumeTaskDDM2', uint64_t, 112), ('dceContextResumeTaskXFB', uint64_t, 120)])
@c.record
class MTFW_COMPUTE_CTX_REGS(c.Struct):
  SIZE = 48
  cdmContextPds0: int
  cdmContextPds1: int
  cdmTerminatePds0: int
  cdmTerminatePds1: int
  cdmResumePds0: int
  cdmResumePds1: int
MTFW_COMPUTE_CTX_REGS.register_fields([('cdmContextPds0', uint64_t, 0), ('cdmContextPds1', uint64_t, 8), ('cdmTerminatePds0', uint64_t, 16), ('cdmTerminatePds1', uint64_t, 24), ('cdmResumePds0', uint64_t, 32), ('cdmResumePds1', uint64_t, 40)])
enum_MTFW_LOG_SFGROUPS: dict[int, str] = {(MTFW_GROUP_NULL:=0): 'MTFW_GROUP_NULL', (MTFW_GROUP_ERR:=1): 'MTFW_GROUP_ERR', (MTFW_GROUP_DBG:=2): 'MTFW_GROUP_DBG', (MTFW_GROUP_FUNC:=3): 'MTFW_GROUP_FUNC', (MTFW_GROUP_REG:=4): 'MTFW_GROUP_REG', (MTFW_GROUP_MMU:=5): 'MTFW_GROUP_MMU', (MTFW_GROUP_INT:=6): 'MTFW_GROUP_INT', (MTFW_GROUP_CORE:=7): 'MTFW_GROUP_CORE', (MTFW_GROUP_DRV:=8): 'MTFW_GROUP_DRV', (MTFW_GROUP_SCH:=9): 'MTFW_GROUP_SCH', (MTFW_GROUP_NODE:=10): 'MTFW_GROUP_NODE', (MTFW_GROUP_WORK:=11): 'MTFW_GROUP_WORK', (MTFW_GROUP_GPU:=12): 'MTFW_GROUP_GPU', (MTFW_GROUP_PERF:=13): 'MTFW_GROUP_PERF', (MTFW_GROUP_PFM:=14): 'MTFW_GROUP_PFM', (MTFW_GROUP_POW:=15): 'MTFW_GROUP_POW', (MTFW_GROUP_CSW:=16): 'MTFW_GROUP_CSW', (MTFW_GROUP_CCB:=17): 'MTFW_GROUP_CCB', (MTFW_GROUP_CTX:=18): 'MTFW_GROUP_CTX', (MTFW_GROUP_SUBM:=19): 'MTFW_GROUP_SUBM', (MTFW_GROUP_FSCH:=20): 'MTFW_GROUP_FSCH', (MTFW_GROUP_MAIN:=21): 'MTFW_GROUP_MAIN', (MTFW_GROUP_VGPU:=22): 'MTFW_GROUP_VGPU', (MTFW_GROUP_QSEMP:=23): 'MTFW_GROUP_QSEMP', (MTFW_GROUP_RSV24:=24): 'MTFW_GROUP_RSV24', (MTFW_GROUP_RSV25:=25): 'MTFW_GROUP_RSV25', (MTFW_GROUP_RSV26:=26): 'MTFW_GROUP_RSV26', (MTFW_GROUP_RSV27:=27): 'MTFW_GROUP_RSV27', (MTFW_GROUP_RSV28:=28): 'MTFW_GROUP_RSV28', (MTFW_GROUP_RSV29:=29): 'MTFW_GROUP_RSV29', (MTFW_GROUP_RSV30:=30): 'MTFW_GROUP_RSV30', (MTFW_GROUP_RSV31:=31): 'MTFW_GROUP_RSV31', (MTFW_LOG_NUMBER_OF_GROUPS:=32): 'MTFW_LOG_NUMBER_OF_GROUPS'}
@c.record
class MTFW_FWIF_ASSERTBUF(c.Struct):
  SIZE = 408
  path: c.Array[ctypes.c_char, Literal[200]]
  info: c.Array[ctypes.c_char, Literal[200]]
  lineNum: int
MTFW_FWIF_ASSERTBUF.register_fields([('path', c.Array[ctypes.c_char, Literal[200]], 0), ('info', c.Array[ctypes.c_char, Literal[200]], 200), ('lineNum', uint32_t, 400)])
@c.record
class MTFW_FWIF_TRACEBUF_SPACE(c.Struct):
  SIZE = 97440
  tracePointer: int
  bWrap: bool
  traceBuffer: c.Array[ctypes.c_uint32, Literal[24000]]
  assertBuf: MTFW_FWIF_ASSERTBUF
  assertEXTInfo: c.Array[ctypes.c_uint32, Literal[256]]
MTFW_FWIF_TRACEBUF_SPACE.register_fields([('tracePointer', uint32_t, 0), ('bWrap', ctypes.c_bool, 4), ('traceBuffer', c.Array[uint32_t, Literal[24000]], 8), ('assertBuf', MTFW_FWIF_ASSERTBUF, 96008), ('assertEXTInfo', c.Array[uint32_t, Literal[256]], 96416)])
@c.record
class MTFW_FWIF_TRACEBUF(c.Struct):
  SIZE = 97568
  logGroups: c.Array[ctypes.c_uint32, Literal[32]]
  traceBuf: c.Array[MTFW_FWIF_TRACEBUF_SPACE, Literal[1]]
MTFW_FWIF_TRACEBUF.register_fields([('logGroups', c.Array[uint32_t, Literal[32]], 0), ('traceBuf', c.Array[MTFW_FWIF_TRACEBUF_SPACE, Literal[1]], 128)])
@c.record
class MTFW_FWIF_SIGBUF_CTL(c.Struct):
  SIZE = 16
  bufferAddr: int
  leftSizeInRegs: int
MTFW_FWIF_SIGBUF_CTL.register_fields([('bufferAddr', fw_va, 0), ('leftSizeInRegs', uint32_t, 8)])
@c.record
class MTFW_FWIF_LOG_HEADER(c.Struct):
  SIZE = 24
  baseAddr: int
  dataLength: int
  dataOffset: int
MTFW_FWIF_LOG_HEADER.register_fields([('baseAddr', uint64_t, 0), ('dataLength', uint64_t, 8), ('dataOffset', uint64_t, 16)])
@c.record
class MTFW_FWIF_LOG_ENTRY(c.Struct):
  SIZE = 20
  groups: int
  paramsNum: int
  lineIdx: int
  fileNameLen: int
  textLen: int
MTFW_FWIF_LOG_ENTRY.register_fields([('groups', uint32_t, 0), ('paramsNum', uint32_t, 4), ('lineIdx', uint32_t, 8), ('fileNameLen', uint32_t, 12), ('textLen', uint32_t, 16)])
MTFW_FWIF_NODE_TYPE: dict[int, str] = {(MTFW_NODE_TYPE_GP:=0): 'MTFW_NODE_TYPE_GP', (MTFW_NODE_TYPE_TDM:=1): 'MTFW_NODE_TYPE_TDM', (MTFW_NODE_TYPE_UQ:=2): 'MTFW_NODE_TYPE_UQ', (MTFW_NODE_TYPE_CDM:=3): 'MTFW_NODE_TYPE_CDM', (MTFW_NODE_TYPE_CE:=4): 'MTFW_NODE_TYPE_CE', (MTFW_NODE_TYPE_KMD_CE:=5): 'MTFW_NODE_TYPE_KMD_CE', (MTFW_NODE_TYPE_MAX_NUM:=6): 'MTFW_NODE_TYPE_MAX_NUM'}
MTFW_LEVEL_QUEUE: TypeAlias = MTFW_NODE_QUEUE
MTFW_FWIF_PERF_EVENT_TYPE: dict[int, str] = {(PERF_REPORTEVENT_RANGE_BEGIN:=1): 'PERF_REPORTEVENT_RANGE_BEGIN', (PERF_EVENT_FW_BG_KICK:=1): 'PERF_EVENT_FW_BG_KICK', (PERF_EVENT_FW_BG_FINISH:=33): 'PERF_EVENT_FW_BG_FINISH', (PERF_EVENT_FW_IRQ_KICK:=2): 'PERF_EVENT_FW_IRQ_KICK', (PERF_EVENT_FW_IRQ_FINISH:=34): 'PERF_EVENT_FW_IRQ_FINISH', (PERF_EVENT_L1_SCH_START:=3): 'PERF_EVENT_L1_SCH_START', (PERF_EVENT_L1_SCH_END:=35): 'PERF_EVENT_L1_SCH_END', (PERF_EVENT_FAST_SCH_START:=4): 'PERF_EVENT_FAST_SCH_START', (PERF_EVENT_FAST_SCH_END:=36): 'PERF_EVENT_FAST_SCH_END', (PERF_EVENT_NODE_START:=5): 'PERF_EVENT_NODE_START', (PERF_EVENT_NODE_END:=37): 'PERF_EVENT_NODE_END', (PERF_EVENT_WORK_SCH_START:=6): 'PERF_EVENT_WORK_SCH_START', (PERF_EVENT_WORK_SCH_END:=38): 'PERF_EVENT_WORK_SCH_END', (PERF_EVENT_DM_SCH_START:=7): 'PERF_EVENT_DM_SCH_START', (PERF_EVENT_DM_SCH_END:=39): 'PERF_EVENT_DM_SCH_END', (PERF_EVENT_DM_KICK_START:=8): 'PERF_EVENT_DM_KICK_START', (PERF_EVENT_DM_KICK_END:=40): 'PERF_EVENT_DM_KICK_END', (PERF_EVENT_PREEMPT_REQ:=9): 'PERF_EVENT_PREEMPT_REQ', (PERF_EVENT_PREEMPT_ACK:=41): 'PERF_EVENT_PREEMPT_ACK', (PERF_EVENT_FENCE_NOTIFY:=10): 'PERF_EVENT_FENCE_NOTIFY', (PERF_EVENT_SUBM_SCH_START:=11): 'PERF_EVENT_SUBM_SCH_START', (PERF_EVENT_SUBM_SCH_END:=43): 'PERF_EVENT_SUBM_SCH_END', (PERF_EVENT_PMOOM_TAPAUSE:=16): 'PERF_EVENT_PMOOM_TAPAUSE', (PERF_EVENT_PMOOM_TARESUME:=48): 'PERF_EVENT_PMOOM_TARESUME', (PERF_EVENT_TA_KICK:=17): 'PERF_EVENT_TA_KICK', (PERF_EVENT_TA_FINISH:=49): 'PERF_EVENT_TA_FINISH', (PERF_EVENT_3D_KICK:=18): 'PERF_EVENT_3D_KICK', (PERF_EVENT_3D_FINISH:=50): 'PERF_EVENT_3D_FINISH', (PERF_EVENT_CDM_KICK:=19): 'PERF_EVENT_CDM_KICK', (PERF_EVENT_CDM_FINISH:=51): 'PERF_EVENT_CDM_FINISH', (PERF_EVENT_3DSPM_KICK:=20): 'PERF_EVENT_3DSPM_KICK', (PERF_EVENT_3DSPM_FINISH:=52): 'PERF_EVENT_3DSPM_FINISH', (PERF_EVENT_TDM_KICK:=21): 'PERF_EVENT_TDM_KICK', (PERF_EVENT_TDM_FINISH:=53): 'PERF_EVENT_TDM_FINISH', (PERF_EVENT_3DRTA_KICK:=22): 'PERF_EVENT_3DRTA_KICK', (PERF_EVENT_3DRTA_FINISH:=54): 'PERF_EVENT_3DRTA_FINISH', (PERF_EVENT_CE_KICK:=23): 'PERF_EVENT_CE_KICK', (PERF_EVENT_CE_FINISH:=55): 'PERF_EVENT_CE_FINISH', (PERF_EVENT_PFM_KICK:=24): 'PERF_EVENT_PFM_KICK', (PERF_EVENT_PFM_FINISH:=56): 'PERF_EVENT_PFM_FINISH', (PERF_EVENT_DEBUG_A_KICK:=25): 'PERF_EVENT_DEBUG_A_KICK', (PERF_EVENT_DEBUG_A_FINISH:=57): 'PERF_EVENT_DEBUG_A_FINISH', (PERF_EVENT_DEBUG_B_KICK:=26): 'PERF_EVENT_DEBUG_B_KICK', (PERF_EVENT_DEBUG_B_FINISH:=58): 'PERF_EVENT_DEBUG_B_FINISH', (PERF_EVENT_DEBUG_C_KICK:=27): 'PERF_EVENT_DEBUG_C_KICK', (PERF_EVENT_DEBUG_C_FINISH:=59): 'PERF_EVENT_DEBUG_C_FINISH', (PERF_EVENT_DEBUG_D_KICK:=28): 'PERF_EVENT_DEBUG_D_KICK', (PERF_EVENT_DEBUG_D_FINISH:=60): 'PERF_EVENT_DEBUG_D_FINISH', (PERF_EVENT_GROUP_LIMIT:=31): 'PERF_EVENT_GROUP_LIMIT', (PERF_REPORTEVENT_RANGE_END:=60): 'PERF_REPORTEVENT_RANGE_END', (PERF_EVENT_FORCE_DWORD:=2147483647): 'PERF_EVENT_FORCE_DWORD'}
MTFW_FWIF_PERF_FILTER: dict[int, str] = {(PERF_FILTER_INVALID_PACKET:=0): 'PERF_FILTER_INVALID_PACKET', (PERF_FILTER_VALID_FW_PACKET:=1): 'PERF_FILTER_VALID_FW_PACKET', (PERF_FILTER_VALID_HW_PACKET:=3): 'PERF_FILTER_VALID_HW_PACKET', (PERF_FILTER_VALID_COUNTER_PACKET:=4): 'PERF_FILTER_VALID_COUNTER_PACKET', (PERF_FILTER_SCH_DATA_PACKET:=5): 'PERF_FILTER_SCH_DATA_PACKET', (PERF_FILTER_SUBM_DATA_PACKET:=6): 'PERF_FILTER_SUBM_DATA_PACKET', (PERF_FILTER_FORCE_DWORD:=2147483647): 'PERF_FILTER_FORCE_DWORD'}
@c.record
class MTFW_FWIF_PERF_EVENT_FW_DATA(c.Struct):
  SIZE = 40
  domain: int
  fwThreadId: int
  fenceId: int
  id: int
  frameId: int
  cbId: int
  processId: int
  queueId: int
MTFW_FWIF_PERF_EVENT_FW_DATA.register_fields([('domain', uint32_t, 0), ('fwThreadId', uint32_t, 4), ('fenceId', uint32_t, 8), ('id', uint32_t, 12), ('frameId', uint64_t, 16), ('cbId', uint64_t, 24), ('processId', uint32_t, 32), ('queueId', uint32_t, 36)])
@c.record
class MTFW_FWIF_PERF_EVENT_HW_DATA(c.Struct):
  SIZE = 128
  frameId: int
  cbId: int
  ccbItemToken: int
  fenceId: int
  processId: int
  coreId: int
  coreMask: int
  queueId: int
  commandFlags: int
  kickSemaphoreBufferAddr: int
  kickSemaphore: MTFW_FWIF_PERF_EVENT_HW_DATA_kickSemaphore
@c.record
class MTFW_FWIF_PERF_EVENT_HW_DATA_kickSemaphore(c.Struct):
  SIZE = 72
  dependentValue: c.Array[ctypes.c_uint64, Literal[8]]
  finishValue: int
MTFW_FWIF_PERF_EVENT_HW_DATA_kickSemaphore.register_fields([('dependentValue', c.Array[uint64_t, Literal[8]], 0), ('finishValue', uint64_t, 64)])
MTFW_FWIF_PERF_EVENT_HW_DATA.register_fields([('frameId', uint64_t, 0), ('cbId', uint64_t, 8), ('ccbItemToken', uint64_t, 16), ('fenceId', uint32_t, 24), ('processId', uint32_t, 28), ('coreId', uint32_t, 32), ('coreMask', uint32_t, 36), ('queueId', uint32_t, 40), ('commandFlags', uint32_t, 44), ('kickSemaphoreBufferAddr', uint64_t, 48), ('kickSemaphore', MTFW_FWIF_PERF_EVENT_HW_DATA_kickSemaphore, 56)])
@c.record
class MTFW_FWIF_PERF_EVENT_SUBM_DATA(c.Struct):
  SIZE = 64
  processId: int
  fenceId: int
  ccbItemToken: int
  frameId: int
  cbId: int
  qSempWaitMask: int
  qSempKickedMask: int
  submFlags: int
  submAttribute: int
  submCmd: int
  submState: int
MTFW_FWIF_PERF_EVENT_SUBM_DATA.register_fields([('processId', uint32_t, 0), ('fenceId', uint32_t, 4), ('ccbItemToken', uint64_t, 8), ('frameId', uint64_t, 16), ('cbId', uint64_t, 24), ('qSempWaitMask', uint64_t, 32), ('qSempKickedMask', uint64_t, 40), ('submFlags', uint32_t, 48), ('submAttribute', uint32_t, 52), ('submCmd', uint32_t, 56), ('submState', uint32_t, 60)])
@c.record
class MTFW_FWIF_PERF_EVENT_SCH_DATA(c.Struct):
  SIZE = 112
  schQCounter64: c.Array[ctypes.c_uint64, Literal[4]]
  schQCounter32: c.Array[ctypes.c_uint32, Literal[8]]
  schQCounter: c.Array[ctypes.c_uint16, Literal[16]]
  queueCtxflags: int
  queueId: int
  scheType: int
MTFW_FWIF_PERF_EVENT_SCH_DATA.register_fields([('schQCounter64', c.Array[uint64_t, Literal[4]], 0), ('schQCounter32', c.Array[uint32_t, Literal[8]], 32), ('schQCounter', c.Array[uint16_t, Literal[16]], 64), ('queueCtxflags', uint32_t, 96), ('queueId', uint32_t, 100), ('scheType', uint32_t, 104)])
@c.record
class MTFW_FWIF_PERF_EVENT_HEADER(c.Struct):
  SIZE = 32
  timestamp: int
  signature: int
  flags: MTFW_FWIF_PERF_EVENT_HEADER_flags
  size: int
  ordinal: int
  event: int
  filter: int
@c.record
class MTFW_FWIF_PERF_EVENT_HEADER_flags(c.Struct):
  SIZE = 4
  field: MTFW_FWIF_PERF_EVENT_HEADER_flags_field
  value: int
@c.record
class MTFW_FWIF_PERF_EVENT_HEADER_flags_field(c.Struct):
  SIZE = 4
  eventVersion: int
  osid: int
MTFW_FWIF_PERF_EVENT_HEADER_flags_field.register_fields([('eventVersion', uint32_t, 0, 7, 0), ('osid', uint32_t, 0, 4, 7)])
MTFW_FWIF_PERF_EVENT_HEADER_flags.register_fields([('field', MTFW_FWIF_PERF_EVENT_HEADER_flags_field, 0), ('value', uint32_t, 0)])
MTFW_FWIF_PERF_EVENT_HEADER.register_fields([('timestamp', uint64_t, 0), ('signature', uint32_t, 8), ('flags', MTFW_FWIF_PERF_EVENT_HEADER_flags, 12), ('size', uint32_t, 16), ('ordinal', uint32_t, 20), ('event', ctypes.c_uint32, 24), ('filter', ctypes.c_uint32, 28)])
@c.record
class MTFW_FWIF_PERF_EVENT_DATA(c.Struct):
  SIZE = 128
  hardware: MTFW_FWIF_PERF_EVENT_HW_DATA
  firmware: MTFW_FWIF_PERF_EVENT_FW_DATA
  schData: MTFW_FWIF_PERF_EVENT_SCH_DATA
  submData: MTFW_FWIF_PERF_EVENT_SUBM_DATA
MTFW_FWIF_PERF_EVENT_DATA.register_fields([('hardware', MTFW_FWIF_PERF_EVENT_HW_DATA, 0), ('firmware', MTFW_FWIF_PERF_EVENT_FW_DATA, 0), ('schData', MTFW_FWIF_PERF_EVENT_SCH_DATA, 0), ('submData', MTFW_FWIF_PERF_EVENT_SUBM_DATA, 0)])
@c.record
class MTFW_FWIF_PERF_EVENT(c.Struct):
  SIZE = 184
  header: MTFW_FWIF_PERF_EVENT_HEADER
  timer: int
  ordinal: int
  osid: int
  event: int
  filter: int
  data: MTFW_FWIF_PERF_EVENT_DATA
MTFW_FWIF_PERF_EVENT.register_fields([('header', MTFW_FWIF_PERF_EVENT_HEADER, 0), ('timer', uint64_t, 32), ('ordinal', uint32_t, 40), ('osid', uint32_t, 44), ('event', ctypes.c_uint32, 48), ('filter', ctypes.c_uint32, 52), ('data', MTFW_FWIF_PERF_EVENT_DATA, 56)])
@c.record
class MTFW_FWIF_PERF_EVENT_BUF_CTL(c.Struct):
  SIZE = 942104
  perfEvents: c.Array[MTFW_FWIF_PERF_EVENT, Literal[5120]]
  perfEventWriteOffset: int
  perfEventReadOffset: int
  perfFilter: int
  hwPerfFilter: int
MTFW_FWIF_PERF_EVENT_BUF_CTL.register_fields([('perfEvents', c.Array[MTFW_FWIF_PERF_EVENT, Literal[5120]], 0), ('perfEventWriteOffset', uint32_t, 942080), ('perfEventReadOffset', uint32_t, 942084), ('perfFilter', uint32_t, 942088), ('hwPerfFilter', uint64_t, 942096)])
MTFW_FWIF_QUEUE_TYPE: dict[int, str] = {(MTFW_FWIF_QUEUE_TYPE_GP:=0): 'MTFW_FWIF_QUEUE_TYPE_GP', (MTFW_FWIF_QUEUE_TYPE_UQ:=1): 'MTFW_FWIF_QUEUE_TYPE_UQ', (MTFW_FWIF_QUEUE_TYPE_TDM:=2): 'MTFW_FWIF_QUEUE_TYPE_TDM', (MTFW_FWIF_QUEUE_TYPE_CDM:=3): 'MTFW_FWIF_QUEUE_TYPE_CDM', (MTFW_FWIF_QUEUE_TYPE_CE:=4): 'MTFW_FWIF_QUEUE_TYPE_CE', (MTFW_FWIF_QUEUE_TYPE_KMD_CE:=5): 'MTFW_FWIF_QUEUE_TYPE_KMD_CE', (MTFW_FWIF_QUEUE_TYPE_MAX:=6): 'MTFW_FWIF_QUEUE_TYPE_MAX'}
@c.record
class MTFW_FW_SCH_BUFFER(c.Struct):
  SIZE = 4096
  schCtx: MTFW_FWIF_SCH_CTX
  schBuffer: c.Array[ctypes.c_ubyte, Literal[4096]]
MTFW_FW_SCH_BUFFER.register_fields([('schCtx', MTFW_FWIF_SCH_CTX, 0), ('schBuffer', c.Array[uint8_t, Literal[4096]], 0)])
MTFW_Q_SEM_FINISH_TYPE: dict[int, str] = {(MTFW_Q_SEM_FINISH_TYPE_INC:=0): 'MTFW_Q_SEM_FINISH_TYPE_INC', (MTFW_Q_SEM_FINISH_TYPE_BINARY:=1): 'MTFW_Q_SEM_FINISH_TYPE_BINARY', (MTFW_Q_SEM_FINISH_TYPE_SIGNAL:=2): 'MTFW_Q_SEM_FINISH_TYPE_SIGNAL', (MTFW_Q_SEM_FINISH_TYPE_EXTEND:=3): 'MTFW_Q_SEM_FINISH_TYPE_EXTEND'}
MTFW_Q_SEM_DEPENDENT_TYPE: dict[int, str] = {(MTFW_Q_SEM_DEPENDENT_TYPE_NORMAL:=0): 'MTFW_Q_SEM_DEPENDENT_TYPE_NORMAL', (MTFW_Q_SEM_DEPENDENT_TYPE_SIGNAL:=1): 'MTFW_Q_SEM_DEPENDENT_TYPE_SIGNAL'}
MTFW_SUBMISSION_REGION_TYPE: dict[int, str] = {(MTFW_SUB_REGION_START:=0): 'MTFW_SUB_REGION_START', (MTFW_SUB_REGION_NOP:=1): 'MTFW_SUB_REGION_NOP', (MTFW_SUB_REGION_CDM:=2): 'MTFW_SUB_REGION_CDM', (MTFW_SUB_REGION_TDM:=3): 'MTFW_SUB_REGION_TDM', (MTFW_SUB_REGION_CE:=4): 'MTFW_SUB_REGION_CE', (MTFW_SUB_REGION_GFX:=5): 'MTFW_SUB_REGION_GFX', (MTFW_SUB_REGION_SPR:=6): 'MTFW_SUB_REGION_SPR', (MTFW_SUB_REGION_MAX_KICK_TYPE:=7): 'MTFW_SUB_REGION_MAX_KICK_TYPE', (MTFW_SUB_REGION_K_SEM_BUFFER:=8): 'MTFW_SUB_REGION_K_SEM_BUFFER', (MTFW_SUB_REGION_CSW_BUFFER:=9): 'MTFW_SUB_REGION_CSW_BUFFER', (MTFW_SUB_REGION_ERROR_BUFFER:=10): 'MTFW_SUB_REGION_ERROR_BUFFER', (MTFW_SUB_REGION_TRACE_INFO:=11): 'MTFW_SUB_REGION_TRACE_INFO', (MTFW_SUB_REGION_PFM_CFG:=12): 'MTFW_SUB_REGION_PFM_CFG', (MTFW_SUB_REGION_GEOM_IDX_FETCH_CFG:=13): 'MTFW_SUB_REGION_GEOM_IDX_FETCH_CFG', (MTFW_SUB_REGION_SCHEDULE_TABLE:=14): 'MTFW_SUB_REGION_SCHEDULE_TABLE', (MTFW_SUB_REGION_DBG_CSW_CFG:=15): 'MTFW_SUB_REGION_DBG_CSW_CFG', (MTFW_SUB_REGION_SPACE:=16): 'MTFW_SUB_REGION_SPACE', (MTFW_SUB_REGION_NUMS:=17): 'MTFW_SUB_REGION_NUMS', (MTFW_SUB_REGION_INVALID:=17): 'MTFW_SUB_REGION_INVALID'}
@c.record
class MTFW_SUBMISSION_HEAD(c.Struct):
  SIZE = 88
  flags: MTFW_FWIF_HEAD
  cmdFlags: MTFW_SUBMISSION_HEAD_cmdFlags
  timestampAddr: int
  fwCswBufer: int
  priority: int
  submissionCmd: int
  submissionQueueId: int
  retryCount: int
  subRegionDescStartOffset: int
  subRegionDescNums: int
  frameId: int
  cbId: int
  processId: int
  reserve: c.Array[ctypes.c_uint32, Literal[4]]
@c.record
class MTFW_SUBMISSION_HEAD_cmdFlags(c.Struct):
  SIZE = 4
  field: MTFW_SUBMISSION_HEAD_cmdFlags_field
  value: int
@c.record
class MTFW_SUBMISSION_HEAD_cmdFlags_field(c.Struct):
  SIZE = 4
  rsv: int
MTFW_SUBMISSION_HEAD_cmdFlags_field.register_fields([('rsv', uint32_t, 0, 1, 0)])
MTFW_SUBMISSION_HEAD_cmdFlags.register_fields([('field', MTFW_SUBMISSION_HEAD_cmdFlags_field, 0), ('value', uint32_t, 0)])
MTFW_SUBMISSION_HEAD.register_fields([('flags', MTFW_FWIF_HEAD, 0), ('cmdFlags', MTFW_SUBMISSION_HEAD_cmdFlags, 4), ('timestampAddr', gpu_va, 8), ('fwCswBufer', gpu_va, 16), ('priority', uint32_t, 24), ('submissionCmd', ctypes.c_uint32, 28), ('submissionQueueId', uint32_t, 32), ('retryCount', uint32_t, 36), ('subRegionDescStartOffset', uint32_t, 40), ('subRegionDescNums', uint32_t, 44), ('frameId', uint64_t, 48), ('cbId', uint64_t, 56), ('processId', uint32_t, 64), ('reserve', c.Array[uint32_t, Literal[4]], 68)])
@c.record
class MTFW_SUBMISSION_REGION_DESC(c.Struct):
  SIZE = 24
  regionType: int
  regionStart: MTFW_SUBMISSION_REGION_DESC_regionStart
  regionSize: int
@c.record
class MTFW_SUBMISSION_REGION_DESC_regionStart(c.Struct):
  SIZE = 8
  offset: int
  gpuAddr: int
  fwAddr: int
  value: int
MTFW_SUBMISSION_REGION_DESC_regionStart.register_fields([('offset', uint32_t, 0), ('gpuAddr', gpu_va, 0), ('fwAddr', fw_va, 0), ('value', uint64_t, 0)])
MTFW_SUBMISSION_REGION_DESC.register_fields([('regionType', ctypes.c_uint32, 0), ('regionStart', MTFW_SUBMISSION_REGION_DESC_regionStart, 8), ('regionSize', uint32_t, 16)])
MTFW_KICK_SEMAPHORE_UQ: dict[int, str] = {(KICK_SEMAPHORE_IDX_TDM:=0): 'KICK_SEMAPHORE_IDX_TDM', (KICK_SEMAPHORE_IDX_SPACE:=0): 'KICK_SEMAPHORE_IDX_SPACE', (KICK_SEMAPHORE_IDX_GEOM:=1): 'KICK_SEMAPHORE_IDX_GEOM', (KICK_SEMAPHORE_IDX_SPR:=1): 'KICK_SEMAPHORE_IDX_SPR', (KICK_SEMAPHORE_IDX_PIXEL:=2): 'KICK_SEMAPHORE_IDX_PIXEL', (KICK_SEMAPHORE_IDX_CDM:=3): 'KICK_SEMAPHORE_IDX_CDM', (KICK_SEMAPHORE_IDX_CE:=4): 'KICK_SEMAPHORE_IDX_CE', (KICK_SEMAPHORE_IDX_MAX:=5): 'KICK_SEMAPHORE_IDX_MAX'}
@c.record
class MTFW_KICK_SEMAPHORE(c.Struct):
  SIZE = 80
  dependentValue: c.Array[ctypes.c_uint64, Literal[8]]
  finishValue: int
  finishValueMask: int
  reserve: c.Array[ctypes.c_ubyte, Literal[3]]
MTFW_KICK_SEMAPHORE.register_fields([('dependentValue', c.Array[uint64_t, Literal[8]], 0), ('finishValue', uint64_t, 64), ('finishValueMask', uint8_t, 72), ('reserve', c.Array[uint8_t, Literal[3]], 73)])
@c.record
class MTFW_KICK_EVENT(c.Struct):
  SIZE = 16
  addr: int
  value: int
MTFW_KICK_EVENT.register_fields([('addr', gpu_va, 0), ('value', uint64_t, 8)])
@c.record
class MTFW_KICK_EVENT_PARA(c.Struct):
  SIZE = 32
  dependKickEventArray: int
  finishKickEventArray: int
  dependKickEventNum: int
  finishKickEventNum: int
  dependKickPollTimeoutSec: int
MTFW_KICK_EVENT_PARA.register_fields([('dependKickEventArray', gpu_va, 0), ('finishKickEventArray', gpu_va, 8), ('dependKickEventNum', uint32_t, 16), ('finishKickEventNum', uint32_t, 20), ('dependKickPollTimeoutSec', uint32_t, 24)])
@c.record
class MTFW_KICK_TRACE(c.Struct):
  SIZE = 16
  startTimestamp: int
  runTime: int
  Info: MTFW_KICK_TRACE_Info
@c.record
class MTFW_KICK_TRACE_Info(c.Struct):
  SIZE = 4
  gfxField: MTFW_KICK_TRACE_Info_gfxField
  value: int
@c.record
class MTFW_KICK_TRACE_Info_gfxField(c.Struct):
  SIZE = 4
  rtaNums: int
  oomCounter: int
MTFW_KICK_TRACE_Info_gfxField.register_fields([('rtaNums', uint32_t, 0, 8, 0), ('oomCounter', uint32_t, 1, 8, 0)])
MTFW_KICK_TRACE_Info.register_fields([('gfxField', MTFW_KICK_TRACE_Info_gfxField, 0), ('value', uint32_t, 0)])
MTFW_KICK_TRACE.register_fields([('startTimestamp', uint64_t, 0), ('runTime', uint32_t, 8), ('Info', MTFW_KICK_TRACE_Info, 12)])
MTFW_SUBMISSION_TRACE: TypeAlias = MTFW_KICK_TRACE
@c.record
class MTFW_KICK_HEAD(c.Struct):
  SIZE = 192
  flags: MTFW_FWIF_HEAD
  kickCtrl: MTFW_KICK_HEAD_kickCtrl
  kickSize: int
  kickTimeoutMs: int
  retryTimeoutUs: int
  kickSemaphore: MTFW_KICK_SEMAPHORE
  kickEvent: MTFW_KICK_EVENT_PARA
  traceAddr: int
  timeStampAddr: int
  optionalRegCfg: int
  reserve: c.Array[ctypes.c_uint64, Literal[4]]
@c.record
class MTFW_KICK_HEAD_kickCtrl(c.Struct):
  SIZE = 4
  field: MTFW_KICK_HEAD_kickCtrl_field
  value: int
@c.record
class MTFW_KICK_HEAD_kickCtrl_field(c.Struct):
  SIZE = 4
  mmuInvalid: int
MTFW_KICK_HEAD_kickCtrl_field.register_fields([('mmuInvalid', uint32_t, 0, 1, 0)])
MTFW_KICK_HEAD_kickCtrl.register_fields([('field', MTFW_KICK_HEAD_kickCtrl_field, 0), ('value', uint32_t, 0)])
MTFW_KICK_HEAD.register_fields([('flags', MTFW_FWIF_HEAD, 0), ('kickCtrl', MTFW_KICK_HEAD_kickCtrl, 4), ('kickSize', uint32_t, 8), ('kickTimeoutMs', uint32_t, 12), ('retryTimeoutUs', uint32_t, 16), ('kickSemaphore', MTFW_KICK_SEMAPHORE, 24), ('kickEvent', MTFW_KICK_EVENT_PARA, 104), ('traceAddr', gpu_va, 136), ('timeStampAddr', gpu_va, 144), ('optionalRegCfg', uint64_t, 152), ('reserve', c.Array[uint64_t, Literal[4]], 160)])
@c.record
class MTFW_KICK_SCH_CTRL(c.Struct):
  SIZE = 16
  field: MTFW_KICK_SCH_CTRL_field
  value: int
  reserve: c.Array[ctypes.c_uint32, Literal[4]]
@c.record
class MTFW_KICK_SCH_CTRL_field(c.Struct):
  SIZE = 4
  submissionSerial: int
  contextSerial: int
  globalSerial: int
  submQueueSerial: int
  forceInOrder: int
  kickExclusive: int
  inDmExclusive: int
  issueNum: int
  kickMode: int
  geomAheadNum: int
  fromKmd: int
  kickEvent: int
  dmExclusive: int
MTFW_KICK_SCH_CTRL_field.register_fields([('submissionSerial', uint32_t, 0, 1, 0), ('contextSerial', uint32_t, 0, 1, 1), ('globalSerial', uint32_t, 0, 1, 2), ('submQueueSerial', uint32_t, 0, 1, 3), ('forceInOrder', uint32_t, 0, 1, 4), ('kickExclusive', uint32_t, 0, 1, 5), ('inDmExclusive', uint32_t, 0, 1, 6), ('issueNum', uint32_t, 0, 8, 7), ('kickMode', uint32_t, 1, 4, 7), ('geomAheadNum', uint32_t, 2, 8, 3), ('fromKmd', uint32_t, 3, 1, 3), ('kickEvent', uint32_t, 3, 1, 4), ('dmExclusive', uint32_t, 3, 1, 5)])
MTFW_KICK_SCH_CTRL.register_fields([('field', MTFW_KICK_SCH_CTRL_field, 0), ('value', uint32_t, 0), ('reserve', c.Array[uint32_t, Literal[4]], 0)])
@c.record
class MTFW_CBUFFER_CTRL(c.Struct):
  SIZE = 16
  writeOffset: int
  readOffset: int
MTFW_CBUFFER_CTRL.register_fields([('writeOffset', uint64_t, 0), ('readOffset', uint64_t, 8)])
MTFW_KICK_OPT_REG_CFG_METHOD: dict[int, str] = {(MTFW_KICK_OPT_REG_CFG_IGNORE:=0): 'MTFW_KICK_OPT_REG_CFG_IGNORE', (MTFW_KICK_OPT_REG_CFG_USER:=1): 'MTFW_KICK_OPT_REG_CFG_USER', (MTFW_KICK_OPT_REG_CFG_FW:=2): 'MTFW_KICK_OPT_REG_CFG_FW', (MTFW_KICK_OPT_REG_CFG_EXTEND:=3): 'MTFW_KICK_OPT_REG_CFG_EXTEND'}
@c.record
class MTFW_KICK_SCHEDULE_TABLE(c.Struct):
  SIZE = 8
  value: int
  schTable: c.Array[ctypes.c_ubyte, Literal[8]]
MTFW_KICK_SCHEDULE_TABLE.register_fields([('value', uint64_t, 0), ('schTable', c.Array[uint8_t, Literal[8]], 0)])
MTFW_FWIF_PRIORITY: dict[int, str] = {(MTFW_PRIORITY_LOWEST:=0): 'MTFW_PRIORITY_LOWEST', (MTFW_PRIORITY_0:=0): 'MTFW_PRIORITY_0', (MTFW_PRIORITY_1:=1): 'MTFW_PRIORITY_1', (MTFW_PRIORITY_HIGHEST:=1): 'MTFW_PRIORITY_HIGHEST', (MTFW_PRIORITY_LEVEL_NUMS:=2): 'MTFW_PRIORITY_LEVEL_NUMS'}
MTFW_FWIF_GPU_UTIL_STATE: dict[int, str] = {(GPU_UTIL_STATE_IDX_IDLE:=0): 'GPU_UTIL_STATE_IDX_IDLE', (GPU_UTIL_STATE_IDX_ACTIVE:=1): 'GPU_UTIL_STATE_IDX_ACTIVE', (GPU_UTIL_STATE_IDX_MAX:=2): 'GPU_UTIL_STATE_IDX_MAX'}
MTFW_FWIF_GPU_UTIL_ITEM_INDEX: dict[int, str] = {(GPU_UTIL_ITEM_IDX_OVERALL_UTIL:=0): 'GPU_UTIL_ITEM_IDX_OVERALL_UTIL', (GPU_UTIL_ITEM_IDX_TDM:=1): 'GPU_UTIL_ITEM_IDX_TDM', (GPU_UTIL_ITEM_IDX_GEOM:=2): 'GPU_UTIL_ITEM_IDX_GEOM', (GPU_UTIL_ITEM_IDX_3D:=3): 'GPU_UTIL_ITEM_IDX_3D', (GPU_UTIL_ITEM_IDX_CDM:=4): 'GPU_UTIL_ITEM_IDX_CDM', (GPU_UTIL_ITEM_IDX_CE:=5): 'GPU_UTIL_ITEM_IDX_CE', (GPU_UTIL_ITEM_IDX_MAX:=6): 'GPU_UTIL_ITEM_IDX_MAX'}
@c.record
class MTFW_FWIF_GPU_UTIL_ITEM(c.Struct):
  SIZE = 32
  latestUpdateState: int
  latestUpdateTimestamp: int
  statsCounters: c.Array[ctypes.c_uint64, Literal[2]]
MTFW_FWIF_GPU_UTIL_ITEM.register_fields([('latestUpdateState', uint64_t, 0), ('latestUpdateTimestamp', uint64_t, 8), ('statsCounters', c.Array[uint64_t, Literal[2]], 16)])
@c.record
class MTFW_FWIF_GPU_UTIL_BUF(c.Struct):
  SIZE = 1216
  gpuDmUtil: c.Array[MTFW_FWIF_GPU_UTIL_ITEM, Literal[6]]
  fwScheCounterBuf: c.Array[ctypes.c_uint32, Literal[256]]
MTFW_FWIF_GPU_UTIL_BUF.register_fields([('gpuDmUtil', c.Array[MTFW_FWIF_GPU_UTIL_ITEM, Literal[6]], 0), ('fwScheCounterBuf', c.Array[uint32_t, Literal[256]], 192)])
MTFW_FWIF_ABI_TYPE: dict[int, str] = {(MTFW_FWIF_ABI_TYPE_RESERVE:=0): 'MTFW_FWIF_ABI_TYPE_RESERVE', (MTFW_FWIF_ABI_TYPE_GEN:=1): 'MTFW_FWIF_ABI_TYPE_GEN', (MTFW_FWIF_ABI_TYPE_SUBMISSION:=2): 'MTFW_FWIF_ABI_TYPE_SUBMISSION', (MTFW_FWIF_ABI_TYPE_KICK:=3): 'MTFW_FWIF_ABI_TYPE_KICK'}
@c.record
class MTFW_FWIF_VERSION(c.Struct):
  SIZE = 16
  version: MTFW_FWIF_VERSION_version
  abiEnum: int
  abiName: c.Array[ctypes.c_ubyte, Literal[8]]
@c.record
class MTFW_FWIF_VERSION_version(c.Struct):
  SIZE = 4
  field: MTFW_FWIF_VERSION_version_field
  value: int
@c.record
class MTFW_FWIF_VERSION_version_field(c.Struct):
  SIZE = 4
  abiType: int
  minVer: int
  maxVer: int
MTFW_FWIF_VERSION_version_field.register_fields([('abiType', uint32_t, 0, 8, 0), ('minVer', uint32_t, 1, 12, 0), ('maxVer', uint32_t, 2, 12, 4)])
MTFW_FWIF_VERSION_version.register_fields([('field', MTFW_FWIF_VERSION_version_field, 0), ('value', uint32_t, 0)])
MTFW_FWIF_VERSION.register_fields([('version', MTFW_FWIF_VERSION_version, 0), ('abiEnum', uint32_t, 4), ('abiName', c.Array[uint8_t, Literal[8]], 8)])
MTFWIF_INIT_CFG_DISABLE_CE = 0x00000001
MTFWIF_INIT_CFG_PREFETCH_SHIFT = 1
MTFWIF_INIT_CFG_PREFETCH_BITMASK = (0x00000003 << MTFWIF_INIT_CFG_PREFETCH_SHIFT)
MTFWIF_INIT_CFG_SIZE = 2048
MTFWIF_FW_BUILD_VERSION_SIZE = 8
MTFW_COREMEM_SIZE = (0x18000)
MTFW_NUM_OS = 1
MTFW_HOST_OS = (0)
MTFW_FW_HEAP_SIZE = (1 << 23)
MTFW_EXT_OSID_CCB_SIZE = (128)
MTFW_EXT_OSID_SIZE = (8)
MTFW_VIRTUALISATION_REG_SIZE_PER_OS = (0x10000)
MTFW_VGPU_INIT_FIXED_SIZE = 450000
MTFW_VGPU_TRACEBUF_FIXED_SIZE = 97560
MTFW_MAX_DM_NUM = 6
MTFW_DEPEND_Q_SEM_NUMS = 32
MTFW_FINISH_Q_SEM_NUMS = 8
MTFW_CCB_QUEUE_LENGTH = 256
MTFW_FWIF_LAST_WORD_MAGIC_KEY = 0x4d545946
MTFW_FWIF_REG_DUMP_MAGIC_KEY = 0x4d54
MTFW_FWIF_FAULT_LOG_TRACE_NUM = 64
MTFW_FWIF_GEN_VER_1 = 1
MTFW_FWIF_GEN_VER = MTFW_FWIF_GEN_VER_1
MTFW_RSRV_PM_PAGE_CNT = 1536
MTFW_SCG_PB_COUNT = 2
MTFW_MCG2_PB_COUNT = 4
MTFW_MCG3_PB_COUNT = 6
MTFW_MCG4_PB_COUNT = 8
MTFW_MCG5_PB_COUNT = 10
MTFW_MCG6_PB_COUNT = 12
MTFW_MCG7_PB_COUNT = 14
MTFW_MCG8_PB_COUNT = 16
MTFW_FWIF_CONFIG = MTFW_FWIF_CONFIG_GEN1
X = lambda A,B: A, # type: ignore
MTFW_FWIF_TRACE_BUFFER_SIZE = 24000
MTFW_FWIF_TRACE_BUFFER_ASSERT_SIZE = 200
MTFW_FWIF_THREAD_NUM = 1
MTFW_FWIF_ASSERT_EXTINFO_SIZE = 256
MTFW_NODE_QUEUE_LENGTH = 64
MTFW_PERF_EVENT_MAX = (5120)
RGX_HWPERF_EVENT_MASK_VALUE = lambda e: (UINT64_C(1) << (e)) # type: ignore
PERF_EVENT_FINISH_OFFSET = 0x20
MTFW_FWIF_PERF_EVENT_SIG = 0x46575030
MTFW_FWIF_PERF_EVENT_ISVALID = lambda sig: ((sig) == MTFW_FWIF_PERF_EVENT_SIG) # type: ignore
MTFW_FWIF_QUEUE_MAX = 8
MTFW_SUBM_RETRY_DEFAULT_COUNT = 3
MTFW_KICK_TIMEOUT_DEFAULT_SEC = 200
MTFW_KICK_RETRY_TIMEOUT_US = 1000000
MTFW_Q_SEM_ATTR_MASK = (7)
MTFW_Q_SEM_ADDRESS_MASK = (~(MTFW_Q_SEM_ATTR_MASK))
MTFW_Q_SEM_ATTR_WITH_SHADOW = (1 << 2)
MTFW_Q_SEM_ATTR_ERR_DIFFISON = (1 << 1)
MTFW_Q_SEM_ATTR_SET_ZERO = (1 << 2)
MTFW_Q_SEM_TYPE_MASK = (3)
MTFW_Q_SEM_VALUE_INVALID = 0x8000000000000000
MTFW_Q_SEM_ARRAY_SIZE = 3
MTFW_SUBMISSION_NAME_NOP = "S-NOP"
MTFW_SUBMISSION_NAME_PREEMPTION = "S-PREE"
MTFW_SUBMISSION_NAME_UQ = "S-UQ"
MTFW_SUBMISSION_NAME_TDM = "S-TDM"
MTFW_SUBMISSION_NAME_CDM = "S-CDM"
MTFW_SUBMISSION_NAME_CE = "S-CE"
MTFW_SUBMISSION_NAME_CDM_STREAM = "S-CDMS"
MTFW_SUBMISSION_NAME_GFX = "S-GFX"
MTFW_SUBMISSION_GPU_CMD_NUMS = (MTFW_SUBMISSION_CMD_MAX_NUMS - MTFW_SUBMISSION_CMD_GPU_START)
MTFW_KICK_NAME_NOP = "R-NOP"
MTFW_KICK_NAME_CDM = "R-CDM"
MTFW_KICK_NAME_TDM = "R-TDM"
MTFW_KICK_NAME_CE = "R-CE"
MTFW_KICK_NAME_GFX = "R-GFX"
MTFW_KICK_NAME_SPR = "R-SPR"
MTFW_KICK_NAME_SPACE = "R-SPACE"
MTFW_KICK_NAME_PFM = "R-PFM"
MTFW_KICK_SEMAPHORE_MAX_NUMS = 8
MTFW_KICK_OPT_REG_CFG_MASK = 0x3
MTFW_KICK_OPT_REG_CFG_SET = lambda cfg,cfgIdx: (((cfg) & MTFW_KICK_OPT_REG_CFG_MASK) << (2 * (cfgIdx))) # type: ignore
MTFW_KICK_OPT_REG_CFG_GET = lambda optCfg,cfgIdx: ((optCfg) >> (2 * (cfgIdx)) & MTFW_KICK_OPT_REG_CFG_MASK) # type: ignore
MTFW_KICK_OPT_REG_BIT_IDX_TO_CFG_IDX = lambda bitIdx: ((bitIdx) >> 1) # type: ignore
MTFW_SCHEDULE_TABLE_MAX_NUM = 8
MTFW_ALIGN = lambda n: __attribute__((aligned(n))) # type: ignore
RGXFW_ALIGN = lambda n: MTFW_ALIGN(n) # type: ignore
MTFW_FWIF_VERSION_VALUE = lambda abiType,minVer,maxVer: (((abiType)&0xff) | (((minVer)&0xfff) << 8) | (((maxVer)&0xfff) << 20)) # type: ignore
MTWF_FWIF_NAME_GEN = "F-GEN"
MTFW_FWIF_VERSION_ITEM_MAX_NUM = 256