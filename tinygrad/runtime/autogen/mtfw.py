# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
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
mt_fwif_t: TypeAlias = struct_MTFW_FWIF_TAG
@c.record
class struct_MTFW_FWIF_TAG_flags(c.Struct):
  SIZE = 4
  field: struct_MTFW_FWIF_TAG_flags_field
  value: int
@c.record
class struct_MTFW_FWIF_TAG_flags_field(c.Struct):
  SIZE = 4
  version: int
uint32_t: TypeAlias = ctypes.c_uint32
struct_MTFW_FWIF_TAG_flags_field.register_fields([('version', uint32_t, 0, 4, 0)])
struct_MTFW_FWIF_TAG_flags.register_fields([('field', struct_MTFW_FWIF_TAG_flags_field, 0), ('value', uint32_t, 0)])
uint8_t: TypeAlias = ctypes.c_ubyte
fw_va: TypeAlias = ctypes.c_uint64
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
uint64_t: TypeAlias = ctypes.c_uint64
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
@c.record
class MTFW_FWIF_STARTUP_INFO(c.Struct):
  SIZE = 4
  field: MTFW_FWIF_STARTUP_INFO_field
  value: int
mt_fwif_startup_info_t: TypeAlias = MTFW_FWIF_STARTUP_INFO
@c.record
class MTFW_FWIF_STARTUP_INFO_field(c.Struct):
  SIZE = 4
  magicValue: int
  threadNum: int
MTFW_FWIF_STARTUP_INFO_field.register_fields([('magicValue', uint32_t, 0, 8, 0), ('threadNum', uint32_t, 1, 1, 0)])
MTFW_FWIF_STARTUP_INFO.register_fields([('field', MTFW_FWIF_STARTUP_INFO_field, 0), ('value', uint32_t, 0)])
@c.record
class MTFW_FWIF_WATCHDOG(c.Struct):
  SIZE = 16
  t0WatchDogCounter: int
  t1WatchDogCounter: int
  t0State: int
  t1State: int
mt_fwif_watchdog_t: TypeAlias = MTFW_FWIF_WATCHDOG
MTFW_FWIF_WATCHDOG.register_fields([('t0WatchDogCounter', uint32_t, 0), ('t1WatchDogCounter', uint32_t, 4), ('t0State', uint32_t, 8), ('t1State', uint32_t, 12)])
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
mt_fwif_fec_boot_cfg_t: TypeAlias = struct__MTFW_FWIF_FEC_BOOT_CFG_
struct__MTFW_FWIF_FEC_BOOT_CFG_.register_fields([('fwifFwVaLow', uint32_t, 0), ('fwifFwVaHi', uint32_t, 4), ('fwifHeapSize', uint32_t, 8), ('fwvaPageTableRootLow', uint32_t, 12), ('fwvaPageTableRootHi', uint32_t, 16), ('reservePPageAddrLow', uint32_t, 20), ('reservePPageAddrHi', uint32_t, 24), ('reservePPageNum', uint32_t, 28)])
mt_fwif_config_gen1_t: TypeAlias = MTFW_FWIF_CONFIG_GEN1
mt_fwif_head_t: TypeAlias = MTFW_FWIF_HEAD
mt_ring_ctrl_t: TypeAlias = MTFW_RING_CTRL
mt_ccb_queue_t: TypeAlias = MTFW_CCB_QUEUE
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
mt_ccb_item_t: TypeAlias = MTFW_CCB_ITEM
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
MTFW_CCB_ITEM_PARA: TypeAlias = union__MTFW_CCB_ITEM_PARA_
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
class MTFW_SUBMISSION_Q_SEM_PARA(c.Struct):
  SIZE = 48
  pageTableRootAddr: int
  pageTableGeneration: int
  writeQSem: MTFW_Q_SEMAPHORE
MTFW_SUBMISSION_Q_SEM_PARA.register_fields([('pageTableRootAddr', gpu_pa, 0), ('pageTableGeneration', uint64_t, 8), ('writeQSem', MTFW_Q_SEMAPHORE, 16)])
@c.record
class MTFW_SUBMISSION_SCHEDULE_PARA(c.Struct):
  SIZE = 32
  pageTableRootAddr: int
  pageTableGeneration: int
  ccbItemToken: int
  ccbType: int
  submissionQueueId: int
MTFW_FWIF_CCB_TYPE: dict[int, str] = {(MTFW_CCB_TYPE_GP:=0): 'MTFW_CCB_TYPE_GP', (MTFW_CCB_TYPE_UQ:=1): 'MTFW_CCB_TYPE_UQ', (MTFW_CCB_TYPE_TDM:=2): 'MTFW_CCB_TYPE_TDM', (MTFW_CCB_TYPE_CDM:=3): 'MTFW_CCB_TYPE_CDM', (MTFW_CCB_TYPE_CE:=4): 'MTFW_CCB_TYPE_CE', (MTFW_CCB_TYPE_MAX_NUM:=5): 'MTFW_CCB_TYPE_MAX_NUM'}
MTFW_SUBMISSION_SCHEDULE_PARA.register_fields([('pageTableRootAddr', gpu_pa, 0), ('pageTableGeneration', uint64_t, 8), ('ccbItemToken', uint64_t, 16), ('ccbType', ctypes.c_uint32, 24), ('submissionQueueId', uint32_t, 28)])
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
union__MTFW_CCB_ITEM_PARA_.register_fields([('submissionPara', MTFW_SUBMISSION_PARA, 0), ('qSempPara', MTFW_SUBMISSION_Q_SEM_PARA, 0), ('schPara', MTFW_SUBMISSION_SCHEDULE_PARA, 0), ('logCfg', MTFW_FWIF_TRACE_LOG_CONFIG, 0), ('perfRequest', MTFW_FWIF_PERF_REQUEST, 0), ('gpuUtilCfg', MTFW_FWIF_GPU_UTIL_CONFIG, 0), ('vmInfo', MTFW_FWIF_VM_SCH_INFO, 0), ('haltCfg', MTFW_FWIF_FIRMWARE_HALT, 0), ('invalidTlbCfg', MTFW_SUBMISSION_INVALID_TLB, 0), ('pcSamplerCfg', MTFW_FWIF_PC_SAMPLER_CFG, 0)])
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
MTFW_CCB_ITEM.register_fields([('flags', MTFW_FWIF_HEAD, 0), ('attribute', MTFW_SUBMISSION_ATTRIBUTE, 4), ('submissionCmd', ctypes.c_uint32, 8), ('priority', uint32_t, 12), ('ccbItemToken', uint64_t, 16), ('hostPrivateData', uint64_t, 24), ('frameId', uint64_t, 32), ('cbId', uint64_t, 40), ('processId', uint32_t, 48), ('ccbItemPara', MTFW_CCB_ITEM_PARA, 56), ('fwBuffer', MTFW_FW_CONTEXT, 1392)])
mt_ccb_resp_t: TypeAlias = MTFW_CCB_RESP
mt_ccb_item_para_t: TypeAlias = union__MTFW_CCB_ITEM_PARA_
mt_submission_para_t: TypeAlias = MTFW_SUBMISSION_PARA
mt_submission_q_sem_para_t: TypeAlias = MTFW_SUBMISSION_Q_SEM_PARA
mt_q_semaphore_t: TypeAlias = MTFW_Q_SEMAPHORE
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
mt_submission_head_t: TypeAlias = MTFW_SUBMISSION_HEAD
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
mt_submission_region_desc_t: TypeAlias = MTFW_SUBMISSION_REGION_DESC
MTFW_SUBMISSION_REGION_TYPE: dict[int, str] = {(MTFW_SUB_REGION_START:=0): 'MTFW_SUB_REGION_START', (MTFW_SUB_REGION_NOP:=1): 'MTFW_SUB_REGION_NOP', (MTFW_SUB_REGION_CDM:=2): 'MTFW_SUB_REGION_CDM', (MTFW_SUB_REGION_TDM:=3): 'MTFW_SUB_REGION_TDM', (MTFW_SUB_REGION_CE:=4): 'MTFW_SUB_REGION_CE', (MTFW_SUB_REGION_GFX:=5): 'MTFW_SUB_REGION_GFX', (MTFW_SUB_REGION_SPR:=6): 'MTFW_SUB_REGION_SPR', (MTFW_SUB_REGION_MAX_KICK_TYPE:=7): 'MTFW_SUB_REGION_MAX_KICK_TYPE', (MTFW_SUB_REGION_K_SEM_BUFFER:=8): 'MTFW_SUB_REGION_K_SEM_BUFFER', (MTFW_SUB_REGION_CSW_BUFFER:=9): 'MTFW_SUB_REGION_CSW_BUFFER', (MTFW_SUB_REGION_ERROR_BUFFER:=10): 'MTFW_SUB_REGION_ERROR_BUFFER', (MTFW_SUB_REGION_TRACE_INFO:=11): 'MTFW_SUB_REGION_TRACE_INFO', (MTFW_SUB_REGION_PFM_CFG:=12): 'MTFW_SUB_REGION_PFM_CFG', (MTFW_SUB_REGION_GEOM_IDX_FETCH_CFG:=13): 'MTFW_SUB_REGION_GEOM_IDX_FETCH_CFG', (MTFW_SUB_REGION_SCHEDULE_TABLE:=14): 'MTFW_SUB_REGION_SCHEDULE_TABLE', (MTFW_SUB_REGION_DBG_CSW_CFG:=15): 'MTFW_SUB_REGION_DBG_CSW_CFG', (MTFW_SUB_REGION_SPACE:=16): 'MTFW_SUB_REGION_SPACE', (MTFW_SUB_REGION_NUMS:=17): 'MTFW_SUB_REGION_NUMS', (MTFW_SUB_REGION_INVALID:=17): 'MTFW_SUB_REGION_INVALID'}
@c.record
class MTFW_SUBMISSION_REGION_DESC_regionStart(c.Struct):
  SIZE = 8
  offset: int
  gpuAddr: int
  fwAddr: int
  value: int
MTFW_SUBMISSION_REGION_DESC_regionStart.register_fields([('offset', uint32_t, 0), ('gpuAddr', gpu_va, 0), ('fwAddr', fw_va, 0), ('value', uint64_t, 0)])
MTFW_SUBMISSION_REGION_DESC.register_fields([('regionType', ctypes.c_uint32, 0), ('regionStart', MTFW_SUBMISSION_REGION_DESC_regionStart, 8), ('regionSize', uint32_t, 16)])
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
mt_kick_head_t: TypeAlias = MTFW_KICK_HEAD
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
@c.record
class MTFW_KICK_SEMAPHORE(c.Struct):
  SIZE = 80
  dependentValue: c.Array[ctypes.c_uint64, Literal[8]]
  finishValue: int
  finishValueMask: int
  reserve: c.Array[ctypes.c_ubyte, Literal[3]]
MTFW_KICK_SEMAPHORE.register_fields([('dependentValue', c.Array[uint64_t, Literal[8]], 0), ('finishValue', uint64_t, 64), ('finishValueMask', uint8_t, 72), ('reserve', c.Array[uint8_t, Literal[3]], 73)])
@c.record
class MTFW_KICK_EVENT_PARA(c.Struct):
  SIZE = 32
  dependKickEventArray: int
  finishKickEventArray: int
  dependKickEventNum: int
  finishKickEventNum: int
  dependKickPollTimeoutSec: int
MTFW_KICK_EVENT_PARA.register_fields([('dependKickEventArray', gpu_va, 0), ('finishKickEventArray', gpu_va, 8), ('dependKickEventNum', uint32_t, 16), ('finishKickEventNum', uint32_t, 20), ('dependKickPollTimeoutSec', uint32_t, 24)])
MTFW_KICK_HEAD.register_fields([('flags', MTFW_FWIF_HEAD, 0), ('kickCtrl', MTFW_KICK_HEAD_kickCtrl, 4), ('kickSize', uint32_t, 8), ('kickTimeoutMs', uint32_t, 12), ('retryTimeoutUs', uint32_t, 16), ('kickSemaphore', MTFW_KICK_SEMAPHORE, 24), ('kickEvent', MTFW_KICK_EVENT_PARA, 104), ('traceAddr', gpu_va, 136), ('timeStampAddr', gpu_va, 144), ('optionalRegCfg', uint64_t, 152), ('reserve', c.Array[uint64_t, Literal[4]], 160)])
@c.record
class MTFW_KICK_SCH_CTRL(c.Struct):
  SIZE = 16
  field: MTFW_KICK_SCH_CTRL_field
  value: int
  reserve: c.Array[ctypes.c_uint32, Literal[4]]
mt_kick_sch_ctrl_t: TypeAlias = MTFW_KICK_SCH_CTRL
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
mt_kick_semaphore_t: TypeAlias = MTFW_KICK_SEMAPHORE
mt_kick_event_para_t: TypeAlias = MTFW_KICK_EVENT_PARA
@c.record
class MTFW_KICK_EVENT(c.Struct):
  SIZE = 16
  addr: int
  value: int
mt_kick_event_t: TypeAlias = MTFW_KICK_EVENT
MTFW_KICK_EVENT.register_fields([('addr', gpu_va, 0), ('value', uint64_t, 8)])
@c.record
class MTFW_FWIF_VERSION(c.Struct):
  SIZE = 16
  version: MTFW_FWIF_VERSION_version
  abiEnum: int
  abiName: c.Array[ctypes.c_ubyte, Literal[8]]
mt_fwif_version_t: TypeAlias = MTFW_FWIF_VERSION
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
