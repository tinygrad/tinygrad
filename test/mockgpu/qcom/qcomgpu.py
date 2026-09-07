"""A630 compute PM4: validate a complete command, then execute resumable actions."""
# The KGSL driver owns contexts, queued command words, and the AddressSpace.
# It supplies a MemoryTransaction to this module for one execution segment.
# PM4 turns those words into validated actions; a launch delegates its image,
# constants, and workgroup geometry to IR3. PM4/global accesses use the transaction.
# The driver publishes staged writes and records progress after this module returns.

from dataclasses import dataclass
import functools, math, operator, time
from typing import Literal
from tinygrad.runtime.autogen import mesa
from test.mockgpu.qcom.image import ImageBindings, decode_image_descriptor, validate_sampler


class PM4Fault(RuntimeError):
  pass


def require(condition, message):
  if not condition:
    raise PM4Fault(message)


@dataclass(frozen=True)
class Packet:
  # Framing is decoded here; opcode/register semantics are checked during planning.
  # A type-4 target is a register address; a type-7 target is an opcode.
  kind:int
  target:int
  words:tuple[int, ...]


def decode_packets(words:tuple[int, ...]) -> tuple[Packet, ...]:
  require(
    all(type(word) is int and 0 <= word <= 0xffffffff for word in words),
    "PM4 words must be unsigned 32-bit integers",
  )
  packets, cursor = [], 0

  while cursor < len(words):
    header, kind = words[cursor], words[cursor] >> 28
    if kind == 7:
      count, target = header & 0x3fff, header >> 16 & 0x7f
      expected = mesa.CP_TYPE7_PKT | count | ((1 ^ (count.bit_count() & 1)) << 15)
      expected |= target << 16 | ((1 ^ (target.bit_count() & 1)) << 23)
    elif kind == 4:
      count, target = header & 0x7f, header >> 8 & 0x3ffff
      require(count > 0 and target + count <= 0x40000, "PM4 register packet extent is invalid")
      expected = mesa.CP_TYPE4_PKT | count | ((1 ^ (count.bit_count() & 1)) << 7)
      expected |= target << 8 | ((1 ^ (target.bit_count() & 1)) << 27)
    else:
      raise PM4Fault(f"unsupported PM4 packet type {kind}")

    # Reconstructing the full header checks parity and every reserved bit together.
    require(header == expected, "PM4 header parity or reserved bits are invalid")
    end = cursor + count + 1
    require(end <= len(words), "truncated PM4 packet")
    packets.append(Packet(kind, target, tuple(words[cursor + 1:end])))
    cursor = end
  return tuple(packets)


def field(value:int, name:str) -> int:
  return (value & getattr(mesa, name + "__MASK")) >> getattr(mesa, name + "__SHIFT")


# Register admission is finite and explicit. Fixed values select supported modes;
# masks admit individual fields whose relationships are checked by launch().
FIXED_REGISTERS = {
  mesa.REG_A6XX_SP_UPDATE_CNTL: (0, mesa.A6XX_SP_UPDATE_CNTL_CS_STATE | mesa.A6XX_SP_UPDATE_CNTL_CS_UAV),
  mesa.REG_A6XX_SP_CS_TSIZE: (0x80,),
  mesa.REG_A6XX_SP_CS_USIZE: (0x40,),
  mesa.REG_A6XX_SP_MODE_CNTL: (
    mesa.ISAMMODE_GL << mesa.A6XX_SP_MODE_CNTL_ISAMMODE__SHIFT | mesa.A6XX_SP_MODE_CNTL_CONSTANT_DEMOTION_ENABLE,
    mesa.ISAMMODE_CL << mesa.A6XX_SP_MODE_CNTL_ISAMMODE__SHIFT,
  ),
  mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK: (mesa.A6XX_SP_PERFCTR_SHADER_MASK_CS,),
  mesa.REG_A6XX_TPL1_MODE_CNTL: (
    mesa.ISAMMODE_GL << mesa.A6XX_TPL1_MODE_CNTL_ISAMMODE__SHIFT,
    mesa.ISAMMODE_CL << mesa.A6XX_TPL1_MODE_CNTL_ISAMMODE__SHIFT,
  ),
  **dict.fromkeys((
    mesa.REG_A6XX_TPL1_DBG_ECO_CNTL,
    mesa.REG_A6XX_SP_CS_BOOLEAN_CF_MASK,
    mesa.REG_A6XX_SP_CS_NDRANGE_2,
    mesa.REG_A6XX_SP_CS_NDRANGE_4,
    mesa.REG_A6XX_SP_CS_NDRANGE_6,
  ), (0,)),
  **dict.fromkeys((
    mesa.REG_A6XX_SP_REG_PROG_ID_0,
    mesa.REG_A6XX_SP_REG_PROG_ID_1,
    mesa.REG_A6XX_SP_REG_PROG_ID_2,
  ), (0xfcfcfcfc,)),
  mesa.REG_A6XX_SP_REG_PROG_ID_3: (0xfc,),
}

# Each row names one exact register and its admitted fields; no register range is inferred.
REGISTER_MASKS = {
  getattr(mesa, f"REG_A6XX_SP_CS_{register}"): functools.reduce(
    operator.or_,
    (getattr(mesa, f"A6XX_SP_CS_{register}_{name}") for name in fields),
    0,
  )
  for register, fields in {
    "CONST_CONFIG": ("CONSTLEN__MASK", "ENABLED"),
    "CNTL_0": ("HALFREGFOOTPRINT__MASK", "FULLREGFOOTPRINT__MASK", "BRANCHSTACK__MASK", "THREADSIZE__MASK"),
    "CNTL_1": ("SHARED_SIZE__MASK", "CONSTANTRAMMODE__MASK"),
    "PVT_MEM_PARAM": ("MEMSIZEPERITEM__MASK",),
    "PVT_MEM_SIZE": ("TOTALPVTMEMSIZE__MASK",),
    "PVT_MEM_STACK_OFFSET": ("OFFSET__MASK",),
    "WGE_CNTL": ("LINEARLOCALIDREGID__MASK", "THREADSIZE__MASK"),
  }.items()
}

# These words carry full addresses, dimensions, or fields interpreted at launch.
REGISTER_MASKS.update(dict.fromkeys((
  mesa.REG_A6XX_SP_CS_BASE,
  mesa.REG_A6XX_SP_CS_BASE + 1,
  mesa.REG_A6XX_SP_CS_PVT_MEM_BASE,
  mesa.REG_A6XX_SP_CS_PVT_MEM_BASE + 1,
  mesa.REG_A6XX_SP_CS_INSTR_SIZE,
  mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET,
  mesa.REG_A6XX_SP_CS_NDRANGE_0,
  mesa.REG_A6XX_SP_CS_NDRANGE_1,
  mesa.REG_A6XX_SP_CS_NDRANGE_3,
  mesa.REG_A6XX_SP_CS_NDRANGE_5,
  mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
  mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X,
  mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Y,
  mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Z,
), 0xffffffff))
REGISTER_MASKS[mesa.REG_A6XX_SP_CS_CONFIG] = (
  mesa.A6XX_SP_CS_CONFIG_ENABLED | mesa.A6XX_SP_CS_CONFIG_NTEX__MASK |
  mesa.A6XX_SP_CS_CONFIG_NSAMP__MASK | mesa.A6XX_SP_CS_CONFIG_NUAV__MASK
)
# Resource bases are optional: their count fields decide whether a launch uses
# them. They must not become required register state for ordinary buffer kernels.
RESOURCE_BASES = (
  mesa.REG_A6XX_SP_CS_SAMPLER_BASE, mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE,
  mesa.REG_A6XX_SP_CS_UAV_BASE, mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE,
)
OPTIONAL_REGISTERS = {register + part for register in RESOURCE_BASES for part in (0, 1)}
REGISTER_MASKS.update(dict.fromkeys(OPTIONAL_REGISTERS, 0xffffffff))
REGISTER_NAMES = {
  value:name for name,value in vars(mesa).items()
  if name.startswith("REG_A6XX_") and type(value) is int
}


def register_write(registers:dict[int, int], address:int, value:int):
  name = REGISTER_NAMES.get(address, hex(address))
  if address in FIXED_REGISTERS:
    require(value in FIXED_REGISTERS[address], f"unsupported {name} configuration")
  elif address in REGISTER_MASKS:
    require(value & ~REGISTER_MASKS[address] == 0, f"unsupported {name} fields")
  else:
    raise PM4Fault(f"unsupported PM4 register {name}")

  # Only the command-local planning state changes here; no guest write is staged.
  registers[address] = value


def address64(low:int, high:int, alignment:int=4) -> int:
  address = low | high << 32
  require(address > 0 and address % alignment == 0, f"PM4 address must be nonzero and aligned to {alignment} bytes")
  return address


@dataclass(frozen=True)
class MemoryAction:
  # The operation selects a footprint and residual behavior. Planning validates
  # the footprint; application samples live data/time or stages a write.
  operation:Literal["wait", "store", "timestamp"]
  address:int
  value:int=0

  @property
  def size(self) -> int:
    return 8 if self.operation == "timestamp" else 4

  def apply(self, memory) -> bool:
    # False means this action is blocked; its cursor must be retained by the driver.
    if self.operation == "wait":
      return int.from_bytes(memory.read(self.address, self.size), "little") >= self.value

    # Timestamps use 19.2 MHz counter units and are sampled during application.
    value = time.perf_counter_ns() * 192 // 10000 if self.operation == "timestamp" else self.value
    memory.write(self.address, value.to_bytes(self.size, "little"))
    return True


@dataclass(frozen=True)
class Launch:
  # This immutable snapshot records one launch's validated inputs. Later register
  # packets cannot alter its geometry, resource sizes, or selected shader image.
  image:bytes
  image_address:int
  constants_address:int
  constants_size:int

  groups:tuple[int, int, int]
  local:tuple[int, int, int]

  # ABI positions for workgroup ID, local ID, and workgroup-size values.
  wgid:int
  lid:int
  wgsize:int

  shared_bytes:int
  private_bytes:int
  image_bindings:ImageBindings=ImageBindings()
  image_uploads:tuple[tuple[int, bytes], ...]=()
  opencl:bool=False
  wgoffset:int=0xfc
  entry_pc:int=0

  def apply(self, memory) -> bool:
    from test.mockgpu.qcom import emu

    # Prior actions may have staged writes. The shader must still match the plan;
    # constants are read now so those prior writes are visible to this launch.
    require(memory.read(self.image_address, len(self.image)) == self.image, "shader image changed after PM4 prevalidation")
    for address, data in self.image_uploads:
      require(memory.read(address, len(data)) == data, "image configuration changed after PM4 prevalidation")
    emu.execute(
      self.image, memory.read(self.constants_address, self.constants_size), self.groups, self.local, self.wgid,
      self.lid, memory, shared_bytes=self.shared_bytes, private_bytes=self.private_bytes, wgsize=self.wgsize,
      image_bindings=self.image_bindings,
      opencl=self.opencl, wgoffset=self.wgoffset, constant_demotion=not self.opencl,
      entry_pc=self.entry_pc,
    )
    return True


def image_resources(registers:dict[int, int], loads:dict[tuple[int, int], tuple[int, int]], memory):
  config = registers[mesa.REG_A6XX_SP_CS_CONFIG]
  require(config & mesa.A6XX_SP_CS_CONFIG_ENABLED, "compute configuration is disabled")
  texture_count = field(config, "A6XX_SP_CS_CONFIG_NTEX")
  sampler_count = field(config, "A6XX_SP_CS_CONFIG_NSAMP")
  output_count = field(config, "A6XX_SP_CS_CONFIG_NUAV")
  uploads = []

  def table(block, kind, count, register, unit_size, alignment, prefetch=None):
    if count == 0:
      return b""
    require(register in registers and register + 1 in registers, "missing image resource base")
    address = address64(registers[register], registers[register + 1], alignment)
    expected_units = count if prefetch is None else min(prefetch, count)
    require(loads.get((block, kind)) == (address, expected_units), "image resource load and configuration disagree")
    memory.validate(address, count * unit_size)
    data = memory.read(address, count * unit_size)
    uploads.append((address, data))
    return data

  sampler_data = table(mesa.SB6_CS_TEX, mesa.ST_SHADER, sampler_count, mesa.REG_A6XX_SP_CS_SAMPLER_BASE, 16, 16)
  texture_data = table(mesa.SB6_CS_TEX, mesa.ST_CONSTANTS, texture_count, mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, 64, 64, 16)
  output_data = table(mesa.SB6_CS_SHADER, mesa.ST6_UAV, output_count, mesa.REG_A6XX_SP_CS_UAV_BASE, 64, 16)
  # The CL producer reserves a trailing zero sampler record. Keep it in the
  # upload snapshot, but do not admit an instruction that tries to execute it.
  active_sampler_count = sampler_count
  while active_sampler_count and sampler_data[(active_sampler_count - 1) * 16:active_sampler_count * 16] == bytes(16):
    active_sampler_count -= 1
  for offset in range(0, active_sampler_count * 16, 16):
    validate_sampler(sampler_data[offset:offset + 16])
  if sampler_count:
    register = mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE
    require(register in registers and register + 1 in registers, "missing image border color base")
    address = address64(registers[register], registers[register + 1], 64)
    memory.validate(address, 4096)
    border = memory.read(address, 4096)
    require(not any(border), "unsupported nonzero image border color")
    uploads.append((address, border))

  bindings = ImageBindings(
    tuple(decode_image_descriptor(texture_data[offset:offset + 64], True) for offset in range(0, len(texture_data), 64)),
    tuple(decode_image_descriptor(output_data[offset:offset + 64], False) for offset in range(0, len(output_data), 64)),
    active_sampler_count,
  )
  bindings.validate(memory)
  return bindings, tuple(uploads)


def launch(registers:dict[int, int], loads:dict[tuple[int, int], tuple[int, int]], groups:tuple[int, int, int],
           memory, *, opencl:bool=False) -> Launch:
  # A register file is local to this command. No state is inferred from a prior
  # submission or from the shape/length of a known command stream.
  required = (set(REGISTER_MASKS) | set(FIXED_REGISTERS)) - OPTIONAL_REGISTERS - {mesa.REG_A6XX_SP_UPDATE_CNTL}
  missing = required - registers.keys()
  require(
    not missing,
    "incomplete PM4 compute register state: " + ", ".join(REGISTER_NAMES.get(reg, hex(reg)) for reg in sorted(missing)),
  )
  shader_key, constant_key = (mesa.SB6_CS_SHADER, mesa.ST_SHADER), (mesa.SB6_CS_SHADER, mesa.ST_CONSTANTS)
  require(shader_key in loads and constant_key in loads, "compute launch needs shader and constant loads")
  # Compiler conventions are selected together: CL uses packed half constants
  # and CP_RUN_OPENCL builtin state; Mesa IR3 uses constant demotion and CP_EXEC_CS.
  mode = mesa.ISAMMODE_CL if opencl else mesa.ISAMMODE_GL
  expected_sp = mode << mesa.A6XX_SP_MODE_CNTL_ISAMMODE__SHIFT
  if not opencl:
    expected_sp |= mesa.A6XX_SP_MODE_CNTL_CONSTANT_DEMOTION_ENABLE
  require(registers[mesa.REG_A6XX_SP_MODE_CNTL] == expected_sp and
          registers[mesa.REG_A6XX_TPL1_MODE_CNTL] == mode << mesa.A6XX_TPL1_MODE_CNTL_ISAMMODE__SHIFT,
          "compute dispatch and compiler modes disagree")

  # Dispatch counts, register dimensions, and local workgroup shape must agree.
  range_control = registers[mesa.REG_A6XX_SP_CS_NDRANGE_0]
  require(field(range_control, "A6XX_SP_CS_NDRANGE_0_KERNELDIM") == 3, "unsupported compute kernel dimensions")
  local = (
    field(range_control, "A6XX_SP_CS_NDRANGE_0_LOCALSIZEX") + 1,
    field(range_control, "A6XX_SP_CS_NDRANGE_0_LOCALSIZEY") + 1,
    field(range_control, "A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ") + 1,
  )
  declared_groups = tuple(registers[reg] for reg in (
    mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X,
    mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Y,
    mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Z,
  ))
  global_size = tuple(registers[reg] for reg in (
    mesa.REG_A6XX_SP_CS_NDRANGE_1,
    mesa.REG_A6XX_SP_CS_NDRANGE_3,
    mesa.REG_A6XX_SP_CS_NDRANGE_5,
  ))
  require(all(value > 0 for value in groups + global_size), "compute dimensions must be positive")
  require(
    groups == declared_groups == tuple((size + block - 1) // block for size, block in zip(global_size, local)),
    "compute dispatch and NDRANGE dimensions disagree",
  )

  # The register footprint bounds how many invocations can share one workgroup.
  control = registers[mesa.REG_A6XX_SP_CS_CNTL_0]
  require(field(control, "A6XX_SP_CS_CNTL_0_THREADSIZE") == mesa.THREAD64, "unsupported compute thread size")
  full = field(control, "A6XX_SP_CS_CNTL_0_FULLREGFOOTPRINT")
  half = field(control, "A6XX_SP_CS_CNTL_0_HALFREGFOOTPRINT")
  max_threads = min(1024, (384 * 32 // (max(1, full + (half + 1) // 2) * 128)) * 128)
  require(math.prod(local) <= max_threads, "compute workgroup exceeds its register resource limit")

  # Decode shared/private capacities, and validate the producer's mapped stack.
  # IR3 uses the per-invocation private capacity rather than the native stack address.
  control = registers[mesa.REG_A6XX_SP_CS_CNTL_1]
  require(field(control, "A6XX_SP_CS_CNTL_1_CONSTANTRAMMODE") == mesa.CONSTLEN_256, "unsupported compute constant RAM mode")
  shared_encoding = field(control, "A6XX_SP_CS_CNTL_1_SHARED_SIZE")
  shared_bytes = (shared_encoding + 1) * 1024 if shared_encoding else 32768

  # Per-invocation private capacity is in 512-byte blocks; stack offsets use 2 KiB.
  private_units = field(registers[mesa.REG_A6XX_SP_CS_PVT_MEM_PARAM], "A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM")
  require(registers[mesa.REG_A6XX_SP_CS_PVT_MEM_SIZE] == private_units * 256, "inconsistent compute private memory size")
  private_address = address64(
    registers[mesa.REG_A6XX_SP_CS_PVT_MEM_BASE], registers[mesa.REG_A6XX_SP_CS_PVT_MEM_BASE + 1],
  )
  per_sp_bytes = registers[mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET] << 11
  stack_bytes = per_sp_bytes * 4 # Preserve the producer's conservative four-times-per-SP allocation policy.
  require(stack_bytes > 0, "compute private stack allocation is missing")
  memory.validate(private_address, stack_bytes, write=True)

  # Shader load units are 128 bytes. Constant load units are vec4s (16 bytes),
  # while CONSTLEN encodes groups of four vec4s. Validate each in its own units.
  shader_address, shader_units = loads[shader_key]
  require(
    shader_address == address64(registers[mesa.REG_A6XX_SP_CS_BASE], registers[mesa.REG_A6XX_SP_CS_BASE + 1], 128),
    "shader load and program base disagree",
  )
  require(shader_units == registers[mesa.REG_A6XX_SP_CS_INSTR_SIZE], "shader load and instruction size disagree")

  constants_address, constant_units = loads[constant_key]
  constants_size = constant_units * 16
  constant_config = registers[mesa.REG_A6XX_SP_CS_CONST_CONFIG]
  constant_vec4s = field(constant_config, "A6XX_SP_CS_CONST_CONFIG_CONSTLEN") << 2
  require(
    constant_config & mesa.A6XX_SP_CS_CONST_CONFIG_ENABLED and 0 < constant_vec4s <= 256,
    "constant configuration is disabled or exceeds the RAM bank",
  )
  require(constants_size <= constant_vec4s * 16, "constant load exceeds its declared configuration")
  memory.validate(constants_address, constants_size)

  image = memory.read(shader_address, shader_units * 128)
  entry_pc = registers[mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET]
  from test.mockgpu.qcom import emu
  # Check every encoding, then the reachable constant spans against the actual
  # upload. A short upload is valid; its contents stay live until application.
  emu.validate_constant_footprint(emu.decode(image), constants_size // 4,
                                constant_demotion=not opencl, entry_pc=entry_pc)
  image_bindings, image_uploads = image_resources(registers, loads, memory)
  emu.validate_image_bindings(emu.decode(image), image_bindings)

  # The compiler ABI selects invocation-register and constant positions; 0xfc
  # denotes an unused position. These coordinates seed the IR3 workgroup state.
  config = registers[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
  wgid = field(config, "A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID")
  lid = field(config, "A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID")
  wgsize = field(config, "A6XX_SP_CS_CONST_CONFIG_0_WGSIZECONSTID")
  wgoffset = field(config, "A6XX_SP_CS_CONST_CONFIG_0_WGOFFSETCONSTID")
  emu.validate_invocation_registers(wgid, lid)
  if opencl:
    emu.validate_invocation_registers(wgsize, wgoffset)
    require(constants_size >= 32 * 4, "OpenCL builtin constant range is incomplete")
  else:
    require(wgoffset == 0xfc, "workgroup offsets are unsupported")
    require(wgsize == 0xfc or (wgsize + 3) * 4 <= constants_size, "workgroup size constant range is invalid")

  wave = registers[mesa.REG_A6XX_SP_CS_WGE_CNTL]
  require(
    field(wave, "A6XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID") == 0xfc and
    field(wave, "A6XX_SP_CS_WGE_CNTL_THREADSIZE") == mesa.THREAD64,
    "unsupported compute workgroup execution mode",
  )
  return Launch(
    image, shader_address, constants_address, constants_size, groups, local, wgid, lid, wgsize, shared_bytes, private_units * 512,
    image_bindings, image_uploads,
    opencl=opencl, wgoffset=wgoffset, entry_pc=entry_pc,
  )


def compile_plan(words:tuple[int, ...], memory) -> tuple[MemoryAction|Launch, ...]:
  # Walk the entire command before applying any action. Register/load declarations
  # update this local state; effect packets append immutable actions in order.
  registers:dict[int, int] = {}
  loads:dict[tuple[int, int], tuple[int, int]] = {}
  actions:list[MemoryAction|Launch] = []
  compute_mode = False

  try:
    for packet in decode_packets(words):
      if packet.kind == 4:
        for index, value in enumerate(packet.words):
          register_write(registers, packet.target + index, value)
        continue

      opcode, payload = packet.target, packet.words
      if opcode in (mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES):
        require(len(payload) == 0, "invalid wait packet length")
      elif opcode == mesa.CP_SET_MARKER:
        require(payload == (mesa.RM6_COMPUTE,), "unsupported PM4 marker mode")
        compute_mode = True
      elif opcode == mesa.CP_EVENT_WRITE:
        if payload == (mesa.CACHE_INVALIDATE,):
          continue
        require(len(payload) == 4 and payload[0] == mesa.CACHE_FLUSH_TS, "unsupported PM4 event write")
        actions.append(MemoryAction(operation="store", address=address64(payload[1], payload[2]), value=payload[3]))
      elif opcode == mesa.CP_WAIT_REG_MEM:
        control = (
          mesa.WRITE_GE << mesa.CP_WAIT_REG_MEM_0_FUNCTION__SHIFT | mesa.POLL_MEMORY << mesa.CP_WAIT_REG_MEM_0_POLL__SHIFT
        )
        require(len(payload) == 6 and payload[0] == control and payload[4] == 0xffffffff, "unsupported PM4 memory wait")
        actions.append(MemoryAction(operation="wait", address=address64(payload[1], payload[2]), value=payload[3]))
      elif opcode == mesa.CP_REG_TO_MEM:
        control = (
          mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER | 2 << mesa.CP_REG_TO_MEM_0_CNT__SHIFT | mesa.CP_REG_TO_MEM_0_64B
        )
        require(len(payload) == 3 and payload[0] == control, "unsupported PM4 register-to-memory transfer")
        actions.append(MemoryAction(operation="timestamp", address=address64(payload[1], payload[2], 8)))
      elif opcode == mesa.CP_LOAD_STATE6_FRAG:
        require(len(payload) == 3, "unsupported PM4 state load length")
        control = payload[0]
        kind = field(control, "CP_LOAD_STATE6_0_STATE_TYPE")
        block = field(control, "CP_LOAD_STATE6_0_STATE_BLOCK")
        units = field(control, "CP_LOAD_STATE6_0_NUM_UNIT")
        alignment = {
          (mesa.SB6_CS_SHADER, mesa.ST_SHADER): 128,
          (mesa.SB6_CS_SHADER, mesa.ST_CONSTANTS): 16,
          (mesa.SB6_CS_SHADER, mesa.ST6_UAV): 16,
          (mesa.SB6_CS_TEX, mesa.ST_SHADER): 16,
          (mesa.SB6_CS_TEX, mesa.ST_CONSTANTS): 64,
        }.get((block, kind))
        if alignment is None:
          raise PM4Fault("unsupported PM4 resource load")
        require(
          field(control, "CP_LOAD_STATE6_0_STATE_SRC") == mesa.SS6_INDIRECT and
          field(control, "CP_LOAD_STATE6_0_DST_OFF") == 0 and units > 0,
          "unsupported PM4 resource load",
        )
        loads[(block, kind)] = address64(payload[1], payload[2], alignment), units
      elif opcode == mesa.CP_EXEC_CS:
        require(len(payload) == 4 and payload[0] == 0 and compute_mode, "unsupported PM4 compute dispatch")
        actions.append(launch(registers, loads, (payload[1], payload[2], payload[3]), memory))
      elif opcode == mesa.CP_RUN_OPENCL:
        require(payload == (0,) and compute_mode, "unsupported PM4 OpenCL dispatch")
        groups = (
          registers.get(mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X, 0),
          registers.get(mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Y, 0),
          registers.get(mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Z, 0),
        )
        actions.append(launch(registers, loads, groups, memory, opencl=True))
      else:
        raise PM4Fault(f"unsupported PM4 opcode {opcode:#x}")

    # Even a fault in a late action's footprint must reject before a valid prefix
    # writes. Launch resources were validated when each launch snapshot was built.
    for action in actions:
      if isinstance(action, MemoryAction):
        memory.validate(action.address, action.size, write=action.operation != "wait")
  except (ValueError, RuntimeError) as error:
    if isinstance(error, PM4Fault):
      raise
    raise PM4Fault(f"PM4 prevalidation failed: {error}") from error
  return tuple(actions)


def execute_command(words:tuple[int, ...], memory, timestamp:int, start:int=0) -> tuple[bool, int]:
  # Revalidate the whole command on each resume, against the current transaction.
  # The cursor counts actions, not packets: the driver commits a successful prefix
  # and resumes at a blocked wait without repeating earlier stores or timestamps.
  # The timestamp argument is retained for the driver interface; timestamp actions
  # sample their own time in MemoryAction.apply().
  actions = compile_plan(words, memory)
  require(type(start) is int and 0 <= start <= len(actions), "invalid PM4 action cursor")

  for cursor in range(start, len(actions)):
    if not actions[cursor].apply(memory):
      return False, cursor
  return True, len(actions)
