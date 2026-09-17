import ctypes, itertools, struct, time
from collections import defaultdict
from collections.abc import Callable
from tinygrad.runtime.autogen import mesa
from test.mockgpu.qcom.emu import Image, Memory, run_workgroup

def field(word:int, name:str) -> int: return (word & getattr(mesa, name+'__MASK')) >> getattr(mesa, name+'__SHIFT')
def address(words:list[int], offset:int=0) -> int: return words[offset] | words[offset+1] << 32

class QCOMGPU:
  def __init__(self, ranges:Callable[[], tuple[tuple[int, int], ...]]):
    self.ranges, self.registers, self.constants = ranges, defaultdict[int, int](int), [0]*1024
    self.program_address = self.program_size = self.submissions = self.dispatches = 0
    self.state_blocks:dict[tuple[int, int, int], tuple[int, ...]] = {}

  def submit(self, pointer:int, size:int):
    memory = Memory(self.ranges())
    memory.check(pointer, size)
    words = list(struct.unpack('<'+'I'*(size//4), ctypes.string_at(pointer, size)))
    cursor = 0
    while cursor < len(words):
      header = words[cursor]
      kind = header >> 28
      count = header & (0x7f if kind == 4 else 0x3fff)
      values = words[cursor+1:cursor+1+count]
      if len(values) != count: raise ValueError('Truncated A630 command packet')
      if kind == 4:
        register = (header >> 8) & 0x3ffff
        self.registers.update((register+i, value) for i,value in enumerate(values))
      elif kind == 7: self.packet((header >> 16) & 0x7f, values, memory)
      else: raise ValueError(f'Unsupported A630 packet type {kind}')
      cursor += count+1
    self.submissions += 1

  def packet(self, opcode:int, values:list[int], memory:Memory):
    if opcode in (mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES, mesa.CP_SET_MARKER): return
    if opcode == mesa.CP_EVENT_WRITE:
      if field(values[0], 'CP_EVENT_WRITE_0_EVENT') == mesa.CACHE_FLUSH_TS: memory.write(address(values, 1), 4, values[3])
    elif opcode == mesa.CP_REG_TO_MEM:
      memory.write(address(values, 1), 8, time.perf_counter_ns()*192//10000)
    elif opcode == mesa.CP_WAIT_REG_MEM:
      if (memory.read(address(values, 1), 4) & values[4]) < (values[3] & values[4]):
        raise RuntimeError('A630 command is waiting on an unsatisfied memory dependency')
    elif opcode == mesa.CP_LOAD_STATE6_FRAG:
      state = values[0]
      block, kind, units = (field(state, 'CP_LOAD_STATE6_0_'+name) for name in ('STATE_BLOCK', 'STATE_TYPE', 'NUM_UNIT'))
      if field(state, 'CP_LOAD_STATE6_0_STATE_SRC') != mesa.SS6_INDIRECT: raise ValueError('Unsupported direct A630 state load')
      pointer = address(values, 1)
      if block == mesa.SB6_CS_SHADER and kind == mesa.ST_CONSTANTS:
        offset = field(state, 'CP_LOAD_STATE6_0_DST_OFF')*4
        memory.check(pointer, units*16)
        self.constants[offset:offset+units*4] = struct.unpack('<'+'I'*(units*4), ctypes.string_at(pointer, units*16))
      elif block == mesa.SB6_CS_SHADER and kind == mesa.ST_SHADER:
        self.program_address, self.program_size = pointer, units*128
      elif (block,kind) in ((mesa.SB6_CS_SHADER,mesa.ST6_UAV), (mesa.SB6_CS_TEX,mesa.ST_CONSTANTS), (mesa.SB6_CS_TEX,mesa.ST_SHADER)):
        stride = 16 if kind == mesa.ST_SHADER else 64
        memory.check(pointer, units*stride)
        offset = field(state, 'CP_LOAD_STATE6_0_DST_OFF')
        for i in range(units):
          self.state_blocks[block,kind,offset+i] = struct.unpack('<'+'I'*(stride//4), ctypes.string_at(pointer+i*stride, stride))
      else: raise ValueError(f'Unsupported A630 state block/type {block}/{kind}')
    elif opcode == mesa.CP_EXEC_CS: self.execute(tuple(values[1:4]), memory)
    elif opcode == mesa.CP_RUN_OPENCL:
      self.execute(tuple(self.registers[mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X+i] for i in range(3)), memory)
    else: raise ValueError(f'Unsupported A630 command {opcode:#x}')

  def images(self, memory:Memory, textures:bool) -> tuple[Image, ...]:
    count = field(self.registers[mesa.REG_A6XX_SP_CS_CONFIG], 'A6XX_SP_CS_CONFIG_'+('NTEX' if textures else 'NUAV'))
    block, kind = (mesa.SB6_CS_TEX, mesa.ST_CONSTANTS) if textures else (mesa.SB6_CS_SHADER, mesa.ST6_UAV)
    base = mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE if textures else mesa.REG_A6XX_SP_CS_UAV_BASE
    pointer = self.registers[base] | self.registers[base+1] << 32
    result = []
    for index in range(count):
      words = self.state_blocks.get((block,kind,index))
      if words is None:
        memory.check(pointer+index*64, 64)
        words = struct.unpack('<16I', ctypes.string_at(pointer+index*64, 64))
      fmt = field(words[0], 'A6XX_TEX_CONST_0_FMT')
      if fmt not in (mesa.FMT6_32_32_32_32_FLOAT, mesa.FMT6_16_16_16_16_FLOAT): raise ValueError(f'Unsupported A630 image format {fmt}')
      swizzle = tuple(field(words[0], 'A6XX_TEX_CONST_0_SWIZ_'+axis) for axis in 'XYZW') if textures else (0,1,2,3)
      result.append(Image(words[4] | words[5] << 32, field(words[1], 'A6XX_TEX_CONST_1_WIDTH'), field(words[1], 'A6XX_TEX_CONST_1_HEIGHT'),
                          field(words[2], 'A6XX_TEX_CONST_2_PITCH'), fmt == mesa.FMT6_16_16_16_16_FLOAT, swizzle))
    return tuple(result)

  def execute(self, groups:tuple[int, ...], memory:Memory):
    geometry = self.registers[mesa.REG_A6XX_SP_CS_NDRANGE_0]
    local_size = tuple(field(geometry, 'A6XX_SP_CS_NDRANGE_0_LOCALSIZE'+axis)+1 for axis in 'XYZ')
    config = self.registers[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
    wgid, wgsz, wgoff, lid = (field(config, 'A6XX_SP_CS_CONST_CONFIG_0_'+name)
                             for name in ('WGIDCONSTID', 'WGSIZECONSTID', 'WGOFFSETCONSTID', 'LOCALIDREGID'))
    linear = field(self.registers[mesa.REG_A6XX_SP_CS_WGE_CNTL], 'A6XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID')
    start = self.registers[mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET]
    memory.check(self.program_address, self.program_size)
    # Keep the whole image: OpenCL kernels can call helper functions preceding their entry point.
    program = ctypes.string_at(self.program_address, self.program_size)
    shared_size = (field(self.registers[mesa.REG_A6XX_SP_CS_CNTL_1], 'A6XX_SP_CS_CNTL_1_SHARED_SIZE')+1)*1024
    private_size = field(self.registers[mesa.REG_A6XX_SP_CS_PVT_MEM_PARAM], 'A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM')*512
    textures, images = self.images(memory, True), self.images(memory, False)
    constant_demotion = bool(self.registers[mesa.REG_A6XX_SP_MODE_CNTL] & mesa.A6XX_SP_MODE_CNTL_CONSTANT_DEMOTION_ENABLE)
    for group in itertools.product(*(range(count) for count in groups)):
      workgroup = []
      for local in itertools.product(*(range(count) for count in local_size)):
        registers = {lid+i: value for i,value in enumerate(local)} if lid != 0xfc else {}
        # Despite their names, these fields select GPRs, not entries in constant RAM (Mesa a6xx.xml).
        for base,values in ((wgid, group), (wgsz, local_size), (wgoff, tuple(g*l for g,l in zip(group, local_size)))):
          if base != 0xfc: registers.update((base+i, value) for i,value in enumerate(values))
        if linear != 0xfc: registers[linear] = local[0]+local_size[0]*(local[1]+local_size[1]*local[2])
        workgroup.append(registers)
      run_workgroup(program, tuple(self.constants), memory, workgroup, shared_size, private_size, textures, images, start, constant_demotion)
    self.dispatches += 1
