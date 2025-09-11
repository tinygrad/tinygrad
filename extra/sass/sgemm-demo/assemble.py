import argparse, io, sys, struct, dataclasses, re
from extra.sass.assembler.CuAsmParser import CuAsmParser, CuAsmSection, CuAsmSegment
from extra.sass.assembler.CuKernelAssembler import CuKernelAssembler
from extra.sass.assembler.CuInsAssemblerRepos import CuInsAssemblerRepos
from extra.sass.assembler.CuSMVersion import CuSMVersion
from elftools.elf.structs import ELFStructs
from typing import Optional, Union

c_ControlCodesPattern = re.compile(r'B(0|-)(1|-)(2|-)(3|-)(4|-)(5|-):R[0-5\-]:W[0-5\-]:(Y|-):S\d{2}')
c_ControlStringLen = 19
class CuControlCode:
  def __init__(self, code):
    c_waitbar, c_readbar, c_writebar, c_yield, c_stall = CuControlCode.splitCode(code)

    self.Barrier = c_waitbar
    self.Read  = c_readbar
    self.Write = c_writebar
    self.Yield = c_yield
    self.Stall = c_stall
  
  def isYield(self):
    ''' If yield flag is set(code=0).'''
    return self.Yield == 0
  
  def getStallCount(self):
    ''' Get stall count.'''
    return self.Stall
  
  def getReadSB(self):
    ''' Get read scoreboard id, return None if not set.'''
    return None if self.Read == 7 else self.Read
  
  def getWriteSB(self):
    ''' Get write scoreboard id, return None if not set.'''
    return None if self.Write == 7 else self.Write
  
  def getBarrierSet(self):
    ''' Get a set of waiting scoreboards, return empty set if waiting on none.'''
    return {i for i in range(6) if (self.Barrier & (1<<i)>0)}

  @staticmethod
  def splitCode(code):
    ''' Split control codes into parts.

    # c.f. : https://github.com/NervanaSystems/maxas/wiki/Control-Codes
    #      : https://arxiv.org/abs/1903.07486
    # reuse  waitbar  rbar  wbar  yield   stall
    #  0000   000000   000   000      0    0000
    #
    # NOTE : It is known that for some instructions(HMMA.SP), reuse may use some other bits.
    #        And for some other instructions(TEXS/BRA/...), the reuse bits may be used for other encoding fields.
    #        Since reuses are displayed as explicit modifiers, we will not split the reuse field any more.
    #        Other fields will be extracted and encoded as control codes.
    # TODO : Maybe we can treat those control fields as normal modifier?        
    '''

    c_stall    = (code & 0x0000f) >> 0
    c_yield    = (code & 0x00010) >> 4
    c_writebar = (code & 0x000e0) >> 5  # write dependency barrier 
    c_readbar  = (code & 0x00700) >> 8  # read  dependency barrier 
    c_waitbar  = (code & 0x1f800) >> 11 # wait on dependency barrier

    return c_waitbar, c_readbar, c_writebar, c_yield, c_stall    
  
  @staticmethod
  def splitCode2(code):
    ''' Split control codes into parts.

        Mostly same as splitCode, but with yield/stall combined.     
    '''

    c_ystall   = (code & 0x0001f) >> 0
    c_writebar = (code & 0x000e0) >> 5  # write dependency barrier
    c_readbar  = (code & 0x00700) >> 8  # read  dependency barrier
    c_waitbar  = (code & 0x1f800) >> 11 # wait on dependency barrier

    return c_waitbar, c_readbar, c_writebar, c_ystall

  @staticmethod
  def mergeCode(c_waitbar, c_readbar, c_writebar, c_yield, c_stall):
    code = c_waitbar<<11
    code += c_readbar<<8
    code += c_writebar<<5
    code += c_yield<<4
    code += c_stall
    return code

  @staticmethod
  def decode(code):
    c_waitbar, c_readbar, c_writebar, c_yield, c_stall = CuControlCode.splitCode(code)

    s_yield = '-' if c_yield !=0 else 'Y'
    s_writebar = '-' if c_writebar == 7 else '%d'%c_writebar
    s_readbar = '-' if c_readbar == 7 else '%d'%c_readbar
    s_waitbar = ''.join(['-' if (c_waitbar & (2**i)) == 0 else '%d'%i for i in range(6)])
    s_stall = '%02d' % c_stall

    return 'B%s:R%s:W%s:%s:S%s' % (s_waitbar, s_readbar, s_writebar, s_yield, s_stall)

  @staticmethod
  def encode(s):
    if not c_ControlCodesPattern.match(s):
        raise ValueError('Invalid control code strings: %s !!!'%s)

    s_waitbar, s_readbar, s_writebar, s_yield, s_stall = tuple(s.split(':'))

    waitbar_tr = str.maketrans('012345-','1111110')

    c_waitbar = int(s_waitbar[:0:-1].translate(waitbar_tr), 2)
    c_readbar = int(s_readbar[1].replace('-', '7'))
    c_writebar = int(s_writebar[1].replace('-','7'))
    c_yield = int(s_yield!='Y')
    c_stall = int(s_stall[1:])

    code = CuControlCode.mergeCode(c_waitbar, c_readbar, c_writebar, c_yield, c_stall)

    return code

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outfile", help="cubin file")
args = parser.parse_args()
output_filename = args.outfile
EIATTR_TAGS = {
  'EIATTR_REGCOUNT': b'\x04\x2f',        # Format 4, Type 47
  'EIATTR_FRAME_SIZE': b'\x04\x11',      # Format 4, Type 48  
  'EIATTR_MIN_STACK_SIZE': b'\x04\x12',   # Format 4, Type 49
  'EIATTR_CUDA_API_VERSION': b'\x04\x37',
  'EIATTR_SW2861232_WAR': b'\x01\x35',
  'EIATTR_PARAM_CBANK': b'\x04\x0a',
  'EIATTR_CBANK_PARAM_SIZE': b'\x03\x19',
  'EIATTR_KPARAM_INFO': b'\x04\x17',
  'EIATTR_MAXREG_COUNT': b'\x03\x1b',
  'EIATTR_EXIT_INSTR_OFFSETS': b'\x04\x1c',
  'EIATTR_MAX_THREADS': b'\x04\x05',
}
EIATTR_FMT = {
  'EIFMT_NVAL': 1,
  'EIFMT_SVAL': 4,
  'EIFMT_HVAL': 3,
  'EIFMT_XVAL': 6,
}
class NvInfoAttribute:
  def __init__(self, name, value: Union[int, str, None, bytes, tuple[int, ...]]=None):
    self.name = name
    self.value = value
  def to_bytes(self): 
    tag= EIATTR_TAGS[self.name]
    fmt_code = tag[0] & 0x0f #lower 4 bit of first byte is the format
    value_bytes: bytes = b''
    length_bytes = b''
    if fmt_code == EIATTR_FMT['EIFMT_NVAL']:
      pass
    elif fmt_code == EIATTR_FMT['EIFMT_SVAL']:
      if isinstance(self.value, tuple):
        value_bytes = struct.pack('<' + 'I' * len(self.value), *self.value)
      else:
        assert type(self.value) == int
        value_bytes = struct.pack('<I', self.value)
      length = len(value_bytes)
      length_bytes = struct.pack('<H', length)
    elif fmt_code == EIATTR_FMT['EIFMT_HVAL']:
      assert type(self.value) == int
      value_bytes = struct.pack('<H', self.value)
      length_bytes = b''
    else:
      raise ValueError(f"Unsupported format code: {fmt_code}")
    return tag + length_bytes + value_bytes

"""
mem_spec is likely specifying the memory type
global memory: 2
constant memory: 
shared memory:
"""
@dataclasses.dataclass
class Symbol:
  name: str
  name_offset: int
  value: int
  size: int
  bind: str
  _type: str
  mem_spec: int
  visibility: str
  shndx: Union[int, str]
  
  def to_bytes(self):
    bind_val = {'STB_LOCAL': 0, 'STB_GLOBAL': 1, 'STB_WEAK': 2}[self.bind]
    type_val = {'STT_NOTYPE': 0, 'STT_OBJECt': 1, 'STT_FUNC': 2, 'STT_SECTION': 3}[self._type]
    info = (bind_val<<4) | type_val
    vis_val = {'STV_DEFAULT': 0, 'STV_INTERVAL': 1, 'STV_HIDDEN': 2, 'STV_PROTECTED': 3}[self.visibility]
    st_other = (self.mem_spec << 3) + vis_val
    shndx = self.shndx if type(self.shndx) == int else 0
    ret = struct.pack('<I', self.name_offset) + \
      struct.pack('<B', info) + \
      struct.pack('<B', st_other) + \
      struct.pack('<H', shndx) + \
      struct.pack('<Q', self.value) + \
      struct.pack('<Q', self.size)
    return ret

@dataclasses.dataclass
class KernelParameter:
  index: int
  ordinal: int
  offset: int
  size: int
  log_align: int
  space: int
  cbank: int
  param_space: str
  def encode(self) -> tuple[int, int, int]:
    word0 = self.index
    word1 = self.ordinal + (self.offset << 16)
    param_space_value = {
      "cbank": 0x2
    }
    word2 = (param_space_value[self.param_space]<<20) + (self.cbank<<12)
    return (word0, word1, word2)


def alignTo(pos, align):
    ''' Padding current position to given alignment.
    
        Return: tuple(newpos, padsize)
    '''

    if align==0 or align==1:
        return pos, 0

    npos = ((pos + align -1 ) // align) * align
    return npos, npos-pos

def printb(b: bytes, prefix: str='', offset:int=0):
  for i, _b in enumerate(b):
    if i % 16 == 0: print(f"{i+offset:#6x}", end=" ")
    print(f"\033[34m{prefix}{_b:02x}\033[0m", end=" ")
    if (i+1) % 8 == 0: print("  ", end="")
    if (i+1) % 16 == 0: print()
    if i == len(b) -1 and (i+1) % 16 != 0: print()
def peek(i: int, ins: list[str]):
  if i < len(ins):
    line = ins[i]
    if not line.endswith(":"):
      return i
    return peek(i+1, ins)
  return i

def encode_instruction(s: str):
  arch = CuSMVersion(86)
  repo = CuInsAssemblerRepos(InsAsmDict="/home/alvy/gbin/tinygrad2/tinygrad/runtime/support/assembler/InsAsmRepos/DefaultInsAsmRepos.sm_86.txt",
                             arch=arch)
  kasm = CuKernelAssembler()
  kasm.m_InsAsmRepos = repo
  s_list = s.split("\n")
  ccode_list = []
  icode_list = []
  labels = {}
  
  p_textline = re.compile(r'\[([\w:-]+)\](.*)')
  m_label     = re.compile(r'([a-zA-Z0-9._$@#]+?)\s*:\s*(.*)')  # "#" for offset label auto rename
  m_directive = re.compile(r'(\.[a-zA-Z0-9_]+)\s*(.*)')
  label_regex = re.compile(r'([a-zA-Z0-9_\.]+):')
  p_ins_label = re.compile(r'`\(([^\)]+)\)')
  ins_idx = 0
  for line in s_list:
    nline = CuAsmParser.stripComments(line).strip()
    if p_textline.match(nline) is not None:
      ins_idx += 1
      continue
    if (res:=label_regex.match(nline)) is not None:
      label = res.groups()[0]
      assert label is not None
      labels[label] = ins_idx * 16
  ins_idx = 0
  for line in s_list:
    p_textline = re.compile(r'\[([\w:-]+)\](.*)')
    m_label     = re.compile(r'([a-zA-Z0-9._$@#]+?)\s*:\s*(.*)')  # "#" for offset label auto rename
    nline = CuAsmParser.stripComments(line).strip()
    m_directive = re.compile(r'(\.[a-zA-Z0-9_]+)\s*(.*)')
    if len(nline)==0 or (m_label.match(nline) is not None) or (m_directive.match(nline) is not None):
      continue
    res = p_textline.match(nline)
    assert res is not None, f"invalid: {nline}"
    ccode_s = res.groups()[0]
    icode_s = res.groups()[1]
    c_ControlCodesPattern = re.compile(r'B(0|-)(1|-)(2|-)(3|-)(4|-)(5|-):R[0-5\-]:W[0-5\-]:(Y|-):S\d{2}')
    res2 = c_ControlCodesPattern.match(ccode_s)
    assert res2 is not None, f"invalid: {ccode_s=}"
    addr = ins_idx * 16
    c_icode_s = icode_s
    ccode = CuControlCode.encode(ccode_s)

    r2 = p_ins_label.search(icode_s)
    if r2:
      label = r2.groups()[0]
      label_val = labels[label]
      icode_s = p_ins_label.sub('%#x'%label_val, icode_s)
    icode = repo.assemble(addr, icode_s)
    ccode_list.append(ccode)
    icode_list.append(icode)
    ins_idx += 1
  b = arch.mergeCtrlCodes_7x_8x(icode_list, ccode_list)
  return b

def assemble(
  kernel_name: str,
  register_count: int,
  parameters: list[KernelParameter],
  exit_instr_offset: int,
  threadDim: tuple[int, int, int],
  textBytes: bytes,
  param_cbank_val: tuple[int, int],
  param_size: int,
  constants_section_size: int,
  rel_debug_frame_bytes: bytes,
  debug_frame_bytes: bytes,
  text_kernel_info: int,
  ):
  output = io.BytesIO()
  data_file_header = io.BytesIO()
  data_section = io.BytesIO()
  data_section_header = io.BytesIO()
  data_program_header = io.BytesIO()

  CubinELFStructs = ELFStructs(little_endian=True, elfclass=64)
  CubinELFStructs.create_basic_structs()
  CubinELFStructs.create_advanced_structs()
  defaultCubinFileHeader = CubinELFStructs.Elf_Ehdr.parse(bytes.fromhex(''.join([
                          '7f454c460201013307000000000000000200be00650000000000000000000000',
                          'c09000000000000000890000000000004b054b0040003800030040001f000100'])))
  mFileHeader = defaultCubinFileHeader.copy()
  mFileHeader['e_ident']['EI_OSABI']      = 51
  mFileHeader['e_ident']['EI_ABIVERSION'] = 7
  mFileHeader['e_type']                   = 'ET_EXEC'
  mFileHeader['e_machine']                = 'EM_CUDA'
  mFileHeader['e_version']                = 129
  mFileHeader['e_entry']                  = 0
  mFileHeader['e_phoff']                  = 0 #2560
  mFileHeader['e_shoff']                  = 0 #1792
  mFileHeader['e_flags']                  = 5637462
  mFileHeader['e_ehsize']                 = 64
  mFileHeader['e_phentsize']              = 56
  mFileHeader['e_phnum']                  = 3
  mFileHeader['e_shentsize']              = 64
  mFileHeader['e_shnum']                  = 12
  mFileHeader['e_shstrndx']               = 1

  shtnull = CuAsmSection('', '0', ['SHT_NULL'])
  shtnull.header['name'] = 0x0
  shtnull.header['type'] = "SHT_NULL"
  shtnull.header['flags'] = 0
  shtnull.header['addr'] = 0
  shtnull.header['offset'] = 0
  shtnull.header['size'] = 0
  shtnull.header['link'] = 0
  shtnull.header['info'] = 0
  shtnull.header['entsize'] = 0

  section_names = [
    "",           # First entry must be empty string (null byte)
    ".shstrtab",
    ".strtab",
    ".symtab",
    ".symtab_shndx",
    ".nv.info",
    f".text.{kernel_name}",
    f".nv.info.{kernel_name}",
    f".nv.shared.{kernel_name}",
    f".nv.constant0.{kernel_name}",
    f".rel.nv.constant0.{kernel_name}",
    ".debug_frame",
    ".rel.debug_frame",
    ".rela.debug_frame",
    ".nv.callgraph",
    ".nv.prototype",
    ".nv.rel.action"
  ]
  shstrtab_offset_map = {}
  shstrtab_current_offset = 0
  for name in section_names:
    shstrtab_offset_map[name] = shstrtab_current_offset
    shstrtab_current_offset += len(name.encode()) + 1
  shstrtab = CuAsmSection('shstrtab', '0', ['SHT_STRTAB'])
  shstrtab.header['name'] = shstrtab_offset_map['.shstrtab']
  shstrtab.header['type'] = "SHT_STRTAB"
  shstrtab.header['flags'] = 0          # val=0
  shstrtab.header['addr'] = 0           # val=0
  shstrtab.offset = 0
  shstrtab.header['size'] = sum(len(s)+1 for s in section_names)
  shstrtab.header['link'] = 0           # val=0
  shstrtab.header['info'] = 0           # val=0
  shstrtab.header['entsize'] = 0        # val=0

  shstrtab.emitAlign(1)
  shstrtab.emitAlign(1)
  for name in section_names:
    shstrtab.emitBytes(name.encode() + b'\x00')

  strtab_names = [
    "",           
    ".shstrtab",
    ".strtab",
    ".symtab",
    ".symtab_shndx",
    ".nv.info",
    f".text.{kernel_name}",
    f".nv.info.{kernel_name}",
    f".nv.shared.{kernel_name}",
    f".rel.nv.constant0.{kernel_name}",
    f".nv.constant0.{kernel_name}",
    ".debug_frame",
    ".rel.debug_frame",
    ".rela.debug_frame",
    ".nv.callgraph",
    ".nv.prototype",
    ".nv.rel.action",
    f"{kernel_name}"
  ]
  strtab_offset_map = {}
  current_offset = 0
  for name in strtab_names:
    strtab_offset_map[name] = current_offset
    current_offset += len(name.encode()) + 1

  strtab = CuAsmSection('strtab', '0', ['SHT_STRTAB'])
  strtab.header['name'] = shstrtab_offset_map['.strtab']
  strtab.header['type'] = "SHT_STRTAB"
  strtab.header['flags'] = 0           # val=0
  strtab.header['addr'] = 0            # val=0
  strtab.offset = 0
  strtab.header['size'] = sum(len(s) + 1 for s in strtab_names)
  strtab.header['link'] = 0            # val=0
  strtab.header['info'] = 0            # val=0
  strtab.header['entsize'] = 0         # val=0

  strtab.emitAlign(1)
  for s in strtab_names:
    strtab.emitBytes(s.encode() + b'\x00')

  text_size = len(textBytes)
  symbols = {
    '': (0, Symbol('', 0, 0, 0, 'STB_LOCAL', 'STT_NOTYPE', 0, 'STV_DEFAULT', 'SHN_UNDEF')),
    f'.text.{kernel_name}': (1, Symbol(f'.text.{kernel_name}', strtab_offset_map[f'.text.{kernel_name}'], 0, 0, 'STB_LOCAL', 'STT_SECTION', 0, 'STV_DEFAULT', 11)),
    f".nv.constant0.{kernel_name}": (2, Symbol(f".nv.constant0.{kernel_name}", strtab_offset_map[f'.nv.constant0.{kernel_name}'], 0, 0, "STB_LOCAL", "STT_SECTION", 0, 'STV_DEFAULT', 10)),
    f".debug_frame": (3, Symbol(".debug_frame", strtab_offset_map['.debug_frame'],  0, 0, "STB_LOCAL", "STT_SECTION", 0, 'STV_DEFAULT', 4)),
    f".nv.callgraph": (4, Symbol(".nv.callgraph", strtab_offset_map['.nv.callgraph'], 0, 0, "STB_LOCAL", "STT_SECTION", 0, 'STV_DEFAULT', 7)),
    f".nv.rel.action": (5, Symbol(".nv.rel.action", strtab_offset_map['.nv.rel.action'], 0, 0, "STB_LOCAL", "STT_SECTION", 0, 'STV_DEFAULT', 8)),
    f"{kernel_name}": (6, Symbol(f"{kernel_name}", strtab_offset_map[f'{kernel_name}'], 0, text_size, "STB_GLOBAL", "STT_FUNC", 2, 'STV_DEFAULT', 11))
  }
  first_global_symbol = symbols[kernel_name][0]
  index_of_asscoiated_strtab = symbols[f".nv.constant0.{kernel_name}"][0]
  symtab = CuAsmSection('symtab', '0', ['SHT_SYMTAB'])
  symtab.header['name'] = shstrtab_offset_map['.symtab']
  symtab.header['type'] = "SHT_SYMTAB"
  symtab.header['flags'] = 0          # val=0
  symtab.header['addr'] = 0           # val=0
  symtab.offset = 0
  symtab.header['link'] = index_of_asscoiated_strtab
  symtab.header['info'] = first_global_symbol
  symtab.header['entsize'] = 24       # val=24 (size per symbol entry)
  symtab.header['size'] = len(symbols) * symtab.header['entsize']

  symtab.emitAlign(8)
  for idx, symbol in symbols.values():
    symtab.emitBytes(symbol.to_bytes())

  debug_frame = CuAsmSection('debug_frame', '""', ['@progbits'])
  debug_frame.header['name'] = shstrtab_offset_map['.debug_frame']
  debug_frame.header['type'] = "SHT_PROGBITS"
  debug_frame.header['flags'] = 0        # val=0
  debug_frame.header['addr'] = 0         # val=0
  debug_frame.offset = 0
  debug_frame.header['size'] = 0x70      # val=112 (section size)
  debug_frame.header['link'] = 0         # val=0
  debug_frame.header['info'] = 0         # val=0
  debug_frame.header['entsize'] = 0      # val=0

  debug_frame.emitAlign(1)
  debug_frame.emitBytes(debug_frame_bytes)
  nv_info = CuAsmSection('.nv.info', '""', ['@"SHT_CUDA_INFO"'])
  nv_info.header['name'] = shstrtab_offset_map['.nv.info']
  nv_info.header['type'] = 0x70000000      # val=1879048192 (SHT_CUDA_INFO)
  nv_info.header['flags'] = 0              # val=0
  nv_info.header['addr'] = 0               # val=0
  nv_info.offset = 0
  nv_info.header['size'] = 0x24            # val=36 (section size)
  nv_info.header['link'] = 3               # val=3 (associated section index)
  nv_info.header['info'] = 0               # val=0
  nv_info.header['entsize'] = 0            # val=0

  REGISTER_COUNT = register_count
  kernel_symbol_idx = symbols[kernel_name][0]
  reg_count = NvInfoAttribute('EIATTR_REGCOUNT', (kernel_symbol_idx, REGISTER_COUNT))
  nv_info.emitAlign(4)
  nv_info.emitBytes(reg_count.to_bytes())

  frame_size = NvInfoAttribute('EIATTR_FRAME_SIZE', (kernel_symbol_idx, 0))
  nv_info.emitAlign(4)
  nv_info.emitBytes(frame_size.to_bytes())

  min_stack_size = NvInfoAttribute('EIATTR_MIN_STACK_SIZE', (kernel_symbol_idx, 0))
  nv_info.emitAlign(4)
  nv_info.emitBytes(min_stack_size.to_bytes())

  nv_info_kernel = CuAsmSection(f'.nv.info.{kernel_name}', '""', ['@"SHT_CUDA_INFO"', '@""'])
  nv_info_kernel.header['name'] = shstrtab_offset_map[f'.nv.info.{kernel_name}']
  nv_info_kernel.header['type'] = 0x70000000  # val=1879048192 (SHT_CUDA_INFO)
  nv_info_kernel.header['flags'] = 0x40       # val=64 (SHF_ALLOC)
  nv_info_kernel.header['addr'] = 0           # val=0
  nv_info_kernel.offset = 0
  nv_info_kernel.header['size'] = 0x68        # val=104 (section size)
  nv_info_kernel.header['link'] = 3           # val=3 (associated section index)
  nv_info_kernel.header['info'] = 0xb         # val=11 (additional info)
  nv_info_kernel.header['entsize'] = 0        # val=0

  CUDA_API_VERSION = 0x81
  api_version = NvInfoAttribute('EIATTR_CUDA_API_VERSION', CUDA_API_VERSION)
  nv_info_kernel.emitAlign(4)
  nv_info_kernel.emitBytes(api_version.to_bytes())

  workaround = NvInfoAttribute('EIATTR_SW2861232_WAR')
  nv_info_kernel.emitBytes(workaround.to_bytes())

  nv_info_kernel.emitAlign(4)
  param_cbank = NvInfoAttribute('EIATTR_PARAM_CBANK', param_cbank_val)
  nv_info_kernel.emitBytes(param_cbank.to_bytes())

  nv_info_kernel.emitAlign(4)
  cbank_param_size = NvInfoAttribute('EIATTR_CBANK_PARAM_SIZE', param_size)
  nv_info_kernel.emitBytes(cbank_param_size.to_bytes())
  for param in parameters:
    encoded_param = param.encode() 
    kparam_info = NvInfoAttribute('EIATTR_KPARAM_INFO', encoded_param)
    nv_info_kernel.emitAlign(4)
    nv_info_kernel.emitBytes(kparam_info.to_bytes())

  MAX_REGISTER_COUNT = 0xff
  maxreg_count = NvInfoAttribute('EIATTR_MAXREG_COUNT', MAX_REGISTER_COUNT)
  nv_info_kernel.emitAlign(4)
  nv_info_kernel.emitBytes(maxreg_count.to_bytes())

  EXIT_INSTR_OFFSET = exit_instr_offset
  exit_instr_offsets = NvInfoAttribute('EIATTR_EXIT_INSTR_OFFSETS', EXIT_INSTR_OFFSET)
  nv_info_kernel.emitAlign(4)
  nv_info_kernel.emitBytes(exit_instr_offsets.to_bytes())

  THREAD_BLOCK_DIMENSIONS = threadDim
  max_threads = NvInfoAttribute('EIATTR_MAX_THREADS', THREAD_BLOCK_DIMENSIONS)
  nv_info_kernel.emitAlign(4)
  nv_info_kernel.emitBytes(max_threads.to_bytes())

  nv_callgraph = CuAsmSection('.nv.callgraph', '""', ['@"SHT_CUDA_CALLGRAPH"', '@""'])
  nv_callgraph.header['name'] = shstrtab_offset_map['.nv.callgraph']
  nv_callgraph.header['type'] = 0x70000001   # val=1879048193 (SHT_CUDA_CALLGRAPH)
  nv_callgraph.header['flags'] = 0           # val=0
  nv_callgraph.header['addr'] = 0            # val=0
  nv_callgraph.offset = 0
  nv_callgraph.header['size'] = 0x20         # val=32 (section size)
  nv_callgraph.header['link'] = 3            # val=3 (associated section index)
  nv_callgraph.header['info'] = 0            # val=0
  nv_callgraph.header['entsize'] = 8         # val=8 (entry size)

  nv_callgraph.emitAlign(4)
  nv_callgraph.emitBytes(
b'\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00\xfe\xff\xff\xff' + \
b'\x00\x00\x00\x00\xfd\xff\xff\xff\x00\x00\x00\x00\xfc\xff\xff\xff'
    )

  nv_rel_action = CuAsmSection('.nv.rel.action', '""', ['@"SHT_CUDA_RELOCINFO"', '@""'])
  nv_rel_action.header['name'] = shstrtab_offset_map['.nv.rel.action']
  nv_rel_action.header['type'] = 0x7000000b   # val=1879048203 (SHT_CUDA_RELOCINFO)
  nv_rel_action.header['flags'] = 0           # val=0
  nv_rel_action.header['addr'] = 0            # val=0
  nv_rel_action.offset = 0
  nv_rel_action.header['size'] = 0x10         # val=16 (section size)
  nv_rel_action.header['link'] = 0            # val=0
  nv_rel_action.header['info'] = 0            # val=0
  nv_rel_action.header['entsize'] = 8         # val=8 (entry size)
  
  nv_rel_action.emitAlign(8)
  nv_rel_action.emitAlign(8)
  nv_rel_action.emitBytes(b'\x73\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x25\x00\x05\x36')

  rel_debug_frame = CuAsmSection('.rel.debug_frame', '64', ['SHT_REL'])
  rel_debug_frame.header['name'] = shstrtab_offset_map['.rel.debug_frame']
  rel_debug_frame.header['type'] = "SHT_REL" # Standard relocation type
  rel_debug_frame.header['flags'] = 0x40     # val=64 (SHF_ALLOC)
  rel_debug_frame.header['addr'] = 0          # val=0
  rel_debug_frame.offset = 0
  rel_debug_frame.header['size'] = 0x10       # val=16 (section size)
  rel_debug_frame.header['link'] = 3          # val=3 (symbol table index)
  rel_debug_frame.header['info'] = 0x4        # val=4 (target section index)
  rel_debug_frame.header['entsize'] = 16      # val=16 (relocation entry size)
  rel_debug_frame.emitAlign(8)
  rel_debug_frame.emitBytes(rel_debug_frame_bytes)

  nv_constant0_kernel = CuAsmSection(f'.nv.constant0.{kernel_name}', '"a"', ['@progbits', '@""'])
  nv_constant0_kernel.header['name'] = shstrtab_offset_map[f'.nv.constant0.{kernel_name}']
  nv_constant0_kernel.header['type'] = "SHT_PROGBITS"
  nv_constant0_kernel.header['flags'] = 0x42      # val=66 (SHF_ALLOC | SHF_WRITE)
  nv_constant0_kernel.header['addr'] = 0          # val=0
  nv_constant0_kernel.offset = 0
  nv_constant0_kernel.header['size'] = constants_section_size
  nv_constant0_kernel.header['link'] = 0          # val=0
  nv_constant0_kernel.header['info'] = 0xb        # val=11 (additional info)
  nv_constant0_kernel.header['entsize'] = 0       # val=0
  nv_constant0_kernel.emitAlign(4)
  nv_constant0_kernel.emitBytes(b'\x00' * nv_constant0_kernel.header['size'])
  text_kernel_data = textBytes
  text_kernel = CuAsmSection(f'.text.{kernel_name}', '"ax"', ['@progbits'])
  text_kernel.header['name'] = shstrtab_offset_map[f'.text.{kernel_name}']
  text_kernel.header['type'] = "SHT_PROGBITS"
  text_kernel.header['flags'] = 0x6         # val=6 (SHF_ALLOC | SHF_EXECINSTR)
  text_kernel.header['addr'] = 0            # val=0
  #text_kernel.header['offset'] = 0x580      # val=1408 (file offset)
  text_kernel.header['size'] = len(text_kernel_data)
  text_kernel.offset = 0
  text_kernel.header['link'] = 3            # val=3 (symbol table index)
  text_kernel.header['info'] = text_kernel_info
  text_kernel.header['entsize'] = 0         # val=0
  text_kernel.emitAlign(128)

  text_kernel.emitBytes(text_kernel_data)

  sections: dict[str, CuAsmSection] = {
    '.sht_null': shtnull,
    '.shstrtab': shstrtab,
    '.strtab': strtab,
    '.symtab': symtab,
    '.debug_frame': debug_frame,
    '.nv.info': nv_info,
    f'.nv.info.{kernel_name}': nv_info_kernel,
    '.nv.callgraph': nv_callgraph,
    '.nv.rel.action': nv_rel_action,
    '.rel.debug_frame': rel_debug_frame,
    f'.nv.constant0.{kernel_name}': nv_constant0_kernel,
    f'.text.{kernel_name}': text_kernel,
  }
  

  elfheadersize = CubinELFStructs.Elf_Ehdr.sizeof()
  file_offset = elfheadersize
  mem_offset = elfheadersize #64
  for sec in sections.values():
    if sec.name == '': continue
    align = sec.addralign
    assert align
    aligned_offset, fpadsize = alignTo(file_offset, align) 
    gap = aligned_offset - file_offset
    if gap > 0:
      data_section.write(b'\x00' * gap)
      file_offset += gap
    sec.offset = file_offset
    if sec.header['type'] != 'SHT_NOBITS':
      data = sec.getData()
      data_section.write(data)
      file_offset += len(data)

  for section in sections.values():
    section.updateHeader()
    data_section_header.write(section.buildHeader())
  
  segment_pt_phdr = CuAsmSegment("PT_PHDR", 5)
  segment_pt_phdr.header['offset'] = elfheadersize + len(data_section.getbuffer()) + \
    len(data_section_header.getbuffer()) #0xa00
  segment_pt_phdr.header['vaddr'] = 0x0
  segment_pt_phdr.header['paddr'] = 0x0
  segment_pt_phdr.header['filesz'] = mFileHeader['e_phentsize'] * mFileHeader['e_phnum']
  segment_pt_phdr.header['memsz'] = mFileHeader['e_phentsize'] * mFileHeader['e_phnum']
  segment_pt_phdr.header['align'] = 8

  segment_pt_phload = CuAsmSegment("PT_LOAD", 5)
  segment_pt_phload.header['offset'] = sections[f'.nv.constant0.{kernel_name}'].offset
  segment_pt_phload.header['vaddr'] = 0x0
  segment_pt_phload.header['paddr'] = 0x0
  segment_pt_phload.header['filesz'] = sections[f'.text.{kernel_name}'].offset + \
    sections[f'.text.{kernel_name}'].header['size'] - sections[f'.nv.constant0.{kernel_name}'].offset
  segment_pt_phload.header['memsz'] = segment_pt_phload.header['filesz']
  segment_pt_phload.header['align'] = 8

  segment_pt_phload2 = CuAsmSegment("PT_LOAD", 5)
  segment_pt_phload2.header['offset'] = elfheadersize + len(data_section.getbuffer()) + \
    len(data_section_header.getbuffer())
  segment_pt_phload2.header['vaddr'] = 0x0
  segment_pt_phload2.header['paddr'] = 0x0
  segment_pt_phload2.header['filesz'] = mFileHeader['e_phentsize'] * mFileHeader['e_phnum']
  segment_pt_phload2.header['memsz'] = mFileHeader['e_phentsize'] * mFileHeader['e_phnum']
  segment_pt_phload2.header['align'] = 8
  segments = [segment_pt_phdr, segment_pt_phload, segment_pt_phload2]

  for seg in segments:
    seg.updateHeader()
    b = seg.build()
    data_program_header.write(b)
  mFileHeader['e_shoff'] = elfheadersize + len(data_section.getbuffer())
  mFileHeader['e_phoff'] = elfheadersize + len(data_section.getbuffer()) + \
    len(data_section_header.getbuffer())
  
  data_file_header.write(CubinELFStructs.Elf_Ehdr.build(mFileHeader))
  output.write(data_file_header.getbuffer())
  output.write(data_section.getbuffer())
  output.write(data_section_header.getbuffer())
  output.write(data_program_header.getbuffer())
  return output


with open(output_filename, 'wb') as f:
  textBytes = \
  textBytes = encode_instruction(s="""
  kernel:
  .text.kernel:
      [B------:R-:W-:-:S02]         /*0000*/                   MOV R1, c[0x0][0x28] ;
      [B------:R-:W0:-:S01]         /*0010*/                   S2R R7, SR_CTAID.Y ;
      [B------:R-:W-:-:S01]         /*0020*/                   MOV R42, c[0x0][0x170] ;
      [B------:R-:W-:-:S01]         /*0030*/                   ULDC.64 UR4, c[0x0][0x118] ;
      [B------:R-:W-:-:S01]         /*0040*/                   MOV R43, c[0x0][0x174] ;
      [B------:R-:W1:-:S01]         /*0050*/                   S2R R38, SR_CTAID.X ;
      [B------:R-:W-:-:S02]         /*0060*/                   MOV R40, c[0x0][0x168] ;
      [B------:R-:W-:-:S02]         /*0070*/                   MOV R41, c[0x0][0x16c] ;
      [B------:R-:W-:-:S02]         /*0080*/                   MOV R9, RZ ;
      [B------:R-:W-:Y:S02]         /*0090*/                   MOV R44, RZ ;
      [B0-----:R-:W-:-:S02]         /*00a0*/                   SHF.L.U32 R7, R7, 0x9, RZ ;
  .L_x_0:
      [B------:R-:W-:-:S02]         /*00b0*/                   MOV R2, R40 ;
      [B------:R-:W-:-:S02]         /*00c0*/                   MOV R3, R41 ;
      [B------:R-:W-:-:S02]         /*00d0*/                   MOV R4, R42 ;
      [B------:R-:W-:-:S01]         /*00e0*/                   MOV R5, R43 ;
      [B------:R-:W-:Y:S04]         /*00f0*/                   IMAD.WIDE R2, R7, 0x4, R2 ;
      [B-1----:R-:W-:-:S01]         /*0100*/                   IMAD.WIDE R4, R38, 0x4, R4 ;
      [B------:R-:W2:-:S04]         /*0110*/                   LDG.E R6, desc[UR4][R2.64] ;
      [B------:R-:W2:-:S04]         /*0120*/                   LDG.E R0, desc[UR4][R4.64] ;
      [B------:R-:W3:-:S04]         /*0130*/                   LDG.E R8, desc[UR4][R2.64+0x4] ;
      [B------:R-:W3:-:S04]         /*0140*/                   LDG.E R11, desc[UR4][R4.64+0x800] ;
      [B------:R-:W4:-:S04]         /*0150*/                   LDG.E R13, desc[UR4][R4.64+0x1000] ;
      [B------:R-:W4:-:S04]         /*0160*/                   LDG.E R10, desc[UR4][R2.64+0x8] ;
      [B------:R-:W5:-:S04]         /*0170*/                   LDG.E R15, desc[UR4][R4.64+0x1800] ;
      [B------:R-:W5:-:S04]         /*0180*/                   LDG.E R12, desc[UR4][R2.64+0xc] ;
      [B------:R-:W5:-:S04]         /*0190*/                   LDG.E R17, desc[UR4][R4.64+0x2000] ;
      [B------:R-:W5:-:S04]         /*01a0*/                   LDG.E R14, desc[UR4][R2.64+0x10] ;
      [B------:R-:W5:-:S04]         /*01b0*/                   LDG.E R19, desc[UR4][R4.64+0x2800] ;
      [B------:R-:W5:-:S04]         /*01c0*/                   LDG.E R16, desc[UR4][R2.64+0x14] ;
      [B------:R-:W5:-:S04]         /*01d0*/                   LDG.E R21, desc[UR4][R4.64+0x3000] ;
      [B------:R-:W5:-:S04]         /*01e0*/                   LDG.E R18, desc[UR4][R2.64+0x18] ;
      [B------:R-:W5:-:S04]         /*01f0*/                   LDG.E R23, desc[UR4][R4.64+0x3800] ;
      [B------:R-:W5:-:S04]         /*0200*/                   LDG.E R20, desc[UR4][R2.64+0x1c] ;
      [B------:R-:W5:-:S04]         /*0210*/                   LDG.E R25, desc[UR4][R4.64+0x4000] ;
      [B------:R-:W5:-:S04]         /*0220*/                   LDG.E R22, desc[UR4][R2.64+0x20] ;
      [B------:R-:W5:-:S04]         /*0230*/                   LDG.E R27, desc[UR4][R4.64+0x4800] ;
      [B------:R-:W5:-:S04]         /*0240*/                   LDG.E R24, desc[UR4][R2.64+0x24] ;
      [B------:R-:W5:-:S04]         /*0250*/                   LDG.E R29, desc[UR4][R4.64+0x5000] ;
      [B------:R-:W5:-:S04]         /*0260*/                   LDG.E R26, desc[UR4][R2.64+0x28] ;
      [B------:R-:W5:-:S04]         /*0270*/                   LDG.E R31, desc[UR4][R4.64+0x5800] ;
      [B------:R-:W5:-:S04]         /*0280*/                   LDG.E R28, desc[UR4][R2.64+0x2c] ;
      [B------:R-:W5:-:S04]         /*0290*/                   LDG.E R33, desc[UR4][R4.64+0x6000] ;
      [B------:R-:W5:-:S04]         /*02a0*/                   LDG.E R30, desc[UR4][R2.64+0x30] ;
      [B------:R-:W5:-:S04]         /*02b0*/                   LDG.E R35, desc[UR4][R4.64+0x6800] ;
      [B------:R-:W5:-:S04]         /*02c0*/                   LDG.E R32, desc[UR4][R2.64+0x34] ;
      [B------:R-:W5:-:S04]         /*02d0*/                   LDG.E R37, desc[UR4][R4.64+0x7000] ;
      [B------:R-:W5:-:S04]         /*02e0*/                   LDG.E R34, desc[UR4][R2.64+0x38] ;
      [B------:R-:W5:-:S04]         /*02f0*/                   LDG.E R39, desc[UR4][R4.64+0x7800] ;
      [B------:R-:W5:-:S01]         /*0300*/                   LDG.E R36, desc[UR4][R2.64+0x3c] ;
      [B------:R-:W-:Y:S02]         /*0310*/                   IADD3 R44, R44, 0x10, RZ ;
      [B------:R-:W-:-:S02]         /*0320*/                   IADD3 R42, P1, R42, 0x8000, RZ ;
      [B------:R-:W-:-:S02]         /*0330*/                   ISETP.NE.AND P0, PT, R44, 0x200, PT ;
      [B------:R-:W-:-:S02]         /*0340*/                   IADD3 R40, P2, R40, 0x40, RZ ;
      [B------:R-:W-:-:S02]         /*0350*/                   IADD3.X R43, RZ, R43, RZ, P1, !PT ;
      [B------:R-:W-:-:S01]         /*0360*/                   IADD3.X R41, RZ, R41, RZ, P2, !PT ;
      [B--2---:R-:W-:Y:S04]         /*0370*/                   FFMA R0, R0, R6, R9 ;
      [B---3--:R-:W-:Y:S04]         /*0380*/                   FFMA R0, R11, R8, R0 ;
      [B----4-:R-:W-:Y:S04]         /*0390*/                   FFMA R0, R13, R10, R0 ;
      [B-----5:R-:W-:Y:S04]         /*03a0*/                   FFMA R0, R15, R12, R0 ;
      [B------:R-:W-:Y:S04]         /*03b0*/                   FFMA R0, R17, R14, R0 ;
      [B------:R-:W-:Y:S04]         /*03c0*/                   FFMA R0, R19, R16, R0 ;
      [B------:R-:W-:Y:S04]         /*03d0*/                   FFMA R0, R21, R18, R0 ;
      [B------:R-:W-:Y:S04]         /*03e0*/                   FFMA R0, R23, R20, R0 ;
      [B------:R-:W-:Y:S04]         /*03f0*/                   FFMA R0, R25, R22, R0 ;
      [B------:R-:W-:Y:S04]         /*0400*/                   FFMA R0, R27, R24, R0 ;
      [B------:R-:W-:Y:S04]         /*0410*/                   FFMA R0, R29, R26, R0 ;
      [B------:R-:W-:Y:S04]         /*0420*/                   FFMA R0, R31, R28, R0 ;
      [B------:R-:W-:Y:S04]         /*0430*/                   FFMA R0, R33, R30, R0 ;
      [B------:R-:W-:Y:S04]         /*0440*/                   FFMA R0, R35, R32, R0 ;
      [B------:R-:W-:Y:S04]         /*0450*/                   FFMA R0, R37, R34, R0 ;
      [B------:R-:W-:-:S01]         /*0460*/                   FFMA R9, R39, R36, R0 ;
      [B------:R-:W-:-:S05]         /*0470*/               @P0 BRA `(.L_x_0) ;
      [B------:R-:W-:-:S02]         /*0480*/                   IADD3 R2, R7, R38, RZ ;
      [B------:R-:W-:Y:S05]         /*0490*/                   MOV R3, 0x4 ;
      [B------:R-:W-:Y:S05]         /*04a0*/                   IMAD.WIDE R2, R2, R3, c[0x0][0x160] ;
      [B------:R-:W-:-:S01]         /*04b0*/                   STG.E desc[UR4][R2.64], R9 ;
      [B------:R-:W-:-:S05]         /*04c0*/                   EXIT ;
  .L_x_1:
      [B------:R-:W-:Y:S00]         /*04d0*/                   BRA `(.L_x_1);
      [B------:R-:W-:Y:S00]         /*04e0*/                   NOP;
      [B------:R-:W-:Y:S00]         /*04f0*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0500*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0510*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0520*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0530*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0540*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0550*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0560*/                   NOP;
      [B------:R-:W-:Y:S00]         /*0570*/                   NOP;
  .L_x_2:
  """)

  # elf dump section i=9
  rel_debug_frame_bytes = b"\x44\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x06\x00\x00\x00"

  # elf dump section i=4
  debug_frame_bytes = \
b'\xff\xff\xff\xff\x24\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff' + \
b'\xff\xff\xff\xff\x03\x00\x04\x7c\xff\xff\xff\xff\x0f\x0c\x81\x80' + \
b'\x80\x28\x00\x08\xff\x81\x80\x28\x08\x81\x80\x80\x28\x00\x00\x00' + \
b'\xff\xff\xff\xff\x34\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' + \
b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x02\x00\x00' + \
b'\x00\x00\x00\x00\x04\x04\x00\x00\x00\x04\x5c\x00\x00\x00\x0c\x81' + \
b'\x80\x80\x28\x00\x04\xfc\xff\xff\x3f\x00\x00\x00\x00\x00\x00\x00'

  output = assemble(
    # cuobjdump
    kernel_name='kernel',
    register_count=47,
    param_cbank_val=(2, 0x180160),
    param_size=0x18,
    parameters=[
      KernelParameter(index=0, ordinal=2, offset=0x10, 
                      size=8, log_align=0, space=0,
                      cbank=0x1f, param_space='cbank'),

      KernelParameter(index=0, ordinal=1, offset=0x8, 
                      size=8, log_align=0, space=0,
                      cbank=0x1f, param_space='cbank'),

      KernelParameter(index=0, ordinal=0, offset=0, 
                      size=8, log_align=0, space=0,
                      cbank=0x1f, param_space='cbank')
    ],
    exit_instr_offset=0x4c0,
    threadDim=(1,1,1),

    # elf dump section header 10 'sh_size'
    constants_section_size=0x178,

    # elf dump section header 11 'sh_info'
    text_kernel_info=0x2f000006,

    textBytes=textBytes,
    rel_debug_frame_bytes=rel_debug_frame_bytes,
    debug_frame_bytes=debug_frame_bytes,
    )
  f.write(output.getbuffer())

