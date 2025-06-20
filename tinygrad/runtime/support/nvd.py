# import re
import os, mmap, array
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.helpers import getenv, round_up, DEBUG, to_mv
from tinygrad.runtime.autogen import libc

# MAP_LOCKED = 0x2000
# def alloc_sysmem(size, contigous=False, data:bytes=None):
#   if getattr(alloc_sysmem, "pagemap", None) is None: alloc_sysmem.pagemap = FileIOInterface("/proc/self/pagemap", os.O_RDONLY)
  
#   size = round_up(size, mmap.PAGESIZE)

#   assert not contigous or size <= (2 << 20), "Contiguous allocation is only supported for sizes <= 2 MiB"
#   flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | MAP_LOCKED

#   if contigous and size > 0x1000: flags |= libc.MAP_HUGETLB
#   va = FileIOInterface.anon_mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, flags, 0)
#   assert va != 0xffffffffffffffff, f"Failed to mmap {size} bytes at {hex(va)}"

#   # Read pagemap to get the physical address of each page. The pages are locked.
#   alloc_sysmem.pagemap.seek(va // mmap.PAGESIZE * 8)
#   if data is not None:
#     assert len(data) <= size, f"Data size {len(data)} exceeds allocated size {size}"
#     to_mv(va, len(data))[:] = data

#   return va, [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', alloc_sysmem.pagemap.read(size//mmap.PAGESIZE*8, binary=True))]

# def _download(self, file) -> str:
#   url = f"https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/e8113f665d936d9f30a6d508f3bacd1e148539be/{file}"
#   return fetch(url, subdir="defines").read_text()

# def parse_header(self, file) -> str:
#   if file in self.scanned_files: return
#   self.scanned_files.add(file)

#   txt = self._download(file)

#   PARAM = re.compile(r'#define\s+(\w+)\s*\(\s*(\w+)\s*\)\s*(.+)')
#   CONST = re.compile(r'#define\s+(\w+)\s+([0-9A-Fa-fx]+)')
#   BITFLD = re.compile(r'#define\s+(\w+)\s+(\d+):(\d+)')

#   defs = {}

#   for raw in txt.splitlines():
#     if raw.startswith("#define "):
#     if (m := BITFLD.match(raw)):
#       name, hi, lo = m.groups()
#       self.defs[f"{name}_SHIFT"] = int(lo)
#       self.defs[f"{name}_MASK"]  = ((1 << (int(hi) - int(lo) + 1)) - 1) << int(lo)
#     elif (m := PARAM.match(raw)):
#       name, param, expr = m.groups()
#       expr = expr.strip().rstrip('\\').split('/*')[0].rstrip()
#       assert self.__dict__.get(name) is None, f"Duplicate definition for {name} in {file}"
#       self.__dict__[name] = eval(f"lambda {param}: {expr}")
#     elif (m := CONST.match(raw)):
#       name, value = m.groups()
#       assert self.__dict__.get(name) in [None, int(value, 0)], f"Duplicate definition for {name} in {file}"
#       self.__dict__[name] = int(value, 0)

# def build_reg(self, name, _val=0, **kwargs):
#   for k, v in kwargs.items(): _val |= (v << self.defs[f"{name.upper()}_{k.upper()}_SHIFT"]) & self.defs[f"{name.upper()}_{k.upper()}_MASK"]
#   return _val

# def read_reg(self, name, val, *fields):
#   if len(fields) > 1: return tuple(self.read_num(name, val, f) for f in fields)
#   return (val & self.defs[f"{name}_{fields[0].upper()}_MASK"]) >> self.defs[f"{name}_{fields[0].upper()}_SHIFT"]