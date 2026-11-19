from __future__ import annotations
import dataclasses, os, pathlib, re, subprocess, tempfile, hashlib
from typing import Any
from tinygrad.device import Compiled, Program, Allocator, BufferStorage, BufferSpec, Buffer
from tinygrad.renderer import Compiler, TinyELF
from tinygrad.renderer.cstyle import CUDARenderer, CStyleLanguage
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import round_up, diskcache_get, diskcache_put
from tinygrad.runtime.support.mt.mtdev import MTDev
from tinygrad.runtime.support.mt.launch import MTLaunch, parse_mt_bundle, compress_mt_bundle, decompress_mt_bundle
from tinygrad.runtime.support.mt.compiler import MTCompiler
from tinygrad.runtime.support.mt.elf import MTKernelImage

MUSA_SOURCES:dict[str,str] = {}


def generate_mt_launch(source:str, name:str, args:list[str], global_size:tuple[int,int,int],
                       local_size:tuple[int,int,int]) -> dict[str,Any]:
  td = tempfile.mkdtemp(prefix='mt-launch-')
  src_path = pathlib.Path(td) / 'kernel.mu'
  arg_decls = 'void *out = reinterpret_cast<void *>(uintptr_t(0xe1c2003000ULL)); uint64_t scalar = 1;'
  arg_names = ','.join('&out' if x == 'out' else '&scalar' for x in args)
  main = (f'#include <musa.h>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <unistd.h>\nint main() {{\n'
          f'  musaFunction_t fn = nullptr;\n  musaGetFuncBySymbol(&fn, (const void *)&{name});\n'
          f'  {arg_decls}\n  void *args[] = {{{arg_names}}};\n'
          f'  musaLaunchKernel((const void *)&{name}, dim3{global_size}, dim3{local_size}, args, 0, nullptr);\n'
          f'  musaDeviceSynchronize();\n'
          f'  char done[512];\n  snprintf(done, sizeof(done), "%s/real-capture-done", getenv("MT_CAPTURE_DIR"));\n'
          f'  while (access(done, F_OK) != 0) usleep(1000);\n  _Exit(0);\n}}\n')
  src_path.write_text(source + '\n' + main)
  musa = os.environ.get('MUSA_PATH', '/tmp/mt-research/toolkit/musa_toolkits_4.3.0')
  driver = '/tmp/mt-research/musa-driver-3.3.0/usr/lib/x86_64-linux-gnu'
  host = str(pathlib.Path(td) / 'host')
  include = pathlib.Path(__file__).parent/'support/mt/include'
  cmd = [str(pathlib.Path(musa)/'bin/mcc'), f'--musa-path={musa}', '--offload-arch=mp_21', f'-I{include}', str(src_path), '-o', host,
         f'-L{musa}/lib', f'-L{driver}', '-lmusart', '-lmusa', '-lstdc++']
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  if result.returncode: raise RuntimeError(f'MUSA launch host build failed:\n{result.stdout}{result.stderr}')
  env = os.environ.copy()
  env.update({'MT_CAPTURE_DIR':td, 'MT_MPX':'0xff', 'MT_UMD':'1', 'LD_PRELOAD':mtum_library(),
              'LD_LIBRARY_PATH':f'{musa}/lib:{driver}'})
  result = subprocess.run([host], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
  if 'JOB SUBMIT' not in result.stderr: raise RuntimeError(f'MUSA launch generation failed:\n{result.stdout}{result.stderr}')
  state, lib = parse_mt_bundle(pathlib.Path(td)), MTCompiler(musa).extract_raw(pathlib.Path(host),td)
  for path in pathlib.Path(td).iterdir():
    if path.name != 'real-submission.bin': path.unlink()
  return {'bundle':td,'packet':str(pathlib.Path(td)/'real-submission.bin'),'state':state,'lib':lib}


class MUSACompiler(Compiler):
  def __init__(self):
    super().__init__('mt')
    self.mt = MTCompiler()
  def compile(self, src:str) -> bytes:
    if os.environ.get('MT_GENERATE','1') == '0': lib = self.mt.compile_raw(src)
    else:
      match = re.search(r'extern "C" __global__ void\s+(\w+)\s*\(([^)]*)\)',src)
      if match is None: raise RuntimeError('MUSA kernel signature is missing')
      params = [p for p in match.group(2).split(',') if p.strip()]
      args = ['out' if '*' in p else 'scalar' for p in params]
      cached = generate_mt_launch(src,match.group(1),args,(1,1,1),(1,1,1))
      lib = cached['lib']
      cached['state'] = compress_mt_bundle(cached['state'])
      key = hashlib.sha256(lib).hexdigest()
      diskcache_put('mt_launch_v3',f'{key}_{(1,1,1)}_{(1,1,1)}',cached)
    key = hashlib.sha256(lib).hexdigest()
    MUSA_SOURCES[key] = src
    diskcache_put('mt_sources', key, src)
    return lib

class MUSARenderer(CUDARenderer):
  kernel_typedef = 'extern "C" __global__ void'
  def __init__(self, target):
    CStyleLanguage.__init__(self, target)
    self.compiler, self.tensor_cores = MUSACompiler(), []
  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    return super().render_kernel(function_name, kernel, bufs, uops, ['#include <musa.h>'] + (prefix or []))
  def supported_dtypes(self): return set(dtypes.all) - {dtypes.double}

@dataclasses.dataclass(frozen=True)
class MTBufferMeta:
  va: int
  pa: int
  size: int

class MTAllocator(Allocator['MTDevice']):
  def __init__(self, dev):
    super().__init__(dev)
    self.next_va, self.next_pa = 0x8001000000, 0x80000000
    self.dev.views = {}
    self.free_blocks:list[tuple[int,int,int]] = []

  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    size = round_up(size, 0x1000)
    if self.free_blocks and self.free_blocks[-1][2] == size:
      va, pa, size = self.free_blocks.pop()
    else:
      va, pa = self.next_va, self.next_pa
      self.next_va += size
      self.next_pa += size
    view = self.dev.mt.pci_dev.map_bar(2, off=pa, size=size, fmt='B')
    meta = MTBufferMeta(va, pa, size)
    self.dev.map_buffer(va, pa, size)
    self.dev.views[va] = (view, meta)
    return BufferStorage(va, meta, view)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    meta = storage.meta
    self.dev.views.pop(meta.va, None)
    self.free_blocks.append((meta.va, meta.pa, meta.size))

  def _copyin(self, dest:int, src:memoryview):
    view, meta = self.dev._view(dest)
    off = dest-meta.va
    if off % 4 == 0 and src.nbytes % 4 == 0: view[off:off+src.nbytes] = src.cast('B')
    else:
      start, end = off & ~3, round_up(off+src.nbytes,4)
      data = bytearray(view[start:end])
      data[off-start:off-start+src.nbytes] = src.cast('B')
      view[start:end] = data

  def _copyout(self, dest:memoryview, src:int):
    view, meta = self.dev._view(src)
    off = src-meta.va
    start, end = off & ~3, round_up(off+dest.nbytes,4)
    data = view[start:end]
    dest[:] = data[off-start:off-start+dest.nbytes]

  def _map(self, buf:Buffer) -> BufferStorage: raise NotImplementedError('MT external mappings are not implemented')
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

def mtum_library() -> str:
  path = os.environ.get('MT_UMD', str(pathlib.Path.home()/'.cache/tinygrad/libmtum.so'))
  source = pathlib.Path(__file__).parent/'support/mt/mtum.c'
  if not pathlib.Path(path).exists() or pathlib.Path(path).stat().st_mtime < source.stat().st_mtime:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f'{path}.{os.getpid()}'
    result = subprocess.run(['cc','-shared','-fPIC','-O2','-o',tmp,str(source),'-ldl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode: raise RuntimeError(f'libmtum build failed:\n{result.stdout}{result.stderr}')
    os.replace(tmp,path)
  return path

class MTLaunchGenerator:
  def __init__(self, dev:MTDevice):
    self.dev = dev

  def generate(self, obj:TinyELF, global_size:tuple[int,int,int], local_size:tuple[int,int,int]) -> MTLaunch:
    args = ['out' if addrspace != AddrSpace.ALU else 'scalar' for _,addrspace,_,_ in obj.signature]
    key = hashlib.sha256(obj.lib).hexdigest()
    source = MUSA_SOURCES.get(key) or diskcache_get('mt_sources', key)
    if source is None: raise RuntimeError('MUSA source is missing from the launch-generation cache')
    cached = generate_mt_launch(source,obj.name,args,global_size,local_size)
    return MTLaunch(self.dev.mt,cached['bundle'],cached['packet'],cached['state'])

class MTProgram(Program['MTDevice']):
  def __init__(self, dev:MTDevice, obj:TinyELF, links:dict[str,MTProgram]):
    self.dev, self.name, self.profile_key, self.obj = dev, obj.name, obj.profile_key, obj
    self.image = MTKernelImage.from_elf(obj.lib)
    self.launch:MTLaunch|None = None

  def __call__(self, *args:int, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               wait:bool=False, timeout:int|None=None):
    self.launch = self.dev.get_launch(self.obj, global_size, local_size)
    if os.environ.get('MT_GENERATE','1') == '0' and not os.environ.get('MT_SKIP_LOAD'):
      self.launch.load_kernel(self.image, global_size, local_size)
    self.dev.ensure_started(self.launch)
    self.launch.set_args(tuple(args))
    self.launch.set_grid(global_size, local_size)
    result = self.launch.execute()
    return result.run_time / 25_000_000 if wait else None

class MTDevice(Compiled):
  renderers = [MUSARenderer]
  def __init__(self, device:str=''):
    dev_id = int(device.split(':')[1]) if ':' in device else 0
    self.mt = MTDev.open_usb(dev_id)
    bundle = os.environ.get('MT_LAUNCH_BUNDLE', '/tmp/mt-research/argfresh')
    packet = os.environ.get('MT_LAUNCH_PACKET', '/tmp/mt-research/real-submission-direct.bin')
    self.launch = None if os.environ.get('MT_GENERATE','1') != '0' else MTLaunch(self.mt, bundle, packet)
    self.launches = [] if self.launch is None else [self.launch]
    self.launch_cache:dict[tuple[int,tuple[int,int,int],tuple[int,int,int]],MTLaunch] = {}
    self.generator = MTLaunchGenerator(self)
    self.buffer_maps:list[tuple[int,int,int]] = []
    self.current_launch:MTLaunch|None = None
    self.started = False
    self.views:dict[int,tuple[Any,MTBufferMeta]] = {}
    super().__init__(device, MTAllocator(self), MTProgram, arch='mp_21')

  @property
  def has_copy_queue(self) -> bool: return False

  def get_launch(self, obj:TinyELF, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1)) -> MTLaunch:
    if os.environ.get('MT_GENERATE','1') == '0':
      assert self.launch is not None
      return self.launch
    key = (hash(obj.lib), global_size, local_size)
    if key not in self.launch_cache:
      lib_key = hashlib.sha256(obj.lib).hexdigest()
      cache_key = f'{lib_key}_{global_size}_{local_size}'
      cached = diskcache_get('mt_launch_v3', cache_key)
      if cached is None and (global_size,local_size) != ((1,1,1),(1,1,1)):
        cached = diskcache_get('mt_launch_v3', f'{lib_key}_{(1,1,1)}_{(1,1,1)}')
      if cached is not None and pathlib.Path(cached['packet']).exists():
        state = cached.get('state')
        if isinstance(state,bytes): state = decompress_mt_bundle(state)
        launch = MTLaunch(self.mt, cached['bundle'], cached['packet'], state)
      else:
        launch = self.generator.generate(obj, global_size, local_size)
      self.launch_cache[key] = launch
      self.launches.append(launch)
      for va,pa,size in self.buffer_maps: launch.map_buffer(va,pa,size)
      assert launch.bundle_state is not None
      diskcache_put('mt_launch_v3', cache_key,
                    {'bundle':str(launch.bundle),'packet':str(launch.packet_path),'state':compress_mt_bundle(launch.bundle_state)})
    return self.launch_cache[key]

  def map_buffer(self, va:int, pa:int, size:int):
    self.buffer_maps.append((va,pa,size))
    for launch in self.launches: launch.map_buffer(va,pa,size)

  def ensure_started(self, launch:MTLaunch):
    if self.current_launch is not launch:
      launch.start(timeout=float(os.environ.get('MT_BOOT_TIMEOUT','10')))
      self.current_launch = launch
      self.started = True

  def _view(self, va:int):
    for base, (view, meta) in self.views.items():
      if base <= va < base + meta.size: return view, meta
    raise RuntimeError(f'Unknown MT buffer address {va:#x}')

  def finalize(self):
    super().finalize()
    self.mt.close()
