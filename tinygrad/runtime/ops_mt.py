from __future__ import annotations
import dataclasses, os, pathlib, subprocess, tempfile, shutil, hashlib
from typing import Any
from tinygrad.device import Compiled, Program, Allocator, BufferStorage, BufferSpec, Buffer
from tinygrad.renderer import Compiler, TinyELF
from tinygrad.renderer.cstyle import CUDARenderer, CStyleLanguage
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import round_up, diskcache_get, diskcache_put
from tinygrad.runtime.support.mt.mtdev import MTDev
from tinygrad.runtime.support.mt.launch import MTLaunch
from tinygrad.runtime.support.mt.compiler import MTCompiler
from tinygrad.runtime.support.mt.elf import MTKernelImage

MUSA_SOURCES:dict[str,str] = {}

class MUSACompiler(Compiler):
  def __init__(self):
    super().__init__('mt')
    self.mt = MTCompiler()
  def compile(self, src:str) -> bytes:
    if os.environ.get('MT_TRACE'): print('MT SOURCE\n'+src,flush=True)
    lib = self.mt.compile_raw(src)
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
    view[dest-meta.va:dest-meta.va+src.nbytes] = src.cast('B')

  def _copyout(self, dest:memoryview, src:int):
    view, meta = self.dev._view(src)
    if os.environ.get('MT_TRACE'): print('MT COPYOUT',hex(src),meta,view[:16].hex(),flush=True)
    dest[:] = view[src-meta.va:src-meta.va+dest.nbytes]

  def _map(self, buf:Buffer) -> BufferStorage: raise NotImplementedError('MT external mappings are not implemented')
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

def mtum_library() -> str:
  path = os.environ.get('MT_UMD', str(pathlib.Path.home()/'.cache/tinygrad/libmtum.so'))
  source = pathlib.Path(__file__).parent/'support/mt/mtum.c'
  if not pathlib.Path(path).exists():
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['cc','-shared','-fPIC','-O2','-o',path,str(source),'-ldl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode: raise RuntimeError(f'libmtum build failed:\n{result.stdout}{result.stderr}')
  return path

class MTLaunchGenerator:
  def __init__(self, dev:MTDevice):
    self.dev = dev

  def generate(self, obj:TinyELF, global_size:tuple[int,int,int], local_size:tuple[int,int,int]) -> MTLaunch:
    td = tempfile.mkdtemp(prefix='mt-launch-')
    src_path = pathlib.Path(td) / 'kernel.mu'
    args = []
    for _, addrspace, _, _ in obj.signature:
      args.append('out' if addrspace != AddrSpace.ALU else 'scalar')
    # The generator uses a fixed pointer only to construct vendor launch state. Real buffers replace it at launch.
    arg_decls = 'void *out = reinterpret_cast<void *>(uintptr_t(0xe1c2003000ULL)); uint64_t scalar = 1;'
    arg_names = ','.join('&out' if x == 'out' else '&scalar' for x in args)
    main = (f'#include <musa.h>\n#include <cstdint>\n#include <unistd.h>\nint main() {{\n'
            f'  musaFunction_t fn = nullptr;\n  musaGetFuncBySymbol(&fn, (const void *)&{obj.name});\n'
            f'  {arg_decls}\n  void *args[] = {{{arg_names}}};\n'
            f'  musaLaunchKernel((const void *)&{obj.name}, dim3{global_size}, dim3{local_size}, args, 0, nullptr);\n'
            f'  musaDeviceSynchronize();\n  usleep(1000000);\n  _Exit(0);\n}}\n')
    key = hashlib.sha256(obj.lib).hexdigest()
    source = MUSA_SOURCES.get(key) or diskcache_get('mt_sources', key)
    if source is None: raise RuntimeError('MUSA source is missing from the launch-generation cache')
    src_path.write_text(source + '\n' + main)
    musa = os.environ.get('MUSA_PATH', '/tmp/mt-research/toolkit/musa_toolkits_4.3.0')
    driver = '/tmp/mt-research/musa-driver-3.3.0/usr/lib/x86_64-linux-gnu'
    host = str(pathlib.Path(td) / 'host')
    cmd = [str(pathlib.Path(musa)/'bin/mcc'), f'--musa-path={musa}', '--offload-arch=mp_21', str(src_path), '-o', host,
           f'-L{musa}/lib', f'-L{driver}', '-lmusart', '-lmusa', '-lstdc++']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode: raise RuntimeError(f'MUSA launch host build failed:\n{result.stdout}{result.stderr}')
    env = os.environ.copy()
    env.update({'MT_CAPTURE_DIR':td, 'MT_MPX':'0xff', 'MT_UMD':'1', 'LD_PRELOAD':mtum_library(),
                'LD_LIBRARY_PATH':f'{musa}/lib:{driver}'})
    result = subprocess.run([host], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
    if 'JOB SUBMIT' not in result.stderr: raise RuntimeError(f'MUSA launch generation failed:\n{result.stdout}{result.stderr}')
    for common in pathlib.Path('/tmp/mt-research').glob('real-map-*.bin'):
      if not (pathlib.Path(td)/common.name).exists(): shutil.copy(common, pathlib.Path(td)/common.name)
    return MTLaunch(self.dev.mt, td, str(pathlib.Path(td)/'real-submission.bin'))

class MTProgram(Program['MTDevice']):
  def __init__(self, dev:MTDevice, obj:TinyELF, links:dict[str,MTProgram]):
    self.dev, self.name, self.profile_key, self.obj = dev, obj.name, obj.profile_key, obj
    self.image = MTKernelImage.from_elf(obj.lib)
    self.launch:MTLaunch|None = None
    if os.environ.get('MT_TRACE'): print('MT IMAGE',self.image.metadata.kernel,flush=True)

  def __call__(self, *args:int, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               wait:bool=False, timeout:int|None=None):
    self.launch = self.dev.get_launch(self.obj, global_size, local_size)
    if os.environ.get('MT_GENERATE','1') == '0' and not os.environ.get('MT_SKIP_LOAD'):
      self.launch.load_kernel(self.image, global_size, local_size)
    if os.environ.get('MT_TRACE'): print('MT START',self.launch.bundle,flush=True)
    self.dev.ensure_started(self.launch)
    if os.environ.get('MT_TRACE'): print('MT STARTED',flush=True)
    self.launch.set_args(tuple(args))
    self.launch.set_grid(global_size, local_size)
    if os.environ.get('MT_TRACE'):
      print('MT LAUNCH DIMS',self.name,global_size,local_size,flush=True)
      print('MT LAUNCH ARGS',tuple(hex(a) for a in args),[hex(self.launch.mmu.pages.get(a & ~0xfff,0)) for a in args],flush=True)
      arg_off = self.launch.backing_offsets[0x81e1200000]
      print('MT LAUNCH ARGSTATE',self.launch.backing[arg_off+0x38:arg_off+0x68].hex(),flush=True)
    result = self.launch.execute()
    if os.environ.get('MT_TRACE'):
      const = self.launch.backing_offsets[0x81e1200000]
      dims_off = self.launch.backing_offsets[0x81e1400000]
      print('MT DIMS',global_size,local_size,self.launch.backing[dims_off+0x2c:dims_off+0x40].hex(),flush=True)
      view, meta = self.dev._view(args[-1])
      off = args[-1] - meta.va
      print('MT PROGRAM',self.name,tuple(hex(a) for a in args),result,
            self.launch.backing[const+0x40:const+0x50].hex(),view[off:off+16].hex(),flush=True)
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
      cache_key = f'{hashlib.sha256(obj.lib).hexdigest()}_{global_size}_{local_size}'
      cached = diskcache_get('mt_launch_v3', cache_key)
      if cached is not None and pathlib.Path(cached['packet']).exists():
        launch = MTLaunch(self.mt, cached['bundle'], cached['packet'])
      else:
        launch = self.generator.generate(obj, global_size, local_size)
        diskcache_put('mt_launch_v3', cache_key, {'bundle':str(launch.bundle),'packet':str(launch.packet_path)})
      self.launch_cache[key] = launch
      self.launches.append(launch)
      for va,pa,size in self.buffer_maps: launch.map_buffer(va,pa,size)
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
