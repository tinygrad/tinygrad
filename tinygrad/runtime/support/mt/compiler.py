from __future__ import annotations
import os, pathlib, subprocess, tempfile
from tinygrad.runtime.support.mt.elf import MTKernelImage

DEFAULT_MUSA_PATH = '/tmp/mt-research/toolkit/musa_toolkits_4.3.0'

class MTCompiler:
  def __init__(self, musa_path:str|None=None):
    self.musa_path = musa_path or os.environ.get('MUSA_PATH', DEFAULT_MUSA_PATH)
    self.bin = pathlib.Path(self.musa_path) / 'bin'

  def _run(self, argv:list[str], cwd:str) -> None:
    result = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode: raise RuntimeError(f"MT tool failed ({' '.join(argv)}):\n{result.stdout}{result.stderr}")

  def extract_raw(self, obj:pathlib.Path, cwd:str) -> bytes:
    fatbin = pathlib.Path(cwd) / 'fatbin.bin'
    self._run([str(self.bin/'llvm-objcopy'), f'--dump-section=.mt_fatbin={fatbin}', str(obj)], cwd)
    device = pathlib.Path(cwd) / 'device.out'
    self._run([str(self.bin/'clang-offload-bundler'), '-unbundle', '-type=o',
               '-targets=host-x86_64-unknown-linux,musa-mtgpu-mt-musa--mp_21',
               f'-inputs={fatbin}', f'-outputs={cwd}/host.o,{device}'], cwd)
    return device.read_bytes()

  def compile_raw(self, source:str) -> bytes:
    with tempfile.TemporaryDirectory(prefix='mt-mcc-') as td:
      src, obj = pathlib.Path(td) / 'kernel.mu', pathlib.Path(td) / 'kernel.o'
      src.write_text(source)
      include = pathlib.Path(__file__).parent/'include'
      self._run([str(self.bin/'mcc'), f'--musa-path={self.musa_path}', '--offload-arch=mp_21', f'-I{include}',
                 '-c', str(src), '-o', str(obj)], td)
      return self.extract_raw(obj,td)

  def compile(self, source:str) -> MTKernelImage:
    return MTKernelImage.from_elf(self.compile_raw(source))
