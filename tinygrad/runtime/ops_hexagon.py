import time, ctypes, subprocess, pathlib, tempfile
from typing import List
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import DEBUG
from tinygrad.renderer.cstyle import HVXRenderer


# argv: 0 - .elf, 1 - output, 2+ - input and symbolic
HEXAGON_FOOTER = """
char* load(char* path){{
  FILE* fp = fopen(path, "rb");
  fseek(fp, 0, SEEK_END);
  int sz = (int) ftell(fp);
  char* data = malloc(sz);
  fseek(fp, 0, SEEK_SET);
  fread(data, sz, 1, fp);
  return data;
}}

int main(int argc, char** argv) {{
  FILE* fp = fopen(argv[1], "rb");
  fseek(fp,0,SEEK_END);
  int sz0 = (int)ftell(fp);
  fclose(fp);
  {CALL}
  fp = fopen(argv[1],"wb");
  fwrite(data0,sz0,1,fp);
  return 0;
}}
"""

def gen_func_call(signature: str):
  name = signature.split("(")[0].split(" ")[-1]
  params = signature.split("(")[1].split(")")[0].split(",")
  symbolic = list(i.split(" ")[-1] for i in filter(lambda x: "const int " in x, params))
  types: List[str] = ["float", "unsigned int", "int", "unsigned long long", "long long", "unsigned long", "long",
           "half", "unsigned char", "uchar", "bool", "char", "unsigned short", "short"]
  types = [next(filter(lambda x: x in p, types)) for p in params]
  lines = [f"{types[0]}* data0 = ({types[0]}*)malloc(sz0);"]
  lines += [f"{types[i]}* data{i} = ({types[i]}*)load(argv[{i+1}]);" for i in range(1, len(params) - len(symbolic))]
  lines += [f"int {p} = atoi(argv[{len(params)-len(symbolic)+i+1}]);" for i, p in enumerate(symbolic)]
  call = f"{name}(" + ",".join([f"{i}" for i in map(lambda x: x.split(" ")[-1], params)]) + ");"
  return "\n".join(lines) + "\n" + call

class HexagonCompiler(Compiler):
  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=False, suffix='.hexagon.elf') as outelf, tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.hexagon.c') as outc:  # noqa: E501
      outc.write(src+HEXAGON_FOOTER.format(CALL = gen_func_call(src.splitlines()[11])))
      outc.flush()
      #print("Put generated C code here:", outc.name)
      cc = 'hexagon-clang'
      cmd= f'{cc} {outc.name} -mhvx -mv65 -O2 -Wall -lm -o {outelf.name} '
      if DEBUG>=4: print(cmd)
      subprocess.check_output(args=cmd.split(), stderr=subprocess.DEVNULL if DEBUG<3 else None)
      #print("Finished compiling")
      return pathlib.Path(outelf.name).read_bytes()


class HexagonProgram:
  def __init__(self, name: str, lib: bytes):
    self.name, self.lib = name, lib

  def __call__(self, *bufs, vals=(), wait=False):
    if wait: st = time.monotonic()
    files = []
    with tempfile.NamedTemporaryFile(delete=False, suffix='.v0.hexagon.elf') as outelf:
      outelf.write(self.lib)
      outelf.flush()
    cmd = f'hexagon-sim {outelf.name} -- '
    for (i,buf) in enumerate(bufs):
      f = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{str(i)}.data.buf')
      f.write(ctypes.string_at(ctypes.addressof(buf), ctypes.sizeof(buf)))
      files.append(f)
      #print("buffer size:", ctypes.sizeof(buf))
      cmd += f' {f.name} '
    fbl = ctypes.sizeof(bufs[0])
    for f in files:
      f.flush()
      f.close()
    #print("Vals:", vals)
    for val in vals: cmd += f' {val} '
    #print(cmd)
    r = pathlib.Path(files[0].name).read_bytes()
    subprocess.check_output(args=cmd.split(), stderr=subprocess.DEVNULL if DEBUG<2 else None)
    r = pathlib.Path(files[0].name).read_bytes()
    #print("length of output: ", len(r))
    assert len(r) == fbl, f"expected {fbl} bytes, got {len(r)}"
    ctypes.memmove(ctypes.addressof(bufs[0]), r, len(r))

    if wait: return time.monotonic()-st

class HexagonDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, HVXRenderer(), HexagonCompiler("compile_hexagon"), HexagonProgram)
