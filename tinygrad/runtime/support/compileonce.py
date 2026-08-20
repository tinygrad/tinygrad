import ast, subprocess, sys
from tinygrad.device import CompileError
from tinygrad.helpers import fromimport

# run argv as a one shot compiler: src goes to stdin, the compiled binary comes back on stdout (see __main__ below)
def compile_once(argv:list[str], src:str, env:dict[str,str]|None=None) -> bytes:
  with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env) as p:
    ret, _ = p.communicate(src.encode())
    if p.returncode != 0: raise CompileError(f"Compilation Error: {' '.join(argv)}")
  return ret

if __name__ == "__main__":
  assert len(sys.argv) >= 3, f"usage: {sys.argv[0]} <compiler> <arch> [<args>]"
  compiler = fromimport(*sys.argv[1].split(':'))(sys.argv[2], *(ast.literal_eval(arg) for arg in sys.argv[3:]))
  sys.stdout.buffer.write(compiler.compile(sys.stdin.buffer.read().decode()))
