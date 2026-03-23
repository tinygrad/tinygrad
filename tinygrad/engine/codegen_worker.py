"""Worker function for parallel codegen — separate module required for spawn-based multiprocessing."""
import pickle

def codegen_worker(args:tuple[bytes, str]) -> bytes:
  """Run get_program in a worker process. Takes (ast_bytes, device_str) and returns ProgramSpec bytes."""
  ast_bytes, device_str = args
  ast = pickle.loads(ast_bytes)
  from tinygrad.codegen import get_program
  from tinygrad.device import Device
  return pickle.dumps(get_program(ast, Device[device_str].renderer))
