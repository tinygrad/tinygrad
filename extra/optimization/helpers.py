# stuff needed to unpack a kernel
from tinygrad import Variable
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import getenv, dedup, DEBUG
from tinygrad.codegen.opt.postrange import Scheduler
inf, nan = float('inf'), float('nan')
UOps = Ops  # backwards compat alias

# kernel unpacker
def ast_str_to_ast(ast_str:str) -> UOp: return eval(ast_str)
def ast_str_to_lin(ast_str:str, opts=None): return Scheduler(ast_str_to_ast(ast_str), opts)

# load worlds, a dataset of kernels
import gzip
from pathlib import Path
import random
def load_worlds(filter_reduce=True, filter_noimage=True, filter_novariable=True):
  fn = Path(__file__).parent.parent / "datasets/sops.gz"
  ast_strs = dedup(gzip.open(fn).read().decode('utf-8').strip().split("\n"))
  assert len(ast_strs) >= getenv("MIN_ASTS", 1000), f"dataset size = {len(ast_strs)} is too small"
  if DEBUG >= 1: print(f"loaded {len(ast_strs)=} before filters")
  if filter_reduce: ast_strs = [x for x in ast_strs if "Ops.REDUCE" in x]
  if filter_noimage: ast_strs = [x for x in ast_strs if "dtypes.image" not in x]
  if filter_novariable: ast_strs = [x for x in ast_strs if "DEFINE_VAR" not in x]
  if DEBUG >= 1: print(f"loaded {len(ast_strs)=} after filters {filter_reduce=}, {filter_noimage=}, {filter_novariable=}")
  random.seed(1337)
  random.shuffle(ast_strs)
  return ast_strs
