import sys, sqlite3, pickle
from collections import defaultdict
from tqdm import tqdm

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')
from tinygrad.codegen.optimizer import Opt, OptOps

if __name__ == "__main__":
  fn = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinygrad_cache"
  conn = sqlite3.connect(fn)
  cur = conn.cursor()
  cur.execute("SELECT key,subkey,val FROM time_linearizer")
  grouped = defaultdict(dict)
  for ast,sk,v in tqdm(cur.fetchall()):
    #, opts, allow_test_size, max_global_size = eval(k)
    grouped[ast][sk] = pickle.loads(v)
    #v = pickle.loads(v)
    #grouped[ast][tuple(opts)] = v

  for ast,optc in grouped.items():
    print(len(optc))
    print(optc.items())



