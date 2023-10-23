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
from tinygrad.codegen.linearizer import Linearizer

if __name__ == "__main__":
  fn = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinygrad_cache"
  conn = sqlite3.connect(fn)
  cur = conn.cursor()
  cur.execute("SELECT * FROM time_linearizer")
  grouped = defaultdict(dict)
  for f in tqdm(cur.fetchall()): grouped[f[0]][f[1:-1]] = pickle.loads(f[-1])

  opts_to_outcome = {}

  for ast,sk in grouped.items():
    cnts = defaultdict(int)
    for sks,tm in sk.items():
      if sks[1] != 1: continue
      opts = eval(sks[0])
      cnts[(len(opts), sks[1])] += 1
      opts_to_outcome[(ast, tuple(opts))] = tm
    print(cnts)

  for ast,k in opts_to_outcome:
    if len(k) == 0: continue
    lin = Linearizer(eval(ast))
    for opt in k[:-1]: lin.apply_opt(opt)
    old_tm = min(opts_to_outcome[(ast,k[:-1])])
    new_tm = min(opts_to_outcome[(ast,k)])
    act = k[-1]
    print(f"ratio: {old_tm/new_tm:8.2f}x from {str(act):50s} on {lin.colored_shape()}")


