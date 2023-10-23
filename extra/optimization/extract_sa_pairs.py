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
  cur.execute("SELECT * FROM time_linearizer")
  grouped = defaultdict(dict)
  for f in tqdm(cur.fetchall()): grouped[f[0]][f[1:-1]] = pickle.loads(f[-1])

  for ast,sk in grouped.items():
    print(len(sk))
    for sks,tm in sk.items():
      #print(sks[1], sks[2])
      opts = eval(sks[0])
      #print(opts)
      #ss = sks.split(",")
      #print(ss)

      #print(sks)
      #opts, allow_test_size, max_global_size = eval(sks)
      #print(tm)

    #print(optc.items())



