#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, getenv, tqdm, timeit, partition
from typing import Tuple, List

page_size = 100
table_name = f"process_replay_{getenv('GITHUB_SHA', 'HEAD')}_{VERSION}"
print(table_name)
conn = db_connection()
cur = conn.cursor()
row_count = cur.execute(f"select count(*) from '{table_name}'").fetchone()[0]
timediffs:List[Tuple[str, float, float]] = []
for offset in tqdm(range(0, row_count, page_size)):
  cur.execute(f"SELECT val FROM '{table_name}' LIMIT ? OFFSET ?", (page_size, offset))
  for row in cur.fetchall():
    ast, opts, applied_opts, name, compare_src, ctx, lintime = pickle.loads(row[0])
    with Context(**{k:v for k,v in ctx.items() if k in ContextVar._cache}):
      # try linearize
      try:
        k = Linearizer(*ast, opts=opts)
        for opt in applied_opts: k.apply_opt(opt)
        good_src = k.opts.render(name, k.linearize().uops)
        if ctx.get("TIMEIT"):
          dt = min(timeit(lambda:k.linearize().uops.linearize()) for i in range(5)) - lintime
          if abs(dt) > 1e-3: timediffs.append((name, lintime, dt))
      except Exception as e:
        print("FAILED TO RECREATE KERNEL")
        print(ast)
        print(applied_opts)
        print(e)
        if getenv("ASSERT_PROCESS_REPLAY", 1): raise e
        continue
      # try compare
      try: assert compare_src == good_src
      except AssertionError as e:
        print("PROCESS REPLAY DETECTED CHANGE")
        print(ast)
        print(applied_opts)
        diff = list(difflib.unified_diff(good_src.splitlines(), compare_src.splitlines()))
        for line in diff:
          print(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
        if getenv("ASSERT_PROCESS_REPLAY", 1): raise e

better, worse = partition(sorted(timediffs, key=lambda x: x[2]),lambda x:x[2]<0)
for (name,lintime, dt) in better[:10]: print(f"better:{lintime:.3f} - {-dt:.3f}s {name}")
for (name, lintime, dt) in worse[-1:-10:-1]: print(f"worse: {lintime:.3f} + {dt:.3f}s {name}")