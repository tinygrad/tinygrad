#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle
from tinygrad.codegen.linearizer import Linearizer, time_lin
from tinygrad.helpers import colored, db_connection, VERSION, getenv, to_function_name, tqdm

page_size = 100
conn = db_connection()
cur = conn.cursor()
row_count = cur.execute(f"select count(*) from 'process_replay_{VERSION}'").fetchone()[0]
for offset in tqdm(range(0, row_count, page_size)):
  cur.execute(f"SELECT val FROM 'process_replay_{VERSION}' LIMIT ? OFFSET ?", (page_size, offset))
  for row in cur.fetchall():
    compare_k, compare_src, time_baseline = pickle.loads(row[0])
    k = Linearizer(*compare_k.ast, opts=compare_k.opts)
    for opt in compare_k.applied_opts: k.apply_opt(opt)
    good_src = k.opts.render(to_function_name(compare_k.name), k.linearize().uops)
    time_ = time_lin(k)
    try:
      assert compare_src == good_src, "PROCESS REPLAY DETECTED CHANGE"
      assert (time_-time_baseline)/max(time_baseline, 1e8) < 0.2, f"TIME REPLAY DETECTED CHANGE {time_baseline} ->{time_}"
    except AssertionError as e:
      print(e.args[0])
      print(compare_k.ast)
      print(compare_k.applied_opts)
      diff = list(difflib.unified_diff(good_src.splitlines(), compare_src.splitlines()))
      for line in diff:
        print(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
      if getenv("ASSERT_PROCESS_REPLAY", 1): raise e