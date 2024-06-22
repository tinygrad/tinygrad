#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, time, shutil
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import colored, db_connection, VERSION, getenv, to_function_name, tinytqdm

page_size = 100
conn = db_connection()
cur = conn.cursor()
row_count = cur.execute(f"select count(*) from 'process_replay_{VERSION}'").fetchone()[0]
for offset in tinytqdm(range(0, row_count, page_size)):
  cur.execute(f"SELECT val FROM 'process_replay_{VERSION}' LIMIT ? OFFSET ?", (page_size, offset))
  for row in cur.fetchall():
    compare_k, compare_src, time_ns = pickle.loads(row[0])
    k = Linearizer(*compare_k.ast, opts=compare_k.opts)
    for opt in compare_k.applied_opts: k.apply_opt(opt)
    st = time.perf_counter_ns()
    good_src = k.opts.render(to_function_name(compare_k.name), k.linearize().uops)
    t_diff = time.perf_counter_ns() - st
    if (abs(t_diff-time_ns)/max(time_ns, 1e8) > 0.1):
      info = colored(f"\rPerf: {time_ns*1e-6:6.2f} -> {t_diff*1e-6:6.2f} ms {compare_k.name}", "red" if t_diff > time_ns else "green")
      print('\r'+' '*(shutil.get_terminal_size().columns) +info )
    try: assert compare_src == good_src
    except AssertionError as e:
      print("PROCESS REPLAY DETECTED CHANGE")
      print(compare_k.ast)
      print(compare_k.applied_opts)
      diff = list(difflib.unified_diff(good_src.splitlines(), compare_src.splitlines()))
      for line in diff:
        print(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
      if getenv("ASSERT_PROCESS_REPLAY", 1): raise e