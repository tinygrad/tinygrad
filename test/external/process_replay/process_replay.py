#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.device import Device
from tinygrad.helpers import colored, db_connection, VERSION, getenv, tqdm

page_size = 100
conn = db_connection()
cur = conn.cursor()
row_count = cur.execute(f"select count(*) from 'process_replay_{VERSION}'").fetchone()[0]
for offset in tqdm(range(0, row_count, page_size)):
  cur.execute(f"SELECT val FROM 'process_replay_{VERSION}' LIMIT ? OFFSET ?", (page_size, offset))
  for row in cur.fetchall():
    name, ast, device, applied_opts, compare_uops, compare_src = pickle.loads(row[0])
    k = Linearizer(*ast, opts=Device[device].renderer)
    for opt in applied_opts: k.apply_opt(opt)
    good_src = k.opts.render(name, k.linearize().uops)
    try: assert compare_src == good_src
    except AssertionError as e:
      print("PROCESS REPLAY DETECTED CHANGE")
      print(ast)
      print(applied_opts)
      diff = list(difflib.unified_diff(good_src.splitlines(), compare_src.splitlines()))
      for line in diff:
        print(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
      if getenv("ASSERT_PROCESS_REPLAY", 1): raise e
