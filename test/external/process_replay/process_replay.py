#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, multiprocessing, os, logging
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, getenv, tqdm

PAGE_SIZE = 100
TABLE_NAME = f"process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}_{VERSION}"
ASSERT_DIFF = getenv("ASSERT_PROCESS_REPLAY", int((k:="[run_process_replay]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)))
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
assert MAX_DIFF_PCT < 100
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format='%(processName)s: %(message)s')

def process_replay(offset:int):
  if early_stop.is_set(): return
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    ast, opts, applied_opts, name, compare_src, ctx = pickle.loads(row[0])
    with Context(**{k:v for k,v in ctx.items() if k in ContextVar._cache}):
      # try linearize
      try:
        k = Kernel(ast, opts=opts)
        for opt in applied_opts: k.apply_opt(opt)
        good_src = k.opts.render(name, k.linearize().uops)
      except Exception as e:
        logging.info("WARN: FAILED TO RECREATE KERNEL")
        logging.info(ast)
        logging.info(applied_opts)
        logging.info(e)
        if ASSERT_DIFF: raise e
        continue
      # try compare
      try: assert compare_src == good_src
      except AssertionError as e:
        changed += 1
        logging.info("PROCESS REPLAY DETECTED CHANGE")
        logging.info(ast)
        logging.info(applied_opts)
        diff = list(difflib.unified_diff(good_src.splitlines(), compare_src.splitlines()))
        for line in diff:
          logging.info(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
        if ASSERT_DIFF: raise e
        if changed > MAX_DIFF_PCT:
          logging.info(f"WARN: detected chanegs in over {MAX_DIFF_PCT}% of kernels. skipping further diff generation.")
          early_stop.set()
          break
  conn.commit()
  cur.close()

if __name__ == "__main__":
  conn = db_connection()
  cur = conn.cursor()
  row_count = cur.execute(f"select count(*) from '{TABLE_NAME}'").fetchone()[0]
  conn.commit()
  cur.close()
  offsets = range(0, row_count, PAGE_SIZE)
  with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: list(tqdm(pool.imap(process_replay, offsets), total=len(offsets)))
