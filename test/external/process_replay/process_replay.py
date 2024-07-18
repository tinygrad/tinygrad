#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, multiprocessing, os, logging, time
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, getenv, tqdm, partition

PAGE_SIZE = 100
TABLE_NAME = f"process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}_{VERSION}"
ASSERT_DIFF = getenv("ASSERT_PROCESS_REPLAY", int((k:="[run_process_replay]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)))
REF = os.getenv("GITHUB_REF_NAME", "")
SKIP_PROCESS_REPLAY = int((k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")) or REF == "master"
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format='%(message)s')

def process_replay(offset:int):
  timediffs = []
  if early_stop.is_set(): return
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    ast, applied_opts = None, None
    # try unpickle and linearize
    try:
      ast, opts, applied_opts, name, compare_src, ctx, comp_time = pickle.loads(row[0])
      with Context(**{k:v for k,v in ctx.items() if k in ContextVar._cache and k != "DEBUG"}):
        k = Kernel(ast, opts=opts)
        for opt in applied_opts: k.apply_opt(opt)
        good_src = k.opts.render(name, k.linearize().uops)
        if comp_time is not None:
          master_time = min(timeit(lambda: k.linearize().uops.linearize()) for _ in range(5))
          if abs(master_time - comp_time) > 1e-3: timediffs.append((name, comp_time, master_time))
    except Exception as e:
      logging.warn("FAILED TO RECREATE KERNEL")
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
        logging.warn(f"detected changes in over {MAX_DIFF_PCT}% of kernels. skipping further diff generation.")
        early_stop.set()
        break
  conn.commit()
  cur.close()
  return timediffs

def timeit(fn):
  st = time.perf_counter()
  fn()
  return time.perf_counter() - st

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)
  conn = db_connection()
  cur = conn.cursor()
  row_count = cur.execute(f"select count(*) from '{TABLE_NAME}'").fetchone()[0]
  conn.commit()
  cur.close()
  offsets = range(0, row_count, PAGE_SIZE)
  with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    timediffs = sum(tqdm(pool.imap(process_replay, offsets), total=len(offsets)), [])
    pool.close()
    pool.join()
  better, worse = partition(sorted(timediffs, key=lambda x: x[1]-x[2]),lambda x:x[1]<x[2])
  for (name, comp_time, master) in better[:10]:print(f"better: {master:.3f} - {master-comp_time:.3f} : {name}")
  for (name, comp_time, master) in worse[-1:-10:-1]: print(f"worse:  {master:.3f} + {comp_time-master:.3f} : {name}")