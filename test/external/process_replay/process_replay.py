#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, multiprocessing, os, logging, sqlite3
from typing import List
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, getenv, tqdm
from tinygrad.ops import LazyOp

PAGE_SIZE = 100
REF = os.getenv("GITHUB_REF_NAME", "")
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
TABLE_NAME = f"process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}_{VERSION}"
REF_TABLE_NAME = f"process_replay_master_{VERSION}"
ASSERT_DIFF = getenv("ASSERT_PROCESS_REPLAY", int((k:="[run_process_replay]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)))
SKIP_PROCESS_REPLAY = (k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "") or REF == "master"
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format='%(message)s')

def process_replay(offset:int, ref_schedule:List[LazyOp]):
  if early_stop.is_set(): return
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    ast, applied_opts = None, None
    # try unpickle and linearize
    try:
      ast, opts, applied_opts, name, compare_src, ctx = pickle.loads(row[0])
      with Context(**{k:v for k,v in ctx.items() if k in ContextVar._cache and k != "DEBUG"}):
        k = Kernel(ast, opts=opts)
        for opt in applied_opts: k.apply_opt(opt)
        good_src = k.opts.render(name, k.linearize().uops)
    except Exception as e:
      logging.warning("FAILED TO RECREATE KERNEL")
      logging.info(ast)
      logging.info(applied_opts)
      logging.info(e)
      if ASSERT_DIFF: raise e
      continue
    # try compare
    if getenv("COMPARE_SCHEDULE") and ast not in ref_schedule:
      with Context(**{k:v for k,v in ctx.items() if k in ContextVar._cache and k != "DEBUG"}):
        print(opts.render(name, Kernel(ast, opts=opts).linearize().uops))
      continue
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
        logging.warning(f"detected changes in over {MAX_DIFF_PCT}% of kernels. skipping further diff generation.")
        early_stop.set()
        break
  conn.commit()
  cur.close()

def get_ref_schedule(offset:int, ref_schedule):
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{REF_TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  for row in cur.fetchall(): ref_schedule.append(pickle.loads(row[0])[0])

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)
  conn = db_connection()
  cur = conn.cursor()
  ref_schedule = multiprocessing.Manager().list()
  if getenv("COMPARE_SCHEDULE"):
    row_count = cur.execute(f"select count(*) from '{REF_TABLE_NAME}'").fetchone()[0]
    processes = []
    for i in tqdm(range(0, row_count, PAGE_SIZE)):
      processes.append(p:=multiprocessing.Process(target=get_ref_schedule, args=(i, ref_schedule)))
      p.start()
    for p in processes: p.join()
  try: row_count = cur.execute(f"select count(*) from '{TABLE_NAME}'").fetchone()[0]
  except sqlite3.OperationalError:
    logging.warning(f"{TABLE_NAME} isn't accessible in master, did DB_VERSION change?")
    exit(0)
  conn.commit()
  cur.close()
  processes = []
  for i in tqdm(range(0, row_count, PAGE_SIZE)):
    processes.append(p:=multiprocessing.Process(target=process_replay, args=(i, ref_schedule)))
    p.start()
  for p in processes: p.join()
