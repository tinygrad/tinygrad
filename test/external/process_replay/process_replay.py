#!/usr/bin/env python3
# compare kernels created by HEAD against master
import os, multiprocessing, logging, pickle, sqlite3
from typing import Callable, List, cast
from tinygrad.helpers import VERSION, Context, ContextVar, db_connection, getenv, tqdm
from tinygrad.codegen.kernel import Kernel
from test.external.process_replay.helpers import print_diff

# *** process replay settings

# internal
PAGE_SIZE = 100
REF = os.getenv("GITHUB_REF_NAME", "")
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
RUN_ID = os.getenv("GITHUB_RUN_ID", "HEAD")
TABLE_NAME = f"process_replay_{RUN_ID}_{getenv('GITHUB_RUN_ATTEMPT')}_{VERSION}"
os.environ["RUN_PROCESS_REPLAY"] = "0"
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format="%(message)s")

# user config
ASSERT_DIFF = getenv("ASSERT_PROCESS_REPLAY", int((k:="[run_process_replay]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)))
SKIP_PROCESS_REPLAY = (k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")
if REF == "master": SKIP_PROCESS_REPLAY = True

# *** differs

def diff_schedule(offset:int) -> bool:
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM 'schedule_diff_{VERSION}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    changed += 1
    buf, asts = pickle.loads(row[0])
    if len(asts) == 1:
      logging.info(f"{buf} was folded")
      logging.info(asts[0])
    else: print_diff(asts[0], asts[1])
  return bool(changed)

def diff_kernel(offset:int) -> bool:
  if early_stop.is_set(): return True
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    # try unpickle
    try: ast, opts, applied_opts, name, compare_src, ctx = pickle.loads(row[0])
    except Exception as e:
      logging.warning(f"FAILED TO UNPICKLE OBJECTS {e}")
      if ASSERT_DIFF: return True
      continue
    # try linearize
    try:
      with Context(**{k:v for k,v in ctx.items() if k in ContextVar._cache and k != "DEBUG"}):
        k = Kernel(ast, opts=opts)
        for opt in applied_opts: k.apply_opt(opt)
        # NOTE: replay with the captured renderer, not the one in master
        good_src = k.opts.render(name, cast(List,k.to_program().uops))
    except Exception as e:
      logging.warning(f"FAILED TO RECREATE KERNEL {e}")
      logging.info(ast)
      logging.info(applied_opts)
      if ASSERT_DIFF: return True
      continue
    # diff kernels
    try: assert compare_src == good_src
    except AssertionError:
      changed += 1
      logging.info("PROCESS REPLAY DETECTED CHANGE")
      logging.info(ast)
      logging.info(applied_opts)
      print_diff(good_src, compare_src)
      if ASSERT_DIFF: return True
      if changed > MAX_DIFF_PCT:
        logging.warning(f"detected changes in over {MAX_DIFF_PCT}% of kernels. skipping further diff generation.")
        early_stop.set()
        break
  conn.commit()
  cur.close()
  return bool(changed)

# *** differ runners with multiprocessing

def _run_differ(row_count:int, differ:Callable[[int], bool]) -> None:
  with multiprocessing.get_context("spawn").Pool(multiprocessing.cpu_count(), maxtasksperchild=16) as pool:
    inputs = list(range(0, row_count, PAGE_SIZE))
    changed: List[bool] = list(tqdm(pool.imap_unordered(differ, inputs), total=len(inputs)))
    pool.close()
    pool.join()
    pool.terminate()
    if any(changed) and ASSERT_DIFF: raise AssertionError("process replay detected changes")

def process_replay_schedule() -> None:
  conn = db_connection()
  cur = conn.cursor()
  try: has_diff = cur.execute(f"select name from sqlite_master where type='table' and name='schedule_diff_{VERSION}'").fetchone()
  except sqlite3.OperationalError:
    logging.warning(f"schedule_diff_{VERSION} isn't accessible in master, did DB_VERSION change?")
    return
  if has_diff:
    row_count = cur.execute(f"select count(*) from 'schedule_diff_{VERSION}'").fetchone()[0]
    if row_count != 0: logging.info("***** schedule diff")
    conn.commit()
    cur.close()
    _run_differ(row_count, diff_schedule)

def process_replay_kernel() -> None:
  conn = db_connection()
  cur = conn.cursor()
  try: row_count = cur.execute(f"select count(*) from '{TABLE_NAME}'").fetchone()[0]
  except sqlite3.OperationalError:
    logging.warning(f"{TABLE_NAME} isn't accessible in master, did DB_VERSION change?")
    return None
  conn.commit()
  cur.close()
  _run_differ(row_count, diff_kernel)

# *** main loop

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)

  logging.info("***** schedule diff")
  try: process_replay_schedule()
  except Exception as e:
    if ASSERT_DIFF: raise e
    logging.error(f"schedule diff err {e}")

  logging.info("***** kernel diff")
  try: process_replay_kernel()
  except Exception as e:
    if ASSERT_DIFF: raise e
    logging.error(f"kernel diff err {e}")
