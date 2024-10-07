#!/usr/bin/env python3
# compare kernels created by HEAD against master
import os, multiprocessing, logging, pickle, sqlite3, difflib
from typing import Callable, List, Tuple, Union, cast
from tinygrad.engine.schedule import full_ast_rewrite
from tinygrad.helpers import VERSION, Context, ContextVar, colored, db_connection, getenv, tqdm
from tinygrad.codegen.kernel import Kernel
from test.external.process_replay.helpers import print_diff

# *** process replay settings

# internal
PAGE_SIZE = 100
REF = os.getenv("GITHUB_REF_NAME", "")
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
TABLE_NAME = f"process_replay_{VERSION}"
os.environ["RUN_PROCESS_REPLAY"] = "0"
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format="%(message)s")

# user config
ASSERT_FLAGS = {"[pr]", "[run_process_replay]"}
ASSERT_DIFF = int(any(flag in os.getenv("COMMIT_MESSAGE", flag) or flag in os.getenv("PR_TITLE", flag) for flag in ASSERT_FLAGS))
if not getenv("ASSERT_PROCESS_REPLAY", 1): ASSERT_DIFF = 0
SKIP_PROCESS_REPLAY = (k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")
COMPARE_SCHEDULE = getenv("COMPARE_SCHEDULE", 1)
if REF == "master": SKIP_PROCESS_REPLAY = True

# *** differs

def diff_schedule(offset:int) -> bool:
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM 'schedule_{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    # try unpickle
    try: raw_ast, ctx, compare_ast = pickle.loads(row[0])
    except Exception as e:
      logging.warning(f"FAILED TO UNPICKLE OBJECTS {e}")
      if ASSERT_DIFF: return True
      continue
    # try full_ast_rewrite
    try: good_ast = full_ast_rewrite(raw_ast, ctx)
    except Exception as e:
      logging.warning(f"FAILED TO DO AST REWRITE {e}")
      logging.info(raw_ast)
      logging.info(ctx)
      if ASSERT_DIFF: return True
      continue
    # diff asts
    try: assert compare_ast == good_ast
    except AssertionError:
      logging.info("PROCESS REPLAY DETECTED CHANGE")
      logging.info(raw_ast)
      logging.info(ctx)
      print_diff(good_ast, compare_ast)
  return bool(changed)

def diff_kernel(offset:int) -> Union[Tuple[int, int], bool]:
  if early_stop.is_set(): return True
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM 'kernel_{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  additions, deletions, changed = 0, 0, 0
  for row in cur.fetchall():
    # try unpickle
    try: ast, opts, applied_opts, name, compare_src, ctx = pickle.loads(row[0])
    except Exception as e:
      logging.warning(f"FAILED TO UNPICKLE OBJECTS {e}")
      if ASSERT_DIFF: return True
      continue
    # try linearize
    try:
      with Context(**{k:v for k,v in ctx.ctx_vars.items() if k in ContextVar._cache and k != "DEBUG"}):
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
      logging.info("PROCESS REPLAY DETECTED CHANGE")
      logging.info(ast)
      logging.info(applied_opts)
      logging.info(ctx.loc)
      print_diff(good_src, compare_src)
      changes = list(difflib.unified_diff(str(good_src).splitlines(), str(compare_src).splitlines()))
      additions += len([x for x in changes if x.startswith("+")])
      deletions += len([x for x in changes if x.startswith("-")])
      if ASSERT_DIFF: return additions, deletions
      if changed > MAX_DIFF_PCT:
        logging.warning(f"detected changes in over {MAX_DIFF_PCT}% of kernels. skipping further diff generation.")
        early_stop.set()
        break
  conn.commit()
  cur.close()
  return additions, deletions

# *** generic runner for executing fxn across all rows of a table in parallel

def _pmap(row_count:int, fxn:Callable[[int], Union[bool, Tuple[int, int]]], maxtasksperchild:int=16) -> None:
  with multiprocessing.get_context("spawn").Pool(multiprocessing.cpu_count(), maxtasksperchild=maxtasksperchild) as pool:
    inputs = list(range(0, row_count, PAGE_SIZE))
    ret: List[Union[bool, Tuple[int, int]]] = list(tqdm(pool.imap_unordered(fxn, inputs), total=len(inputs)))
    pool.close()
    pool.join()
    pool.terminate()
    changed = [bool(x[0] or x[1]) if isinstance(x, tuple) else x for x in ret]
    insertion, deletions = [x[0] for x in ret if isinstance(x, tuple)], [x[1] for x in ret if isinstance(x, tuple)]
    logging.info(f"{sum(changed)} kernels changed")
    if len(insertion) != 0: logging.info(colored(f"{sum(insertion)} insertions(+)", "green"))
    if len(deletions) != 0: logging.info(colored(f"{sum(deletions)} deletions(-)", "red"))
    if any(changed) and ASSERT_DIFF: raise AssertionError("process replay detected changes")

# *** process replay parallel differ runners

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
    _pmap(row_count, diff_schedule)

def process_replay_kernel() -> None:
  conn = db_connection()
  cur = conn.cursor()
  try: row_count = cur.execute(f"select count(*) from 'kernel_{TABLE_NAME}'").fetchone()[0]
  except sqlite3.OperationalError:
    logging.warning(f"kernel_{TABLE_NAME} isn't accessible in master, did DB_VERSION change?")
    return None
  conn.commit()
  cur.close()
  _pmap(row_count, diff_kernel)

# *** main loop

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)

  if COMPARE_SCHEDULE:
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
