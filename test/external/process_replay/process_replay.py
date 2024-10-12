#!/usr/bin/env python3
# compare kernels created by HEAD against master
import os, multiprocessing, logging, pickle, sqlite3, difflib, functools
from typing import List, Tuple, Union, cast
from tinygrad.engine.schedule import full_ast_rewrite
from tinygrad.helpers import VERSION, Context, ContextVar, colored, db_connection, getenv, tqdm
from tinygrad.codegen.kernel import Kernel
from test.external.process_replay.helpers import print_diff
from tinygrad.renderer import Program

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
if REF == "master": SKIP_PROCESS_REPLAY = True

# *** recreators

def recreate(compare, *args, **kwargs) -> str:
  if isinstance(x:=args[0], Kernel):
    k = Kernel(x.ast, opts=x.opts)
    for opt in x.applied_opts: k.apply_opt(opt)
    # NOTE: replay with the captured renderer, not the one in master
    return k.opts.render(compare.function_name, cast(List,k.to_program().uops))
  return str(full_ast_rewrite(*args, **kwargs))

# *** diff a "good" recreation against the generated version

def diff(offset:int) -> Union[Tuple[int, int], bool]:
  if early_stop.is_set(): return True
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  additions, deletions, changed = 0, 0, 0
  for row in cur.fetchall():
    # try unpickle
    try: args, kwargs, ctx, compare = pickle.loads(row[0])
    except Exception as e:
      logging.warning(f"FAILED TO UNPICKLE OBJECTS {e}")
      if ASSERT_DIFF: return True
      continue
    # try recreate
    try:
      with Context(**{k:v for k,v in ctx[0].items() if k in ContextVar._cache and k != "DEBUG"}): good = recreate(compare, *args, **kwargs)
      compare = compare.src if isinstance(compare, Program) else str(compare)
    except Exception as e:
      logging.warning(f"FAILED TO RECREATE KERNEL {e}")
      if isinstance(x:=args[0], Kernel): args = [x.ast, x.applied_opts] # TODO: can Kernel have __repr__
      for x in args[:-1]: logging.info(x)
      if ASSERT_DIFF: return True
      continue
    # diff kernels
    try: assert compare == good
    except AssertionError:
      logging.info("PROCESS REPLAY DETECTED CHANGE")
      if isinstance(x:=args[0], Kernel): args = [x.ast, x.applied_opts]
      for x in args: logging.info(x)
      print_diff(good, compare)
      changes = list(difflib.unified_diff(good.splitlines(), compare.splitlines()))
      additions += len([x for x in changes if x.startswith("+")])
      deletions += len([x for x in changes if x.startswith("-")])
      if ASSERT_DIFF: return additions, deletions
      if changed > MAX_DIFF_PCT:
        logging.warning(f"detected changes in over {MAX_DIFF_PCT}% of samples. skipping further diff generation.")
        early_stop.set()
        break
  conn.commit()
  cur.close()
  return additions, deletions

# *** generic runner for executing fxn across all rows of a table in parallel

def main() -> None:
  conn = db_connection()
  cur = conn.cursor()
  try: row_count = cur.execute(f"select count(*) from '{TABLE_NAME}'").fetchone()[0]
  except sqlite3.OperationalError:
    logging.warning(f"{TABLE_NAME} isn't accessible in master, did DB_VERSION change?")
    return None
  conn.commit()
  cur.close()
  with multiprocessing.get_context("spawn").Pool(multiprocessing.cpu_count(), maxtasksperchild=16) as pool:
    inputs = list(range(0, row_count, PAGE_SIZE))
    ret: List[Union[bool, Tuple[int, int]]] = list(tqdm(pool.imap_unordered(functools.partial(diff), inputs), total=len(inputs)))
    pool.close()
    pool.join()
    pool.terminate()
    changed = [bool(x[0] or x[1]) if isinstance(x, tuple) else x for x in ret]
    insertion, deletions = [x[0] for x in ret if isinstance(x, tuple)], [x[1] for x in ret if isinstance(x, tuple)]
    logging.info(f"{sum(changed)} kernels changed")
    if sum(insertion) != 0: logging.info(colored(f"{sum(insertion)} insertions(+)", "green"))
    if sum(deletions) != 0: logging.info(colored(f"{sum(deletions)} deletions(-)", "red"))
    if any(changed) and ASSERT_DIFF: raise AssertionError("process replay detected changes")

# *** main loop

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)

  try: main()
  except Exception as e:
    if ASSERT_DIFF: raise e
    logging.error(f"diff err {e}")
