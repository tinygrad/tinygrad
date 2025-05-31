#!/usr/bin/env python3
# compare kernels created by HEAD against master
import os, multiprocessing, logging, pickle, sqlite3, difflib, warnings, itertools
from typing import Callable, cast
from tinygrad.helpers import VERSION, Context, ContextVar, colored, db_connection, getenv, tqdm, to_function_name
from tinygrad.engine.grouper import get_kernelize_map
from tinygrad.codegen.kernel import Kernel
from tinygrad.uop.ops import UOp, Ops

# *** process replay settings

# internal
PAGE_SIZE = getenv("PAGE_SIZE", 100)
REF = os.getenv("GITHUB_REF_NAME", "")
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
TABLE_NAME = f"process_replay_{VERSION}"
os.environ["CAPTURE_PROCESS_REPLAY"] = "0"
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format="%(message)s")
MAX_LINES = 500
def trunc_log(x):
  if len(lines:=repr(x).splitlines()) > MAX_LINES: lines = lines[:MAX_LINES]+[f"WARN: truncated string with {len(lines)} lines"]
  logging.info("\n".join(lines))

# user config
ASSERT_DIFF = int((flag:="[pr]") in os.getenv("COMMIT_MESSAGE", flag) or flag in os.getenv("PR_TITLE", flag))
if not getenv("ASSERT_PROCESS_REPLAY", 1): ASSERT_DIFF = 0
SKIP_PROCESS_REPLAY = (k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")
if REF == "master": SKIP_PROCESS_REPLAY = True
class ProcessReplayWarning(Warning): pass

# *** recreators

def recreate_sched(ret:dict[UOp, UOp], big_sink:UOp) -> bool:
  UOp.unique_num = itertools.count(max([u.arg for u in big_sink.toposort() if u.op is Ops.UNIQUE], default=0)+1)
  new_sink = big_sink.substitute(get_kernelize_map(big_sink))
  new_asts = "\n".join([repr(u.arg.ast) for u in new_sink.toposort() if u.op is Ops.KERNEL])
  old_asts = "\n".join([repr(u.arg.ast) for u in ret[big_sink].toposort() if u.op is Ops.KERNEL])
  return new_asts, old_asts

def recreate_kernel(ret:Kernel, s:Kernel, name_override=None, ast_transform=None) -> bool:
  k = Kernel(s.ast, opts=s.opts).apply_opts(s.applied_opts)
  # NOTE: replay with the captured renderer, not the one in master
  new_src = k.opts.render(cast(list,k.to_program(to_function_name(ret.name)).uops))
  old_src = ret.opts.render(ret.uops)
  return new_src, old_src

differs: dict[str, Callable[..., tuple[str, str]]] = {"get_kernelize_map":recreate_sched, "linearize":recreate_kernel}

# *** replay all rows starting from the offset and print generated diff

def diff(offset:int) -> None:
  if ASSERT_DIFF: warnings.filterwarnings("error", category=ProcessReplayWarning)
  if early_stop.is_set(): return None
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  changed = 0
  for row in cur.fetchall():
    if changed > MAX_DIFF_PCT:
      warnings.warn(f"detected changes in over {MAX_DIFF_PCT}%. skipping further diff generation.", ProcessReplayWarning)
      early_stop.set()
      break
    # try unpickle and unpack
    try: name, args, kwargs, ctx_vals, loc, ret = pickle.loads(row[0])
    except Exception as e:
      changed += 1
      warnings.warn(f"FAILED TO UNPICKLE OBJECTS {e}", ProcessReplayWarning)
      continue
    if (differ:=differs.get(name)) is None: continue
    # try recreate
    try:
      ctx_vars = {k:v.value for k,v in ctx_vals.items() if k != "DEBUG" and (var:=ContextVar._cache.get(k)) is not None and var.value != v.value}
      with Context(**ctx_vars): good, compare = differ(ret, *args, **kwargs)
    except Exception as e:
      changed += 1
      if ctx_vars: logging.info(ctx_vars)
      for x in args: trunc_log(x)
      for k,v in kwargs.items(): trunc_log((k, v))
      warnings.warn(f"FAILED TO RECREATE KERNEL {e}", ProcessReplayWarning)
      continue
    # diff kernels
    try: assert good == compare
    except AssertionError:
      changed += 1
      if ctx_vars: logging.info(ctx_vars)
      for x in args: trunc_log(x)
      for k,v in kwargs.items(): trunc_log((k, v))
      changes = list(difflib.unified_diff(good.splitlines(), compare.splitlines()))
      logging.info("\n".join(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None) for line in changes))
      warnings.warn("PROCESS REPLAY DETECTED CHANGE", ProcessReplayWarning)
  conn.commit()
  cur.close()

# *** generic runner for executing fxn across all rows of a table in parallel

def _pmap(maxtasksperchild:int=16) -> None:
  conn = db_connection()
  cur = conn.cursor()
  try: row_count = cur.execute(f"select count(*) from '{TABLE_NAME}'").fetchone()[0]
  except sqlite3.OperationalError:
    warnings.warn(f"{TABLE_NAME} isn't accessible in master, did DB_VERSION change?", ProcessReplayWarning)
    return None
  conn.commit()
  cur.close()
  with multiprocessing.get_context("spawn").Pool(multiprocessing.cpu_count(), maxtasksperchild=maxtasksperchild) as pool:
    inputs = list(range(0, row_count, PAGE_SIZE))
    list(tqdm(pool.imap_unordered(diff, inputs), total=len(inputs)))
    pool.close()
    pool.join()
    pool.terminate()

# *** main loop

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)

  print(f"running process replay with {ASSERT_DIFF=}")
  try: _pmap()
  except ProcessReplayWarning: exit(1)
  except Exception as e:
    if ASSERT_DIFF: raise e
    logging.error(f"diff err {e}")
