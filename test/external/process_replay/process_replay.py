#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, multiprocessing, os, logging, sqlite3, requests
from tabulate import tabulate
from datetime import datetime
from typing import Dict, List, cast
from test.external.process_replay.utils import print_diff
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, getenv, tqdm

# *** process replay settings
PAGE_SIZE = 100
REF = os.getenv("GITHUB_REF_NAME", "")
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
RUN_ID = os.getenv("GITHUB_RUN_ID", "HEAD")
TABLE_NAME = f"process_replay_{RUN_ID}_{getenv('GITHUB_RUN_ATTEMPT')}_{VERSION}"
ASSERT_DIFF = getenv("ASSERT_PROCESS_REPLAY", int((k:="[run_process_replay]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)))
COMPARE_SCHEDULE = getenv("COMPARE_SCHEDULE", int((k:="[compare_schedule]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")))
SKIP_PROCESS_REPLAY = (k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")
if REF == "master": SKIP_PROCESS_REPLAY = True
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format='%(message)s')
# *** github settings
BASE_URL = f"https://api.github.com/repos/{os.getenv('GITHUB_REPOSITORY', 'tinygrad/tinygrad')}"
GH_HEADERS = {"Authorization": f"Bearer {os.getenv('GH_TOKEN', '')}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

def diff_kernel(offset:int) -> bool:
  if early_stop.is_set(): return True
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
        # NOTE: replay with the captured renderer, not the one in master
        good_src = k.opts.render(name, cast(List,k.to_program().uops))
    except Exception as e:
      logging.warning("FAILED TO RECREATE KERNEL")
      logging.info(ast)
      logging.info(applied_opts)
      logging.info(e)
      if ASSERT_DIFF: return True
      continue
    try: assert compare_src == good_src
    except AssertionError:
      changed += 1
      logging.info("PROCESS REPLAY DETECTED CHANGE")
      logging.info(ast)
      logging.info(applied_opts)
      diff = list(difflib.unified_diff(good_src.splitlines(), compare_src.splitlines()))
      for line in diff:
        logging.info(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
      if ASSERT_DIFF: return True
      if changed > MAX_DIFF_PCT:
        logging.warning(f"detected changes in over {MAX_DIFF_PCT}% of kernels. skipping further diff generation.")
        early_stop.set()
        break
  conn.commit()
  cur.close()
  return bool(changed)

def print_ast_diff(offset:int):
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM 'schedule_diff_{VERSION}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  for row in cur.fetchall():
    buf, asts = pickle.loads(row[0])
    if len(asts) == 1:
      logging.info(f"{buf} was folded")
      logging.info(asts[0])
    else: print_diff(asts[0], asts[1])

def get_step_times(data) -> Dict[str, float]:
  tms: Dict[str, float] = {}
  for step in data["steps"][4:]:
    # last task
    if step["name"] == "Run actions/upload-artifact@v4": break
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    tm = datetime.strptime(step["completed_at"], fmt) - datetime.strptime(step["started_at"], fmt)
    tms[step["name"]] = tm.total_seconds()
  return tms

def process_replay():
  # *** speed diff (for benchmarks)
  if REF == "update_benchmark":
    name = {"testmacbenchmark": "Mac", "testnvidiabenchmark": "tinybox green", "testmorenvidiabenchmark": "tinybox green Training",
            "testamdbenchmark": "tinybox red", "testmoreamdbenchmark": "tinybox red Training",
            "testqualcommbenchmark": "comma Benchmark"}[os.environ["GITHUB_JOB"]]
    compare_jobs = requests.get(f"{BASE_URL}/actions/runs/{RUN_ID}/jobs", headers=GH_HEADERS).json()["jobs"]
    compare_job = next(j for j in compare_jobs if j["name"] == f"{name} Benchmark")
    ref_runs = requests.get(f"{BASE_URL}/actions/workflows/benchmark.yml/runs?per_page=1&branch=master&status=success", headers=GH_HEADERS).json()
    ref_jobs = requests.get(f"{BASE_URL}/actions/runs/{ref_runs['workflow_runs'][0]['id']}/jobs").json()["jobs"]
    ref_job = next(j for j in ref_jobs if j["name"] == f"{name} Benchmark")
    logging.info(f"comparing speed for {compare_job['id']} against {ref_job['id']}")
    compare_tms = get_step_times(compare_job)
    ref_tms = get_step_times(ref_job)
    diff = [[k, f"{v}s", f"{compare_tms[k]}s", f"{(((v-compare_tms[k])/v)*100):7.2f}%"] for k,v in ref_tms.items() if v>0]
    logging.info(tabulate(diff, headers=["job", "master", "compare", "diff"]))

  # *** schedule diff
  if COMPARE_SCHEDULE:
    conn = db_connection()
    cur = conn.cursor()
    try: has_diff = cur.execute(f"select name from sqlite_master where type='table' and name='schedule_diff_{VERSION}'").fetchone()
    except sqlite3.OperationalError:
      logging.warning(f"schedule_diff_{VERSION} isn't accessible in master, did DB_VERSION change?")
      exit(0)
    if has_diff:
      row_count = cur.execute(f"select count(*) from 'schedule_diff_{VERSION}'").fetchone()[0]
      conn.commit()
      cur.close()
      with multiprocessing.get_context("spawn").Pool(multiprocessing.cpu_count(), maxtasksperchild=16) as pool:
        inputs = list(range(0, row_count, PAGE_SIZE))
        list(tqdm(pool.imap_unordered(print_ast_diff, inputs), total=len(inputs)))
        pool.close()
        pool.join()
        pool.terminate()
        if ASSERT_DIFF: raise Exception("kernel process replay detected changes")

  # *** kernel diff
  conn = db_connection()
  cur = conn.cursor()
  try: row_count = cur.execute(f"select count(*) from '{TABLE_NAME}'").fetchone()[0]
  except sqlite3.OperationalError:
    logging.warning(f"{TABLE_NAME} isn't accessible in master, did DB_VERSION change?")
    exit(0)
  conn.commit()
  cur.close()
  with multiprocessing.get_context("spawn").Pool(multiprocessing.cpu_count(), maxtasksperchild=16) as pool:
    inputs = list(range(0, row_count, PAGE_SIZE))
    changed = list(tqdm(pool.imap_unordered(diff_kernel, inputs), total=len(inputs)))
    pool.close()
    pool.join()
    pool.terminate()
    if any(changed) and ASSERT_DIFF: raise Exception("kernel process replay detected changes")

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)
  try: process_replay()
  except Exception as e:
    if ASSERT_DIFF: raise e
