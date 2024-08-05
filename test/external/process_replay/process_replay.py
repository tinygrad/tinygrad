#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, multiprocessing, os, logging, sqlite3, requests, io, zipfile
from tabulate import tabulate
from datetime import datetime
from typing import Dict, List, cast
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import Device
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, getenv, temp, tqdm
from tinygrad.ops import LazyOp

# *** process replay settings
PAGE_SIZE = 100
REF = os.getenv("GITHUB_REF_NAME", "")
MAX_DIFF_PCT = getenv("PROCESS_REPLAY_MAX_DIFF_PCT", 20)
RUN_ID = os.getenv("GITHUB_RUN_ID", "HEAD")
TABLE_NAME = f"process_replay_{RUN_ID}_{getenv('GITHUB_RUN_ATTEMPT')}_{VERSION}"
ASSERT_DIFF = getenv("ASSERT_PROCESS_REPLAY", int((k:="[run_process_replay]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)))
if REF == "master": ASSERT_DIFF = False
COMPARE_SCHEDULE = getenv("COMPARE_SCHEDULE", int((k:="[compare_schedule]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")))
SKIP_PROCESS_REPLAY = (k:="[skip_process_replay]") in os.getenv("COMMIT_MESSAGE", "") or k in os.getenv("PR_TITLE", "")
TEMP_DIR = temp("process_replay")
early_stop = multiprocessing.Event()
logging.basicConfig(level=logging.INFO, format='%(message)s')
# *** github settings
BASE_URL = f"https://api.github.com/repos/{os.getenv('GITHUB_REPOSITORY', 'tinygrad/tinygrad')}"
GH_HEADERS = {"Authorization": f"Bearer {os.getenv('GH_TOKEN', '')}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

def diff_kernel(offset:int, ref_schedule:List[LazyOp], kernel_changed):
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
        # NOTE: replay with the captured renderer, not the one in master
        good_src = k.opts.render(name, cast(List,k.to_program().uops))
    except Exception as e:
      logging.warning("FAILED TO RECREATE KERNEL")
      logging.info(ast)
      logging.info(applied_opts)
      logging.info(e)
      kernel_changed.value = True
      if ASSERT_DIFF: raise e
      continue
    # try compare
    if COMPARE_SCHEDULE and ast not in ref_schedule:
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
      kernel_changed.value = True
      if ASSERT_DIFF: raise e
      if changed > MAX_DIFF_PCT:
        logging.warning(f"detected changes in over {MAX_DIFF_PCT}% of kernels. skipping further diff generation.")
        early_stop.set()
        break
  conn.commit()
  cur.close()

def get_ref_schedule(offset:int, ref_table_name:str, ref_schedule):
  conn = sqlite3.connect("/tmp/process_replay/process_replay.db")
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{ref_table_name}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  for row in cur.fetchall(): ref_schedule.append(pickle.loads(row[0])[0])
  conn.commit()
  cur.close()

def download_artifact(run_id:str, name:str, dest:str):
  res = requests.get(f"{BASE_URL}/actions/runs/{run_id}/artifacts?name={name}", headers=GH_HEADERS)
  assert res.status_code == 200, f"download failed {res.status_code} {res.json()}"
  download_url = res.json()["artifacts"][0]["archive_download_url"]
  res = requests.get(download_url, headers=GH_HEADERS)
  assert res.status_code == 200, f"download failed {res.status_code}"
  with io.BytesIO(res.content) as zip_content:
    with zipfile.ZipFile(zip_content, "r") as zip_ref: zip_ref.extractall(dest)

def _get_times(data) -> Dict[str, float]:
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
            "testamdbenchmark": "tinybox red", "testmoreamdbenchmark": "tinybox red Training"}[os.environ["GITHUB_JOB"]]
    compare_jobs = requests.get(f"{BASE_URL}/actions/runs/{RUN_ID}/jobs", headers=GH_HEADERS).json()["jobs"]
    compare_job = next(j for j in compare_jobs if j["name"] == f"{name} Benchmark")
    ref_runs = requests.get(f"{BASE_URL}/actions/workflows/benchmark.yml/runs?per_page=1&branch=master&status=success", headers=GH_HEADERS).json()
    ref_jobs = requests.get(f"{BASE_URL}/actions/runs/{ref_runs['workflow_runs'][0]['id']}/jobs").json()["jobs"]
    ref_job = next(j for j in ref_jobs if j["name"] == f"{name} Benchmark")
    logging.info(f"comparing speed for {compare_job['id']} against {ref_job['id']}")
    compare_tms = _get_times(compare_job)
    ref_tms = _get_times(ref_job)
    diff = [[k, f"{v}s", f"{compare_tms[k]}s", f"{(((v-compare_tms[k])/v)*100):7.2f}%"] for k,v in ref_tms.items() if v>0]
    logging.info(tabulate(diff, headers=["job", "master", "compare", "diff"]))

  # *** schedule diff
  ref_schedule = multiprocessing.Manager().list()
  if COMPARE_SCHEDULE:
    logging.info("fetching process replay reference")
    # TODO: make this run_id dynamic
    download_artifact("10253655789", f"process_replay_{os.getenv('BACKEND'), Device.DEFAULT}.db", f"{TEMP_DIR}/schedule")
    ref_conn = sqlite3.connect(f"{TEMP_DIR}/schedule/process_replay.db")
    ref_table_name = ref_conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchone()[0]
    row_count = ref_conn.execute(f"select count(*) from '{ref_table_name}'").fetchone()[0]
    processes = []
    for i in tqdm(range(0, row_count, PAGE_SIZE)):
      processes.append(p:=multiprocessing.Process(target=get_ref_schedule, args=(i, ref_table_name, ref_schedule)))
      p.start()
    for p in processes: p.join()
    ref_conn.close()
  conn = db_connection()
  cur = conn.cursor()

  # *** kernel diff
  try: row_count = cur.execute(f"select count(*) from '{TABLE_NAME}'").fetchone()[0]
  except sqlite3.OperationalError:
    logging.warning(f"{TABLE_NAME} isn't accessible in master, did DB_VERSION change?")
    exit(0)
  conn.commit()
  cur.close()
  processes = []
  changed = multiprocessing.Manager().Value('b', False)
  for i in tqdm(range(0, row_count, PAGE_SIZE)):
    processes.append(p:=multiprocessing.Process(target=diff_kernel, args=(i, ref_schedule, changed)))
    p.start()
  for p in processes: p.join()
  if changed.value and ASSERT_DIFF: raise Exception("process replay detected changes")

if __name__ == "__main__":
  if SKIP_PROCESS_REPLAY:
    logging.info("skipping process replay.")
    exit(0)
  try: process_replay()
  except Exception as e:
    if ASSERT_DIFF: raise e
