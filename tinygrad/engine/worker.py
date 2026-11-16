import multiprocessing, multiprocessing.util, multiprocessing.pool, atexit, signal, sys, os, threading, contextlib, subprocess, importlib
from multiprocessing.context import SpawnContext, SpawnProcess
from tinygrad.helpers import Context, getenv, PARALLEL, BEAM

# generic pool of worker processes for parallel compilation, shared by kernel lowering and BEAM search

# workers should not open devices and should ignore ctrl c and should not launch VIZ
def _init_worker():
  Context(ALLOW_DEVICE_USAGE=0, VIZ=0, TRACK_MATCH_STATS=0).__enter__()
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  # (WORKER_NICE: below the processes they compile for. off: Qwen C2's start-up on tr4 was 2.3 s faster without, 126.8 vs 129.1 s)
  if (nice:=getenv("WORKER_NICE", 0)) and hasattr(os, "nice"): os.nice(nice)

# spawn reimports the user's __main__ before _init_worker, replaying top-level code and potentially creating pools recursively.
# A spec named "__main__" makes multiprocessing.spawn skip that import.
_spawn_lock = threading.Lock()
@contextlib.contextmanager
def _without_main():
  main = sys.modules["__main__"]
  with _spawn_lock:
    saved = main.__spec__
    try:
      main.__spec__ = importlib.machinery.ModuleSpec("__main__", None)
      yield
    finally: main.__spec__ = saved

# spawn execs with the live environ, other threads changing os.environ (pytest) can break it: https://github.com/python/cpython/issues/157613
def _spawnv_passfds(path, args, passfds):
  (p:=subprocess.Popen(args, executable=path, pass_fds=passfds, env=getattr(os.environ, "_data").copy())).returncode = 0 # multiprocessing reaps it
  return p.pid
multiprocessing.util.spawnv_passfds = _spawnv_passfds

class _WorkerProcess(SpawnProcess):
  @staticmethod
  def _Popen(process_obj):
    with _without_main(): return SpawnProcess._Popen(process_obj)

class _WorkerContext(SpawnContext): Process = _WorkerProcess

worker_pool = None
def get_worker_pool():
  global worker_pool
  if multiprocessing.current_process().daemon or PARALLEL == 0: return None
  if worker_pool is None:
    # (BEAM's workers are replaced every 16 tasks (their memory); compiling's live on: a new one starts cold (its pattern matchers compiled again))
    worker_pool = _WorkerContext().Pool(PARALLEL.value, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16 if BEAM.value else 0) or None)
    @atexit.register
    def close_pool(pool=worker_pool): pool.close()
  return worker_pool

def terminate_worker_pool():
  global worker_pool
  if worker_pool is not None: worker_pool.terminate()
  worker_pool = None

def kill_worker_pool():
  """at a stop (SIGTERM to the process group: the workers die with it): the workers are killed, and the pool's exit finalizer is skipped. It
  would wait forever for the queue lock an idle worker held when it died (Pool._help_stuff_finish), and the process would never end"""
  global worker_pool
  if worker_pool is None: return
  # (first its handler stops replacing dead workers: else every killed worker comes back, a fresh spawn the join below waits for)
  worker_pool._worker_handler._state = multiprocessing.pool.TERMINATE  # type: ignore[attr-defined]
  worker_pool._change_notifier.put(None)  # type: ignore[attr-defined]
  worker_pool._worker_handler.join(5)  # type: ignore[attr-defined]
  for p in (ps:=list(worker_pool._pool)): p.kill()  # type: ignore[attr-defined]
  for p in ps: p.join(1)
  worker_pool._terminate.cancel()  # type: ignore[attr-defined]
  worker_pool = None
