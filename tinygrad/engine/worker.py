import multiprocessing, atexit, signal
from tinygrad.helpers import Context, getenv, PARALLEL

# generic pool of worker processes for parallel compilation, shared by kernel lowering and BEAM search

# workers should not open devices and should ignore ctrl c and should not launch VIZ
def _init_worker():
  Context(ALLOW_DEVICE_USAGE=0, VIZ=0, TRACK_MATCH_STATS=0).__enter__()
  signal.signal(signal.SIGINT, signal.SIG_IGN)

worker_pool = None
def get_worker_pool():
  global worker_pool
  if multiprocessing.current_process().daemon or PARALLEL == 0: return None
  if worker_pool is None:
    worker_pool = multiprocessing.get_context("spawn").Pool(PARALLEL.value, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16))
    @atexit.register
    def close_pool(pool=worker_pool): pool.close()
  return worker_pool

def terminate_worker_pool():
  global worker_pool
  if worker_pool is not None: worker_pool.terminate()
  worker_pool = None
