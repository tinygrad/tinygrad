"""Bounded NumPy prefetch; Tensor construction stays on the consumer thread."""
from collections import deque
import threading
import time


class PrefetchArrays:
  """Single consumer, explicit context/close required; depth counts queued+inflight.

  The worker only advances the NumPy iterator; Tensor construction stays on the consumer thread.
  The caller owns an immutable dataset. close() joins in-flight I/O up to close_timeout,
  since Python cannot cancel a blocked read.
  """
  def __init__(self, iterator, *, depth=2, item_bytes_limit=1<<20, close_timeout=5.0):
    assert type(depth) is int and 1 <= depth <= 4
    assert type(item_bytes_limit) is int and item_bytes_limit > 0 and close_timeout > 0
    self.iterator, self.depth, self.item_bytes_limit, self.close_timeout = iter(iterator),depth,item_bytes_limit,close_timeout
    self.owner = threading.get_ident()
    self.cv, self.queue = threading.Condition(),deque()
    self.stop = self.done = self.inflight = False
    self.error = None
    self.produced = self.consumed = self.max_queued = self.max_queued_bytes = 0
    self.produce_s = self.wait_s = 0.0
    self.thread = threading.Thread(target=self._run,name='gptoss-numpy-prefetch',daemon=True)
    self.thread.start()

  def _run(self):
    import numpy as np
    try:
      while True:
        with self.cv:
          self.cv.wait_for(lambda:self.stop or len(self.queue)<self.depth)
          if self.stop: return
          self.inflight = True
        started = time.perf_counter()
        try: value = next(self.iterator)
        except StopIteration: return
        finally: self.produce_s += time.perf_counter()-started
        assert type(value) is np.ndarray and value.dtype.kind != 'O', 'worker must return plain numerical ndarray'
        assert value.nbytes <= self.item_bytes_limit, 'prefetch item exceeds reviewed host-memory bound'
        with self.cv:
          self.inflight = False
          if self.stop: return
          self.queue.append(value)
          self.produced += 1
          self.max_queued = max(self.max_queued,len(self.queue))
          self.max_queued_bytes = max(self.max_queued_bytes,sum(x.nbytes for x in self.queue))
          self.cv.notify_all()
        del value
    except BaseException as error:
      with self.cv: self.error = error.with_traceback(None)
    finally:
      try:
        if hasattr(self.iterator,'close'): self.iterator.close()
      finally:
        with self.cv:
          self.iterator = None
          self.inflight = False
          self.done = True
          self.cv.notify_all()

  def __iter__(self): return self

  def __next__(self):
    assert threading.get_ident() == self.owner, 'consume on creator thread'
    started = time.perf_counter()
    with self.cv:
      self.cv.wait_for(lambda:self.stop or self.queue or self.done)
      self.wait_s += time.perf_counter()-started
      if self.stop: raise StopIteration
      if self.queue:
        value = self.queue.popleft()
        self.consumed += 1
        self.cv.notify_all()
        return value
      error, self.error = self.error,None
    self.close()
    if error is not None: raise error
    raise StopIteration

  def close(self):
    assert threading.get_ident() == self.owner, 'close on creator thread'
    with self.cv:
      self.stop = True
      self.queue.clear()
      self.error = None
      self.cv.notify_all()
    self.thread.join(self.close_timeout)
    if self.thread.is_alive(): raise RuntimeError('prefetch worker did not finish current batch before close deadline')

  def __enter__(self): return self
  def __exit__(self,*_): self.close()

  def next_tensor(self):
    """The original Tensor construction stays on the consuming/main thread."""
    from tinygrad import Tensor
    return Tensor(next(self),device='NPY')

  def statistics(self):
    with self.cv:
      return dict(produced=self.produced,consumed=self.consumed,max_queued=self.max_queued,
        max_queued_bytes=self.max_queued_bytes,produce_s=self.produce_s,consumer_wait_s=self.wait_s,
        stopped=self.stop,done=self.done,worker_alive=self.thread.is_alive())


class PrefetchTensors:
  """Keep every Tensor construction on the training thread; close explicitly."""
  def __init__(self, iterator, *, depth=2):
    assert threading.get_ident() == threading.main_thread().ident
    self.arrays = PrefetchArrays(iterator, depth=depth)

  def __iter__(self): return self

  def __next__(self): return self.arrays.next_tensor()

  def close(self): self.arrays.close()
